"""
Pipeline Manager - Quản lý toàn bộ luồng xử lý
Single Source of Truth: meeting.json

Pipeline:
[1] Preprocessing → [2] Unified Chunking → [3] Whisper → [4] Diarization →
[5] Speaker Assignment → [6] Merge & Normalize → [7] Gender → 
[7.5] Spell Check (LLM) → [8] meeting.json → [9] LLM Analysis → [10] Export
"""
import os
import json
import time
import sys
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
# Import config
import sys
# ✅ BACKEND_DIR should be 'backend' folder, not 'backend/src'
BACKEND_DIR = Path(__file__).parent.parent  # backend/src/pipeline_manager.py → backend
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from config import (
        PIPELINE_CONFIG, 
        WHISPER_MODEL, 
        WHISPER_BEAM_SIZE, 
        WHISPER_VAD_FILTER, 
        USE_GPU,
        OUTPUT_DIR,
        HF_TOKEN as CONFIG_HF_TOKEN
    )
except ImportError:
    # Fallback config if import fails
    PIPELINE_CONFIG = {
        "chunk_duration_minutes": 10,
        "enable_vad": True,
        "enable_gender": True,
        "enable_spell_check": True,
        "enable_llm": True,
        "min_speakers": None,
        "max_speakers": None
    }
    WHISPER_MODEL = "medium"
    WHISPER_BEAM_SIZE = 5
    WHISPER_VAD_FILTER = True
    USE_GPU = False
    OUTPUT_DIR = None
    CONFIG_HF_TOKEN = None
# ✅ FIX: Import với absolute path từ src
try:
    from src.stage1_preprocessing import AudioPreprocessor
    from src.stage2_chunking import UnifiedChunker
    from src.stage3_whisper import WhisperProcessor
    from src.stage4_diarization import DiarizationProcessor
    from src.stage5_speaker_assignment import SpeakerAssigner
    from src.stage6_merge_normalize import MergeNormalizer
    from src.stage7_gender import GenderClassifier
    from src.stage8_validation import MeetingValidator
    from src.stage9_llm import LLMAnalyzer
except ImportError:
    # ✅ FALLBACK: Try relative import
    try:
        from stage1_preprocessing import AudioPreprocessor
        from stage2_chunking import UnifiedChunker
        from stage3_whisper import WhisperProcessor
        from stage4_diarization import DiarizationProcessor
        from stage5_speaker_assignment import SpeakerAssigner
        from stage6_merge_normalize import MergeNormalizer
        from stage7_gender import GenderClassifier
        from stage8_validation import MeetingValidator
        from stage9_llm import LLMAnalyzer
    except ImportError as e:
        raise ImportError(f"Cannot import stage processors: {e}")

class MeetingPipeline:
    """
    Quản lý toàn bộ pipeline xử lý meeting
    
    Attributes:
        audio_path: Đường dẫn file audio
        output_dir: Thư mục output
        meeting_data: SSoT - chứa tất cả thông tin
        config: Pipeline configuration
    """
    
    def __init__(self, audio_path: str, output_dir: str = None):
        self.audio_path = audio_path
        
        # ✅ Use OUTPUT_DIR from config if not specified
        if output_dir is None:
            if OUTPUT_DIR is not None:
                output_dir = str(OUTPUT_DIR)
            else:
                # Fallback to backend/data/outputs
                src_dir = Path(__file__).parent  # backend/src
                output_dir = str(src_dir.parent / "data" / "outputs")
        
        self.base_output_dir = output_dir  # Base outputs folder
        self.meeting_id = self._generate_meeting_id()
        
        # ✅ Create dedicated folder for this audio file
        audio_folder_name = Path(audio_path).stem  # Get filename without extension
        self.output_dir = os.path.join(output_dir, audio_folder_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # ✅ SSoT - Single Source of Truth
        self.meeting_data = {
            "metadata": {
                "meeting_id": self.meeting_id,
                "audio_file": audio_path,
                "created_at": datetime.now().isoformat(),
                "pipeline_version": "2.0",
                "status": "initializing"
            },
            "preprocessing": {},
            "chunks": [],
            "whisper": {
                "segments": [],
                "words": [],
                "language": None
            },
            "diarization": {
                "timeline": [],
                "speakers": {}
            },
            "speakers": {},  # Gender + metadata
            "segments": [],  # Final segments with all info
            "llm": {
                "summary": None,
                "tasks": [],
                "insights": []
            }
        }
        
        # ✅ Load config from config.py
        self.config = PIPELINE_CONFIG.copy()
        
        # ✅ Stage timing tracking
        self.stage_timings = {}
        
        # ✅ Debug mode: save intermediate stage outputs
        self.debug_mode = self.config.get("debug_mode", False)
        
        print(f"\n{'='*80}")
        print(f"🎯 MEETING PIPELINE INITIALIZED")
        print(f"{'='*80}")
        print(f"   Meeting ID: {self.meeting_id}")
        print(f"   Audio: {Path(audio_path).name}")
        print(f"   Output Folder: {self.output_dir}")
        print(f"{'='*80}\n")
    
    def _generate_meeting_id(self) -> str:
        """Generate unique meeting ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        basename = Path(self.audio_path).stem
        return f"{basename}_{timestamp}"
    
    def _save_meeting_json(self):
        """Save meeting.json (SSoT)"""
        # ✅ Use meeting_id in filename for consistency
        meeting_json_path = os.path.join(self.output_dir, f"{self.meeting_id}_meeting.json")
        
        try:
            print(f"      💾 Saving meeting.json...")
            
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Count data size
            segments_count = len(self.meeting_data.get("whisper", {}).get("segments", []))
            print(f"      📊 Serializing {segments_count} segments...")
            
            with open(meeting_json_path, 'w', encoding='utf-8') as f:
                json.dump(self.meeting_data, f, ensure_ascii=False, indent=2)
            
            print(f"      ✅ meeting.json saved successfully")
            return meeting_json_path
            
        except Exception as e:
            print(f"      ❌ Failed to save meeting.json: {e}", flush=True)
            print(f"      📍 Path: {meeting_json_path}", flush=True)
            import traceback
            traceback.print_exc()
            raise
    
    def _save_stage_output(self, stage_name: str, data: dict, file_type: str = 'json'):
        """Save intermediate stage outputs (only if debug_mode enabled)"""
        if not self.debug_mode:
            return None
        
        try:
            filename = f"{stage_name}.{file_type}"
            output_path = os.path.join(self.output_dir, filename)
            
            if file_type == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            elif file_type == 'txt':
                with open(output_path, 'w', encoding='utf-8') as f:
                    if isinstance(data, dict):
                        for key, value in data.items():
                            f.write(f"{key}: {value}\n")
                    elif isinstance(data, list):
                        for item in data:
                            f.write(f"{item}\n")
                    else:
                        f.write(str(data))
            
            print(f"      💾 Saved: {filename}")
            return output_path
            
        except Exception as e:
            print(f"      ⚠️ Failed to save {stage_name}: {e}", flush=True)
            return None
    
    def _update_status(self, status: str, stage: str = None):
        """Update pipeline status"""
        self.meeting_data["metadata"]["status"] = status
        if stage:
            self.meeting_data["metadata"]["current_stage"] = stage
        self.meeting_data["metadata"]["last_updated"] = datetime.now().isoformat()
        self._save_meeting_json()
    
    def get_meeting_json_path(self) -> str:
        """Get path to meeting.json file"""
        return os.path.join(self.output_dir, f"{self.meeting_id}_meeting.json")
    
    # ==================== STAGE RUNNERS ====================
    
    def run_stage1_preprocessing(self):
        """[1] Audio Preprocessing"""
        self._update_status("processing", "preprocessing")
        
        preprocessor = AudioPreprocessor(self.audio_path)
        result = preprocessor.process(
            enable_vad=self.config["enable_vad"]
        )
        
        # Update SSoT
        self.meeting_data["preprocessing"] = result
        self.preprocessed_audio = result["output_path"]
        
        print(f"   ✅ Preprocessing complete")
        print(f"      - Output: {result['output_path']}")
        print(f"      - Duration: {result['duration']:.1f}s")
        if self.config["enable_vad"]:
            print(f"      - Speech regions: {len(result['vad_regions'])}")
        
        # Save stage output
        self._save_stage_output('stage1_preprocessing', result, 'json')
        self._save_meeting_json()
    
    def run_stage2_chunking(self):
        """[2] Unified Chunking"""
        self._update_status("processing", "chunking")
        
        chunker = UnifiedChunker(self.preprocessed_audio)
        chunks = chunker.create_chunks(
            chunk_duration_minutes=self.config["chunk_duration_minutes"],
            vad_regions=self.meeting_data["preprocessing"].get("vad_regions")
        )
        
        # Update SSoT
        self.meeting_data["chunks"] = chunks
        
        print(f"   ✅ Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"      Chunk {i+1}: [{chunk['start']:.1f}s - {chunk['end']:.1f}s]")
        
        # Save stage output
        self._save_stage_output('stage2_chunks', {'chunks': chunks, 'count': len(chunks)}, 'json')
        self._save_meeting_json()
    
    def run_stage3_whisper(self):
        """[3] Whisper Transcription"""
        self._update_status("processing", "whisper")
        
        # ✅ Use faster-whisper with config
        device_mode = "cuda" if USE_GPU else "cpu"
        
        # Get model settings from config (can be overridden by processing mode)
        whisper_model = self.config.get("whisper_model", WHISPER_MODEL)
        whisper_beam_size = self.config.get("whisper_beam_size", WHISPER_BEAM_SIZE)
        whisper_vad_filter = self.config.get("whisper_vad_filter", WHISPER_VAD_FILTER)
        
        print(f"   🤖 Using Whisper Model: {whisper_model}")
        print(f"   📊 Using Beam Size: {whisper_beam_size}")
        
        whisper = WhisperProcessor(
            model_size=whisper_model,
            device=device_mode,  # ✅ Force device from config
            beam_size=whisper_beam_size,
            vad_filter=whisper_vad_filter
        )
        result = whisper.process(
            audio_path=self.preprocessed_audio,
            chunks=self.meeting_data["chunks"]
        )
        
        # Delete whisper object to free memory/GPU
        del whisper
        import gc
        gc.collect()
        
        # ✅ Save words to TEMP FILE (not memory, not JSON) to avoid OS killing process
        import pickle
        import os
        words_temp_path = os.path.join(self.output_dir, "_temp_words.pkl")
        with open(words_temp_path, 'wb') as f:
            pickle.dump(result["words"], f)
        
        # Update SSoT - but DON'T store words in JSON
        self.meeting_data["whisper"]["segments"] = result["segments"]
        self.meeting_data["whisper"]["words"] = []  # Empty - not saved to disk
        self.meeting_data["whisper"]["language"] = result["language"]
        self.meeting_data["whisper"]["word_count"] = len(result["words"])
        
        print(f"   ✅ Transcription complete")
        print(f"      - Segments: {len(result['segments'])}")
        print(f"      - Words: {len(result['words'])} (saved to temp file)")
        
        # Clear result from memory immediately
        del result
        
        # Save outputs
        stage_output = {
            "segments": self.meeting_data["whisper"]["segments"],
            "word_count": self.meeting_data["whisper"]["word_count"],
            "language": self.meeting_data["whisper"]["language"]
        }
        self._save_stage_output('stage3_whisper', stage_output, 'json')
        self._save_meeting_json()
        
        # Force cleanup and release GPU memory
        import gc
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
    
    def run_stage4_diarization(self):
        """[4] Speaker Diarization"""
        self._update_status("processing", "diarization")
        
        diarizer = DiarizationProcessor()
        result = diarizer.process(
            audio_path=self.preprocessed_audio,
            chunks=self.meeting_data["chunks"],
            min_speakers=self.config["min_speakers"],
            max_speakers=self.config["max_speakers"]
        )
        
        # Update SSoT
        self.meeting_data["diarization"]["timeline"] = result["timeline"]
        self.meeting_data["diarization"]["speakers"] = result["speakers"]
        
        print(f"   ✅ Diarization complete")
        print(f"      - Speakers: {len(result['speakers'])}")
        print(f"      - Timeline entries: {len(result['timeline'])}")
        
        # Save stage output
        self._save_stage_output('stage4_diarization', result, 'json')
        self._save_meeting_json()
    
    def run_stage5_speaker_assignment(self):
        """[5] Speaker Assignment (word-level voting)"""
        self._update_status("processing", "speaker_assignment")
        
        # ✅ Load words from temp file
        import pickle
        import os
        words_temp_path = os.path.join(self.output_dir, "_temp_words.pkl")
        
        if not os.path.exists(words_temp_path):
            print(f"   ⚠️ Warning: Temp words file not found")
            whisper_words = []
        else:
            with open(words_temp_path, 'rb') as f:
                whisper_words = pickle.load(f)
        
        if not whisper_words:
            print(f"   ⚠️ Warning: No whisper words available")
            # Fallback to empty assignment
            segments = self.meeting_data["whisper"]["segments"]
            for seg in segments:
                seg["speaker"] = "UNKNOWN"
            self.meeting_data["segments_raw"] = segments
        else:
            assigner = SpeakerAssigner()
            segments = assigner.assign(
                whisper_segments=self.meeting_data["whisper"]["segments"],
                whisper_words=whisper_words,  # From temp file
                diarization_timeline=self.meeting_data["diarization"]["timeline"]
            )
            self.meeting_data["segments_raw"] = segments
        
        # ✅ Delete temp file after use
        if os.path.exists(words_temp_path):
            os.remove(words_temp_path)
            print(f"   🗑️ Temp words file deleted")
        
        print(f"   ✅ Speaker assignment complete")
        print(f"      - Assigned segments: {len(self.meeting_data['segments_raw'])}")
        
        # Save stage output
        self._save_stage_output('stage5_speaker_assignment', {'segments': self.meeting_data["segments_raw"], 'count': len(self.meeting_data["segments_raw"])}, 'json')
        self._save_meeting_json()
    
    def run_stage6_merge_normalize(self):
        """[6] Merge & Normalize - TẬP TRUNG TẤT CẢ LOGIC"""
        self._update_status("processing", "merge_normalize")
        
        normalizer = MergeNormalizer()
        result = normalizer.process(
            segments=self.meeting_data["segments_raw"]
        )
        
        # Update SSoT - FINAL SEGMENTS
        self.meeting_data["segments"] = result["segments"]
        self.meeting_data["processing_stats"] = result["stats"]
        
        print(f"   ✅ Merge & normalize complete")
        print(f"      - Input: {result['stats']['input_count']} segments")
        print(f"      - After merge: {result['stats']['after_merge']} segments")
        print(f"      - After dedup: {result['stats']['after_dedup']} segments")
        print(f"      - Final: {result['stats']['final_count']} segments")
        
        # Save stage output
        self._save_stage_output('stage6_merge_normalize', result, 'json')
        self._save_meeting_json()
    
    def run_stage7_gender(self):
        """[7] Gender Classification (optional, fallback)"""
        if not self.config["enable_gender"]:
            print(f"\n   ⏭️  Skipping gender classification (disabled)")
            return
        
        self._update_status("processing", "gender")
        
        gender_clf = GenderClassifier()
        speakers_gender = gender_clf.classify(
            audio_path=self.preprocessed_audio,
            speakers=self.meeting_data["diarization"]["speakers"],
            segments=self.meeting_data["segments"]
        )
        
        # Update SSoT - gán vào speakers metadata
        for speaker_id, gender in speakers_gender.items():
            if speaker_id in self.meeting_data["speakers"]:
                self.meeting_data["speakers"][speaker_id]["gender"] = gender
            else:
                self.meeting_data["speakers"][speaker_id] = {"gender": gender}
        
        print(f"   ✅ Gender classification complete")
        for speaker, gender in speakers_gender.items():
            emoji = "👨" if gender == "Male" else "👩" if gender == "Female" else "❓"
            print(f"      {emoji} {speaker}: {gender}")
        
        # Save stage output
        self._save_stage_output('stage7_gender', {'speakers_gender': speakers_gender}, 'json')
        self._save_meeting_json()
    
    def run_stage7_5_spell_check(self):
        """[7.5] LLM Spell Check & Grammar Correction (Optional)"""
        if not self.config["enable_spell_check"]:
            print(f"\n   ⏭️  Skipping spell check (disabled)")
            return
        
        self._update_status("processing", "spell_check")
        
        # Initialize LLM analyzer
        llm_analyzer = LLMAnalyzer()
        
        if not llm_analyzer.available:
            print(f"   ⚠️  LLM not available - skipping spell check")
            print(f"   💡 To enable spell check: install Ollama and pull model")
            return
        
        # Get segments to correct
        segments = self.meeting_data.get("segments", [])
        
        if not segments:
            print(f"   ⚠️  No segments found to spell check")
            return
        
        print(f"   📝 Correcting {len(segments)} segments...")
        
        # Run spell check
        corrected_segments = llm_analyzer.spell_check_segments(segments)
        
        # Update SSoT
        self.meeting_data["segments"] = corrected_segments
        
        # Count corrections
        num_corrected = sum(1 for seg in corrected_segments if 'text_original' in seg)
        
        print(f"   ✅ Spell check complete")
        print(f"      - Total segments: {len(segments)}")
        print(f"      - Corrected: {num_corrected}")
        print(f"      - Unchanged: {len(segments) - num_corrected}")
        
        # Save stage output
        self._save_stage_output('stage7.5_spell_check', {
            'corrected_segments': corrected_segments,
            'stats': {
                'total': len(segments),
                'corrected': num_corrected,
                'unchanged': len(segments) - num_corrected
            }
        }, 'json')
        self._save_meeting_json()
    
    def run_stage8_meeting_json(self):
        """
        [8] Meeting.json Validation & Export
        
        Checkpoint stage:
        - Validate meeting.json structure
        - Export to multiple formats (JSON, TXT, CSV)
        - Create backup
        - Generate statistics report
        """
        self._update_status("processing", "meeting_json_export")
        
        # ✅ Use Stage 8 processor
        validator = MeetingValidator()
        result = validator.process(
            meeting_data=self.meeting_data,
            output_dir=self.output_dir,
            meeting_id=self.meeting_id
        )
        
        # Update metadata with stats
        self.meeting_data["metadata"]["statistics"] = result['stats']
        
        # Save final meeting.json
        self._save_meeting_json()
        
        print(f"\n   ✅ Stage 8 complete:")
        print(f"      - Validation: {'✅ Passed' if result['valid'] else '❌ Failed'}")
        print(f"      - Exports: {len(result['exports'])} files")
        for format_name, path in result['exports'].items():
            print(f"         • {format_name}: {os.path.basename(path)}")
    
    def run_stage9_llm(self):
        """[9] LLM Analysis & Insights"""
        if not self.config["enable_llm"]:
            print(f"\n   ⏭️  Skipping LLM analysis (disabled)")
            return
        
        self._update_status("processing", "llm_analysis")
        
        llm_analyzer = LLMAnalyzer()
        llm_result = llm_analyzer.analyze(
            segments=self.meeting_data["segments"],
            speakers=self.meeting_data["speakers"],
            metadata=self.meeting_data["metadata"]
        )
        
        # Update SSoT
        self.meeting_data["llm"]["summary"] = llm_result.get("summary", "")
        self.meeting_data["llm"]["tasks"] = llm_result.get("tasks", [])
        self.meeting_data["llm"]["insights"] = llm_result.get("insights", [])
        
        print(f"   ✅ LLM analysis complete")
        print(f"      - Summary: {len(llm_result.get('summary', ''))} chars")
        print(f"      - Tasks: {len(llm_result.get('tasks', []))}")
        print(f"      - Insights: {len(llm_result.get('insights', []))}")
        
        # Save stage output
        self._save_stage_output('stage9_llm', llm_result, 'json')
        self._save_meeting_json()
    
    def _validate_meeting_json(self) -> Dict:
        """
        Validate meeting.json structure
        
        Checks:
        - Required fields exist
        - Data types are correct
        - Timestamps are valid
        - No duplicate segment IDs
        """
        errors = []
        
        # Check required top-level keys
        required_keys = ["metadata", "segments", "speakers", "diarization", "whisper"]
        for key in required_keys:
            if key not in self.meeting_data:
                errors.append(f"Missing required key: {key}")
        
        # Check segments
        if "segments" in self.meeting_data:
            segments = self.meeting_data["segments"]
            
            if not isinstance(segments, list):
                errors.append("segments must be a list")
            else:
                # Check segment structure
                for i, seg in enumerate(segments):
                    if "start_time" not in seg or "end_time" not in seg:
                        errors.append(f"Segment {i} missing time fields")
                    
                    if "text" not in seg:
                        errors.append(f"Segment {i} missing text field")
                    
                    if "speaker" not in seg:
                        errors.append(f"Segment {i} missing speaker field")
                    
                    # Validate timestamps
                    if "start_time" in seg and "end_time" in seg:
                        if seg["end_time"] <= seg["start_time"]:
                            errors.append(f"Segment {i} has invalid timestamps")
        
        # Check speakers
        if "speakers" in self.meeting_data:
            if not isinstance(self.meeting_data["speakers"], dict):
                errors.append("speakers must be a dict")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def _calculate_comprehensive_stats(self) -> Dict:
        """Calculate comprehensive statistics"""
        segments = self.meeting_data.get("segments", [])
        words = self.meeting_data.get("whisper", {}).get("words", [])
        speakers = self.meeting_data.get("speakers", {})
        
        total_duration = 0
        speaker_time = {}
        word_count_per_speaker = {}
        
        for seg in segments:
            duration = seg["end_time"] - seg["start_time"]
            total_duration += duration
            
            speaker = seg["speaker"]
            speaker_time[speaker] = speaker_time.get(speaker, 0) + duration
            
            word_count = len(seg.get("text", "").split())
            word_count_per_speaker[speaker] = word_count_per_speaker.get(speaker, 0) + word_count
        
        return {
            "total_segments": len(segments),
            "total_duration": total_duration,
            "total_words": len(words),
            "total_text_words": sum(len(seg.get("text", "").split()) for seg in segments),
            "num_speakers": len(speakers),
            "speaker_time": speaker_time,
            "word_count_per_speaker": word_count_per_speaker,
            "avg_segment_duration": total_duration / len(segments) if segments else 0
        }
    
    def _export_to_txt(self, output_path: str):
        """Export meeting to readable TXT format"""
        segments = self.meeting_data.get("segments", [])
        speakers = self.meeting_data.get("speakers", {})
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("="*80 + "\n")
            f.write(f"MEETING TRANSCRIPT\n")
            f.write(f"Meeting ID: {self.meeting_data['metadata']['meeting_id']}\n")
            f.write(f"Date: {self.meeting_data['metadata']['created_at']}\n")
            f.write("="*80 + "\n\n")
            
            # Transcript
            for seg in segments:
                speaker = seg.get("speaker_display", seg.get("speaker", "UNKNOWN"))
                time_str = seg.get("time_str", f"{seg['start_time']:.1f}s")
                text = seg.get("text", "")
                
                # Get gender emoji
                speaker_id = seg.get("speaker")
                gender = speakers.get(speaker_id, {}).get("gender", "Unknown")
                emoji = "👨" if gender == "Male" else "👩" if gender == "Female" else "❓"
                
                f.write(f"[{time_str}] {emoji} {speaker}:\n")
                f.write(f"{text}\n\n")
    
    def _export_to_csv(self, output_path: str):
        """Export segments to CSV format"""
        import csv
        
        segments = self.meeting_data.get("segments", [])
        speakers = self.meeting_data.get("speakers", {})
        
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "Segment ID", "Start Time", "End Time", "Duration", 
                "Speaker", "Gender", "Text", "Word Count"
            ])
            
            # Data
            for seg in segments:
                speaker_id = seg.get("speaker", "UNKNOWN")
                gender = speakers.get(speaker_id, {}).get("gender", "Unknown")
                
                writer.writerow([
                    seg.get("segment_id", ""),
                    seg.get("start_time", 0),
                    seg.get("end_time", 0),
                    seg.get("duration", 0),
                    speaker_id,
                    gender,
                    seg.get("text", ""),
                    seg.get("word_count", 0)
                ])
    
    def _generate_stats_report(self, output_path: str, stats: Dict):
        """Generate comprehensive statistics report"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("MEETING STATISTICS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Basic stats
            f.write("📊 BASIC STATISTICS:\n")
            f.write(f"   Total Segments: {stats['total_segments']}\n")
            f.write(f"   Total Duration: {stats['total_duration']:.1f}s ({stats['total_duration']/60:.1f} min)\n")
            f.write(f"   Total Words: {stats['total_text_words']}\n")
            f.write(f"   Avg Segment Duration: {stats['avg_segment_duration']:.1f}s\n")
            f.write(f"   Number of Speakers: {stats['num_speakers']}\n\n")
            
            # Speaker stats
            f.write("👥 SPEAKER STATISTICS:\n")
            for speaker, duration in sorted(stats['speaker_time'].items(), key=lambda x: x[1], reverse=True):
                percentage = (duration / stats['total_duration'] * 100) if stats['total_duration'] > 0 else 0
                word_count = stats['word_count_per_speaker'].get(speaker, 0)
                
                f.write(f"   {speaker}:\n")
                f.write(f"      Speaking Time: {duration:.1f}s ({percentage:.1f}%)\n")
                f.write(f"      Words: {word_count}\n")
                f.write(f"      Avg Words/Minute: {(word_count / (duration/60)):.1f}\n\n")
            
            # Pipeline info
            f.write("⚙️ PIPELINE INFO:\n")
            f.write(f"   Pipeline Version: {self.meeting_data['metadata']['pipeline_version']}\n")
            f.write(f"   Created At: {self.meeting_data['metadata']['created_at']}\n")
            f.write(f"   Audio File: {self.meeting_data['metadata']['audio_file']}\n\n")
            
            # Processing stats
            if "processing_stats" in self.meeting_data:
                proc_stats = self.meeting_data["processing_stats"]
                f.write("🔧 PROCESSING STATISTICS:\n")
                f.write(f"   Input Segments: {proc_stats.get('input_count', 0)}\n")
                f.write(f"   After Merge: {proc_stats.get('after_merge', 0)}\n")
                f.write(f"   After Dedup: {proc_stats.get('after_dedup', 0)}\n")
                f.write(f"   Final Count: {proc_stats.get('final_count', 0)}\n")
                f.write(f"   Removed by Quality: {proc_stats.get('removed_by_quality', 0)}\n")
    
    # ==================== STAGE TIMING WRAPPER ====================
    
    def _run_stage(self, stage_name: str, stage_func, *args, **kwargs):
        """Wrapper to track and print timing for each stage"""
        print(f"\n{'='*80}")
        print(f"[{stage_name}]")
        print(f"{'='*80}\n")
        
        stage_start = time.time()
        result = stage_func(*args, **kwargs)
        stage_elapsed = time.time() - stage_start
        
        self.stage_timings[stage_name] = stage_elapsed
        print(f"\n   ⏱️  {stage_name} completed in {stage_elapsed:.2f}s ({stage_elapsed/60:.2f} min)")
        
        return result
    
    def _print_timing_summary(self):
        """Print summary of all stage timings"""
        print(f"\n{'='*80}")
        print(f"⏱️  STAGE TIMING SUMMARY")
        print(f"{'='*80}\n")
        
        total_time = sum(self.stage_timings.values())
        
        for stage_name, elapsed in self.stage_timings.items():
            percentage = (elapsed / total_time * 100) if total_time > 0 else 0
            print(f"   {stage_name:35} {elapsed:8.2f}s ({percentage:5.1f}%)")
        
        print(f"   {'-'*80}")
        print(f"   {'TOTAL':35} {total_time:8.2f}s ({total_time/60:6.2f} min)")
        print(f"\n{'='*80}\n")
    
    # ==================== MAIN RUN (UPDATED) ====================
    
    def run(self):
        """Run toàn bộ pipeline"""
        pipeline_start = time.time()
        
        try:
            # Stage 1: Preprocessing
            self._run_stage("STAGE 1: Audio Preprocessing", self.run_stage1_preprocessing)
            
            # Stage 2: Unified Chunking
            self._run_stage("STAGE 2: Unified Chunking", self.run_stage2_chunking)
            
            # Stage 3: Whisper
            self._run_stage("STAGE 3: Whisper Transcription", self.run_stage3_whisper)
            
            # Small delay to let OS cleanup memory
            import time as time_module
            time_module.sleep(0.5)
            
            # Stage 4: Diarization
            self._run_stage("STAGE 4: Speaker Diarization", self.run_stage4_diarization)
            
            # Stage 5: Speaker Assignment
            self._run_stage("STAGE 5: Speaker Assignment", self.run_stage5_speaker_assignment)
            
            # Stage 6: Merge & Normalize
            self._run_stage("STAGE 6: Merge & Normalize", self.run_stage6_merge_normalize)
            
            # Stage 7: Gender (optional)
            if self.config.get("enable_gender", True):
                self._run_stage("STAGE 7: Gender Classification", self.run_stage7_gender)
            
            # Stage 7.5: Spell Check (optional, LLM-based)
            if self.config.get("enable_spell_check", True):
                self._run_stage("STAGE 7.5: LLM Spell Check", self.run_stage7_5_spell_check)
            
            # ✅ Stage 8: Meeting.json Validation & Export
            self._run_stage("STAGE 8: Validation & Export", self.run_stage8_meeting_json)
            
            # Stage 9: LLM (optional)
            if self.config.get("enable_llm", True):
                self._run_stage("STAGE 9: LLM Analysis", self.run_stage9_llm)
            
            # Final status
            self._update_status("completed")
            
            pipeline_elapsed = time.time() - pipeline_start
            
            # Print timing summary
            self._print_timing_summary()
            
            print(f"\n{'='*80}")
            print(f"✅ PIPELINE COMPLETED SUCCESSFULLY")
            print(f"{'='*80}")
            print(f"   Meeting ID: {self.meeting_id}")
            print(f"   Total time: {pipeline_elapsed:.1f}s ({pipeline_elapsed/60:.1f} min)")
            print(f"   Meeting JSON: {self.get_meeting_json_path()}")
            print(f"{'='*80}\n")
            
            return self.meeting_data
            
        except Exception as e:
            self._update_status("failed")
            print(f"\n❌ Pipeline failed: {e}")
            
            # Print full traceback for debugging
            import traceback
            print(f"\n📋 Full traceback:")
            traceback.print_exc()
            
            # Print timing summary even on failure
            if self.stage_timings:
                print(f"\n⏱️  Stages completed before failure:")
                for stage_name, elapsed in self.stage_timings.items():
                    print(f"   {stage_name}: {elapsed:.2f}s")
            
            raise

# ==================== USAGE ====================
if __name__ == "__main__":
    import sys
    
    # Load .env file để lấy HF_TOKEN khi chạy standalone
    try:
        from dotenv import load_dotenv
        env_path = BACKEND_DIR / ".env"
        
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✅ Loaded .env from: {env_path}")
            
            # Priority: .env > config.py
            hf_token = os.getenv("HF_TOKEN") or CONFIG_HF_TOKEN
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token
                print(f"✅ HF_TOKEN loaded: {hf_token[:10]}...{hf_token[-5:]}")
            else:
                print(f"⚠️  Warning: HF_TOKEN not found in .env or config.py")
        else:
            # Use config.py value if .env doesn't exist
            print(f"⚠️  .env file not found at {env_path}")
            if CONFIG_HF_TOKEN:
                os.environ["HF_TOKEN"] = CONFIG_HF_TOKEN
                print(f"✅ HF_TOKEN loaded from config.py")
            else:
                print(f"⚠️  Warning: HF_TOKEN not in config.py either")
    except ImportError:
        print("⚠️  python-dotenv not installed, skipping .env loading")
    
    if len(sys.argv) < 2:
        print("\nUsage: python pipeline_manager.py <audio_file>")
        print("\nExample:")
        print("  python pipeline_manager.py 'D:\\Audio\\meeting.mp3'")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    # Verify audio file exists
    if not os.path.exists(audio_file):
        print(f"❌ Error: Audio file not found: {audio_file}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"🎯 AMITA PIPELINE - Standalone Mode")
    print(f"{'='*80}")
    print(f"Audio file: {audio_file}")
    print(f"Output dir: backend/src/outputs/")
    print(f"{'='*80}\n")
    
    # Create pipeline (output_dir will default to backend/src/outputs)
    pipeline = MeetingPipeline(audio_file)
    
    # Configure - Use settings from config.py via PIPELINE_CONFIG
    # No need to override here, MeetingPipeline already uses PIPELINE_CONFIG
    
    # Run
    try:
        result = pipeline.run()
        print(f"\n✅ Meeting JSON saved at: {pipeline.get_meeting_json_path()}")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
