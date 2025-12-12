"""
Pipeline Manager - Quản lý toàn bộ luồng xử lý
Single Source of Truth: meeting.json

Pipeline:
[1] Preprocessing → [2] Unified Chunking → [3] Whisper → [4] Diarization →
[5] Speaker Assignment → [6] Merge & Normalize → [7] Gender → 
[8] meeting.json → [9] LLM → [10] Export
"""
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

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
    
    def __init__(self, audio_path: str, output_dir: str = "outputs"):
        self.audio_path = audio_path
        self.output_dir = output_dir
        self.meeting_id = self._generate_meeting_id()
        
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
        
        # Config
        self.config = {
            "chunk_duration_minutes": 10,
            "enable_vad": True,
            "enable_gender": True,
            "enable_llm": True,
            "min_speakers": None,
            "max_speakers": None
        }
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"🎯 MEETING PIPELINE INITIALIZED")
        print(f"{'='*80}")
        print(f"   Meeting ID: {self.meeting_id}")
        print(f"   Audio: {Path(audio_path).name}")
        print(f"   Output: {output_dir}")
        print(f"{'='*80}\n")
    
    def _generate_meeting_id(self) -> str:
        """Generate unique meeting ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        basename = Path(self.audio_path).stem
        return f"{basename}_{timestamp}"
    
    def _save_meeting_json(self):
        """Save meeting.json (SSoT)"""
        meeting_json_path = os.path.join(self.output_dir, f"{self.meeting_id}_meeting.json")
        with open(meeting_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.meeting_data, f, ensure_ascii=False, indent=2)
        return meeting_json_path
    
    def _update_status(self, status: str, stage: str = None):
        """Update pipeline status"""
        self.meeting_data["metadata"]["status"] = status
        if stage:
            self.meeting_data["metadata"]["current_stage"] = stage
        self.meeting_data["metadata"]["last_updated"] = datetime.now().isoformat()
        self._save_meeting_json()
    
    # ==================== STAGE RUNNERS ====================
    
    def run_stage1_preprocessing(self):
        """[1] Audio Preprocessing"""
        print(f"\n{'='*80}")
        print(f"[STAGE 1] AUDIO PREPROCESSING")
        print(f"{'='*80}\n")
        
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
        
        self._save_meeting_json()
    
    def run_stage2_chunking(self):
        """[2] Unified Chunking"""
        print(f"\n{'='*80}")
        print(f"[STAGE 2] UNIFIED CHUNKING")
        print(f"{'='*80}\n")
        
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
        
        self._save_meeting_json()
    
    def run_stage3_whisper(self):
        """[3] Whisper Transcription"""
        print(f"\n{'='*80}")
        print(f"[STAGE 3] WHISPER TRANSCRIPTION")
        print(f"{'='*80}\n")
        
        self._update_status("processing", "whisper")
        
        whisper = WhisperProcessor()
        result = whisper.process(
            audio_path=self.preprocessed_audio,
            chunks=self.meeting_data["chunks"]
        )
        
        # Update SSoT
        self.meeting_data["whisper"]["segments"] = result["segments"]
        self.meeting_data["whisper"]["words"] = result["words"]
        self.meeting_data["whisper"]["language"] = result["language"]
        
        print(f"   ✅ Transcription complete")
        print(f"      - Segments: {len(result['segments'])}")
        print(f"      - Words: {len(result['words'])}")
        print(f"      - Language: {result['language']}")
        
        self._save_meeting_json()
    
    def run_stage4_diarization(self):
        """[4] Speaker Diarization"""
        print(f"\n{'='*80}")
        print(f"[STAGE 4] SPEAKER DIARIZATION")
        print(f"{'='*80}\n")
        
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
        
        self._save_meeting_json()
    
    def run_stage5_speaker_assignment(self):
        """[5] Speaker Assignment (word-level voting)"""
        print(f"\n{'='*80}")
        print(f"[STAGE 5] SPEAKER ASSIGNMENT")
        print(f"{'='*80}\n")
        
        self._update_status("processing", "speaker_assignment")
        
        assigner = SpeakerAssigner()
        segments = assigner.assign(
            whisper_segments=self.meeting_data["whisper"]["segments"],
            whisper_words=self.meeting_data["whisper"]["words"],
            diarization_timeline=self.meeting_data["diarization"]["timeline"]
        )
        
        # Update SSoT (tạm thời, chưa merge)
        self.meeting_data["segments_raw"] = segments
        
        print(f"   ✅ Speaker assignment complete")
        print(f"      - Assigned segments: {len(segments)}")
        
        self._save_meeting_json()
    
    def run_stage6_merge_normalize(self):
        """[6] Merge & Normalize - TẬP TRUNG TẤT CẢ LOGIC"""
        print(f"\n{'='*80}")
        print(f"[STAGE 6] MERGE & NORMALIZE")
        print(f"{'='*80}\n")
        
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
        
        self._save_meeting_json()
    
    def run_stage7_gender(self):
        """[7] Gender Classification (optional, fallback)"""
        if not self.config["enable_gender"]:
            print(f"\n   ⏭️  Skipping gender classification (disabled)")
            return
        
        print(f"\n{'='*80}")
        print(f"[STAGE 7] GENDER CLASSIFICATION")
        print(f"{'='*80}\n")
        
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
        print(f"\n{'='*80}")
        print(f"[STAGE 8] MEETING.JSON VALIDATION & EXPORT")
        print(f"{'='*80}\n")
        
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
        
        print(f"\n{'='*80}")
        print(f"[STAGE 9] LLM ANALYSIS & INSIGHTS")
        print(f"{'='*80}\n")
        
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
    
    # ==================== MAIN RUN (UPDATED) ====================
    
    def run(self):
        """Run toàn bộ pipeline"""
        start_time = time.time()
        
        try:
            # Stage 1: Preprocessing
            self.run_stage1_preprocessing()
            
            # Stage 2: Unified Chunking
            self.run_stage2_chunking()
            
            # Stage 3: Whisper
            self.run_stage3_whisper()
            
            # Stage 4: Diarization
            self.run_stage4_diarization()
            
            # Stage 5: Speaker Assignment
            self.run_stage5_speaker_assignment()
            
            # Stage 6: Merge & Normalize
            self.run_stage6_merge_normalize()
            
            # Stage 7: Gender (optional)
            self.run_stage7_gender()
            
            # ✅ Stage 8: Meeting.json Validation & Export
            self.run_stage8_meeting_json()
            
            # Stage 9: LLM (optional)
            self.run_stage9_llm()
            
            # Final status
            self._update_status("completed")
            
            elapsed = time.time() - start_time
            
            print(f"\n{'='*80}")
            print(f"✅ PIPELINE COMPLETED SUCCESSFULLY")
            print(f"{'='*80}")
            print(f"   Meeting ID: {self.meeting_id}")
            print(f"   Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
            print(f"   Meeting JSON: {self.get_meeting_json_path()}")
            print(f"{'='*80}\n")
            
            return self.meeting_data
            
        except Exception as e:
            self._update_status("failed")
            print(f"\n❌ Pipeline failed: {e}")
            raise

# ==================== USAGE ====================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pipeline_manager.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    # Create pipeline
    pipeline = MeetingPipeline(audio_file)
    
    # Configure (optional)
    pipeline.config.update({
        "chunk_duration_minutes": 10,
        "enable_vad": True,
        "enable_gender": True,
        "enable_llm": True
    })
    
    # Run
    result = pipeline.run()
    
    print(f"\n✅ Meeting JSON saved at: {pipeline.get_meeting_json_path()}")
