"""
Stage 8: Meeting.json Validation & Export
Checkpoint stage - Đảm bảo meeting.json hợp lệ và export nhiều formats
"""
import os
import shutil
import csv
from typing import Dict, List


class MeetingValidator:
    """
    Meeting.json Validator & Exporter
    
    Features:
        - Validate meeting.json structure
        - Export to multiple formats (TXT, CSV)
        - Create backup
        - Generate statistics report
    """
    
    def __init__(self):
        pass
    
    def process(self, meeting_data: Dict, output_dir: str, meeting_id: str) -> Dict:
        """
        Main processing pipeline
        
        Args:
            meeting_data: Meeting data (SSoT)
            output_dir: Output directory
            meeting_id: Meeting ID
        
        Returns:
            Dict: {
                valid: bool,
                exports: Dict[str, str],
                stats: Dict
            }
        """
        print(f"   🔍 Validating meeting.json structure...")
        
        # Step 1: Validate
        validation_result = self.validate_structure(meeting_data)
        
        if not validation_result["valid"]:
            print(f"   ❌ Validation failed:")
            for error in validation_result["errors"]:
                print(f"      - {error}")
            raise ValueError("Meeting.json validation failed")
        
        print(f"   ✅ Validation passed")
        
        # Step 2: Calculate stats
        print(f"   📊 Calculating statistics...")
        stats = self.calculate_stats(meeting_data)
        
        print(f"      - Total segments: {stats['total_segments']}")
        print(f"      - Total duration: {stats['total_duration']:.1f}s")
        print(f"      - Total words: {stats['total_words']}")
        print(f"      - Speakers: {stats['num_speakers']}")
        
        # Step 3: Export to formats
        print(f"   📤 Exporting to additional formats...")
        exports = {}
        
        # TXT export
        txt_path = os.path.join(output_dir, f"{meeting_id}_transcript.txt")
        self.export_to_txt(meeting_data, txt_path)
        exports['txt'] = txt_path
        print(f"      ✅ Text: {meeting_id}_transcript.txt")
        
        # CSV export
        csv_path = os.path.join(output_dir, f"{meeting_id}_segments.csv")
        self.export_to_csv(meeting_data, csv_path)
        exports['csv'] = csv_path
        print(f"      ✅ CSV: {meeting_id}_segments.csv")
        
        # Step 4: Create backup
        json_path = os.path.join(output_dir, f"{meeting_id}_meeting.json")
        backup_path = os.path.join(output_dir, f"{meeting_id}_meeting_backup.json")
        
        if os.path.exists(json_path):
            shutil.copy2(json_path, backup_path)
            exports['backup'] = backup_path
            print(f"   💾 Backup created: {meeting_id}_meeting_backup.json")
        
        # Step 5: Generate report
        report_path = os.path.join(output_dir, f"{meeting_id}_report.txt")
        self.generate_report(meeting_data, stats, report_path)
        exports['report'] = report_path
        print(f"   📋 Stats report: {meeting_id}_report.txt")
        
        return {
            'valid': True,
            'exports': exports,
            'stats': stats
        }
    
    def validate_structure(self, meeting_data: Dict) -> Dict:
        """
        Validate meeting.json structure
        
        Checks:
        - Required fields exist
        - Data types are correct
        - Timestamps are valid
        - No duplicate segment IDs
        
        Returns:
            Dict: {valid: bool, errors: List[str]}
        """
        errors = []
        
        # Check required top-level keys
        required_keys = ["metadata", "segments", "speakers", "diarization", "whisper"]
        for key in required_keys:
            if key not in meeting_data:
                errors.append(f"Missing required key: {key}")
        
        # Validate metadata
        if "metadata" in meeting_data:
            metadata = meeting_data["metadata"]
            required_meta_keys = ["meeting_id", "created_at", "pipeline_version"]
            for key in required_meta_keys:
                if key not in metadata:
                    errors.append(f"Missing metadata.{key}")
        
        # Validate segments
        if "segments" in meeting_data:
            segments = meeting_data["segments"]
            
            if not isinstance(segments, list):
                errors.append("segments must be a list")
            else:
                seen_ids = set()
                
                for i, seg in enumerate(segments):
                    # Check required fields
                    if "start_time" not in seg or "end_time" not in seg:
                        errors.append(f"Segment {i} missing time fields")
                    
                    if "text" not in seg:
                        errors.append(f"Segment {i} missing text field")
                    
                    if "speaker" not in seg:
                        errors.append(f"Segment {i} missing speaker field")
                    
                    # Validate timestamps
                    if "start_time" in seg and "end_time" in seg:
                        if seg["end_time"] <= seg["start_time"]:
                            errors.append(f"Segment {i} has invalid timestamps: end <= start")
                        
                        if seg["start_time"] < 0:
                            errors.append(f"Segment {i} has negative start_time")
                    
                    # Check for duplicate segment IDs
                    if "segment_id" in seg:
                        seg_id = seg["segment_id"]
                        if seg_id in seen_ids:
                            errors.append(f"Duplicate segment_id: {seg_id}")
                        seen_ids.add(seg_id)
        
        # Validate speakers
        if "speakers" in meeting_data:
            if not isinstance(meeting_data["speakers"], dict):
                errors.append("speakers must be a dict")
        
        # Validate whisper
        if "whisper" in meeting_data:
            whisper = meeting_data["whisper"]
            if "segments" in whisper and not isinstance(whisper["segments"], list):
                errors.append("whisper.segments must be a list")
            if "words" in whisper and not isinstance(whisper["words"], list):
                errors.append("whisper.words must be a list")
        
        # Validate diarization
        if "diarization" in meeting_data:
            diar = meeting_data["diarization"]
            if "timeline" in diar and not isinstance(diar["timeline"], list):
                errors.append("diarization.timeline must be a list")
            if "speakers" in diar and not isinstance(diar["speakers"], dict):
                errors.append("diarization.speakers must be a dict")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def calculate_stats(self, meeting_data: Dict) -> Dict:
        """
        Calculate comprehensive statistics
        
        Returns:
            Dict: Statistics about the meeting
        """
        segments = meeting_data.get("segments", [])
        words = meeting_data.get("whisper", {}).get("words", [])
        speakers = meeting_data.get("speakers", {})
        
        total_duration = 0
        speaker_time = {}
        word_count_per_speaker = {}
        speaker_segments = {}
        
        for seg in segments:
            # Duration
            duration = seg.get("end_time", 0) - seg.get("start_time", 0)
            total_duration += duration
            
            # Speaker stats
            speaker = seg.get("speaker", "UNKNOWN")
            speaker_time[speaker] = speaker_time.get(speaker, 0) + duration
            speaker_segments[speaker] = speaker_segments.get(speaker, 0) + 1
            
            # Word count
            word_count = len(seg.get("text", "").split())
            word_count_per_speaker[speaker] = word_count_per_speaker.get(speaker, 0) + word_count
        
        return {
            "total_segments": len(segments),
            "total_duration": total_duration,
            "total_words": len(words),
            "total_text_words": sum(len(seg.get("text", "").split()) for seg in segments),
            "num_speakers": len(speakers),
            "speaker_time": speaker_time,
            "speaker_segments": speaker_segments,
            "word_count_per_speaker": word_count_per_speaker,
            "avg_segment_duration": total_duration / len(segments) if segments else 0,
            "avg_words_per_segment": sum(word_count_per_speaker.values()) / len(segments) if segments else 0
        }
    
    def export_to_txt(self, meeting_data: Dict, output_path: str):
        """
        Export meeting to readable TXT format
        
        Format:
        ================================================================================
        MEETING TRANSCRIPT
        Meeting ID: xxx
        Date: xxx
        ================================================================================
        
        [00:00] 👨 Speaker 01:
        Text content...
        
        [00:15] 👩 Speaker 02:
        Text content...
        """
        segments = meeting_data.get("segments", [])
        speakers = meeting_data.get("speakers", {})
        metadata = meeting_data.get("metadata", {})
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("MEETING TRANSCRIPT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Meeting ID: {metadata.get('meeting_id', 'Unknown')}\n")
            f.write(f"Date: {metadata.get('created_at', 'Unknown')}\n")
            f.write(f"Pipeline Version: {metadata.get('pipeline_version', 'Unknown')}\n")
            f.write(f"Audio File: {metadata.get('audio_file', 'Unknown')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Transcript
            for seg in segments:
                speaker = seg.get("speaker_display", seg.get("speaker", "UNKNOWN"))
                time_str = seg.get("time_str", self._format_time(seg.get("start_time", 0)))
                text = seg.get("text", "")
                
                # Get gender emoji
                speaker_id = seg.get("speaker")
                gender = speakers.get(speaker_id, {}).get("gender", "Unknown")
                emoji = "👨" if gender == "Male" else "👩" if gender == "Female" else "❓"
                
                f.write(f"[{time_str}] {emoji} {speaker}:\n")
                f.write(f"{text}\n\n")
            
            # Footer stats
            stats = self.calculate_stats(meeting_data)
            f.write("\n" + "=" * 80 + "\n")
            f.write("STATISTICS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total Segments: {stats['total_segments']}\n")
            f.write(f"Total Duration: {self._format_time(stats['total_duration'])}\n")
            f.write(f"Number of Speakers: {stats['num_speakers']}\n")
    
    def export_to_csv(self, meeting_data: Dict, output_path: str):
        """
        Export segments to CSV format
        
        Columns:
        - Segment ID
        - Start Time
        - End Time
        - Duration
        - Speaker
        - Gender
        - Text
        - Word Count
        """
        segments = meeting_data.get("segments", [])
        speakers = meeting_data.get("speakers", {})
        
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "Segment ID",
                "Start Time (s)",
                "End Time (s)",
                "Duration (s)",
                "Speaker",
                "Gender",
                "Text",
                "Word Count"
            ])
            
            # Data rows
            for i, seg in enumerate(segments):
                speaker_id = seg.get("speaker", "UNKNOWN")
                gender = speakers.get(speaker_id, {}).get("gender", "Unknown")
                
                start_time = seg.get("start_time", 0)
                end_time = seg.get("end_time", 0)
                duration = end_time - start_time
                text = seg.get("text", "")
                word_count = len(text.split())
                
                writer.writerow([
                    seg.get("segment_id", i),
                    f"{start_time:.2f}",
                    f"{end_time:.2f}",
                    f"{duration:.2f}",
                    speaker_id,
                    gender,
                    text,
                    word_count
                ])
    
    def generate_report(self, meeting_data: Dict, stats: Dict, output_path: str):
        """
        Generate comprehensive statistics report
        
        Includes:
        - Basic stats
        - Speaker stats
        - Pipeline info
        - Processing stats
        """
        metadata = meeting_data.get("metadata", {})
        processing_stats = meeting_data.get("processing_stats", {})
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("MEETING STATISTICS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Basic statistics
            f.write("📊 BASIC STATISTICS:\n")
            f.write(f"   Total Segments: {stats['total_segments']}\n")
            f.write(f"   Total Duration: {stats['total_duration']:.1f}s ({stats['total_duration']/60:.1f} min)\n")
            f.write(f"   Total Words: {stats['total_text_words']}\n")
            f.write(f"   Avg Segment Duration: {stats['avg_segment_duration']:.1f}s\n")
            f.write(f"   Avg Words/Segment: {stats['avg_words_per_segment']:.1f}\n")
            f.write(f"   Number of Speakers: {stats['num_speakers']}\n\n")
            
            # Speaker statistics
            f.write("👥 SPEAKER STATISTICS:\n")
            for speaker, duration in sorted(stats['speaker_time'].items(), key=lambda x: x[1], reverse=True):
                percentage = (duration / stats['total_duration'] * 100) if stats['total_duration'] > 0 else 0
                word_count = stats['word_count_per_speaker'].get(speaker, 0)
                num_segments = stats['speaker_segments'].get(speaker, 0)
                
                f.write(f"\n   {speaker}:\n")
                f.write(f"      Speaking Time: {duration:.1f}s ({percentage:.1f}%)\n")
                f.write(f"      Number of Turns: {num_segments}\n")
                f.write(f"      Total Words: {word_count}\n")
                
                if duration > 0:
                    wpm = (word_count / (duration / 60))
                    f.write(f"      Avg Words/Minute: {wpm:.1f}\n")
            
            # Pipeline info
            f.write("\n" + "=" * 80 + "\n")
            f.write("⚙️ PIPELINE INFO:\n")
            f.write(f"   Pipeline Version: {metadata.get('pipeline_version', 'Unknown')}\n")
            f.write(f"   Meeting ID: {metadata.get('meeting_id', 'Unknown')}\n")
            f.write(f"   Created At: {metadata.get('created_at', 'Unknown')}\n")
            f.write(f"   Audio File: {metadata.get('audio_file', 'Unknown')}\n")
            f.write(f"   Status: {metadata.get('status', 'Unknown')}\n")
            
            # Processing stats
            if processing_stats:
                f.write("\n" + "=" * 80 + "\n")
                f.write("🔧 PROCESSING STATISTICS:\n")
                f.write(f"   Input Segments: {processing_stats.get('input_count', 0)}\n")
                f.write(f"   After Merge: {processing_stats.get('after_merge', 0)}\n")
                f.write(f"   After Dedup: {processing_stats.get('after_dedup', 0)}\n")
                f.write(f"   After Clean: {processing_stats.get('after_clean', 0)}\n")
                f.write(f"   Final Count: {processing_stats.get('final_count', 0)}\n")
                f.write(f"   Removed by Quality: {processing_stats.get('removed_by_quality', 0)}\n")
                f.write(f"   Fillers Removed: {processing_stats.get('fillers_removed', 0)}\n")
            
            f.write("\n" + "=" * 80 + "\n")
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds to MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
