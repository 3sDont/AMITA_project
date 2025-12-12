"""
AMITA System - Single Page App

✅ VERSION 2.0: Sử dụng pipeline mới thống nhất
"""
import streamlit as st
import os
import sys
import json
import time
import html
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

import config
import utils

# ✅ Optional imports
try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    RECORDING_AVAILABLE = True
except ImportError:
    RECORDING_AVAILABLE = False

try:
    from integrations import get_trello_client, get_notion_client, get_clickup_client
    INTEGRATIONS_AVAILABLE = True
    has_trello = True
    has_notion = True
    has_clickup = True
except ImportError:
    INTEGRATIONS_AVAILABLE = False
    has_trello = False
    has_notion = False
    has_clickup = False

# ✅ IMPORT SHARED UTILITIES
try:
    from src.text_utils import format_duration, truncate_text
    from src.data_processor import get_segments_stats
    TEXT_UTILS_AVAILABLE = True
except ImportError:
    TEXT_UTILS_AVAILABLE = False

# ✅ IMPORT NEW PIPELINE V2.0
try:
    from src.pipeline_manager import MeetingPipeline
    NEW_PIPELINE_AVAILABLE = True
except ImportError as e:
    NEW_PIPELINE_AVAILABLE = False
    print(f"⚠️  New pipeline not available: {e}")

# ======================= PAGE CONFIG =======================
st.set_page_config(
    page_title="AMITA v2.0 - AI Meeting Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======================= SESSION STATE =======================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''
if 'audio_path' not in st.session_state:
    st.session_state['audio_path'] = None
if 'processing_done' not in st.session_state:
    st.session_state['processing_done'] = False
if 'meeting_data' not in st.session_state:
    st.session_state['meeting_data'] = None
if 'transcript' not in st.session_state:
    st.session_state['transcript'] = []
if 'summary' not in st.session_state:
    st.session_state['summary'] = None
if 'highlights' not in st.session_state:
    st.session_state['highlights'] = []
if 'next_actions' not in st.session_state:
    st.session_state['next_actions'] = []
if 'tasks' not in st.session_state:
    st.session_state['tasks'] = []
# ✅ Recording states (initialize even if disabled to avoid KeyError)
if 'is_recording' not in st.session_state:
    st.session_state['is_recording'] = False
if 'recording_data' not in st.session_state:
    st.session_state['recording_data'] = []
if 'recording_start_time' not in st.session_state:
    st.session_state['recording_start_time'] = None
# ✅ ADD: Pipeline mode (FIX MISSING KEY ERROR)
if 'pipeline_mode' not in st.session_state:
    st.session_state['pipeline_mode'] = 'v2' if NEW_PIPELINE_AVAILABLE else 'v1'

# ======================= CUSTOM CSS =======================
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .card {
        border-radius: 10px;
        padding: 1.5rem;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .transcript-box {
        max-height: 500px;
        overflow-y: auto;
        padding: 1.5rem;
        background-color: white;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
    }
    .transcript-box::-webkit-scrollbar {
        width: 8px;
    }
    .transcript-box::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    .transcript-box::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 10px;
    }
    .transcript-box::-webkit-scrollbar-thumb:hover {
        background: #5568d3;
    }
    .speaker-line {
        margin: 0.8rem 0;
        padding: 0.8rem;
        border-left: 3px solid #667eea;
        background-color: #f8f9fa;
        border-radius: 5px;
    }
    .highlight {
        background-color: #fff3cd;
        padding: 0.2rem 0.4rem;
        border-radius: 3px;
        font-weight: 500;
    }
    .action-item {
        padding: 0.8rem;
        margin: 0.5rem 0;
        background-color: white;
        border-left: 4px solid #28a745;
        border-radius: 5px;
    }
    .task-card {
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: white;
    }
    
    /* ✅ Thêm style cho how_to */
    .task-card em {
        color: #6c757d;
        font-size: 0.9rem;
        display: block;
        margin-top: 0.5rem;
        padding: 0.5rem;
        background-color: #f8f9fa;
        border-left: 3px solid #17a2b8;
        border-radius: 4px;
    }
    
    .priority-high { border-left: 4px solid #dc3545; }
    .priority-medium { border-left: 4px solid #ffc107; }
    .priority-low { border-left: 4px solid #28a745; }
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* ✅ Recording UI - Inspired by design */
    .recording-container {
        background: linear-gradient(135deg, #2d3561 0%, #1e2338 100%);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .recording-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(102,126,234,0.1) 0%, transparent 70%);
        animation: pulse 3s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.3; }
    }
    
    .recording-status {
        position: relative;
        z-index: 1;
    }
    
    .record-button-wrapper {
        display: inline-block;
        position: relative;
        margin: 2rem 0;
    }
    
    .record-timer {
        font-size: 3.5rem;
        font-weight: bold;
        color: #ff4444;
        margin: 1.5rem 0;
        font-family: 'Courier New', monospace;
        text-shadow: 0 0 20px rgba(255,68,68,0.5);
    }
    
    .record-instruction {
        color: #a8b0d3;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .recording-pulse {
        animation: recording-pulse 1.5s ease-in-out infinite;
    }
    
    @keyframes recording-pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    /* Progress bar at top */
    .recording-progress {
        position: absolute;
        top: 0;
        left: 0;
        height: 4px;
        background: linear-gradient(90deg, #00ff88 0%, #00cc88 100%);
        transition: width 0.3s ease;
        box-shadow: 0 0 10px rgba(0,255,136,0.5);
    }
</style>
""", unsafe_allow_html=True)

# ======================= LOGIN SYSTEM =======================
def login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="main-title">🎙️ AMITA System</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">AI Meeting Intelligence & Task Automation</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔐 Login", use_container_width=True, type="primary"):
                    if username and password:
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = username
                        st.rerun()
                    else:
                        st.error("Please enter username and password")
            with col_b:
                if st.button("👤 Register", use_container_width=True):
                    st.info("Registration feature coming soon")
            st.markdown('</div>', unsafe_allow_html=True)

# ======================= PROCESS AUDIO =======================
def process_audio(audio_path):
    """
    ✅ NEW PIPELINE: Unified processing
    
    Pipeline Flow:
    [1] Preprocessing → [2] Unified Chunking → [3] Whisper → 
    [4] Diarization → [5] Speaker Assignment → [6] Merge & Normalize → 
    [7] Gender → [8] Validation & Export → [9] LLM
    
    Returns: meeting.json (SSoT)
    """
    if not NEW_PIPELINE_AVAILABLE:
        st.error("❌ Pipeline v2.0 not available. Please check installation.")
        return False
    
    try:
        # ✅ Create progress container
        progress_container = st.container()
        
        with progress_container:
            st.markdown("### 🚀 Processing Pipeline v2.0")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # ✅ Create pipeline
            status_text.text("Initializing pipeline...")
            pipeline = MeetingPipeline(audio_path, output_dir="outputs")
            
            # ✅ Configure pipeline
            pipeline.config.update({
                "chunk_duration_minutes": 10,
                "enable_vad": True,
                "enable_gender": True,
                "enable_llm": config.ENABLE_LLM_ANALYSIS,
                "min_speakers": config.MIN_SPEAKERS,
                "max_speakers": config.MAX_SPEAKERS
            })
            
            progress_bar.progress(10)
            
            # ✅ Stage 1: Preprocessing
            status_text.text("🎵 Stage 1/9: Audio Preprocessing...")
            pipeline.run_stage1_preprocessing()
            progress_bar.progress(20)
            
            # ✅ Stage 2: Chunking
            status_text.text("✂️ Stage 2/9: Unified Chunking...")
            pipeline.run_stage2_chunking()
            progress_bar.progress(30)
            
            # ✅ Stage 3: Whisper
            status_text.text("🎤 Stage 3/9: Whisper Transcription...")
            pipeline.run_stage3_whisper()
            progress_bar.progress(45)
            
            # ✅ Stage 4: Diarization
            status_text.text("👥 Stage 4/9: Speaker Diarization...")
            pipeline.run_stage4_diarization()
            progress_bar.progress(60)
            
            # ✅ Stage 5: Speaker Assignment
            status_text.text("🎯 Stage 5/9: Speaker Assignment...")
            pipeline.run_stage5_speaker_assignment()
            progress_bar.progress(70)
            
            # ✅ Stage 6: Merge & Normalize
            status_text.text("🔧 Stage 6/9: Merge & Normalize...")
            pipeline.run_stage6_merge_normalize()
            progress_bar.progress(80)
            
            # ✅ Stage 7: Gender
            if pipeline.config["enable_gender"]:
                status_text.text("👨👩 Stage 7/9: Gender Classification...")
                pipeline.run_stage7_gender()
            progress_bar.progress(85)
            
            # ✅ Stage 8: Validation & Export
            status_text.text("✅ Stage 8/9: Validation & Export...")
            pipeline.run_stage8_meeting_json()
            progress_bar.progress(90)
            
            # ✅ Stage 9: LLM
            if pipeline.config["enable_llm"]:
                status_text.text("🤖 Stage 9/9: LLM Analysis...")
                pipeline.run_stage9_llm()
            progress_bar.progress(95)
            
            # ✅ Finalize
            pipeline._update_status("completed")
            progress_bar.progress(100)
            status_text.text("✅ Pipeline completed successfully!")
            
            # ✅ Get meeting data
            meeting_data = pipeline.meeting_data
            
            # ✅ Update session state
            st.session_state['meeting_data'] = meeting_data
            st.session_state['transcript'] = meeting_data.get('segments', [])
            
            # Extract LLM results
            llm_data = meeting_data.get('llm', {})
            if llm_data.get('summary'):
                st.session_state['summary'] = llm_data['summary']
                
                # Extract highlights from summary
                highlights = []
                for line in llm_data['summary'].split('\n'):
                    if any(keyword in line.lower() for keyword in ['quyết định', 'quan trọng', 'chính', 'key', 'main']):
                        highlights.append(line.strip())
                st.session_state['highlights'] = highlights[:5]
            
            if llm_data.get('tasks'):
                st.session_state['tasks'] = llm_data['tasks']
                next_actions = [t['task'] for t in llm_data['tasks'][:3]]
                st.session_state['next_actions'] = next_actions
            
            st.session_state['processing_done'] = True
            
            # ✅ Show completion message
            st.success(f"✅ Processing complete! Meeting ID: {meeting_data['metadata']['meeting_id']}")
            
            return True
        
    except Exception as e:
        st.error(f"❌ Processing failed: {str(e)}")
        import traceback
        with st.expander("Show error details"):
            st.code(traceback.format_exc())
        return False

# ======================= MAIN APP =======================
def main_app():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="main-title">🎙️ AMITA System v2.0</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="subtitle">Welcome, {st.session_state["username"]}!</div>', unsafe_allow_html=True)
    
    with col2:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.rerun()
    
    # ✅ Pipeline info badge (SIMPLIFIED - no mode selector)
    if NEW_PIPELINE_AVAILABLE:
        st.info("🚀 Using **Pipeline v2.0** (Unified Architecture)")
    else:
        st.warning("⚠️ Pipeline v2.0 not available. Please install dependencies or check file structure.")
        with st.expander("Show missing modules"):
            st.code("""
Required files in src/:
- stage1_preprocessing.py
- stage2_chunking.py  
- stage3_whisper.py
- stage4_diarization.py
- stage5_speaker_assignment.py
- stage6_merge_normalize.py
- stage7_gender.py
- stage8_validation.py
- stage9_llm.py
- pipeline_manager.py
            """)
    
    st.divider()
    
    # ======================= AUDIO INPUT SECTION =======================
    st.subheader("🎤 Audio Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔴 Record Audio (Live Meeting)**")
        
        if RECORDING_AVAILABLE:
            # ✅ Beautiful Recording UI
            if st.session_state['is_recording']:
                # Calculate elapsed time
                elapsed = time.time() - st.session_state['recording_start_time']
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                
                # Progress bar animation (0-100% based on time, reset every 60s for visual effect)
                progress_percent = int((elapsed % 60) / 60 * 100)
                
                st.markdown(f"""
                <div class="recording-container">
                    <div class="recording-progress" style="width: {progress_percent}%;"></div>
                    <div class="recording-status">
                        <div class="record-timer recording-pulse">
                            {minutes:02d}:{seconds:02d}
                        </div>
                        <div class="record-instruction">
                            🎙️ Recording in progress...
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Control buttons
                col_a, col_b, col_c = st.columns([1, 2, 1])
                with col_b:
                    if st.button("⏹️ Stop Recording", use_container_width=True, type="primary", key="stop_rec"):
                        # Generate filename
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"meeting_recording_{timestamp}.wav"
                        filepath = os.path.join("uploads", filename)
                        
                        # Save recording
                        with st.spinner("💾 Saving recording..."):
                            os.makedirs("uploads", exist_ok=True)
                            
                            recording_data = st.session_state.get('recording_data', [])
                            if recording_data:
                                import numpy as np
                                import soundfile as sf
                                recording = np.concatenate(recording_data, axis=0)
                                sf.write(filepath, recording, 16000)
                                
                                st.session_state['audio_path'] = filepath
                                st.session_state['is_recording'] = False
                                st.session_state['recording_data'] = []
                                
                                file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                                st.success(f"✅ Recording saved: {filename} ({file_size_mb:.1f} MB)")
                                st.info(f"📁 Saved to: uploads/{filename}")
                                st.rerun()
            else:
                # Idle state - Show record button
                st.markdown("""
                <div class="recording-container">
                    <div class="recording-status">
                        <div class="record-instruction">
                            Click the button to start recording
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Start button
                col_a, col_b, col_c = st.columns([1, 2, 1])
                with col_b:
                    if st.button("🎙️ Start Recording", use_container_width=True, type="primary", key="start_rec"):
                        st.session_state['is_recording'] = True
                        st.session_state['recording_start_time'] = time.time()
                        st.session_state['recording_data'] = []
                        
                        # Start recording in background
                        def record_audio_background():
                            """Background recording thread"""
                            import sounddevice as sd
                            def callback(indata, frames, time_info, status):
                                if st.session_state['is_recording']:
                                    st.session_state['recording_data'].append(indata.copy())
                            
                            with sd.InputStream(samplerate=16000, channels=1, callback=callback):
                                while st.session_state['is_recording']:
                                    sd.sleep(100)
                        
                        import threading
                        thread = threading.Thread(target=record_audio_background, daemon=True)
                        thread.start()
                        st.session_state['recording_thread'] = thread
                        
                        st.rerun()
        else:
            st.warning("⚠️ Recording disabled. Install required packages:")
            st.code("pip install sounddevice soundfile", language="bash")
    
    with col2:
        st.markdown("**📤 Upload Audio File**")
        uploaded_file = st.file_uploader(
            "Choose file",
            type=['mp3', 'wav', 'm4a', 'aac', 'ogg', 'flac'],
            label_visibility="collapsed"
        )
        if uploaded_file:
            os.makedirs("uploads", exist_ok=True)
            audio_path = os.path.join("uploads", uploaded_file.name)
            with open(audio_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.session_state['audio_path'] = audio_path
            
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.success(f"✅ Uploaded: {uploaded_file.name} ({file_size_mb:.1f} MB)")
    
    # ======================= AUDIO PLAYER =======================
    if st.session_state['audio_path']:
        st.divider()
        st.subheader("🎵 Audio Player")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.audio(st.session_state['audio_path'])
        with col2:
            if st.button("🚀 Process Audio", use_container_width=True, type="primary"):
                success = process_audio(st.session_state['audio_path'])
                if success:
                    time.sleep(1)
                    st.rerun()
        with col3:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state['audio_path'] = None
                st.session_state['processing_done'] = False
                st.session_state['meeting_data'] = None
                st.rerun()
    
    # ======================= RESULTS SECTION =======================
    if st.session_state['processing_done']:
        st.divider()
        
        # ✅ Meeting Overview Metrics
        if st.session_state.get('meeting_data'):
            meeting_data = st.session_state['meeting_data']
            
            st.subheader("📊 Meeting Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                segments = meeting_data.get('segments', [])
                st.metric("📝 Segments", len(segments))
            with col2:
                speakers = meeting_data.get('diarization', {}).get('speakers', {})
                st.metric("👥 Speakers", len(speakers))
            with col3:
                tasks = meeting_data.get('llm', {}).get('tasks', [])
                st.metric("✅ Tasks", len(tasks))
            with col4:
                stats = meeting_data.get('metadata', {}).get('statistics', {})
                duration = stats.get('total_duration', 0)
                st.metric("⏱️ Duration", f"{duration/60:.1f} min")
            
            # ✅ Download meeting.json (SSoT)
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                meeting_json = json.dumps(meeting_data, ensure_ascii=False, indent=2)
                st.download_button(
                    "📥 Download meeting.json",
                    data=meeting_json,
                    file_name=f"meeting_{meeting_data['metadata']['meeting_id']}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col_b:
                # Check if TXT export exists
                meeting_id = meeting_data['metadata']['meeting_id']
                txt_path = f"outputs/{meeting_id}_transcript.txt"
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        txt_content = f.read()
                    st.download_button(
                        "📥 Download transcript.txt",
                        data=txt_content,
                        file_name=f"{meeting_id}_transcript.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            
            with col_c:
                # Check if CSV export exists
                csv_path = f"outputs/{meeting_id}_segments.csv"
                if os.path.exists(csv_path):
                    with open(csv_path, 'r', encoding='utf-8') as f:
                        csv_content = f.read()
                    st.download_button(
                        "📥 Download segments.csv",
                        data=csv_content,
                        file_name=f"{meeting_id}_segments.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            st.divider()
        
        # ======================= TRANSCRIPT & SUMMARY =======================
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("💬 Transcript")
            
            transcript_html = '<div class="transcript-box">'
            
            for seg in st.session_state['transcript']:
                speaker = seg.get('speaker_display', seg.get('speaker', 'UNKNOWN'))
                text = seg.get('text', '')
                
                # Get gender
                if st.session_state.get('meeting_data'):
                    speaker_id = seg.get('speaker')
                    speakers_data = st.session_state['meeting_data'].get('speakers', {})
                    gender = speakers_data.get(speaker_id, {}).get('gender', 'Unknown')
                else:
                    gender = seg.get('gender', 'Unknown')
                
                gender_emoji = "👨" if gender == 'Male' else "👩" if gender == 'Female' else "❓"
                
                # Format time
                if TEXT_UTILS_AVAILABLE:
                    start = seg.get('start_time', 0)
                    time_str = format_duration(start)
                    text_preview = truncate_text(text, max_length=200)
                else:
                    start = seg.get('start_time', 0)
                    time_str = f"{start:.1f}s"
                    text_preview = text[:200] + ('...' if len(text) > 200 else '')
                
                # Escape HTML to prevent rendering HTML tags in transcript
                text_preview_escaped = html.escape(text_preview)
                speaker_escaped = html.escape(speaker)
                
                transcript_html += f"""
                <div class="speaker-line">
                    <strong>{gender_emoji} {speaker_escaped}</strong> <small>({time_str})</small><br>
                    {text_preview_escaped}
                </div>
                """
            
            transcript_html += '</div>'
            st.markdown(transcript_html, unsafe_allow_html=True)
        
        with col2:
            st.subheader("📋 Summary")
            
            # Highlights
            if st.session_state['highlights']:
                st.markdown("**🔆 Key Highlights**")
                for highlight in st.session_state['highlights']:
                    st.markdown(f'<div class="highlight">{highlight}</div>', unsafe_allow_html=True)
                st.markdown("")
            
            # Next Actions
            if st.session_state['next_actions']:
                st.markdown("**⚡ Next Actions**")
                for action in st.session_state['next_actions']:
                    st.markdown(f'<div class="action-item">• {action}</div>', unsafe_allow_html=True)
            
            # Full summary
            if st.session_state['summary']:
                with st.expander("📄 Full Summary"):
                    st.text_area("Summary Content", st.session_state['summary'], height=200, label_visibility="collapsed")
        
        # ======================= TASKS SECTION =======================
        st.divider()
        st.subheader("✅ Tasks")
        
        if st.session_state['tasks']:
            tasks = st.session_state['tasks']
            
            # Filter
            col1, col2 = st.columns([3, 1])
            with col1:
                filter_priority = st.multiselect(
                    "Filter by priority",
                    ["high", "medium", "low"],
                    default=["high", "medium", "low"],
                    label_visibility="collapsed"
                )
            
            filtered_tasks = [t for t in tasks if t.get('priority', 'medium') in filter_priority]
            
            # Display tasks
            for i, task in enumerate(filtered_tasks, 1):
                priority = task.get('priority', 'medium')
                priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(priority, '⚪')
                how_to = task.get('how_to', '')
                
                st.markdown(f"""
                <div class="task-card priority-{priority}">
                    <strong>{priority_emoji} Task {i}:</strong> {task['task']}<br>
                    <small>👤 {task.get('assigned_to', 'Unassigned')} | 📅 {task.get('deadline', 'No deadline')} | ⚡ {priority.upper()}</small>
                    {'<br><br><em>💡 Hướng dẫn: ' + how_to + '</em>' if how_to else ''}
                </div>
                """, unsafe_allow_html=True)
            
            # Export options
            st.markdown("**🔗 Export Tasks**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                tasks_json = json.dumps(filtered_tasks, ensure_ascii=False, indent=2)
                st.download_button(
                    "📥 JSON",
                    data=tasks_json,
                    file_name=f"tasks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                if INTEGRATIONS_AVAILABLE and has_trello:
                    if st.button("📤 Trello", use_container_width=True):
                        with st.spinner("Exporting..."):
                            try:
                                trello = get_trello_client(
                                    os.getenv("TRELLO_API_KEY"),
                                    os.getenv("TRELLO_TOKEN"),
                                    os.getenv("TRELLO_BOARD_ID")
                                )
                                urls = trello.export_tasks(filtered_tasks, st.session_state.get('summary', ''))
                                st.success(f"✅ Exported {len(urls)} cards!")
                            except Exception as e:
                                st.error(f"❌ {str(e)}")
            
            with col3:
                if INTEGRATIONS_AVAILABLE and has_notion:
                    if st.button("📤 Notion", use_container_width=True):
                        with st.spinner("Exporting..."):
                            try:
                                notion = get_notion_client(
                                    os.getenv("NOTION_API_KEY"),
                                    os.getenv("NOTION_DATABASE_ID")
                                )
                                urls = notion.export_tasks(filtered_tasks, st.session_state.get('summary', ''))
                                st.success(f"✅ Exported {len(urls)} pages!")
                            except Exception as e:
                                st.error(f"❌ {str(e)}")
            
            with col4:
                if INTEGRATIONS_AVAILABLE and has_clickup:
                    if st.button("📤 ClickUp", use_container_width=True):
                        with st.spinner("Exporting..."):
                            try:
                                clickup = get_clickup_client(
                                    os.getenv("CLICKUP_API_KEY"),
                                    os.getenv("CLICKUP_LIST_ID")
                                )
                                urls = clickup.export_tasks(filtered_tasks, st.session_state.get('summary', ''))
                                st.success(f"✅ Exported {len(urls)} tasks!")
                            except Exception as e:
                                st.error(f"❌ {str(e)}")
        else:
            st.info("💡 No tasks found. Make sure LLM Analysis is enabled in config.")


# ======================= MAIN ROUTER =======================
if not st.session_state['logged_in']:
    login_page()
else:
    if not NEW_PIPELINE_AVAILABLE:
        st.error("❌ Pipeline is not available. Please check your installation.")
        st.code("pip install -r requirements.txt", language="bash")
    else:
        main_app()

st.divider()
st.caption("© 2024 AMITA System - AI Meeting Intelligence & Task Automation")
