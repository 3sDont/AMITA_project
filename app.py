"""
AMITA System
"""
import streamlit as st
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# ✅ Load .env TRƯỚC khi import config
load_dotenv()

import config
import utils

# ✅ Import integrations chỉ khi cần
try:
    from integrations import get_trello_client, get_notion_client, get_clickup_client
    INTEGRATIONS_AVAILABLE = True
except ImportError as e:
    INTEGRATIONS_AVAILABLE = False
    # Không warning ở đây, sẽ warning khi user thực sự cần

# ======================= PAGE CONFIG =======================
st.set_page_config(
    page_title="AMITA System",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ Initialize session state
if 'processing_done' not in st.session_state:
    st.session_state['processing_done'] = False
if 'summary' not in st.session_state:
    st.session_state['summary'] = None
if 'tasks' not in st.session_state:
    st.session_state['tasks'] = []
if 'combined_result' not in st.session_state:
    st.session_state['combined_result'] = []

# ======================= CUSTOM CSS =======================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .upload-box {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
    }
    
    .task-card {
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
        border-radius: 5px;
    }
    
    .high-priority { border-left-color: #dc3545; }
    .medium-priority { border-left-color: #ffc107; }
    .low-priority { border-left-color: #28a745; }
</style>
""", unsafe_allow_html=True)

# ======================= SIDEBAR =======================
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/667eea/ffffff?text=AMITA", use_column_width=True)  # ✅ Changed from use_column_width
    st.title("⚙️ Settings")
    
    # Pipeline settings
    st.subheader("🎤 Pipeline")
    use_gpu = st.checkbox("Use GPU", value=config.USE_GPU, help="Tăng tốc 3-5x nếu có GPU NVIDIA")
    whisper_model = st.selectbox(
        "Whisper Model",
        ["small", "medium", "large"],
        index=["small", "medium", "large"].index(config.WHISPER_MODEL) if config.WHISPER_MODEL in ["small", "medium", "large"] else 1,
        help="Medium = cân bằng, Large = chính xác nhất"
    )
    
    enable_llm = st.checkbox("Enable LLM Analysis", value=config.ENABLE_LLM_ANALYSIS)
    
    st.divider()
    
    # ✅ Integration settings - READ FROM ENV
    st.subheader("🔗 Integrations")
    
    # Check which integrations are configured
    has_trello = bool(os.getenv("TRELLO_API_KEY") and os.getenv("TRELLO_TOKEN") and os.getenv("TRELLO_BOARD_ID"))
    has_notion = bool(os.getenv("NOTION_API_KEY") and os.getenv("NOTION_DATABASE_ID"))
    has_clickup = bool(os.getenv("CLICKUP_API_KEY") and os.getenv("CLICKUP_LIST_ID"))
    
    if has_trello:
        st.success("✅ Trello configured")
        enable_trello = st.checkbox("Enable Trello Export", value=True)
    else:
        st.info("ℹ️  Trello not configured")
        enable_trello = False
    
    if has_notion:
        st.success("✅ Notion configured")
        enable_notion = st.checkbox("Enable Notion Export", value=True)
    else:
        st.info("ℹ️  Notion not configured")
        enable_notion = False
    
    if has_clickup:
        st.success("✅ ClickUp configured")
        enable_clickup = st.checkbox("Enable ClickUp Export", value=True)
    else:
        st.info("ℹ️  ClickUp not configured")
        enable_clickup = False
    
    st.divider()
    st.caption("💡 Configure API keys in .env file")
    st.caption("See .env.example for template")

# ======================= MAIN APP =======================
st.markdown('<div class="main-header">AMITA System</div>', unsafe_allow_html=True)
st.markdown("**Phân tích cuộc họp tự động với AI: Transcription + Speaker Diarization + Task Extraction + Export**")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "📊 Results", "✅ Tasks", "🔗 Export"])

# ======================= TAB 1: UPLOAD =======================
with tab1:
    st.header("📤 Upload Audio File")
    
    st.info("💡 **Giới hạn:** File tối đa 200MB. File lớn hơn → chạy CLI: `python main.py`")
    
    uploaded_file = st.file_uploader(
        "Chọn file audio (MP3, WAV, M4A, AAC, OGG, FLAC)",
        type=['mp3', 'wav', 'm4a', 'aac', 'ogg', 'flac'],
        help="File tối đa 200MB",
        key="audio_uploader"
    )
    
    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        
        if file_size_mb > 200:
            st.error(f"❌ File quá lớn ({file_size_mb:.1f} MB > 200 MB)")
            st.warning("💡 Compress trước hoặc chạy CLI: `python main.py`")
            st.stop()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📁 Filename", uploaded_file.name)
        with col2:
            st.metric("📏 Size", f"{file_size_mb:.1f} MB")
        with col3:
            st.metric("📎 Type", uploaded_file.type or "unknown")
        
        # Save file
        try:
            os.makedirs("uploads", exist_ok=True)
            audio_path = os.path.join("uploads", uploaded_file.name)
            
            if os.path.exists(audio_path):
                st.warning(f"⚠️  File đã tồn tại, sẽ ghi đè")
            
            with st.spinner("📤 Đang upload..."):
                with open(audio_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
            
            st.success(f"✅ Upload thành công: {file_size_mb:.1f} MB")
            
            # Verify
            if not os.path.exists(audio_path):
                st.error("❌ File không lưu được")
                st.stop()
            
        except Exception as e:
            st.error(f"❌ Lỗi upload: {str(e)}")
            st.stop()
        
        # Process button
        if st.button("🚀 Bắt đầu phân tích", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_container = st.expander("📋 Logs", expanded=True)
            
            try:
                with log_container:
                    st.text("🔄 Starting pipeline...")
                    
                    # ✅ Verify file readable
                    try:
                        import soundfile as sf
                        info = sf.info(audio_path)
                        st.text(f"✅ Valid audio: {info.duration:.1f}s, {info.samplerate}Hz\n")
                    except Exception as e:
                        st.error(f"❌ File không đọc được: {str(e)}")
                        st.stop()
                    
                    # Update config
                    config.AUDIO_FILE = audio_path
                    config.WHISPER_MODEL = whisper_model
                    config.USE_GPU = use_gpu
                    config.ENABLE_LLM_ANALYSIS = enable_llm
                    
                    basename = os.path.splitext(os.path.basename(audio_path))[0]
                    config.WHISPER_CACHE = f"outputs/whisper_output_{basename}.json"
                    config.DIARIZATION_CACHE = f"outputs/diarization_output_{basename}.json"
                    config.GENDER_CACHE = f"outputs/gender_output_{basename}.json"
                    config.COMBINING_CACHE = f"outputs/combining_output_{basename}.json"
                    
                    os.makedirs("outputs", exist_ok=True)
                    
                    # Stage 0: Preprocessing
                    status_text.text("Stage 0/5: Audio Preprocessing...")
                    progress_bar.progress(10)
                    st.text("🎵 Preprocessing...")
                    
                    from src import audio_processor
                    enhanced_path = f"outputs/{basename}_enhanced.wav"
                    processed_audio = audio_processor.enhance_audio(audio_path, enhanced_path)
                    st.text(f"   ✅ Enhanced: {processed_audio}")
                    
                    # Stage 1: Diarization
                    status_text.text("Stage 1/5: Diarization...")
                    progress_bar.progress(25)
                    st.text("👥 Diarization...")
                    
                    from src import diarization
                    diar_result = diarization.diarize_audio(
                        processed_audio,
                        min_speakers=config.MIN_SPEAKERS,
                        max_speakers=config.MAX_SPEAKERS
                    )
                    st.text(f"   ✅ {len(set(s['speaker'] for s in diar_result))} speakers")
                    
                    # Stage 2: Gender
                    status_text.text("Stage 2/5: Gender...")
                    progress_bar.progress(40)
                    st.text("👤 Gender classification...")
                    
                    from src import gender_classifier
                    gender_result = gender_classifier.classify_gender(processed_audio)
                    st.text(f"   ✅ {len(gender_result)} speakers classified")
                    
                    # Stage 3: Whisper
                    status_text.text("Stage 3/5: Transcription...")
                    progress_bar.progress(60)
                    st.text("🎤 Transcribing...")
                    
                    from src import whisper
                    whisper_result = whisper.transcribe_audio_optimized(processed_audio, 10)
                    st.text(f"   ✅ {len(whisper_result.get('segments', []))} segments")
                    
                    # Stage 4: Combining
                    status_text.text("Stage 4/5: Combining...")
                    progress_bar.progress(80)
                    st.text("🔗 Combining...")
                    
                    from src import combiner
                    combined_result = combiner.combine_results()
                    st.text(f"   ✅ {len(combined_result)} final segments")
                    
                    # Stage 5: LLM
                    if enable_llm:
                        status_text.text("Stage 5/5: LLM Analysis...")
                        progress_bar.progress(90)
                        st.text("🧠 LLM analyzing...")
                        
                        # ✅ Check Ollama availability
                        try:
                            from src import llm_applying
                            llm_result = llm_applying.run_pipeline(
                                config.COMBINING_CACHE,
                                config.OLLAMA_MODEL,
                                "outputs"
                            )
                            
                            st.session_state['summary'] = llm_result['summary']
                            st.session_state['tasks'] = llm_result['tasks']
                            st.text(f"   ✅ {len(llm_result['tasks'])} tasks")
                        except Exception as llm_error:
                            st.warning(f"⚠️  LLM failed: {str(llm_error)}")
                            st.warning("Pipeline tiếp tục, bỏ qua LLM")
                            enable_llm = False
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Done!")
                    st.text("🎉 Complete!")
                    
                    st.session_state['processing_done'] = True
                    st.session_state['audio_path'] = audio_path
                    st.session_state['combined_result'] = combined_result
                    
                st.success("🎉 Hoàn tất! → Tab Results")
                st.balloons()
                
            except Exception as e:
                status_text.text("❌ Error!")
                st.error(f"❌ Lỗi: {str(e)}")
                
                with st.expander("🐛 Debug", expanded=True):
                    import traceback
                    st.code(traceback.format_exc())

# ======================= TAB 2: RESULTS =======================
with tab2:
    st.header("📊 Kết quả phân tích")
    
    if st.session_state.get('processing_done'):
        # Summary
        if st.session_state.get('summary'):
            st.subheader("📋 Tóm tắt cuộc họp")
            st.info(st.session_state['summary'])
        
        # Transcript preview
        st.subheader("💬 Transcript (Preview 10 segments đầu)")
        
        combined = st.session_state.get('combined_result', [])
        
        for i, seg in enumerate(combined[:10]):
            gender_emoji = "👨" if seg.get('gender') == 'Male' else "👩"
            
            st.markdown(f"""
            **{gender_emoji} {seg['speaker']}** ({seg['start']:.1f}s - {seg['end']:.1f}s)  
            _{seg['text']}_
            """)
        
        if len(combined) > 10:
            st.caption(f"... và {len(combined) - 10} segments khác")
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # Download transcript JSON
            json_str = json.dumps(combined, ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 Download Transcript (JSON)",
                data=json_str,
                file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Download summary
            if st.session_state.get('summary'):
                st.download_button(
                    label="📥 Download Summary (TXT)",
                    data=st.session_state['summary'],
                    file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    else:
        st.info("👆 Upload file audio ở tab Upload để bắt đầu")

# ======================= TAB 3: TASKS =======================
with tab3:
    st.header("✅ Danh sách công việc")
    
    if st.session_state.get('tasks'):
        tasks = st.session_state['tasks']
        
        # Filter by priority
        filter_priority = st.multiselect(
            "Lọc theo độ ưu tiên",
            ["high", "medium", "low"],
            default=["high", "medium", "low"]
        )
        
        filtered_tasks = [t for t in tasks if t.get('priority', 'medium') in filter_priority]
        
        st.caption(f"Hiển thị {len(filtered_tasks)}/{len(tasks)} tasks")
        
        # Display tasks
        for i, task in enumerate(filtered_tasks, 1):
            priority = task.get('priority', 'medium')
            priority_class = f"{priority}-priority"
            
            priority_emoji = {
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢'
            }.get(priority, '⚪')
            
            st.markdown(f"""
            <div class="task-card {priority_class}">
                <h4>{priority_emoji} Task {i}: {task['task']}</h4>
                <p><strong>👤 Assigned:</strong> {task.get('assigned_to') or 'Chưa rõ'}</p>
                <p><strong>📅 Deadline:</strong> {task.get('deadline') or 'Chưa rõ'}</p>
                <p><strong>⚡ Priority:</strong> {priority.upper()}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Download tasks
        tasks_json = json.dumps(filtered_tasks, ensure_ascii=False, indent=2)
        st.download_button(
            label="📥 Download Tasks (JSON)",
            data=tasks_json,
            file_name=f"tasks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    else:
        st.info("👆 Chạy phân tích LLM để trích xuất tasks")

# ======================= TAB 4: EXPORT =======================
with tab4:
    st.header("🔗 Export")
    
    if not st.session_state.get('tasks'):
        st.warning("⚠️  Chưa có tasks. Chạy LLM analysis trước.")
    else:
        tasks = st.session_state['tasks']
        summary = st.session_state.get('summary', '')
        
        st.info(f"📊 {len(tasks)} tasks sẵn sàng export")
        
        # ✅ Check integrations available
        if not INTEGRATIONS_AVAILABLE:
            st.warning("⚠️  Integration modules chưa cài. Install: `pip install -r requirements_web.txt`")
        else:
            # Trello
            if enable_trello and has_trello:
                if st.button("📤 Export to Trello", use_container_width=True):
                    with st.spinner("Creating Trello cards..."):
                        try:
                            trello = get_trello_client(
                                os.getenv("TRELLO_API_KEY"),
                                os.getenv("TRELLO_TOKEN"),
                                os.getenv("TRELLO_BOARD_ID")
                            )
                            urls = trello.export_tasks(tasks, summary)
                            st.success(f"✅ {len(urls)} cards created!")
                            for url in urls:
                                st.markdown(f"- [Card]({url})")
                        except Exception as e:
                            st.error(f"❌ {str(e)}")
            
            # Notion
            if enable_notion and has_notion:
                if st.button("📤 Export to Notion", use_container_width=True):
                    with st.spinner("Creating Notion pages..."):
                        try:
                            notion = get_notion_client(
                                os.getenv("NOTION_API_KEY"),
                                os.getenv("NOTION_DATABASE_ID")
                            )
                            urls = notion.export_tasks(tasks, summary)
                            st.success(f"✅ {len(urls)} pages created!")
                            for url in urls:
                                st.markdown(f"- [Page]({url})")
                        except Exception as e:
                            st.error(f"❌ {str(e)}")
            
            # ClickUp
            if enable_clickup and has_clickup:
                if st.button("📤 Export to ClickUp", use_container_width=True):
                    with st.spinner("Creating ClickUp tasks..."):
                        try:
                            clickup = get_clickup_client(
                                os.getenv("CLICKUP_API_KEY"),
                                os.getenv("CLICKUP_LIST_ID")
                            )
                            urls = clickup.export_tasks(tasks, summary)
                            st.success(f"✅ {len(urls)} tasks created!")
                            for url in urls:
                                st.markdown(f"- [Task]({url})")
                        except Exception as e:
                            st.error(f"❌ {str(e)}")

st.divider()
st.caption("AMITA System")
