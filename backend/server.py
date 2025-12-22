"""
FastAPI Backend for AMITA System
Connects React UI with Python Pipeline
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
import os
from pathlib import Path
import shutil
import json
from datetime import datetime
import time
import asyncio
from dotenv import load_dotenv
import config

# Load environment variables from .env file
load_dotenv()

# Add backend/src to path for local modules
sys.path.insert(0, str(config.BACKEND_DIR / "src"))

# Set HF_TOKEN from environment variable for diarization
if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    print("✅ HF_TOKEN loaded from .env")

# Import AMITA modules from backend/src
try:
    # Workaround for torchaudio backend issue
    import warnings
    warnings.filterwarnings('ignore')
    
    # Import pipeline manager and stages from local src/
    from pipeline_manager import MeetingPipeline
    print("✅ MeetingPipeline loaded from backend/src")
    
    from stage1_preprocessing import AudioPreprocessor
    print("✅ AudioPreprocessor loaded")
    
    from stage3_whisper import WhisperProcessor
    print("✅ WhisperProcessor loaded")
    
    try:
        from stage4_diarization import DiarizationProcessor
        print("✅ DiarizationProcessor loaded")
        DIARIZATION_AVAILABLE = True
    except Exception as e:
        print(f"⚠️  DiarizationProcessor not available: {e}")
        DIARIZATION_AVAILABLE = False
    
    try:
        from stage7_gender import GenderClassifier
        print("✅ GenderClassifier loaded")
        GENDER_AVAILABLE = True
    except Exception as e:
        print(f"⚠️  GenderClassifier not available: {e}")
        GENDER_AVAILABLE = False
    
    try:
        from stage9_llm import LLMAnalyzer
        print("✅ LLMAnalyzer loaded")
        LLM_AVAILABLE = True
    except Exception as e:
        print(f"⚠️  LLMAnalyzer not available: {e}")
        LLM_AVAILABLE = False
    
    PIPELINE_ENABLED = True
    print("✅ AMITA Pipeline loaded successfully!")
    
except Exception as e:
    PIPELINE_ENABLED = False
    DIARIZATION_AVAILABLE = False
    GENDER_AVAILABLE = False
    LLM_AVAILABLE = False
    print(f"⚠️  AMITA Pipeline not available: {e}")
    import traceback
    traceback.print_exc()

app = FastAPI(title="AMITA API", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create data directories from config
config.DATA_DIR.mkdir(exist_ok=True)
config.UPLOAD_DIR.mkdir(exist_ok=True)
config.OUTPUT_DIR.mkdir(exist_ok=True)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "AMITA API is running", "version": "1.0.0"}


@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Upload audio file"""
    try:
        # Validate file type
        if not any(file.filename.endswith(ext) for ext in config.ALLOWED_AUDIO_FORMATS):
            raise HTTPException(400, f"Invalid file type. Supported: {', '.join(config.ALLOWED_AUDIO_FORMATS)}")
        
        # Save uploaded file
        file_path = config.UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Validate file was written successfully
        file_size = file_path.stat().st_size
        if file_size == 0:
            file_path.unlink()  # Delete empty file
            raise HTTPException(400, "Upload failed: File is empty")
        
        if file_size < 1024:  # Less than 1KB is suspicious
            file_path.unlink()  # Delete corrupted file
            raise HTTPException(400, f"Upload failed: File too small ({file_size} bytes), possibly corrupted")
        
        return JSONResponse({
            "success": True,
            "filename": file.filename,
            "path": str(file_path),
            "size": file_size
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")


@app.post("/api/process")
async def process_audio(data: dict):
    """Process audio through AMITA pipeline"""
    try:
        filename = data.get("filename")
        if not filename:
            raise HTTPException(400, "Filename is required")
        
        audio_path = config.UPLOAD_DIR / filename
        if not audio_path.exists():
            raise HTTPException(404, f"Audio file not found: {audio_path}")
        
        print(f"🎵 Processing audio: {filename}")
        print(f"📂 Full path: {audio_path}")
        print(f"📊 File size: {audio_path.stat().st_size} bytes")
        
        if PIPELINE_ENABLED:
            # REAL PIPELINE - Process with AMITA MeetingPipeline
            try:
                # Use centralized output directory from config
                output_dir = config.OUTPUT_DIR
                
                # Initialize pipeline
                print(f"🎯 Initializing MeetingPipeline for {filename}...")
                pipeline = MeetingPipeline(
                    audio_path=str(audio_path),
                    output_dir=str(output_dir)
                )
                
                # Configure pipeline
                pipeline.config.update({
                    "chunk_duration_minutes": 10,
                    "enable_vad": True,
                    "enable_gender": GENDER_AVAILABLE,
                    "enable_llm": LLM_AVAILABLE and hasattr(config, 'ENABLE_LLM_ANALYSIS') and config.ENABLE_LLM_ANALYSIS,
                    "min_speakers": getattr(config, 'MIN_SPEAKERS', None),
                    "max_speakers": getattr(config, 'MAX_SPEAKERS', None)
                })
                
                # Run full pipeline
                print("🚀 Running full pipeline...")
                pipeline.run()
                
                # Extract results from meeting_data
                segments = pipeline.meeting_data.get("segments", [])
                speakers_info = pipeline.meeting_data.get("speakers", {})
                llm_data = pipeline.meeting_data.get("llm", {})
                metadata = pipeline.meeting_data.get("metadata", {})
                
                print(f"📊 Pipeline results: {len(segments)} segments, {len(speakers_info)} speakers")
                
                # Format transcript for UI (limited by config)
                transcript = []
                #for seg in segments[:config.MAX_TRANSCRIPT_SEGMENTS]:
                for seg in segments:
                    # Try multiple keys for start time (pipeline uses 'start_time', some stages use 'start')
                    start_time = seg.get('start_time', seg.get('start', 0))
                    text = seg.get('text', '').strip()
                    if not text:  # Skip empty segments
                        continue
                    transcript.append({
                        "time": f"{int(start_time//60):02d}:{int(start_time%60):02d}",
                        "speaker": seg.get('speaker_display', seg.get('speaker', 'Speaker 1')),
                        "text": text
                    })
                
                print(f"📝 Formatted {len(transcript)} transcript entries")
                
                # Get summary and tasks from LLM or default
                summary = llm_data.get('summary') or "Meeting transcription completed successfully using AMITA Pipeline."
                tasks = llm_data.get('tasks', [])
                
                # Debug: Check if tasks are empty
                print(f"📊 LLM Data Keys: {list(llm_data.keys())}")
                print(f"📊 Tasks from LLM: {len(tasks)} tasks")
                if len(tasks) == 0:
                    print(f"⚠️  No tasks from LLM, using fallback")
                    tasks = [
                        "Review the complete transcript",
                        "Identify key action items",
                        "Share with meeting participants"
                    ]
                
                # Calculate duration
                duration = 0
                if segments:
                    duration = segments[-1].get('end', 0)
                
                # Count speakers
                num_speakers = len(speakers_info)
                
                return JSONResponse({
                    "success": True,
                    "processed_at": datetime.now().isoformat(),
                    "transcript": transcript,
                    "summary": summary,
                    "tasks": tasks,
                    "metadata": {
                        "filename": filename,
                        "duration": duration,
                        "speakers": num_speakers,
                        "segments": len(segments),
                        "pipeline": "AMITA Pipeline v2.0",
                        "pipeline_version": metadata.get("pipeline_version", "2.0"),
                        "meeting_id": metadata.get("meeting_id"),
                        "diarization": DIARIZATION_AVAILABLE,
                        "gender_classification": GENDER_AVAILABLE,
                        "llm_analysis": LLM_AVAILABLE
                    }
                })
            
            except Exception as pipeline_error:
                print(f"❌ Pipeline error: {pipeline_error}")
                import traceback
                traceback.print_exc()
                raise HTTPException(500, f"Pipeline processing failed: {str(pipeline_error)}")
        
        # FALLBACK - Mock data if pipeline not available
        await asyncio.sleep(2)
        
        transcript = [
            {
                "time": "00:00",
                "speaker": "Speaker A",
                "text": f"Processing audio file: {filename}"
            },
            {
                "time": "00:05",
                "speaker": "Speaker B",
                "text": "The AMITA system will extract speakers, transcribe speech, and generate summaries."
            },
            {
                "time": "00:15",
                "speaker": "Speaker A",
                "text": "Key topics will be identified and action items will be extracted automatically."
            },
            {
                "time": "00:25",
                "speaker": "Speaker C",
                "text": "This demonstration shows the UI flow. Pipeline is ready but using demo data."
            }
        ]
        
        summary = f"Audio file '{filename}' uploaded. Pipeline is available but using demo mode. Install required models to enable full processing."
        
        tasks = [
            "Download Whisper model: base",
            "Setup PyAnnote diarization models",
            "Configure LLM for summary generation",
            "Process audio with full pipeline"
        ]
        
        return JSONResponse({
            "success": True,
            "processed_at": datetime.now().isoformat(),
            "transcript": transcript,
            "summary": summary,
            "tasks": tasks,
            "metadata": {
                "filename": filename,
                "filesize": audio_path.stat().st_size,
                "pipeline": "Demo Mode",
                "note": "Pipeline loaded but using demo data"
            }
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Processing failed: {str(e)}")


@app.get("/api/status")
async def get_status():
    """Get system status"""
    if PIPELINE_ENABLED:
        return JSONResponse({
            "status": "running",
            "pipeline": "enabled",
            "whisper_model": config.WHISPER_MODEL,
            "language": config.LANGUAGE,
            "gpu_enabled": config.USE_GPU,
            "llm_enabled": config.ENABLE_LLM_ANALYSIS
        })
    else:
        return JSONResponse({
            "status": "running",
            "pipeline": "demo_mode",
            "note": "AMITA pipeline not fully loaded - using demo data",
            "whisper_model": "base",
            "language": "vi",
            "gpu_enabled": False,
            "llm_enabled": False
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
