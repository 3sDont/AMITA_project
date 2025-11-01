"""Run standard pipeline."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ✅ FIX: Import từ pipeline.standard (không phải pipeline.py)
from pipeline.standard import main as run_standard_pipeline

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Standard Audio Processing Pipeline")
    parser.add_argument("audio_path", type=str, help="Path to audio/video file")
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--chunk-minutes", type=int, default=10)
    
    args = parser.parse_args()
    
    try:
        run_standard_pipeline(
            args.audio_path,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            chunk_minutes=args.chunk_minutes
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
