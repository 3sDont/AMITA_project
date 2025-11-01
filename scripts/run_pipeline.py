"""
Main entry point for running the pipeline.
"""
import sys
import os
import argparse
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    parser = argparse.ArgumentParser(description="Audio Processing Pipeline")
    parser.add_argument("audio_path", type=str, help="Path to audio/video file")
    parser.add_argument("--num-speakers", type=int, default=None, help="Exact number of speakers")
    parser.add_argument("--min-speakers", type=int, default=None, help="Minimum speakers")
    parser.add_argument("--max-speakers", type=int, default=None, help="Maximum speakers")
    parser.add_argument("--chunk-minutes", type=int, default=10, help="Chunk length (minutes)")
    
    args = parser.parse_args()
    
    # Load environment (if any)
    load_dotenv()

    # Prefer using the function-based standard pipeline (src/pipeline/standard.py)
    try:
        from pipeline.standard import main as run_standard_pipeline
    except Exception:
        print("⚠️  Could not import pipeline.standard.main; ensure src/pipeline/standard.py exists")
        raise

    try:
        run_standard_pipeline(
            args.audio_path,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            chunk_minutes=args.chunk_minutes
        )
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
