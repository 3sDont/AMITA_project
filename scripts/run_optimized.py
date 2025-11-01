"""Run optimized pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ✅ FIX: Import từ pipeline.optimized
from pipeline.optimized import main as run_optimized_pipeline

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Pipeline (Cache + Parallel)")
    parser.add_argument("audio_path", type=str)
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--chunk-minutes", type=int, default=10)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    
    args = parser.parse_args()
    
    try:
        run_optimized_pipeline(
            args.audio_path,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            chunk_minutes=args.chunk_minutes,
            use_parallel=args.parallel,
            use_cache=not args.no_cache,
            force_rerun=args.force_rerun
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
