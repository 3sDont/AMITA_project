"""
Tool để xem và so sánh các benchmark logs.
"""
import json
import os
from pathlib import Path
from datetime import datetime
import sys


def load_all_benchmarks(benchmark_dir=None):
    """Load tất cả benchmark files."""
    # ✅ FIX: Mặc định từ data/logs/benchmarks
    if benchmark_dir is None:
        benchmark_dir = os.path.join("data", "logs", "benchmarks")
    
    if not os.path.exists(benchmark_dir):
        print(f"❌ Không tìm thấy thư mục {benchmark_dir}")
        return []
    
    benchmarks = []
    for file in Path(benchmark_dir).glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["_filename"] = file.name
            benchmarks.append(data)
    
    return sorted(benchmarks, key=lambda x: x.get("timestamp", ""), reverse=True)


def format_time(seconds):
    """Format thời gian đẹp."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def compare_benchmarks(benchmarks):
    """So sánh các benchmarks."""
    if not benchmarks:
        print("❌ Không có benchmark nào để so sánh")
        return
    
    print(f"\n{'='*80}")
    print(f"📊 BENCHMARK COMPARISON ({len(benchmarks)} runs)")
    print(f"{'='*80}\n")
    
    # Header
    print(f"{'Run':<5} {'Date':<20} {'Device':<8} {'Duration':<10} {'Total':<10} {'Speedup':<10} {'Method':<12}")
    print(f"{'-'*80}")
    
    for idx, b in enumerate(benchmarks, 1):
        summary = b.get("summary", {})
        system = b.get("system_info", {})
        config = b.get("config", {})
        
        date = b.get("timestamp", "N/A")[:16]
        device = system.get("device", "N/A")
        duration = system.get("audio_duration_minutes", 0)
        total_time = summary.get("total_time_seconds", 0)
        speedup = summary.get("speedup_vs_realtime", 0)
        
        # Xác định method
        steps = b.get("steps", {})
        trans_step = steps.get("transcription", {})
        method = trans_step.get("method", "standard")
        if config.get("use_cache") and trans_step.get("cached"):
            method += "+cache"
        
        print(f"{idx:<5} {date:<20} {device:<8} {duration:<10.1f} {format_time(total_time):<10} "
              f"{speedup:<10.2f}x {method:<12}")
    
    print(f"{'-'*80}\n")
    
    # Chi tiết breakdown
    print("📋 DETAILED BREAKDOWN (last 3 runs):\n")
    for idx, b in enumerate(benchmarks[:3], 1):
        print(f"Run {idx}: {b.get('_filename', 'N/A')}")
        steps = b.get("steps", {})
        
        print(f"   0️⃣  Preprocessing:    {format_time(steps.get('preprocessing', {}).get('time_seconds', 0)):>8}")
        print(f"   1️⃣  Diarization:      {format_time(steps.get('diarization', {}).get('time_seconds', 0)):>8}")
        print(f"   2️⃣  Transcription:    {format_time(steps.get('transcription', {}).get('time_seconds', 0)):>8}")
        print(f"   3️⃣  Gender classify:  {format_time(steps.get('gender_classification', {}).get('time_seconds', 0)):>8}")
        print(f"   4️⃣  Combine:          {format_time(steps.get('combine', {}).get('time_seconds', 0)):>8}")
        print()
    
    # Thống kê tổng hợp
    print("📈 STATISTICS:\n")
    
    # Nhóm theo device
    by_device = {}
    for b in benchmarks:
        device = b.get("system_info", {}).get("device", "unknown")
        if device not in by_device:
            by_device[device] = []
        by_device[device].append(b)
    
    for device, runs in by_device.items():
        avg_speedup = sum(r.get("summary", {}).get("speedup_vs_realtime", 0) for r in runs) / len(runs)
        avg_time = sum(r.get("summary", {}).get("total_time_seconds", 0) for r in runs) / len(runs)
        print(f"   {device.upper()}:")
        print(f"      - Runs: {len(runs)}")
        print(f"      - Avg speedup: {avg_speedup:.2f}x realtime")
        print(f"      - Avg time: {format_time(avg_time)}")
    
    print()


def show_latest(benchmark_dir="benchmarks", n=1):
    """Hiển thị n benchmark gần nhất."""
    benchmarks = load_all_benchmarks(benchmark_dir)
    
    if not benchmarks:
        print("❌ Không có benchmark nào")
        return
    
    for b in benchmarks[:n]:
        print(f"\n{'='*60}")
        print(f"📊 BENCHMARK: {b.get('_filename', 'N/A')}")
        print(f"{'='*60}\n")
        
        print(f"⏰ Timestamp: {b.get('timestamp', 'N/A')}")
        
        system = b.get("system_info", {})
        print(f"\n💻 System Info:")
        for key, value in system.items():
            print(f"   - {key}: {value}")
        
        config = b.get("config", {})
        print(f"\n⚙️  Config:")
        for key, value in config.items():
            print(f"   - {key}: {value}")
        
        steps = b.get("steps", {})
        print(f"\n📋 Steps:")
        for step_name, step_data in steps.items():
            time_sec = step_data.get("time_seconds", 0)
            cached = " (cached)" if step_data.get("cached") else ""
            print(f"   - {step_name}: {format_time(time_sec)}{cached}")
        
        summary = b.get("summary", {})
        print(f"\n📊 Summary:")
        print(f"   - Total time: {format_time(summary.get('total_time_seconds', 0))}")
        print(f"   - Speedup: {summary.get('speedup_vs_realtime', 0):.2f}x realtime")
        if summary.get('cache_saved_time', 0) > 0:
            print(f"   - Cache saved: {format_time(summary['cache_saved_time'])}")


def export_csv(benchmark_dir="benchmarks", output_file="benchmarks.csv"):
    """Export benchmarks ra CSV để phân tích."""
    benchmarks = load_all_benchmarks(benchmark_dir)
    
    if not benchmarks:
        print("❌ Không có benchmark nào")
        return
    
    import csv
    
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "timestamp", "device", "audio_duration_min", 
            "preprocessing_s", "diarization_s", "transcription_s", "gender_s", "combine_s",
            "total_time_s", "speedup", "method", "use_cache"
        ])
        
        # Data
        for b in benchmarks:
            system = b.get("system_info", {})
            steps = b.get("steps", {})
            summary = b.get("summary", {})
            config = b.get("config", {})
            
            writer.writerow([
                b.get("timestamp", ""),
                system.get("device", ""),
                system.get("audio_duration_minutes", 0),
                steps.get("preprocessing", {}).get("time_seconds", 0),
                steps.get("diarization", {}).get("time_seconds", 0),
                steps.get("transcription", {}).get("time_seconds", 0),
                steps.get("gender_classification", {}).get("time_seconds", 0),
                steps.get("combine", {}).get("time_seconds", 0),
                summary.get("total_time_seconds", 0),
                summary.get("speedup_vs_realtime", 0),
                steps.get("transcription", {}).get("method", ""),
                config.get("use_cache", False)
            ])
    
    print(f"✅ Exported to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Viewer")
    parser.add_argument("--compare", action="store_true", help="Compare all benchmarks")
    parser.add_argument("--latest", type=int, default=1, help="Show latest N benchmarks")
    parser.add_argument("--export-csv", type=str, help="Export to CSV file")
    parser.add_argument("--dir", type=str, default="benchmarks", help="Benchmark directory")
    
    args = parser.parse_args()
    
    if args.compare:
        benchmarks = load_all_benchmarks(args.dir)
        compare_benchmarks(benchmarks)
    elif args.export_csv:
        export_csv(args.dir, args.export_csv)
    else:
        show_latest(args.dir, args.latest)
