

import os
import time
import json
import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.execution_reward import (
    execution_reward_batch_sequential,
    execution_reward_batch_parallel,
    execution_reward_timed,
    set_use_cache,
    clear_result_cache
)

def generate_mock_rollouts(num_rollouts: int = 100):
    """Generates heavy queries across multiple databases to properly test true concurrency."""
    print(f"\nGenerating {num_rollouts} heavy rollouts to simulate RLHF query workload...")
    
    real_dbs = glob.glob("data/database/*/*.sqlite")
    if real_dbs:
        print(f"Found {len(real_dbs)} real SQLite databases. Distributing workload...")
    else:
        print("No real databases found. Using simulated paths.")
        
    rollouts = []
    for i in range(num_rollouts):
        if real_dbs:
            db_path = real_dbs[i % len(real_dbs)]
        else:
            db_path = f"data/database/db_{i % 20}/db_{i % 20}.sqlite"
            
        heavy_sql = f"""
        WITH RECURSIVE cnt(x) AS (
            SELECT 1
            UNION ALL
            SELECT x+1 FROM cnt WHERE x < {50000 + (i % 1000)}
        )
        SELECT sum(x) FROM cnt;
        """
        clean_sql = heavy_sql.replace("\n", " ").strip()
        rollouts.append((clean_sql, db_path, clean_sql))
        
    return rollouts

def profile_bottlenecks(rollouts):
    """Profiles CPU usage to identify time spent in parsing, planning, and execution."""
    print("\n" + "="*65)
    print(" 🔍 CPU PROFILING: IDENTIFYING BOTTLENECKS (100 Rollouts)")
    print("="*65)
    
    clear_result_cache()
    set_use_cache(False) # Disable cache to force real work
    
    total_parse = 0.0
    total_plan = 0.0
    total_exec = 0.0
    
    # Profile just the first 100 to get an accurate average without waiting forever
    sample_size = min(100, len(rollouts))
    sample_rollouts = rollouts[:sample_size]
    
    for pred, db, gold in sample_rollouts:
        _, timings = execution_reward_timed(pred, db, gold, measure_plan=True)
        total_parse += timings['parse_s']
        total_plan += timings['plan_s']
        total_exec += timings['exec_s']
        
    total_time = total_parse + total_plan + total_exec
    if total_time == 0: total_time = 0.0001 # Prevent div by zero
    
    print(f"{'Phase':<15} | {'Avg Time (ms)':<15} | {'% of Total CPU':<15}")
    print("-" * 65)
    print(f"{'Regex Parsing':<15} | {(total_parse/sample_size)*1000:<15.2f} | {(total_parse/total_time)*100:<14.1f}%")
    print(f"{'Query Planning':<15} | {(total_plan/sample_size)*1000:<15.2f} | {(total_plan/total_time)*100:<14.1f}%")
    print(f"{'DB Execution':<15} | {(total_exec/sample_size)*1000:<15.2f} | {(total_exec/total_time)*100:<14.1f}%")
    print("="*65 + "\n")

def run_benchmark_for_setting(rollouts, use_cache: bool, max_workers: int):
    set_use_cache(use_cache)
    
    # Sequential
    clear_result_cache()
    start_time = time.perf_counter()
    execution_reward_batch_sequential(rollouts)
    sequential_s = time.perf_counter() - start_time

    # Parallel
    clear_result_cache()
    start_time = time.perf_counter()
    execution_reward_batch_parallel(rollouts, max_workers=max_workers)
    parallel_s = time.perf_counter() - start_time

    speedup = sequential_s / parallel_s if parallel_s > 0 else 0

    return {
        "sequential_s": sequential_s,
        "parallel_s": parallel_s,
        "speedup": speedup
    }

def print_comparison_table(results):
    print("="*65)
    print(f"{'Setting':<16} | {'Sequential (s)':<14} | {'Parallel (s)':<14} | {'Speedup':<10}")
    print("-" * 65)
    for setting, key in [("With Cache", "with_cache"), ("Without Cache", "without_cache")]:
        seq = results[key]['sequential_s']
        par = results[key]['parallel_s']
        spd = results[key]['speedup']
        print(f"{setting:<16} | {seq:<14.4f} | {par:<14.4f} | {spd:<9.2f}x")
    print("="*65 + "\n")

def plot_results(results, output_path: str):
    labels = ['With Cache', 'Without Cache']
    seq_times = [results['with_cache']['sequential_s'], results['without_cache']['sequential_s']]
    par_times = [results['with_cache']['parallel_s'], results['without_cache']['parallel_s']]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, seq_times, width, label='Sequential', color='#4C72B0')
    ax.bar(x + width/2, par_times, width, label='Parallel', color='#DD8452')

    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Text2SQL Reward Execution: Sequential vs Parallel')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)

    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Benchmark SQL Execution Reward")
    parser.add_argument("--n", type=int, default=1000, help="Number of rollouts to benchmark")
    parser.add_argument("--max-workers", type=int, default=20, help="Max workers for parallel execution")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    rollouts = generate_mock_rollouts(args.n)
    
    # NEW: Fulfills Requirement 6
    profile_bottlenecks(rollouts)
    
    print("Starting Main Scalability Benchmarks...")

    print("Running Experiment A: Cache ENABLED...")
    results_with_cache = run_benchmark_for_setting(rollouts, use_cache=True, max_workers=args.max_workers)

    print("Running Experiment B: Cache DISABLED...")
    results_without_cache = run_benchmark_for_setting(rollouts, use_cache=False, max_workers=args.max_workers)

    final_results = {
        "with_cache": results_with_cache,
        "without_cache": results_without_cache
    }

    json_path = "results/task1_results.json"
    with open(json_path, 'w') as f:
        json.dump(final_results, f, indent=4)

    print_comparison_table(final_results)
    plot_results(final_results, "results/task1_plot.png")

if __name__ == "__main__":
    main()
