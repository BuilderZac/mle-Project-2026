"""
Train and Benchmark 3 Distinct Models Against the Random Baseline Floor.

Models compared:
  0. Baseline: `RandomBaselineModel` (Uniformly random inputs - Floor)
  1. Model 1 : `GeneticLinearModel` (Linear Policy trained via Genetic Algorithm)
  2. Model 2 : `QLearningModel` (Value-based Tabular/Tile-Coded RL trained via TD(0) + Replay)
  3. Model 3 : `NeuroEvolutionMLPModel` (2-Layer Nonlinear MLP trained via Cross-Entropy Method)

All models are evaluated on the exact same 100-episode test suite (`base_seed=1000`),
disjoint from their training episodes, and results are saved to `model_comparison.json`.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Dict, List

from baseline_model import (
    BaseModel,
    BenchmarkSummary,
    RandomBaselineModel,
    evaluate_model,
)
from models import (
    GeneticLinearModel,
    NeuroEvolutionMLPModel,
    QLearningModel,
    save_trained_models,
)


def format_comparison_table(
    baseline_summary: BenchmarkSummary,
    model_summaries: List[BenchmarkSummary],
    training_times: Dict[str, float],
) -> str:
    floor_score = baseline_summary.mean_score
    all_summaries = [baseline_summary] + model_summaries

    header = (
        f"{'Model Name':<26} | {'Mean Score':>10} | {'vs Floor':>9} | "
        f"{'Median':>8} | {'Max Score':>9} | {'Mean Dist':>9} | "
        f"{'Mean Steps':>10} | {'Max Diff':>8} | {'Obs Clrd':>8} | {'Train Time':>10}"
    )
    sep = "-" * len(header)
    lines = [
        "\n==================================================================================================================================",
        "                         MODEL BENCHMARK COMPARISON (100 Test Episodes, Seed=1000..1099)",
        "==================================================================================================================================",
        header,
        sep,
    ]

    for s in all_summaries:
        if s.model_name == baseline_summary.model_name:
            vs_floor_str = "FLOOR"
        else:
            pct = ((s.mean_score - floor_score) / max(1e-6, floor_score)) * 100.0
            vs_floor_str = f"{pct:+8.1f}%"

        t_sec = training_times.get(s.model_name, 0.0)
        t_str = f"{t_sec:.2f}s" if t_sec > 0 else "0.00s"

        lines.append(
            f"{s.model_name:<26} | {s.mean_score:10.2f} | {vs_floor_str:>9} | "
            f"{s.median_score:8.2f} | {s.max_score:9.2f} | {s.mean_distance:9.1f} | "
            f"{s.mean_steps:10.1f} | {s.max_difficulty_reached:7.2f}x | "
            f"{s.mean_obstacles_cleared:8.2f} | {t_str:>10}"
        )

    lines.append(sep)
    lines.append("\nDetailed Per-Model Failure Breakdown:")
    for s in all_summaries:
        lines.append(f"  - {s.model_name:<26}: {s.death_counts}")

    return "\n".join(lines)


def run_training_and_comparison(
    eval_episodes: int = 100,
    eval_seed: int = 1000,
) -> Dict[str, object]:
    training_times: Dict[str, float] = {}

    print("1. Evaluating RandomBaselineModel (Floor) on 100 test episodes...")
    baseline_model = RandomBaselineModel(seed=eval_seed)
    baseline_summary = evaluate_model(
        baseline_model, num_episodes=eval_episodes, base_seed=eval_seed
    )
    training_times[baseline_model.name] = 0.0

    print("\n2. Training Model 1: GeneticLinearModel (Genetic Algorithm)...")
    t0 = time.perf_counter()
    linear_model = GeneticLinearModel(seed=42)
    linear_model.train(
        generations=16,
        population_size=28,
        episodes_per_eval=8,
        seed=42,
        verbose=True,
    )
    training_times[linear_model.name] = round(time.perf_counter() - t0, 2)
    linear_summary = evaluate_model(
        linear_model, num_episodes=eval_episodes, base_seed=eval_seed
    )

    print("\n3. Training Model 2: QLearningModel (n-Step TD Q-Learning)...")
    t0 = time.perf_counter()
    q_model = QLearningModel(seed=42)
    q_model.train(
        num_episodes=300,
        learning_rate=0.15,
        gamma=0.96,
        n_step=18,
        seed=42,
        verbose=True,
    )
    training_times[q_model.name] = round(time.perf_counter() - t0, 2)
    q_summary = evaluate_model(
        q_model, num_episodes=eval_episodes, base_seed=eval_seed
    )

    print("\n4. Training Model 3: NeuroEvolutionMLPModel (2-Layer MLP + Cross-Entropy Method)...")
    t0 = time.perf_counter()
    mlp_model = NeuroEvolutionMLPModel(seed=42)
    mlp_model.train(
        generations=16,
        population_size=28,
        elite_frac=0.25,
        episodes_per_eval=8,
        seed=42,
        verbose=True,
    )
    training_times[mlp_model.name] = round(time.perf_counter() - t0, 2)
    mlp_summary = evaluate_model(
        mlp_model, num_episodes=eval_episodes, base_seed=eval_seed
    )

    # Save trained model parameters
    save_trained_models(linear_model, q_model, mlp_model, path="trained_models.json")

    model_summaries = [linear_summary, q_summary, mlp_summary]
    table_str = format_comparison_table(baseline_summary, model_summaries, training_times)
    print(table_str)

    comparison_data = {
        "benchmark_episodes": eval_episodes,
        "eval_base_seed": eval_seed,
        "baseline_floor": asdict(baseline_summary),
        "models": {
            s.model_name: {
                **asdict(s),
                "improvement_over_floor_pct": round(
                    ((s.mean_score - baseline_summary.mean_score) / baseline_summary.mean_score) * 100.0,
                    2,
                ),
                "training_time_sec": training_times[s.model_name],
            }
            for s in model_summaries
        },
    }
    with open("model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)

    return comparison_data


if __name__ == "__main__":
    run_training_and_comparison(eval_episodes=100, eval_seed=1000)
