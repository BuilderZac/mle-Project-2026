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


def formatComparisonTable(
    baselineSummary: BenchmarkSummary,
    modelSummaries: List[BenchmarkSummary],
    trainingTimes: Dict[str, float],
) -> str:
    floorScore = baselineSummary.mean_score
    allSummaries = [baselineSummary] + modelSummaries

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

    for s in allSummaries:
        if s.model_name == baselineSummary.model_name:
            vsFloorStr = "FLOOR"
        else:
            pct = ((s.mean_score - floorScore) / max(1e-6, floorScore)) * 100.0
            vsFloorStr = f"{pct:+8.1f}%"

        tSec = trainingTimes.get(s.model_name, 0.0)
        tStr = f"{tSec:.2f}s" if tSec > 0 else "0.00s"

        lines.append(
            f"{s.model_name:<26} | {s.mean_score:10.2f} | {vsFloorStr:>9} | "
            f"{s.median_score:8.2f} | {s.max_score:9.2f} | {s.mean_distance:9.1f} | "
            f"{s.mean_steps:10.1f} | {s.max_difficulty_reached:7.2f}x | "
            f"{s.mean_obstacles_cleared:8.2f} | {tStr:>10}"
        )

    lines.append(sep)
    lines.append("\nDetailed Per-Model Failure Breakdown:")
    for s in allSummaries:
        lines.append(f"  - {s.model_name:<26}: {s.death_counts}")

    return "\n".join(lines)


def runTrainingAndComparison(
    evalEpisodes: int = 100,
    evalSeed: int = 1000,
) -> Dict[str, object]:
    trainingTimes: Dict[str, float] = {}

    print("1. Evaluating RandomBaselineModel (Floor) on 100 test episodes...")
    baselineModel = RandomBaselineModel(seed=evalSeed)
    baselineSummary = evaluate_model(
        baselineModel, num_episodes=evalEpisodes, base_seed=evalSeed
    )
    trainingTimes[baselineModel.name] = 0.0

    print("\n2. Training Model 1: GeneticLinearModel (Genetic Algorithm)...")
    t0 = time.perf_counter()
    linearModel = GeneticLinearModel(seed=42)
    linearModel.train(
        generations=16,
        population_size=28,
        episodes_per_eval=8,
        seed=42,
        verbose=True,
    )
    trainingTimes[linearModel.name] = round(time.perf_counter() - t0, 2)
    linearSummary = evaluate_model(
        linearModel, num_episodes=evalEpisodes, base_seed=evalSeed
    )

    print("\n3. Training Model 2: QLearningModel (n-Step TD Q-Learning)...")
    t0 = time.perf_counter()
    qModel = QLearningModel(seed=42)
    qModel.train(
        num_episodes=300,
        learning_rate=0.15,
        gamma=0.96,
        n_step=18,
        seed=42,
        verbose=True,
    )
    trainingTimes[qModel.name] = round(time.perf_counter() - t0, 2)
    qSummary = evaluate_model(
        qModel, num_episodes=evalEpisodes, base_seed=evalSeed
    )

    print("\n4. Training Model 3: NeuroEvolutionMLPModel (2-Layer MLP + Cross-Entropy Method)...")
    t0 = time.perf_counter()
    mlpModel = NeuroEvolutionMLPModel(seed=42)
    mlpModel.train(
        generations=16,
        population_size=28,
        elite_frac=0.25,
        episodes_per_eval=8,
        seed=42,
        verbose=True,
    )
    trainingTimes[mlpModel.name] = round(time.perf_counter() - t0, 2)
    mlpSummary = evaluate_model(
        mlpModel, num_episodes=evalEpisodes, base_seed=evalSeed
    )

    # Save trained model parameters
    save_trained_models(linearModel, qModel, mlpModel, path="trainedModels.json")

    modelSummaries = [linearSummary, qSummary, mlpSummary]
    tableStr = formatComparisonTable(baselineSummary, modelSummaries, trainingTimes)
    print(tableStr)

    comparisonData = {
        "benchmarkEpisodes": evalEpisodes,
        "evalBaseSeed": evalSeed,
        "baselineFloor": asdict(baselineSummary),
        "models": {
            s.model_name: {
                **asdict(s),
                "improvementOverFloorPct": round(
                    ((s.mean_score - baselineSummary.mean_score) / baselineSummary.mean_score) * 100.0,
                    2,
                ),
                "trainingTimeSec": trainingTimes[s.model_name],
            }
            for s in modelSummaries
        },
    }
    with open("modelComparison.json", "w", encoding="utf-8") as f:
        json.dump(comparisonData, f, indent=2)

    return comparisonData


if __name__ == "__main__":
    runTrainingAndComparison(evalEpisodes=100, evalSeed=1000)
