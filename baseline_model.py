"""
Baseline Random Input Model and Benchmarking Suite.

Provides:
  - `BaseModel`: Standard interface for all game-playing models.
  - `RandomBaselineModel`: Pure random-input model used as the performance floor
    to compare all future models against.
  - `EvolvableLinearModel`: A lightweight linear policy model that starts from
    random weights and can be mutated/rolled back by `evoHarness`.
  - `evaluate_model` / `run_baseline_benchmark`: Reproducible multi-episode
    evaluation functions that compute floor metrics (mean, median, std, min, max).
"""

from __future__ import annotations

import copy
import json
import math
import random
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from game import (
    ACTION_NAMES,
    NUM_ACTIONS,
    SidescrollerGame,
)


class BaseModel:
    """Interface for all models evaluated in the sidescroller benchmark."""

    name: str = "BaseModel"

    def select_action(self, state: List[float]) -> int:
        """Select a discrete action in [0, NUM_ACTIONS - 1] given the state vector."""
        raise NotImplementedError

    def clone(self) -> "BaseModel":
        """Return a deep copy of this model (used for rollback / regression)."""
        return copy.deepcopy(self)

    def mutate(
        self,
        mutation_rate: float = 0.35,
        mutation_scale: float = 0.6,
        seed: Optional[int] = None,
    ) -> None:
        """Mutate model parameters in-place (for evolutionary replacement)."""
        pass


class RandomBaselineModel(BaseModel):
    """
    Baseline model that ignores state observations and selects uniformly random
    inputs at every game tick. Serves as the performance floor against which all
    future evolved or trained models are compared.
    """

    def __init__(self, seed: Optional[int] = None):
        self.name = "RandomBaselineModel"
        self.seed = seed
        self.rng = random.Random(seed)

    def reset_rng(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = random.Random(seed)
        elif self.seed is not None:
            self.rng = random.Random(self.seed)

    def select_action(self, state: List[float]) -> int:
        return self.rng.randrange(NUM_ACTIONS)


class EvolvableLinearModel(BaseModel):
    """
    A simple linear policy model (`action = argmax(W @ state + b)`) that can be
    mutated and rolled back by `evoHarness` to demonstrate evolutionary survival
    of the fittest over the `RandomBaselineModel` floor.
    """

    STATE_DIM = 16

    def __init__(self, seed: Optional[int] = None, generation: int = 0):
        self.generation = generation
        self.name = f"EvolvableLinearModel(gen={generation})"
        rng = random.Random(seed)
        self.weights: List[List[float]] = [
            [rng.gauss(0.0, 0.5) for _ in range(self.STATE_DIM)]
            for _ in range(NUM_ACTIONS)
        ]
        self.biases: List[float] = [rng.gauss(0.0, 0.2) for _ in range(NUM_ACTIONS)]
        self._mutation_rng = random.Random(seed)

    def select_action(self, state: List[float]) -> int:
        best_action = 0
        best_score = -float("inf")
        for a in range(NUM_ACTIONS):
            w_row = self.weights[a]
            logit = self.biases[a] + sum(w * s for w, s in zip(w_row, state))
            if logit > best_score:
                best_score = logit
                best_action = a
        return best_action

    def mutate(
        self,
        mutation_rate: float = 0.35,
        mutation_scale: float = 0.6,
        seed: Optional[int] = None,
    ) -> None:
        rng = random.Random(seed) if seed is not None else self._mutation_rng
        self.generation += 1
        self.name = f"EvolvableLinearModel(gen={self.generation})"
        for a in range(NUM_ACTIONS):
            for i in range(self.STATE_DIM):
                if rng.random() < mutation_rate:
                    self.weights[a][i] += rng.gauss(0.0, mutation_scale)
            if rng.random() < mutation_rate:
                self.biases[a] += rng.gauss(0.0, mutation_scale)


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark statistics over N evaluation episodes."""
    model_name: str
    num_episodes: int
    mean_score: float
    median_score: float
    std_score: float
    min_score: float
    max_score: float
    mean_distance: float
    mean_steps: float
    mean_max_difficulty: float
    max_difficulty_reached: float
    mean_obstacles_cleared: float
    death_counts: Dict[str, int]

    def format_report(self) -> str:
        lines = [
            f"=== Benchmark Report: {self.model_name} ({self.num_episodes} episodes) ===",
            f"  Score (Floor Metric) : mean={self.mean_score:.2f} | median={self.median_score:.2f} | std={self.std_score:.2f} | min={self.min_score:.2f} | max={self.max_score:.2f}",
            f"  Distance Traveled    : mean={self.mean_distance:.1f} px",
            f"  Survival Time        : mean={self.mean_steps:.1f} ticks",
            f"  Difficulty Reached   : mean={self.mean_max_difficulty:.3f}x | max={self.max_difficulty_reached:.3f}x",
            f"  Obstacles Cleared    : mean={self.mean_obstacles_cleared:.2f}",
            f"  Failure Breakdown    : {self.death_counts}",
        ]
        return "\n".join(lines)


def evaluate_model(
    model: BaseModel,
    num_episodes: int = 50,
    base_seed: int = 1000,
) -> BenchmarkSummary:
    """
    Evaluate `model` across `num_episodes` deterministic level seeds and return
    a `BenchmarkSummary`.
    """
    scores: List[float] = []
    distances: List[float] = []
    steps_list: List[int] = []
    difficulties: List[float] = []
    obstacles_list: List[int] = []
    death_counts: Dict[str, int] = {}

    game = SidescrollerGame()

    for ep in range(num_episodes):
        ep_seed = base_seed + ep
        if isinstance(model, RandomBaselineModel):
            model.reset_rng(seed=ep_seed * 31 + 7)

        state = game.reset(seed=ep_seed)
        while not game.done:
            action = model.select_action(state)
            state, _, _, info = game.step(action)

        scores.append(float(info["score"]))
        distances.append(float(info["distance"]))
        steps_list.append(int(info["steps"]))
        difficulties.append(float(info["difficulty"]))
        obstacles_list.append(int(info["obstacles_cleared"]))
        reason = str(info["death_reason"])
        death_counts[reason] = death_counts.get(reason, 0) + 1

    std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0

    return BenchmarkSummary(
        model_name=model.name,
        num_episodes=num_episodes,
        mean_score=round(statistics.mean(scores), 2),
        median_score=round(statistics.median(scores), 2),
        std_score=round(std_score, 2),
        min_score=round(min(scores), 2),
        max_score=round(max(scores), 2),
        mean_distance=round(statistics.mean(distances), 2),
        mean_steps=round(statistics.mean(steps_list), 2),
        mean_max_difficulty=round(statistics.mean(difficulties), 3),
        max_difficulty_reached=round(max(difficulties), 3),
        mean_obstacles_cleared=round(statistics.mean(obstacles_list), 2),
        death_counts=death_counts,
    )


def run_baseline_benchmark(
    num_episodes: int = 100,
    base_seed: int = 1000,
    save_path: Optional[str] = "baseline_floor.json",
) -> BenchmarkSummary:
    """
    Run the `RandomBaselineModel` across `num_episodes` episodes to establish
    the baseline floor and optionally save the results to JSON.
    """
    baseline = RandomBaselineModel(seed=base_seed)
    summary = evaluate_model(baseline, num_episodes=num_episodes, base_seed=base_seed)

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, indent=2)

    return summary


if __name__ == "__main__":
    summary = run_baseline_benchmark(num_episodes=100, base_seed=1000)
    print(summary.format_report())
