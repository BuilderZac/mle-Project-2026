"""
Evolutionary Model Replacement Harness (`evoHarness`).

Implements the survival-of-the-fittest production model replacement workflow:
  1. Establish the performance floor using `RandomBaselineModel`.
  2. Run the current production model in the infinitely scaling 2D sidescroller.
  3. When the model fails (`modelFailure`), mutate a candidate (`evolve()`).
  4. Test the candidate in `benchmark()` against the baseline floor and current
     production score.
  5. If worse, roll back via `regress()` and mutate again; if better, promote
     the candidate to production.
"""

from __future__ import annotations

from typing import List, Optional

from baseline_model import (
    BaseModel,
    BenchmarkSummary,
    EvolvableLinearModel,
    RandomBaselineModel,
    evaluate_model,
    run_baseline_benchmark,
)
from game import SidescrollerGame


class modelFailure(Exception):
    """Trigger when model needs to evolve"""


class evoHarness:
    def __init__(
        self,
        benchmark_episodes: int = 30,
        max_evolutions: int = 5,
        max_attempts_per_evolution: int = 25,
        seed: int = 42,
    ):
        self.needToEvolve = False
        self.seed = seed
        self.benchmark_episodes = benchmark_episodes
        self.max_evolutions = max_evolutions
        self.max_attempts_per_evolution = max_attempts_per_evolution
        self.evolutions_completed = 0
        self.total_rollbacks = 0
        self.mutation_attempt = 0
        self.attempts_in_current_cycle = 0

        # 1. Establish the Random Baseline Model floor first
        self.baseline_model = RandomBaselineModel(seed=seed)
        self.baseline_summary: BenchmarkSummary = run_baseline_benchmark(
            num_episodes=benchmark_episodes,
            base_seed=1000,
            save_path="baseline_floor.json",
        )
        self.baseline_floor_score: float = self.baseline_summary.mean_score

        # Start production with the Random Baseline Model as the initial floor
        self.production_model: BaseModel = self.baseline_model
        self.backup_model: BaseModel = self.production_model.clone()
        self.candidate_model: Optional[BaseModel] = None

        self.best_benchmark_score: float = self.baseline_floor_score
        self.history: List[dict] = [
            {
                "stage": "baseline_floor",
                "model": self.baseline_model.name,
                "mean_score": self.baseline_summary.mean_score,
                "mean_distance": self.baseline_summary.mean_distance,
                "mean_difficulty": self.baseline_summary.mean_max_difficulty,
            }
        ]

    def run(self) -> bool:
        """
        Run the current production model in the live game environment.
        Because difficulty scales infinitely, the model will eventually fail,
        raising `modelFailure` to trigger evolutionary replacement until
        `max_evolutions` improvements have been completed.
        """
        try:
            game = SidescrollerGame(seed=self.seed + self.evolutions_completed)
            state = game.reset()
            while not game.done:
                action = self.production_model.select_action(state)
                state, _, _, info = game.step(action)

            print(
                f"[PRODUCTION RUN] Model={self.production_model.name} | "
                f"Score={info['score']:.2f} (Floor={self.baseline_floor_score:.2f}) | "
                f"Distance={info['distance']:.1f} | "
                f"MaxDifficulty={info['difficulty']:.2f}x | "
                f"Outcome={info['death_reason']}"
            )

            # Stop evolving once we have completed the target number of evolutionary upgrades
            if self.evolutions_completed >= self.max_evolutions:
                return True

            # Trigger modelFailure when the model dies in the infinitely scaling game
            raise modelFailure(
                f"Model {self.production_model.name} failed at difficulty {info['difficulty']}x "
                f"({info['death_reason']})"
            )
        except modelFailure as exc:
            print(f"  -> modelFailure triggered: {exc}")
            self.needToEvolve = True
            return False

    def evolve(self) -> None:
        """Mutate the current model to create a candidate replacement."""
        if self.needToEvolve:
            self.mutation_attempt += 1
            self.attempts_in_current_cycle += 1
            self.backup_model = self.production_model.clone()

            # Use a unique seed per mutation attempt so rollbacks via regress()
            # never replay the same RNG sequence.
            attempt_seed = self.seed + self.mutation_attempt * 1009

            # If production is still the non-parameterized RandomBaselineModel,
            # initialize an EvolvableLinearModel candidate to evolve beyond the random floor.
            if isinstance(self.production_model, RandomBaselineModel):
                candidate = EvolvableLinearModel(
                    seed=attempt_seed,
                    generation=1,
                )
            else:
                candidate = self.production_model.clone()
                candidate.mutate(
                    mutation_rate=0.35,
                    mutation_scale=0.65,
                    seed=attempt_seed,
                )

            self.candidate_model = candidate

    def benchmark(self) -> bool:
        """
        Benchmark the mutated candidate model against the current best score
        (which is initialized to the RandomBaselineModel floor).
        Returns True and promotes the candidate if it improves performance;
        returns False to trigger `regress()` if it performs worse.
        """
        if self.candidate_model is None:
            return False

        summary = evaluate_model(
            self.candidate_model,
            num_episodes=self.benchmark_episodes,
            base_seed=1000,
        )
        candidate_score = summary.mean_score
        delta_vs_floor = candidate_score - self.baseline_floor_score

        if candidate_score > self.best_benchmark_score:
            print(
                f"  [BENCHMARK PASS] {self.candidate_model.name} mean_score={candidate_score:.2f} "
                f"beats previous best={self.best_benchmark_score:.2f} "
                f"(+{delta_vs_floor:.2f} vs Random Floor {self.baseline_floor_score:.2f})"
            )
            self.production_model = self.candidate_model
            self.best_benchmark_score = candidate_score
            self.needToEvolve = False
            self.attempts_in_current_cycle = 0
            self.evolutions_completed += 1
            self.history.append(
                {
                    "stage": f"evolution_{self.evolutions_completed}",
                    "model": self.production_model.name,
                    "mean_score": summary.mean_score,
                    "mean_distance": summary.mean_distance,
                    "mean_difficulty": summary.mean_max_difficulty,
                }
            )
            return True

        print(
            f"  [BENCHMARK FAIL] {self.candidate_model.name} mean_score={candidate_score:.2f} "
            f"<= current best={self.best_benchmark_score:.2f} -> rolling back (regress)"
        )

        # Safeguard against unbounded rejection loops when near a local optimum
        if self.attempts_in_current_cycle >= self.max_attempts_per_evolution:
            print(
                f"  [MAX ATTEMPTS REACHED] Retaining current best model ({self.production_model.name}, "
                f"mean_score={self.best_benchmark_score:.2f}) after {self.attempts_in_current_cycle} attempts."
            )
            self.regress()
            self.needToEvolve = False
            self.attempts_in_current_cycle = 0
            self.evolutions_completed += 1
            return True

        return False

    def regress(self) -> None:
        """Roll back the failed candidate mutation to the previous production model."""
        self.production_model = self.backup_model.clone()
        self.candidate_model = None
        self.total_rollbacks += 1
