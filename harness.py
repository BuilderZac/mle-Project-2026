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
        benchmarkEpisodes: int = 30,
        maxEvolutions: int = 5,
        maxAttemptsPerEvolution: int = 25,
        seed: int = 42,
    ):
        self.needToEvolve = False
        self.seed = seed
        self.benchmarkEpisodes = benchmarkEpisodes
        self.maxEvolutions = maxEvolutions
        self.maxAttemptsPerEvolution = maxAttemptsPerEvolution
        self.evolutionsCompleted = 0
        self.totalRollbacks = 0
        self.mutationAttempt = 0
        self.attemptsInCurrentCycle = 0

        # 1. Establish the Random Baseline Model floor first
        self.baselineModel = RandomBaselineModel(seed=seed)
        self.baselineSummary: BenchmarkSummary = run_baseline_benchmark(
            num_episodes=benchmarkEpisodes,
            base_seed=1000,
            save_path="baselineFloor.json",
        )
        self.baselineFloorScore: float = self.baselineSummary.mean_score

        # Start production with the Random Baseline Model as the initial floor
        self.productionModel: BaseModel = self.baselineModel
        self.backupModel: BaseModel = self.productionModel.clone()
        self.candidateModel: Optional[BaseModel] = None

        self.bestBenchmarkScore: float = self.baselineFloorScore
        self.history: List[dict] = [
            {
                "stage": "baseline_floor",
                "model": self.baselineModel.name,
                "mean_score": self.baselineSummary.mean_score,
                "mean_distance": self.baselineSummary.mean_distance,
                "mean_difficulty": self.baselineSummary.mean_max_difficulty,
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
            game = SidescrollerGame(seed=self.seed + self.evolutionsCompleted)
            state = game.reset()
            while not game.done:
                action = self.productionModel.select_action(state)
                state, _, _, info = game.step(action)

            print(
                f"[PRODUCTION RUN] Model={self.productionModel.name} | "
                f"Score={info['score']:.2f} (Floor={self.baselineFloorScore:.2f}) | "
                f"Distance={info['distance']:.1f} | "
                f"MaxDifficulty={info['difficulty']:.2f}x | "
                f"Outcome={info['death_reason']}"
            )

            # Stop evolving once we have completed the target number of evolutionary upgrades
            if self.evolutionsCompleted >= self.maxEvolutions:
                return True

            # Trigger modelFailure when the model dies in the infinitely scaling game
            raise modelFailure(
                f"Model {self.productionModel.name} failed at difficulty {info['difficulty']}x "
                f"({info['death_reason']})"
            )
        except modelFailure as exc:
            print(f"  -> modelFailure triggered: {exc}")
            self.needToEvolve = True
            return False

    def evolve(self) -> None:
        """Mutate the current model to create a candidate replacement."""
        if self.needToEvolve:
            self.mutationAttempt += 1
            self.attemptsInCurrentCycle += 1
            self.backupModel = self.productionModel.clone()

            # Use a unique seed per mutation attempt so rollbacks via regress()
            # never replay the same RNG sequence.
            attemptSeed = self.seed + self.mutationAttempt * 1009

            # If production is still the non-parameterized RandomBaselineModel,
            # initialize an EvolvableLinearModel candidate to evolve beyond the random floor.
            if isinstance(self.productionModel, RandomBaselineModel):
                candidate = EvolvableLinearModel(
                    seed=attemptSeed,
                    generation=1,
                )
            else:
                candidate = self.productionModel.clone()
                candidate.mutate(
                    mutation_rate=0.35,
                    mutation_scale=0.65,
                    seed=attemptSeed,
                )

            self.candidateModel = candidate

    def benchmark(self) -> bool:
        """
        Benchmark the mutated candidate model against the current best score
        (which is initialized to the RandomBaselineModel floor).
        Returns True and promotes the candidate if it improves performance;
        returns False to trigger `regress()` if it performs worse.
        """
        if self.candidateModel is None:
            return False

        summary = evaluate_model(
            self.candidateModel,
            num_episodes=self.benchmarkEpisodes,
            base_seed=1000,
        )
        candidateScore = summary.mean_score
        deltaVsFloor = candidateScore - self.baselineFloorScore

        if candidateScore > self.bestBenchmarkScore:
            print(
                f"  [BENCHMARK PASS] {self.candidateModel.name} mean_score={candidateScore:.2f} "
                f"beats previous best={self.bestBenchmarkScore:.2f} "
                f"(+{deltaVsFloor:.2f} vs Random Floor {self.baselineFloorScore:.2f})"
            )
            self.productionModel = self.candidateModel
            self.bestBenchmarkScore = candidateScore
            self.needToEvolve = False
            self.attemptsInCurrentCycle = 0
            self.evolutionsCompleted += 1
            self.history.append(
                {
                    "stage": f"evolution_{self.evolutionsCompleted}",
                    "model": self.productionModel.name,
                    "mean_score": summary.mean_score,
                    "mean_distance": summary.mean_distance,
                    "mean_difficulty": summary.mean_max_difficulty,
                }
            )
            return True

        print(
            f"  [BENCHMARK FAIL] {self.candidateModel.name} mean_score={candidateScore:.2f} "
            f"<= current best={self.bestBenchmarkScore:.2f} -> rolling back (regress)"
        )

        # Safeguard against unbounded rejection loops when near a local optimum
        if self.attemptsInCurrentCycle >= self.maxAttemptsPerEvolution:
            print(
                f"  [MAX ATTEMPTS REACHED] Retaining current best model ({self.productionModel.name}, "
                f"mean_score={self.bestBenchmarkScore:.2f}) after {self.attemptsInCurrentCycle} attempts."
            )
            self.regress()
            self.needToEvolve = False
            self.attemptsInCurrentCycle = 0
            self.evolutionsCompleted += 1
            return True

        return False

    def regress(self) -> None:
        """Roll back the failed candidate mutation to the previous production model."""
        self.productionModel = self.backupModel.clone()
        self.candidateModel = None
        self.totalRollbacks += 1
