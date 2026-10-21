import argparse
import harness
from baseline_model import run_baseline_benchmark
from game import SidescrollerGame, TkinterGameViewer


def __main__(
    benchmark_episodes: int = 30,
    max_evolutions: int = 5,
    watch: bool = False,
):
    model = harness.evoHarness(
        benchmark_episodes=benchmark_episodes,
        max_evolutions=max_evolutions,
    )

    print(model.baseline_summary.format_report())
    print("\n--- Starting Production Model Evolution Loop ---")

    while not model.run():
        model.evolve()
        while not model.benchmark():
            model.regress()
            model.evolve()

    print("\n=== Evolution Summary vs. Random Baseline Floor ===")
    print(f"  Random Baseline Floor Mean Score : {model.baseline_floor_score:.2f}")
    print(f"  Final Production Model           : {model.production_model.name}")
    print(f"  Final Benchmark Mean Score       : {model.best_benchmark_score:.2f}")
    improvement_pct = (
        ((model.best_benchmark_score - model.baseline_floor_score) / max(1e-6, model.baseline_floor_score))
        * 100.0
    )
    print(f"  Improvement Over Random Floor    : +{improvement_pct:.1f}%")
    print(f"  Total Rollbacks (Regressions)    : {model.total_rollbacks}")

    if watch:
        viewer = TkinterGameViewer(SidescrollerGame(), model=model.production_model, human_mode=False)
        viewer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Production Model Evolution Based Replacement - Sidescroller Benchmark"
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only the Random Baseline Model benchmark to establish the floor.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Train all 3 models (GeneticLinear, QLearning, NeuroEvolutionMLP) and compare against the baseline.",
    )
    parser.add_argument(
        "--watch-model",
        choices=["random", "linear", "qlearning", "mlp"],
        help="Watch a specific model ('random', 'linear', 'qlearning', or 'mlp') in the 60 FPS GUI window.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of benchmark episodes per model evaluation.",
    )
    parser.add_argument(
        "--evolutions",
        type=int,
        default=5,
        help="Number of evolutionary replacements to run in the harness.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Open the 2D sidescroller GUI window to watch the evolved model play.",
    )
    parser.add_argument(
        "--watch-random",
        action="store_true",
        help="Immediately open the 2D sidescroller GUI window watching the Random Baseline Model.",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play the 2D sidescroller manually using keyboard controls.",
    )
    args = parser.parse_args()

    if args.play:
        viewer = TkinterGameViewer(SidescrollerGame(), human_mode=True)
        viewer.run()
    elif args.watch_model:
        from baseline_model import RandomBaselineModel
        from models import load_trained_models

        if args.watch_model == "random":
            chosen = RandomBaselineModel(seed=42)
        else:
            trained = load_trained_models()
            if not trained:
                from train_and_compare import run_training_and_comparison
                run_training_and_comparison(eval_episodes=50, eval_seed=1000)
                trained = load_trained_models()
            chosen = trained[args.watch_model]
        viewer = TkinterGameViewer(SidescrollerGame(), model=chosen, human_mode=False)
        viewer.run()
    elif args.watch_random:
        from baseline_model import RandomBaselineModel
        viewer = TkinterGameViewer(
            SidescrollerGame(),
            model=RandomBaselineModel(seed=42),
            human_mode=False,
        )
        viewer.run()
    elif args.compare:
        from train_and_compare import run_training_and_comparison
        run_training_and_comparison(eval_episodes=100, eval_seed=1000)
    elif args.baseline_only:
        summary = run_baseline_benchmark(num_episodes=args.episodes)
        print(summary.format_report())
    else:
        __main__(
            benchmark_episodes=args.episodes,
            max_evolutions=args.evolutions,
            watch=args.watch,
        )
