# mle-Project-2026: Production Model Evolution Based Replacement

## Project Overview
As outlined in [`Group_Assignment_1 (1) (1).pdf`](Group_Assignment_1%20(1)%20(1).pdf), models deployed in production can degrade or fail over time as project goals shift or environmental complexity increases. When zero downtime is required, a failing production model must be automatically mutated, benchmarked against the current baseline/production standard, rolled back (`regress`) if worse, or promoted to production if better.

This repository implements the proof-of-concept environment, baseline model floor, three distinct trained models, and the evolutionary replacement harness:
1. **`game.py` — 2D Sidescroller Platformer with Infinitely Scaling Difficulty**:
   - **Mechanics**: Horizontal sidescrolling platformer with variable-height platforms, pits/gaps, ground spikes (require jumping), overhead barriers (require ducking), and vertically oscillating saws (unlocked as difficulty rises).
   - **Infinitely Scaling Difficulty**: Difficulty scales continuously and without bound as a function of distance traveled (`difficulty = 1 + d/450 + 0.015*(d/450)^1.35`), increasing world scroll speed, widening gaps, shrinking platforms, and increasing hazard density/complexity until any fixed policy eventually reaches its failure point.
   - **Headless & GUI Modes**: Runs headless for fast multi-episode benchmarking or visually at 60 FPS using Python's built-in `tkinter`.
2. **`baseline_model.py` — Random Input Baseline Model (Performance Floor)**:
   - **`RandomBaselineModel`**: Selects uniformly random actions (`NOOP`, `JUMP`, `DUCK`, `LEFT`, `RIGHT`) at each tick, establishing the statistical floor (`baseline_floor.json`) to compare all models against.
3. **`models.py` & `train_and_compare.py` — Three Trained Models**:
   - **`GeneticLinearModel`**: Linear policy (`W @ state + b` over the 16-D raw state vector) trained via a $(\mu, \lambda)$ Genetic Algorithm with tournament selection, uniform crossover, and Gaussian mutation.
   - **`NeuroEvolutionMLPModel`**: 2-layer nonlinear Multi-Layer Perceptron (`10 -> 16 tanh -> 5`) trained via the Cross-Entropy Method (CEM).
   - **`QLearningModel`**: Value-based Reinforcement Learning agent using tactical time-to-contact state discretization and trained via $n$-step Temporal-Difference Q-Learning ($n=18$) with $\epsilon$-greedy exploration.
4. **`harness.py` & `main.py` — Evolutionary Replacement Harness (`evoHarness`)**:
   - Runs the production model until `modelFailure`, mutates candidates via `evolve()`, evaluates them in `benchmark()`, rolls back worse candidates via `regress()`, and promotes superior candidates.

## Benchmark Comparison (100 Test Episodes, Seeds 1000..1099)

| Model | Paradigm | Mean Score | vs. Baseline Floor | Median Score | Max Score | Mean Distance | Mean Steps | Max Difficulty | Mean Obstacles Cleared | Train Time |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`RandomBaselineModel`** | Uniform Random Inputs | `426.19` | **FLOOR** | `280.83` | `1623.22` | `1582.8 px` | `251.8` | `10.26x` | `0.91` | `0.00s` |
| **`GeneticLinearModel`** | Linear Policy + Genetic Algorithm | `639.07` | **+49.9%** | `465.15` | `2287.94` | `2119.1 px` | `321.1` | `12.34x` | `1.64` | `14.00s` |
| **`NeuroEvolutionMLPModel`** | 2-Layer MLP + Cross-Entropy Method | `694.86` | **+63.0%** | `556.59` | `2592.59` | `2229.3 px` | `333.1` | `12.36x` | `1.68` | `26.00s` |
| **`QLearningModel`** | $n$-Step TD Q-Learning (RL) | **`1295.27`** | **+203.9%** | **`1214.78`** | **`2741.39`** | **`3463.7 px`** | **`484.7`** | **`14.51x`** | **`3.91`** | **`0.89s`** |

## Usage

```bash
# Train all 3 models and compare them against the Random Baseline Floor (saves trained_models.json & model_comparison.json)
python train_and_compare.py

# Watch any specific model in the 60 FPS GUI window (press 'N' inside the window to cycle between all models!)
python game.py --watch-model qlearning
python game.py --watch-model mlp
python game.py --watch-model linear
python game.py --watch-model random

# Play the 2D sidescroller yourself (Space/W/Up = Jump, S/Down = Duck, A/D/Left/Right = Move, M/Tab = Toggle AI, N = Next AI Model)
python game.py --play

# Run the evolutionary replacement harness
python main.py
```
