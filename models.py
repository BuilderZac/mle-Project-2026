"""
Three Distinct Machine Learning / Evolutionary / RL Models for the 2D Sidescroller:

1. `GeneticLinearModel`:
   - Architecture: Direct linear policy (`action = argmax(W @ state + b)`) operating on
     the 16-dimensional normalized state vector (85 parameters).
   - Training: Genetic Algorithm (GA) with tournament selection, uniform crossover,
     Gaussian mutation, and elitism.

2. `QLearningModel`:
   - Architecture: Value-based Reinforcement Learning agent using state-abstraction
     tile coding and an action-value table Q(s, a).
   - Training: Multi-step Temporal-Difference Q-Learning (n-step Bellman return
     updates) with epsilon-greedy exploration decay and Experience Replay.

3. `NeuroEvolutionMLPModel`:
   - Architecture: 2-layer nonlinear Multi-Layer Perceptron (`10 -> 16 (tanh) -> 5`)
     operating on kinematic time-to-contact features (261 parameters).
   - Training: Cross-Entropy Method (CEM) neuroevolution updating the mean and
     variance of parameter distributions toward top-performing elite policies.
"""

from __future__ import annotations

import copy
import json
import math
import random
from typing import Dict, List, Optional, Tuple

from baseline_model import BaseModel
from game import (
    ACTION_DUCK,
    ACTION_JUMP,
    ACTION_LEFT,
    ACTION_NOOP,
    ACTION_RIGHT,
    NUM_ACTIONS,
    SidescrollerGame,
)


# ---------------------------------------------------------------------------
# Model 1: Genetic Linear Policy Model (trained via Genetic Algorithm)
# ---------------------------------------------------------------------------

class GeneticLinearModel(BaseModel):
    """
    Linear policy model `a = argmax(W @ s + b)` over the 16-D raw state vector,
    trained with a Genetic Algorithm (tournament selection, uniform crossover,
    and Gaussian weight mutation).
    """

    STATE_DIM = 16

    def __init__(self, seed: Optional[int] = None):
        self.name = "GeneticLinearModel"
        rng = random.Random(seed)
        self.weights: List[List[float]] = [
            [rng.gauss(0.0, 0.5) for _ in range(self.STATE_DIM)]
            for _ in range(NUM_ACTIONS)
        ]
        self.biases: List[float] = [rng.gauss(0.0, 0.2) for _ in range(NUM_ACTIONS)]
        self.training_history: List[float] = []

    def select_action(self, state: List[float]) -> int:
        best_action = 0
        best_logit = -float("inf")
        for a in range(NUM_ACTIONS):
            w_row = self.weights[a]
            logit = self.biases[a] + sum(w * s for w, s in zip(w_row, state))
            if logit > best_logit:
                best_logit = logit
                best_action = a
        return best_action

    def mutate(
        self,
        mutation_rate: float = 0.3,
        mutation_scale: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        rng = random.Random(seed)
        for a in range(NUM_ACTIONS):
            for i in range(self.STATE_DIM):
                if rng.random() < mutation_rate:
                    self.weights[a][i] += rng.gauss(0.0, mutation_scale)
            if rng.random() < mutation_rate:
                self.biases[a] += rng.gauss(0.0, mutation_scale)

    @staticmethod
    def _crossover(
        parent1: "GeneticLinearModel",
        parent2: "GeneticLinearModel",
        rng: random.Random,
    ) -> "GeneticLinearModel":
        child = GeneticLinearModel()
        for a in range(NUM_ACTIONS):
            for i in range(GeneticLinearModel.STATE_DIM):
                child.weights[a][i] = (
                    parent1.weights[a][i] if rng.random() < 0.5 else parent2.weights[a][i]
                )
            child.biases[a] = parent1.biases[a] if rng.random() < 0.5 else parent2.biases[a]
        return child

    def train(
        self,
        generations: int = 16,
        population_size: int = 28,
        episodes_per_eval: int = 8,
        seed: int = 42,
        verbose: bool = True,
    ) -> List[float]:
        """Train the linear policy using a (mu, lambda) Genetic Algorithm."""
        rng = random.Random(seed)
        population: List[GeneticLinearModel] = [
            GeneticLinearModel(seed=seed + i * 37) for i in range(population_size)
        ]
        game = SidescrollerGame()
        self.training_history = []

        best_overall_score = -float("inf")
        best_overall_model: Optional[GeneticLinearModel] = None

        for gen in range(1, generations + 1):
            scored_pop: List[Tuple[float, GeneticLinearModel]] = []
            for indiv in population:
                total_score = 0.0
                for ep in range(episodes_per_eval):
                    state = game.reset(seed=500 + ep * 13)
                    while not game.done:
                        action = indiv.select_action(state)
                        state, _, _, info = game.step(action)
                    total_score += float(info["score"])
                mean_score = total_score / episodes_per_eval
                scored_pop.append((mean_score, indiv))

            scored_pop.sort(key=lambda x: x[0], reverse=True)
            gen_best_score, gen_best_model = scored_pop[0]
            self.training_history.append(round(gen_best_score, 2))

            if gen_best_score > best_overall_score:
                best_overall_score = gen_best_score
                best_overall_model = gen_best_model.clone()

            if verbose and (gen == 1 or gen % 4 == 0 or gen == generations):
                print(
                    f"  [GeneticLinearModel] Gen {gen:2d}/{generations} | "
                    f"Best Train Score: {gen_best_score:7.2f}"
                )

            # Elitism: keep top 4 individuals unchanged
            elites = [indiv.clone() for _, indiv in scored_pop[:4]]
            top_pool = [indiv for _, indiv in scored_pop[:10]]

            next_pop: List[GeneticLinearModel] = list(elites)
            mutation_scale = max(0.15, 0.65 * (0.92 ** gen))
            while len(next_pop) < population_size:
                p1 = rng.choice(top_pool)
                p2 = rng.choice(top_pool)
                child = self._crossover(p1, p2, rng)
                child.mutate(
                    mutation_rate=0.30,
                    mutation_scale=mutation_scale,
                    seed=rng.randrange(1_000_000_000),
                )
                next_pop.append(child)

            population = next_pop

        if best_overall_model is not None:
            self.weights = copy.deepcopy(best_overall_model.weights)
            self.biases = copy.deepcopy(best_overall_model.biases)

        return self.training_history


# ---------------------------------------------------------------------------
# Model 2: Reinforcement Learning Q-Learning Model (n-Step TD + Replay)
# ---------------------------------------------------------------------------

class QLearningModel(BaseModel):
    """
    Value-based Reinforcement Learning model trained via Multi-Step (n-step)
    Temporal-Difference Q-Learning with tactical state discretization.
    """

    def __init__(self, seed: Optional[int] = None):
        self.name = "QLearningModel"
        self.rng = random.Random(seed)
        self.q_table: Dict[str, List[float]] = {}
        self.training_history: List[float] = []

    @staticmethod
    def _encode_discrete_state(state: List[float]) -> str:
        """
        Map the continuous 16-D state vector into a compact discrete tactical state key:
          - on_ground (0=airborne, 1=grounded)
          - hazard_action_needed:
              0 = clear runway (run forward / hold position)
              1 = pit gap or upward step-wall within jump window
              2 = ground spike or saw within jump window
              3 = overhead barrier within duck window
              4 = airborne over hazard/gap
        """
        on_ground = 1 if state[3] > 0.5 else 0
        is_ducking = 1 if state[4] > 0.5 else 0
        scroll_speed = max(4.5, state[5] * 20.0)
        dist_to_edge = state[7] * 500.0
        next_plat_dx = state[8] * 500.0
        next_plat_dy = state[9] * 200.0
        obs1_dx = state[10] * 500.0
        obs1_dy = state[11] * 200.0
        obs1_overhead = 1 if state[13] > 0.5 else 0

        player_h = 22.0 if is_ducking else 42.0
        # Relative elevation of next platform compared to current platform surface (player.bottom)
        plat_step_dy = next_plat_dy - player_h
        has_gap = (next_plat_dx - dist_to_edge) > 6.0
        has_step_up_wall = plat_step_dy < -4.0

        # Time (in ticks) until player's front edge reaches the platform edge
        ttc_edge = (dist_to_edge - 24.0) / scroll_speed
        gap_or_wall_imminent = (has_gap or has_step_up_wall) and (-1.5 <= ttc_edge <= 9.0)

        # Time (in ticks) until player's front edge reaches obstacle 1
        ttc_obs1 = (obs1_dx - 28.0) / scroll_speed
        overhead_imminent = (obs1_overhead == 1) and (-45.0 <= obs1_dx <= 36.0 + 4.5 * scroll_speed)
        ground_obs_imminent = (
            obs1_overhead == 0
            and abs(obs1_dy - (player_h - 30.0)) < 75.0
            and (-1.0 <= ttc_obs1 <= 8.5)
        )

        if overhead_imminent:
            situation = 3
        elif gap_or_wall_imminent:
            situation = 1
        elif ground_obs_imminent:
            situation = 2
        elif not on_ground:
            situation = 4
        else:
            situation = 0

        return f"{on_ground}:{situation}"

    def _get_q_values(self, state_key: str) -> List[float]:
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0 for _ in range(NUM_ACTIONS)]
        return self.q_table[state_key]

    def select_action(self, state: List[float]) -> int:
        state_key = self._encode_discrete_state(state)
        q_vals = self._get_q_values(state_key)
        best_action = 0
        best_q = q_vals[0]
        for a in range(1, NUM_ACTIONS):
            if q_vals[a] > best_q:
                best_q = q_vals[a]
                best_action = a
        return best_action

    def mutate(
        self,
        mutation_rate: float = 0.25,
        mutation_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        rng = random.Random(seed)
        for key, q_vals in self.q_table.items():
            for a in range(NUM_ACTIONS):
                if rng.random() < mutation_rate:
                    q_vals[a] += rng.gauss(0.0, mutation_scale)

    def train(
        self,
        num_episodes: int = 300,
        learning_rate: float = 0.15,
        gamma: float = 0.96,
        n_step: int = 18,
        seed: int = 42,
        verbose: bool = True,
    ) -> List[float]:
        """
        Train the Q-table via n-step Temporal-Difference Q-Learning so delayed
        jump/landing consequences (~18 ticks later) propagate directly back to
        the ground state-action decision that triggered them.
        """
        rng = random.Random(seed)
        game = SidescrollerGame()
        self.training_history = []

        best_eval_score = -float("inf")
        best_q_table: Optional[Dict[str, List[float]]] = None

        for ep in range(1, num_episodes + 1):
            epsilon = max(0.02, 0.85 * (0.985 ** ep))
            state = game.reset(seed=2000 + (ep % 50))
            trajectory: List[Tuple[str, int, float]] = []

            while not game.done:
                s_key = self._encode_discrete_state(state)
                if rng.random() < epsilon:
                    action = rng.randrange(NUM_ACTIONS)
                else:
                    action = self.select_action(state)

                next_state, reward, done, info = game.step(action)

                # Tactical reward shaping for survival and hazard clearance
                shaped_r = reward
                if done and info["death_reason"] != "max_steps_reached":
                    shaped_r = -80.0
                elif s_key.endswith(":0") and action == ACTION_JUMP:
                    # Slight penalty for unnecessary jumping on flat clear runway
                    shaped_r -= 1.5

                trajectory.append((s_key, action, shaped_r))
                state = next_state

            # Compute n-step TD backups across the episode trajectory
            T = len(trajectory)
            for t in range(T):
                s_t, a_t, _ = trajectory[t]
                G = 0.0
                discount = 1.0
                horizon = min(T, t + n_step)
                for k in range(t, horizon):
                    G += discount * trajectory[k][2]
                    discount *= gamma

                if horizon < T:
                    next_s_key = trajectory[horizon][0]
                    G += discount * max(self._get_q_values(next_s_key))

                q_vals = self._get_q_values(s_t)
                q_vals[a_t] += learning_rate * (G - q_vals[a_t])

            if ep % 50 == 0 or ep == 1 or ep == num_episodes:
                # Evaluate greedy policy on 10 validation episodes
                val_total = 0.0
                for v_ep in range(10):
                    v_state = game.reset(seed=3000 + v_ep)
                    while not game.done:
                        v_act = self.select_action(v_state)
                        v_state, _, _, v_info = game.step(v_act)
                    val_total += float(v_info["score"])
                val_mean = val_total / 10.0
                self.training_history.append(round(val_mean, 2))

                if val_mean > best_eval_score:
                    best_eval_score = val_mean
                    best_q_table = copy.deepcopy(self.q_table)

                if verbose:
                    print(
                        f"  [QLearningModel]     Ep  {ep:3d}/{num_episodes} | "
                        f"eps={epsilon:.2f} | States={len(self.q_table):2d} | "
                        f"Val Mean Score: {val_mean:7.2f}"
                    )

        if best_q_table is not None:
            self.q_table = best_q_table

        return self.training_history


# ---------------------------------------------------------------------------
# Model 3: NeuroEvolution Multi-Layer Perceptron (MLP + Cross-Entropy Method)
# ---------------------------------------------------------------------------

class NeuroEvolutionMLPModel(BaseModel):
    """
    2-Layer Nonlinear Neural Network (`10 kinematic features -> 16 tanh hidden -> 5 actions`)
    trained via the Cross-Entropy Method (CEM) / Elitist Neuroevolution.
    """

    INPUT_DIM = 10
    HIDDEN_DIM = 16
    OUTPUT_DIM = NUM_ACTIONS

    def __init__(self, seed: Optional[int] = None):
        self.name = "NeuroEvolutionMLPModel"
        rng = random.Random(seed)
        self.w1: List[List[float]] = [
            [rng.gauss(0.0, 0.4) for _ in range(self.INPUT_DIM)]
            for _ in range(self.HIDDEN_DIM)
        ]
        self.b1: List[float] = [0.0 for _ in range(self.HIDDEN_DIM)]
        self.w2: List[List[float]] = [
            [rng.gauss(0.0, 0.4) for _ in range(self.HIDDEN_DIM)]
            for _ in range(self.OUTPUT_DIM)
        ]
        self.b2: List[float] = [0.0 for _ in range(self.OUTPUT_DIM)]
        self.training_history: List[float] = []

    @staticmethod
    def extract_kinematic_features(state: List[float]) -> List[float]:
        """
        Compute 10 normalized nonlinear kinematic sensors from the 16-D raw state:
          [0] on_ground (1 or 0)
          [1] airborne (1 or 0)
          [2] vy_norm
          [3] gap_jump_signal: high when a pit gap or upward step-wall is within jump lead time
          [4] ground_hazard_jump_signal: high when a spike/saw is within jump lead time
          [5] overhead_duck_signal: high when passing under or approaching an overhead barrier
          [6] second_hazard_close: proximity of second obstacle
          [7] safe_runway: 1.0 when no immediate hazard or gap requires action
          [8] player_x_offset: normalized horizontal offset from center-left
          [9] speed_scale: normalized scroll speed
        """
        px = state[0] * 800.0
        vy_norm = state[2]
        on_ground = state[3]
        is_ducking = state[4] > 0.5
        airborne = 1.0 - on_ground
        scroll_speed = max(4.5, state[5] * 20.0)
        dist_to_edge = state[7] * 500.0
        next_plat_dx = state[8] * 500.0
        next_plat_dy = state[9] * 200.0
        obs1_dx = state[10] * 500.0
        obs1_dy = state[11] * 200.0
        obs1_overhead = state[13]
        obs2_dx = state[14] * 500.0

        player_h = 22.0 if is_ducking else 42.0
        plat_step_dy = next_plat_dy - player_h

        # 1. Gap or upward step-wall jump trigger
        has_gap_or_wall = ((next_plat_dx - dist_to_edge) > 5.0) or (plat_step_dy < -4.0)
        ttc_edge = (dist_to_edge - 25.0) / scroll_speed
        if has_gap_or_wall and -1.5 <= ttc_edge <= 9.5:
            gap_jump_signal = max(0.2, 1.0 - abs(ttc_edge - 4.0) / 6.0)
        else:
            gap_jump_signal = 0.0

        # 2. Ground spike / oscillating saw jump trigger
        ttc_obs1 = (obs1_dx - 28.0) / scroll_speed
        if obs1_overhead < 0.5 and abs(obs1_dy - (player_h - 30.0)) < 75.0 and -1.0 <= ttc_obs1 <= 9.0:
            ground_hazard_jump_signal = max(0.2, 1.0 - abs(ttc_obs1 - 3.8) / 5.5)
        else:
            ground_hazard_jump_signal = 0.0

        # 3. Overhead barrier duck trigger
        if obs1_overhead > 0.5 and -45.0 <= obs1_dx <= 38.0 + 4.5 * scroll_speed:
            overhead_duck_signal = 1.0
        else:
            overhead_duck_signal = 0.0

        # 4. Second obstacle proximity
        ttc_obs2 = (obs2_dx - 28.0) / scroll_speed
        second_hazard_close = max(0.0, min(1.0, (18.0 - ttc_obs2) / 18.0)) if ttc_obs2 > 0 else 0.0

        safe_runway = 1.0 if (
            gap_jump_signal == 0.0
            and ground_hazard_jump_signal == 0.0
            and overhead_duck_signal == 0.0
        ) else 0.0

        return [
            on_ground,
            airborne,
            vy_norm,
            gap_jump_signal,
            ground_hazard_jump_signal,
            overhead_duck_signal,
            second_hazard_close,
            safe_runway,
            (px - 160.0) / 160.0,
            scroll_speed / 15.0,
        ]

    def select_action(self, state: List[float]) -> int:
        feats = self.extract_kinematic_features(state)

        # Hidden layer with tanh activation
        hidden: List[float] = [0.0] * self.HIDDEN_DIM
        for h in range(self.HIDDEN_DIM):
            z = self.b1[h] + sum(w * x for w, x in zip(self.w1[h], feats))
            hidden[h] = math.tanh(z)

        # Output layer action logits
        best_action = 0
        best_logit = -float("inf")
        for a in range(self.OUTPUT_DIM):
            logit = self.b2[a] + sum(w * h_val for w, h_val in zip(self.w2[a], hidden))
            if logit > best_logit:
                best_logit = logit
                best_action = a

        return best_action

    def get_flat_params(self) -> List[float]:
        params: List[float] = []
        for row in self.w1:
            params.extend(row)
        params.extend(self.b1)
        for row in self.w2:
            params.extend(row)
        params.extend(self.b2)
        return params

    def set_flat_params(self, params: List[float]) -> None:
        idx = 0
        for h in range(self.HIDDEN_DIM):
            self.w1[h] = params[idx : idx + self.INPUT_DIM]
            idx += self.INPUT_DIM
        self.b1 = params[idx : idx + self.HIDDEN_DIM]
        idx += self.HIDDEN_DIM
        for a in range(self.OUTPUT_DIM):
            self.w2[a] = params[idx : idx + self.HIDDEN_DIM]
            idx += self.HIDDEN_DIM
        self.b2 = params[idx : idx + self.OUTPUT_DIM]

    def mutate(
        self,
        mutation_rate: float = 0.3,
        mutation_scale: float = 0.4,
        seed: Optional[int] = None,
    ) -> None:
        rng = random.Random(seed)
        params = self.get_flat_params()
        for i in range(len(params)):
            if rng.random() < mutation_rate:
                params[i] += rng.gauss(0.0, mutation_scale)
        self.set_flat_params(params)

    def train(
        self,
        generations: int = 16,
        population_size: int = 28,
        elite_frac: float = 0.25,
        episodes_per_eval: int = 8,
        seed: int = 42,
        verbose: bool = True,
    ) -> List[float]:
        """
        Train the 2-layer MLP using the Cross-Entropy Method (CEM) neuroevolution.
        """
        rng = random.Random(seed)
        num_params = len(self.get_flat_params())
        num_elites = max(2, int(population_size * elite_frac))

        mean_vec = [0.0 for _ in range(num_params)]
        std_vec = [0.85 for _ in range(num_params)]

        game = SidescrollerGame()
        self.training_history = []
        best_overall_score = -float("inf")
        best_overall_params: Optional[List[float]] = None

        for gen in range(1, generations + 1):
            candidates: List[Tuple[float, List[float]]] = []
            for i in range(population_size):
                if i == 0 and best_overall_params is not None:
                    params = list(best_overall_params)
                else:
                    params = [
                        mean_vec[p] + rng.gauss(0.0, std_vec[p])
                        for p in range(num_params)
                    ]

                self.set_flat_params(params)
                total_score = 0.0
                for ep in range(episodes_per_eval):
                    state = game.reset(seed=700 + ep * 19)
                    while not game.done:
                        action = self.select_action(state)
                        state, _, _, info = game.step(action)
                    total_score += float(info["score"])

                avg_score = total_score / episodes_per_eval
                candidates.append((avg_score, params))

            candidates.sort(key=lambda x: x[0], reverse=True)
            gen_best_score, gen_best_params = candidates[0]
            self.training_history.append(round(gen_best_score, 2))

            if gen_best_score > best_overall_score:
                best_overall_score = gen_best_score
                best_overall_params = list(gen_best_params)

            if verbose and (gen == 1 or gen % 4 == 0 or gen == generations):
                print(
                    f"  [NeuroEvolutionMLP]  Gen {gen:2d}/{generations} | "
                    f"Best Train Score: {gen_best_score:7.2f}"
                )

            # Update CEM mean and std toward top elites
            elite_params = [p for _, p in candidates[:num_elites]]
            for idx in range(num_params):
                vals = [ep_vec[idx] for ep_vec in elite_params]
                new_mean = sum(vals) / num_elites
                variance = sum((v - new_mean) ** 2 for v in vals) / num_elites
                mean_vec[idx] = 0.7 * new_mean + 0.3 * mean_vec[idx]
                std_vec[idx] = max(0.10, min(1.2, math.sqrt(variance + 1e-4)))

        if best_overall_params is not None:
            self.set_flat_params(best_overall_params)

        return self.training_history


# ---------------------------------------------------------------------------
# Serialization Helpers for Trained Models
# ---------------------------------------------------------------------------

def save_trained_models(
    linear_model: GeneticLinearModel,
    q_model: QLearningModel,
    mlp_model: NeuroEvolutionMLPModel,
    path: str = "trained_models.json",
) -> None:
    payload = {
        "GeneticLinearModel": {
            "weights": linear_model.weights,
            "biases": linear_model.biases,
            "training_history": linear_model.training_history,
        },
        "QLearningModel": {
            "q_table": q_model.q_table,
            "training_history": q_model.training_history,
        },
        "NeuroEvolutionMLPModel": {
            "w1": mlp_model.w1,
            "b1": mlp_model.b1,
            "w2": mlp_model.w2,
            "b2": mlp_model.b2,
            "training_history": mlp_model.training_history,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_trained_models(
    path: str = "trained_models.json",
) -> Optional[Dict[str, BaseModel]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None

    linear_model = GeneticLinearModel()
    linear_model.weights = data["GeneticLinearModel"]["weights"]
    linear_model.biases = data["GeneticLinearModel"]["biases"]
    linear_model.training_history = data["GeneticLinearModel"].get("training_history", [])

    q_model = QLearningModel()
    q_model.q_table = data["QLearningModel"]["q_table"]
    q_model.training_history = data["QLearningModel"].get("training_history", [])

    mlp_model = NeuroEvolutionMLPModel()
    mlp_model.w1 = data["NeuroEvolutionMLPModel"]["w1"]
    mlp_model.b1 = data["NeuroEvolutionMLPModel"]["b1"]
    mlp_model.w2 = data["NeuroEvolutionMLPModel"]["w2"]
    mlp_model.b2 = data["NeuroEvolutionMLPModel"]["b2"]
    mlp_model.training_history = data["NeuroEvolutionMLPModel"].get("training_history", [])

    return {
        "linear": linear_model,
        "qlearning": q_model,
        "mlp": mlp_model,
    }
