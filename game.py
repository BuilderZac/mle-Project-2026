"""
2D Sidescroller Platformer with Infinitely Scaling Difficulty.

Designed as the benchmark environment for the Production Model Evolution Based
Replacement project. Supports both fast headless execution for model benchmarking
and evolution, and a visual 60 FPS Tkinter renderer for watching models or
human play.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Action Constants
ACTION_NOOP = 0
ACTION_JUMP = 1
ACTION_DUCK = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4

ACTION_NAMES = {
    ACTION_NOOP: "NOOP",
    ACTION_JUMP: "JUMP",
    ACTION_DUCK: "DUCK",
    ACTION_LEFT: "LEFT",
    ACTION_RIGHT: "RIGHT",
}
NUM_ACTIONS = len(ACTION_NAMES)


@dataclass
class Platform:
    """A solid horizontal surface the player can land and run on."""
    x: float
    y: float
    width: float
    height: float = 20.0

    @property
    def right(self) -> float:
        return self.x + self.width


@dataclass
class Obstacle:
    """
    A hazard that terminates the episode on collision.
    kind:
      - 'spike': ground hazard (player must jump over)
      - 'overhead': suspended barrier (player must duck under)
      - 'saw': vertically oscillating hazard (unlocked at higher difficulty)
    """
    x: float
    y: float
    width: float
    height: float
    kind: str = "spike"
    base_y: float = 0.0
    osc_speed: float = 0.0
    osc_amp: float = 0.0
    phase: float = 0.0
    passed: bool = False

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height


@dataclass
class Player:
    """The platformer agent character."""
    x: float = 120.0
    y: float = 300.0
    vx: float = 0.0
    vy: float = 0.0
    width: float = 28.0
    standing_height: float = 42.0
    ducking_height: float = 22.0
    on_ground: bool = False
    is_ducking: bool = False
    alive: bool = True

    @property
    def height(self) -> float:
        return self.ducking_height if self.is_ducking else self.standing_height

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height


class SidescrollerGame:
    """
    2D Sidescroller Platformer environment with infinitely scaling difficulty.

    As the player progresses further (`distance`), `difficulty` increases
    continuously without an upper cap:
      1. World scroll speed increases.
      2. Platform lengths shrink and gaps between platforms widen.
      3. Platform height variations grow more extreme.
      4. Obstacle spawn probabilities and complexity (ground spikes, overhead
         barriers, oscillating saws) scale upward.
    """

    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 450
    GROUND_BASE_Y = 360.0

    # Physics constants
    GRAVITY = 0.75
    FAST_FALL_GRAVITY = 1.65
    JUMP_VELOCITY = -13.5
    HORIZONTAL_ACCEL = 3.5
    MIN_PLAYER_X = 50.0
    MAX_PLAYER_X = 320.0

    # Difficulty scaling constants
    BASE_SCROLL_SPEED = 4.5
    DIFFICULTY_DISTANCE_SCALE = 450.0  # +1.0 difficulty per 450 distance units

    def __init__(self, seed: Optional[int] = None, max_steps: int = 50_000):
        self.seed_value = seed
        self.rng = random.Random(seed)
        self.max_steps = max_steps

        self.player = Player()
        self.platforms: List[Platform] = []
        self.obstacles: List[Obstacle] = []

        self.steps = 0
        self.distance = 0.0
        self.score = 0.0
        self.obstacles_cleared = 0
        self.difficulty = 1.0
        self.scroll_speed = self.BASE_SCROLL_SPEED
        self.done = False
        self.death_reason = ""

        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None) -> List[float]:
        """Reset the game to the initial state and return the initial observation."""
        if seed is not None:
            self.seed_value = seed
            self.rng = random.Random(seed)
        elif self.seed_value is not None:
            self.rng = random.Random(self.seed_value)

        self.steps = 0
        self.distance = 0.0
        self.score = 0.0
        self.obstacles_cleared = 0
        self.difficulty = 1.0
        self.scroll_speed = self.BASE_SCROLL_SPEED
        self.done = False
        self.death_reason = ""

        # Safe starting runway platform
        start_platform = Platform(x=0.0, y=self.GROUND_BASE_Y, width=650.0, height=90.0)
        self.platforms = [start_platform]
        self.obstacles = []

        self.player = Player(
            x=120.0,
            y=start_platform.y - 42.0,
            vx=0.0,
            vy=0.0,
            on_ground=True,
            is_ducking=False,
            alive=True,
        )

        # Populate initial world ahead
        self._spawn_world_ahead()
        return self.get_state()

    def _update_difficulty(self) -> None:
        """
        Compute infinitely scaling difficulty as a function of distance traveled.
        Difficulty starts at 1.0 and grows unboundedly:
          difficulty = 1.0 + (distance / 450) + 0.00015 * (distance / 100)^2
        """
        linear_term = self.distance / self.DIFFICULTY_DISTANCE_SCALE
        quadratic_term = 0.015 * ((self.distance / self.DIFFICULTY_DISTANCE_SCALE) ** 1.35)
        self.difficulty = 1.0 + linear_term + quadratic_term

        # Scroll speed scales unboundedly (sub-linear + linear drift so physics stay playable
        # early on while eventually overwhelming any fixed reaction window)
        self.scroll_speed = (
            self.BASE_SCROLL_SPEED
            + 1.35 * math.log1p(self.difficulty - 1.0)
            + 0.22 * (self.difficulty - 1.0)
        )

    def _spawn_world_ahead(self) -> None:
        """Procedurally generate platforms and obstacles ahead of the camera."""
        while self.platforms[-1].right < self.SCREEN_WIDTH + 600:
            prev = self.platforms[-1]
            diff = self.difficulty

            # Calculate maximum jumpable horizontal gap at current scroll speed
            air_time = (2.0 * abs(self.JUMP_VELOCITY)) / self.GRAVITY
            max_jump_dist = (self.scroll_speed + 1.5) * air_time * 0.72

            # Gap scales with difficulty, starting small and approaching max_jump_dist
            gap_ratio = min(0.88, 0.20 + 0.07 * (diff - 1.0))
            min_gap = max(35.0, min(max_jump_dist * 0.35, 35.0 + 8.0 * (diff - 1.0)))
            max_gap = max(min_gap + 15.0, max_jump_dist * gap_ratio)

            # Decide whether to insert a gap or a step change
            has_gap = self.rng.random() < min(0.85, 0.35 + 0.05 * (diff - 1.0))
            gap = self.rng.uniform(min_gap, max_gap) if has_gap else 0.0

            # Platform width shrinks as difficulty increases
            min_width = max(95.0, 260.0 - 16.0 * (diff - 1.0))
            max_width = max(min_width + 40.0, 480.0 - 24.0 * (diff - 1.0))
            width = self.rng.uniform(min_width, max_width)

            # Platform elevation change scales with difficulty
            max_step_up = min(75.0, 25.0 + 6.0 * (diff - 1.0))
            max_step_down = min(90.0, 30.0 + 8.0 * (diff - 1.0))
            dy = self.rng.uniform(-max_step_up, max_step_down)
            new_y = max(210.0, min(390.0, prev.y + dy))

            new_x = prev.right + gap
            plat = Platform(
                x=new_x,
                y=new_y,
                width=width,
                height=self.SCREEN_HEIGHT - new_y,
            )
            self.platforms.append(plat)

            # Spawn obstacles on this platform based on current difficulty
            self._spawn_obstacles_on_platform(plat, diff)

    def _spawn_obstacles_on_platform(self, plat: Platform, diff: float) -> None:
        """Spawn 0, 1, or multiple obstacles on a newly generated platform."""
        # Leave safe landing and takeoff margins on each platform
        safe_margin_left = max(55.0, 95.0 - 3.0 * (diff - 1.0))
        safe_margin_right = max(50.0, 85.0 - 3.0 * (diff - 1.0))
        usable_length = plat.width - safe_margin_left - safe_margin_right
        if usable_length < 40.0:
            return

        # Obstacle probability scales toward 0.95 as difficulty increases
        spawn_prob = min(0.95, 0.42 + 0.065 * (diff - 1.0))
        if self.rng.random() > spawn_prob:
            return

        # Number of obstacles on a single platform can scale with difficulty
        max_obstacles = max(1, min(3, int(usable_length // 150.0)))
        num_obstacles = 1
        if max_obstacles > 1 and self.rng.random() < min(0.75, 0.15 * (diff - 1.0)):
            num_obstacles = self.rng.randint(2, max_obstacles)

        segment_width = usable_length / num_obstacles
        for i in range(num_obstacles):
            seg_start = plat.x + safe_margin_left + i * segment_width
            obs_x = seg_start + self.rng.uniform(5.0, max(10.0, segment_width - 40.0))

            # Choose obstacle type based on difficulty
            roll = self.rng.random()
            saw_chance = min(0.35, max(0.0, (diff - 2.0) * 0.06))
            overhead_chance = min(0.35, 0.18 + 0.02 * (diff - 1.0))

            if roll < saw_chance:
                # Vertically oscillating saw blade
                obs_w = 24.0
                obs_h = 24.0
                base_y = plat.y - 36.0
                self.obstacles.append(
                    Obstacle(
                        x=obs_x,
                        y=base_y,
                        width=obs_w,
                        height=obs_h,
                        kind="saw",
                        base_y=base_y,
                        osc_speed=0.08 + 0.015 * min(10.0, diff),
                        osc_amp=min(45.0, 18.0 + 2.5 * diff),
                        phase=self.rng.uniform(0.0, math.tau),
                    )
                )
            elif roll < saw_chance + overhead_chance:
                # Overhead barrier: player must duck under it (or jump high if low enough,
                # but positioned so standing hits it while ducking clears it cleanly)
                obs_w = min(65.0, 32.0 + 2.5 * (diff - 1.0))
                obs_h = 34.0
                # Bottom of barrier is 26px above platform: standing (42px) hits it, ducking (22px) clears it
                obs_y = plat.y - 26.0 - obs_h
                self.obstacles.append(
                    Obstacle(
                        x=obs_x,
                        y=obs_y,
                        width=obs_w,
                        height=obs_h,
                        kind="overhead",
                        base_y=obs_y,
                    )
                )
            else:
                # Ground spike: player must jump over it
                obs_w = min(48.0, 22.0 + 2.0 * (diff - 1.0))
                obs_h = min(42.0, 24.0 + 1.5 * (diff - 1.0))
                obs_y = plat.y - obs_h
                self.obstacles.append(
                    Obstacle(
                        x=obs_x,
                        y=obs_y,
                        width=obs_w,
                        height=obs_h,
                        kind="spike",
                        base_y=obs_y,
                    )
                )

    def step(self, action: int) -> Tuple[List[float], float, bool, Dict[str, object]]:
        """
        Advance the simulation by one tick using `action`.

        Returns:
            state: Normalized feature vector for the new state.
            reward: Incremental reward earned on this step.
            done: True if the player died or reached `max_steps`.
            info: Diagnostic dictionary (score, distance, difficulty, death_reason).
        """
        if self.done:
            return self.get_state(), 0.0, True, self._get_info()

        self.steps += 1
        self._update_difficulty()

        prev_bottom = self.player.bottom

        # Handle Ducking state transition (keep feet planted on ground)
        was_ducking = self.player.is_ducking
        self.player.is_ducking = (action == ACTION_DUCK)
        if self.player.on_ground and was_ducking != self.player.is_ducking:
            self.player.y = prev_bottom - self.player.height

        # Handle Jump
        if action == ACTION_JUMP and self.player.on_ground:
            self.player.vy = self.JUMP_VELOCITY
            self.player.on_ground = False

        # Handle horizontal relative adjustments (LEFT / RIGHT)
        if action == ACTION_LEFT:
            self.player.vx = -self.HORIZONTAL_ACCEL
        elif action == ACTION_RIGHT:
            self.player.vx = self.HORIZONTAL_ACCEL
        else:
            # Gently decay relative velocity
            self.player.vx *= 0.6

        # Update player horizontal position within screen bounds
        self.player.x = max(
            self.MIN_PLAYER_X,
            min(self.MAX_PLAYER_X, self.player.x + self.player.vx),
        )

        # Apply gravity (faster fall if ducking in mid-air)
        grav = self.FAST_FALL_GRAVITY if (self.player.is_ducking and not self.player.on_ground) else self.GRAVITY
        self.player.vy = min(18.0, self.player.vy + grav)
        self.player.y += self.player.vy

        # Scroll world left by scroll_speed
        dx = self.scroll_speed
        self.distance += dx
        for plat in self.platforms:
            plat.x -= dx
        for obs in self.obstacles:
            obs.x -= dx
            if obs.kind == "saw":
                obs.y = obs.base_y - abs(math.sin(self.steps * obs.osc_speed + obs.phase)) * obs.osc_amp

        # Platform landing / ground collision check
        self.player.on_ground = False
        for plat in self.platforms:
            # Horizontal overlap between player and platform
            if self.player.right > plat.x + 2.0 and self.player.x < plat.right - 2.0:
                # Landing from above onto platform surface
                if self.player.vy >= 0.0 and prev_bottom <= plat.y + 10.0 and self.player.bottom >= plat.y:
                    self.player.y = plat.y - self.player.height
                    self.player.vy = 0.0
                    self.player.on_ground = True
                    break
                # Side collision against a raised step/cliff
                elif self.player.bottom > plat.y + 8.0 and self.player.y < plat.y + plat.height:
                    self.player.alive = False
                    self.done = True
                    self.death_reason = "crashed_into_platform_wall"
                    break

        # Check pit fall
        if not self.done and self.player.y > self.SCREEN_HEIGHT:
            self.player.alive = False
            self.done = True
            self.death_reason = "fell_in_pit"

        # Check obstacle collisions and count cleared obstacles
        step_bonus = 0.0
        if not self.done:
            for obs in self.obstacles:
                if self._check_aabb_collision(
                    self.player.x + 3.0,
                    self.player.y + 3.0,
                    self.player.width - 6.0,
                    self.player.height - 6.0,
                    obs.x,
                    obs.y,
                    obs.width,
                    obs.height,
                ):
                    self.player.alive = False
                    self.done = True
                    self.death_reason = f"hit_{obs.kind}"
                    break

                if not obs.passed and obs.right < self.player.x:
                    obs.passed = True
                    self.obstacles_cleared += 1
                    step_bonus += 25.0 * self.difficulty

        # Prune off-screen entities and spawn new ones ahead
        self.platforms = [p for p in self.platforms if p.right > -100.0]
        self.obstacles = [o for o in self.obstacles if o.right > -100.0]
        self._spawn_world_ahead()

        # Update score
        step_reward = (dx * 0.2) + step_bonus
        if self.done and not self.player.alive:
            step_reward = -10.0
        else:
            self.score += step_reward

        if self.steps >= self.max_steps and not self.done:
            self.done = True
            self.death_reason = "max_steps_reached"

        return self.get_state(), step_reward, self.done, self._get_info()

    @staticmethod
    def _check_aabb_collision(
        x1: float, y1: float, w1: float, h1: float,
        x2: float, y2: float, w2: float, h2: float,
    ) -> bool:
        return (
            x1 < x2 + w2
            and x1 + w1 > x2
            and y1 < y2 + h2
            and y1 + h1 > y2
        )

    def get_state(self) -> List[float]:
        """
        Return a 16-dimensional normalized observation vector describing:
          [0] player_x_norm
          [1] player_y_norm
          [2] player_vy_norm
          [3] player_on_ground (0 or 1)
          [4] player_is_ducking (0 or 1)
          [5] scroll_speed_norm
          [6] difficulty_norm
          [7] dist_to_gap_ahead_norm (distance to end of current/underfoot platform)
          [8] next_platform_dx_norm
          [9] next_platform_dy_norm
          [10] obs1_dx_norm (nearest upcoming obstacle relative x)
          [11] obs1_dy_norm (nearest upcoming obstacle relative y)
          [12] obs1_height_norm
          [13] obs1_is_overhead (1 if overhead barrier, 0 otherwise)
          [14] obs2_dx_norm (second upcoming obstacle relative x)
          [15] obs2_dy_norm (second upcoming obstacle relative y)
        """
        px = self.player.x
        py = self.player.y

        # Find current platform under or immediately ahead of player
        current_plat: Optional[Platform] = None
        next_plat: Optional[Platform] = None
        for plat in self.platforms:
            if plat.right >= px:
                if current_plat is None:
                    current_plat = plat
                elif next_plat is None:
                    next_plat = plat
                    break

        dist_to_edge = (current_plat.right - px) if current_plat else 0.0
        next_plat_dx = (next_plat.x - px) if next_plat else 400.0
        next_plat_dy = (next_plat.y - py) if next_plat else 0.0

        # Find nearest 2 upcoming obstacles
        upcoming_obs = [o for o in self.obstacles if o.right >= px]
        upcoming_obs.sort(key=lambda o: o.x)

        if len(upcoming_obs) >= 1:
            o1 = upcoming_obs[0]
            obs1_dx = o1.x - px
            obs1_dy = o1.y - py
            obs1_h = o1.height
            obs1_overhead = 1.0 if o1.kind == "overhead" else 0.0
        else:
            obs1_dx = 600.0
            obs1_dy = 0.0
            obs1_h = 0.0
            obs1_overhead = 0.0

        if len(upcoming_obs) >= 2:
            o2 = upcoming_obs[1]
            obs2_dx = o2.x - px
            obs2_dy = o2.y - py
        else:
            obs2_dx = 600.0
            obs2_dy = 0.0

        return [
            px / self.SCREEN_WIDTH,
            py / self.SCREEN_HEIGHT,
            self.player.vy / 20.0,
            1.0 if self.player.on_ground else 0.0,
            1.0 if self.player.is_ducking else 0.0,
            self.scroll_speed / 20.0,
            math.log1p(self.difficulty) / 5.0,
            min(1.0, dist_to_edge / 500.0),
            min(1.0, next_plat_dx / 500.0),
            next_plat_dy / 200.0,
            min(1.0, obs1_dx / 500.0),
            obs1_dy / 200.0,
            obs1_h / 100.0,
            obs1_overhead,
            min(1.0, obs2_dx / 500.0),
            obs2_dy / 200.0,
        ]

    def _get_info(self) -> Dict[str, object]:
        return {
            "steps": self.steps,
            "distance": round(self.distance, 2),
            "score": round(self.score, 2),
            "difficulty": round(self.difficulty, 3),
            "scroll_speed": round(self.scroll_speed, 2),
            "obstacles_cleared": self.obstacles_cleared,
            "death_reason": self.death_reason,
        }


class TkinterGameViewer:
    """
    Visual 60 FPS window for watching a model play or playing manually via keyboard.
    Uses Python's standard library `tkinter` so no external GUI binaries are required.
    """

    def __init__(self, game: SidescrollerGame, model=None, human_mode: bool = False):
        import tkinter as tk
        from baseline_model import RandomBaselineModel
        from models import load_trained_models

        self.tk = tk
        self.game = game

        # Build roster of available AI models for live cycling via 'N'
        self.available_models = [RandomBaselineModel(seed=42)]
        loaded = load_trained_models()
        if loaded:
            self.available_models.extend([
                loaded["linear"],
                loaded["qlearning"],
                loaded["mlp"],
            ])

        if model is None:
            model = self.available_models[-1] if len(self.available_models) > 1 else self.available_models[0]
        elif all(getattr(m, "name", "") != getattr(model, "name", "") for m in self.available_models):
            self.available_models.append(model)

        self.model = model
        self.model_index = next(
            (i for i, m in enumerate(self.available_models) if getattr(m, "name", "") == getattr(model, "name", "")),
            0,
        )
        self.human_mode = human_mode

        self.root = tk.Tk()
        self._update_title()
        self.root.resizable(False, False)

        # Center the window on screen and force it to the foreground on Windows
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        pos_x = max(0, (screen_w - SidescrollerGame.SCREEN_WIDTH) // 2)
        pos_y = max(0, (screen_h - SidescrollerGame.SCREEN_HEIGHT) // 2 - 40)
        self.root.geometry(
            f"{SidescrollerGame.SCREEN_WIDTH}x{SidescrollerGame.SCREEN_HEIGHT}+{pos_x}+{pos_y}"
        )

        self.canvas = tk.Canvas(
            self.root,
            width=SidescrollerGame.SCREEN_WIDTH,
            height=SidescrollerGame.SCREEN_HEIGHT,
            bg="#121624",
            highlightthickness=0,
        )
        self.canvas.pack()

        self.pressed_keys = set()
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.canvas.bind("<Button-1>", lambda _e: self.canvas.focus_set())

        self.episode = 1
        self.best_score = 0.0
        self.last_action = ACTION_NOOP
        self.death_pause_ticks = 0

        # Ensure the window pops up in front of the IDE/terminal on Windows
        self.root.deiconify()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.update_idletasks()
        self.root.focus_force()
        self.canvas.focus_set()
        self.root.after(400, lambda: self.root.attributes("-topmost", False))

    def _update_title(self) -> None:
        title_mode = "Human Player" if self.human_mode else f"AI Model: {getattr(self.model, 'name', 'Agent')}"
        self.root.title(f"MLE 2026 - Infinite Scaling Sidescroller [{title_mode}]")

    def _on_key_press(self, event) -> None:
        key = event.keysym.lower()
        self.pressed_keys.add(key)
        if key == "r":
            self.episode += 1
            self.death_pause_ticks = 0
            self.game.reset()
        elif key in ("tab", "m"):
            self.human_mode = not self.human_mode
            self._update_title()
            if self.game.done:
                self.episode += 1
                self.death_pause_ticks = 0
                self.game.reset()
        elif key == "n":
            self.human_mode = False
            self.model_index = (self.model_index + 1) % len(self.available_models)
            self.model = self.available_models[self.model_index]
            self.best_score = 0.0
            self._update_title()
            self.episode += 1
            self.death_pause_ticks = 0
            self.game.reset()
        elif key == "escape":
            self.root.destroy()

    def _on_key_release(self, event) -> None:
        key = event.keysym.lower()
        self.pressed_keys.discard(key)

    def _get_human_action(self) -> int:
        if "space" in self.pressed_keys or "up" in self.pressed_keys or "w" in self.pressed_keys:
            return ACTION_JUMP
        if "down" in self.pressed_keys or "s" in self.pressed_keys:
            return ACTION_DUCK
        if "left" in self.pressed_keys or "a" in self.pressed_keys:
            return ACTION_LEFT
        if "right" in self.pressed_keys or "d" in self.pressed_keys:
            return ACTION_RIGHT
        return ACTION_NOOP

    def run(self) -> None:
        print(
            "Opening 2D Sidescroller GUI window...\n"
            "  Controls: [W/Up/Space]=Jump  [S/Down]=Duck  [A/Left & D/Right]=Move\n"
            "            [M or Tab]=Toggle Human / AI Model  [R]=Restart  [Esc]=Quit",
            flush=True,
        )
        self._tick()
        self.root.mainloop()

    def _tick(self) -> None:
        if not self.game.done:
            state = self.game.get_state()
            if self.human_mode:
                action = self._get_human_action()
            elif self.model is not None:
                action = self.model.select_action(state)
            else:
                action = ACTION_NOOP

            self.last_action = action
            self.game.step(action)
            self.best_score = max(self.best_score, self.game.score)
        elif not self.human_mode:
            # Pause briefly (~0.6s) on death when watching an AI model so the user can see the outcome
            self.death_pause_ticks += 1
            if self.death_pause_ticks >= 35:
                self.death_pause_ticks = 0
                self.episode += 1
                self.game.reset()

        self._render()
        self.root.after(16, self._tick)  # ~60 FPS

    def _render(self) -> None:
        c = self.canvas
        c.delete("all")

        # Scrolling background grid lines
        grid_offset = int(self.game.distance * 0.4) % 50
        for x in range(-grid_offset, SidescrollerGame.SCREEN_WIDTH, 50):
            c.create_line(x, 0, x, SidescrollerGame.SCREEN_HEIGHT, fill="#1b2136")
        for y in range(0, SidescrollerGame.SCREEN_HEIGHT, 50):
            c.create_line(0, y, SidescrollerGame.SCREEN_WIDTH, y, fill="#1b2136")

        # Draw platforms
        for plat in self.game.platforms:
            c.create_rectangle(
                plat.x, plat.y, plat.right, SidescrollerGame.SCREEN_HEIGHT,
                fill="#29324e", outline="#4a5a8a", width=2,
            )
            c.create_rectangle(
                plat.x, plat.y, plat.right, plat.y + 6,
                fill="#5c7cfa", outline="",
            )

        # Draw obstacles
        for obs in self.game.obstacles:
            if obs.kind == "spike":
                c.create_polygon(
                    obs.x, obs.bottom,
                    obs.x + obs.width / 2.0, obs.y,
                    obs.right, obs.bottom,
                    fill="#ff4d6d", outline="#ff8fa3", width=1,
                )
            elif obs.kind == "overhead":
                c.create_rectangle(
                    obs.x, obs.y, obs.right, obs.bottom,
                    fill="#f77f00", outline="#fcbf49", width=2,
                )
            elif obs.kind == "saw":
                c.create_oval(
                    obs.x, obs.y, obs.right, obs.bottom,
                    fill="#e63946", outline="#f1faee", width=2,
                )

        # Draw player
        p = self.game.player
        player_color = "#06d6a0" if p.alive else "#ef476f"
        c.create_rectangle(
            p.x, p.y, p.right, p.bottom,
            fill=player_color, outline="#ffffff", width=2,
        )

        # HUD overlay
        model_label = "HUMAN (WASD/Arrows/Space)" if self.human_mode else f"AI: {getattr(self.model, 'name', 'Agent')}"
        hud_text = (
            f"Ep: {self.episode}  |  "
            f"Score: {self.game.score:6.1f} (Best: {self.best_score:6.1f})  |  "
            f"Dist: {self.game.distance:6.0f}  |  "
            f"Diff: {self.game.difficulty:4.2f}x  |  "
            f"Spd: {self.game.scroll_speed:4.1f}  |  "
            f"Act: {ACTION_NAMES.get(self.last_action, 'NOOP')}"
        )
        c.create_rectangle(0, 0, SidescrollerGame.SCREEN_WIDTH, 34, fill="#0b0e17", outline="")
        c.create_text(
            12, 17, anchor="w", text=hud_text,
            fill="#e2e8f0", font=("Consolas", 10, "bold"),
        )
        c.create_text(
            12, 48, anchor="w",
            text=f"Mode: {model_label}  |  Cleared: {self.game.obstacles_cleared}  |  [M/Tab] Human/AI  [N] Next AI Model  [R] Restart",
            fill="#94a3b8", font=("Consolas", 10),
        )

        if self.game.done:
            restart_hint = "Press 'R' to Restart" if self.human_mode else "Auto-restarting..."
            c.create_text(
                SidescrollerGame.SCREEN_WIDTH // 2,
                SidescrollerGame.SCREEN_HEIGHT // 2,
                text=f"GAME OVER ({self.game.death_reason})\nScore: {self.game.score:.1f}  |  {restart_hint}",
                fill="#ffffff",
                font=("Consolas", 16, "bold"),
                justify="center",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the 2D Infinite Scaling Sidescroller Platformer.")
    parser.add_argument("--play", action="store_true", help="Play the game manually using keyboard controls (default).")
    parser.add_argument("--watch", "--watch-random", dest="watch_random", action="store_true", help="Watch the random baseline model play in a GUI window.")
    parser.add_argument(
        "--watch-model",
        choices=["random", "linear", "qlearning", "mlp"],
        help="Watch a specific trained model ('random', 'linear', 'qlearning', or 'mlp') in the GUI window.",
    )
    parser.add_argument("--headless", action="store_true", help="Run a single headless test episode without opening a window.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the game environment.")
    args = parser.parse_args()

    game_env = SidescrollerGame(seed=args.seed)
    if args.headless:
        rng = random.Random(args.seed)
        state = game_env.reset()
        while not game_env.done:
            action = rng.randrange(NUM_ACTIONS)
            state, reward, done, info = game_env.step(action)
        print(f"Headless episode complete: {info}")
    elif args.watch_model:
        from baseline_model import RandomBaselineModel
        from models import load_trained_models

        if args.watch_model == "random":
            chosen_model = RandomBaselineModel(seed=args.seed)
        else:
            trained = load_trained_models()
            if not trained:
                print("Trained models not found on disk; training now...")
                from train_and_compare import run_training_and_comparison
                run_training_and_comparison(eval_episodes=50, eval_seed=1000)
                trained = load_trained_models()
            chosen_model = trained[args.watch_model]
        viewer = TkinterGameViewer(game_env, model=chosen_model, human_mode=False)
        viewer.run()
    elif args.watch_random:
        from baseline_model import RandomBaselineModel
        baseline = RandomBaselineModel(seed=args.seed)
        viewer = TkinterGameViewer(game_env, model=baseline, human_mode=False)
        viewer.run()
    else:
        # Default to opening the interactive window in Human Play mode
        viewer = TkinterGameViewer(game_env, human_mode=True)
        viewer.run()
