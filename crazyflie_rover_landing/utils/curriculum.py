"""Curriculum learning manager for progressive difficulty training.

Adapted from crazyflie-mape-crazyflow for the drone-rover landing task.
Uses landing success rate (instead of blue win rate) as the performance metric.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class CurriculumLevel:
    """A single difficulty level in the curriculum.

    Attributes:
        name: Human-readable name for this level.
        params: Dictionary of environment config parameters to override.
        spawn: Spawn configuration dict for this level (optional).
    """
    name: str
    params: dict = field(default_factory=dict)
    spawn: dict = field(default_factory=dict)


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning.

    Attributes:
        enabled: Whether curriculum learning is active.
        advance_threshold: Landing rate required to advance (0.0-1.0).
        window_size: Number of episodes to track for landing rate calculation.
        levels: List of curriculum levels from easiest to hardest.
        allow_regression: Whether to go back to easier levels if performance drops.
        regression_threshold: Landing rate below which to regress (if enabled).
    """
    enabled: bool = True
    advance_threshold: float = 0.65
    window_size: int = 100
    levels: list[CurriculumLevel] = field(default_factory=list)
    allow_regression: bool = False
    regression_threshold: float = 0.3


class CurriculumManager:
    """Manages curriculum learning progression based on landing rate.

    Tracks episode outcomes over a rolling window and advances/regresses
    through difficulty levels based on performance thresholds.

    Attributes:
        config: Curriculum configuration.
        current_level: Index of current difficulty level.
        episode_outcomes: Rolling window of episode outcomes (True = landed).
        total_episodes: Total episodes seen across all levels.
        level_episodes: Episodes completed at each level.
    """

    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.current_level = 1
        self.episode_outcomes: deque[bool] = deque(maxlen=config.window_size)
        self.total_episodes = 0
        self.level_episodes: list[int] = [0] * len(config.levels)
        self._on_level_change_callbacks: list[Callable[[int, CurriculumLevel], None]] = []

    @property
    def current_level_config(self) -> CurriculumLevel:
        """Get the current level configuration."""
        return self.config.levels[self.current_level - 1]

    @property
    def landing_rate(self) -> float:
        """Calculate current landing success rate over the window."""
        if len(self.episode_outcomes) == 0:
            return 0.0
        return sum(self.episode_outcomes) / len(self.episode_outcomes)

    @property
    def is_final_level(self) -> bool:
        """Check if currently on the final (hardest) level."""
        return self.current_level >= len(self.config.levels)

    @property
    def window_filled(self) -> bool:
        """Check if the episode window is fully filled."""
        return len(self.episode_outcomes) >= self.config.window_size

    def on_level_change(self, callback: Callable[[int, CurriculumLevel], None]):
        """Register a callback for level changes."""
        self._on_level_change_callbacks.append(callback)

    def set_level(self, level: int, trigger_callbacks: bool = True):
        """Set the curriculum to a specific level."""
        if level < 1 or level > len(self.config.levels):
            raise ValueError(
                f"Level {level} out of range. Valid range: 1-{len(self.config.levels)}"
            )
        old_level = self.current_level
        self.current_level = level
        self.episode_outcomes.clear()
        if trigger_callbacks and old_level != level:
            for callback in self._on_level_change_callbacks:
                callback(self.current_level, self.current_level_config)
        print(f"[Curriculum] Set to level {level} ({self.current_level_config.name})")

    def check_advancement(self, rate: float) -> bool:
        """Check if landing rate exceeds threshold and advance if so."""
        if self.is_final_level:
            return False
        if rate >= self.config.advance_threshold:
            self._advance_level()
            return True
        return False

    def record_episode(self, landed: bool) -> dict[str, Any]:
        """Record an episode outcome and check for level advancement.

        Args:
            landed: Whether the drone successfully landed this episode.

        Returns:
            Dictionary with curriculum state info.
        """
        self.episode_outcomes.append(landed)
        self.total_episodes += 1
        self.level_episodes[self.current_level - 1] += 1

        advanced = False
        regressed = False
        rate_at_change = None

        if self.window_filled:
            rate = self.landing_rate

            if not self.is_final_level and rate >= self.config.advance_threshold:
                rate_at_change = rate
                self._advance_level()
                advanced = True
            elif (self.config.allow_regression
                  and self.current_level > 1
                  and rate < self.config.regression_threshold):
                rate_at_change = rate
                self._regress_level()
                regressed = True

        return {
            "level": self.current_level,
            "level_name": self.current_level_config.name,
            "landing_rate": rate_at_change if rate_at_change is not None else self.landing_rate,
            "window_episodes": len(self.episode_outcomes),
            "advanced": advanced,
            "regressed": regressed,
        }

    def record_landing_batch(self, landed: np.ndarray) -> dict[str, Any]:
        """Record multiple episode outcomes from parallel environments.

        Args:
            landed: Boolean array of shape (n_worlds,) indicating successful landings.

        Returns:
            Dictionary with curriculum state info.
        """
        result = None
        for did_land in landed:
            result = self.record_episode(bool(did_land))
        return result if result else {
            "level": self.current_level,
            "level_name": self.current_level_config.name,
            "landing_rate": self.landing_rate,
            "window_episodes": len(self.episode_outcomes),
            "advanced": False,
            "regressed": False,
        }

    def _advance_level(self):
        """Advance to the next difficulty level."""
        old_level = self.current_level
        self.current_level = min(self.current_level + 1, len(self.config.levels))
        self.episode_outcomes.clear()
        for callback in self._on_level_change_callbacks:
            callback(self.current_level, self.current_level_config)
        print(f"[Curriculum] Advanced from level {old_level} to {self.current_level} "
              f"({self.current_level_config.name})")

    def _regress_level(self):
        """Regress to an easier difficulty level."""
        old_level = self.current_level
        self.current_level = max(self.current_level - 1, 1)
        self.episode_outcomes.clear()
        for callback in self._on_level_change_callbacks:
            callback(self.current_level, self.current_level_config)
        print(f"[Curriculum] Regressed from level {old_level} to {self.current_level} "
              f"({self.current_level_config.name})")

    def get_env_params(self) -> dict[str, Any]:
        """Get environment parameters for the current level."""
        level = self.current_level_config
        params = dict(level.params)
        if level.spawn:
            params["spawn"] = level.spawn
        return params

    def get_stats(self) -> dict[str, Any]:
        """Get curriculum statistics for logging."""
        return {
            "curriculum/level": self.current_level,
            "curriculum/landing_rate": self.landing_rate,
            "curriculum/total_episodes": self.total_episodes,
            "curriculum/window_episodes": len(self.episode_outcomes),
            "curriculum/level_episodes": self.level_episodes[self.current_level - 1],
        }


def load_curriculum_config(config: dict) -> Optional[CurriculumConfig]:
    """Load curriculum configuration from experiment config dict.

    Args:
        config: Full experiment configuration dictionary.

    Returns:
        CurriculumConfig if curriculum is defined and enabled, None otherwise.
    """
    curriculum_cfg = config.get("curriculum")
    if curriculum_cfg is None or not curriculum_cfg.get("enabled", False):
        return None

    levels = []
    for i, level_dict in enumerate(curriculum_cfg.get("levels", [])):
        name = level_dict.get("name")
        if name is None:
            level_num = level_dict.get("level", i + 1)
            name = f"Level {level_num}"

        # drone_spawn and rover_spawn go into spawn dict; everything else into params
        drone_spawn = level_dict.get("drone_spawn", {})
        rover_spawn = level_dict.get("rover_spawn", {})
        spawn = {}
        if drone_spawn:
            spawn["drone"] = drone_spawn
        if rover_spawn:
            spawn["rover"] = rover_spawn

        params = {
            k: v for k, v in level_dict.items()
            if k not in ("name", "level", "drone_spawn", "rover_spawn")
        }

        levels.append(CurriculumLevel(name=name, params=params, spawn=spawn))

    if not levels:
        print("[Curriculum] Warning: No levels defined, disabling curriculum")
        return None

    return CurriculumConfig(
        enabled=True,
        advance_threshold=curriculum_cfg.get("advance_threshold", 0.65),
        window_size=curriculum_cfg.get("window_size", 100),
        levels=levels,
        allow_regression=curriculum_cfg.get("allow_regression", False),
        regression_threshold=curriculum_cfg.get("regression_threshold", 0.3),
    )
