"""Optional JSON snapshots for the separate live dashboard page.

The main notebook does not need this file. It exists so an external HTML page
can poll `runs/latest.json` while training runs and show live metrics.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from torch import nn

from .environment import ACTION_NAMES, REWARD_MODES, GridWorld
from .policies import Episode
from .plotting import policy_action_probs


def recent_mean(values: list[float], window: int = 50) -> float | None:
    """Mean of the most recent values, or `None` when there are no values."""

    if len(values) == 0:
        return None
    return float(np.mean(values[-window:]))


def json_ready_policy_probs(policy: nn.Module, env: GridWorld) -> list[list[list[float | None]]]:
    """Convert policy probabilities into JSON-safe Python values."""

    probs = policy_action_probs(policy, env)
    return [
        [
            [None if not np.isfinite(value) else round(float(value), 6) for value in probs[row, col]]
            for col in range(env.n)
        ]
        for row in range(env.n)
    ]


def json_ready_stats(stats: dict[str, list[float]]) -> dict[str, list[float]]:
    """Convert NumPy-like metric values into regular floats for JSON."""

    return {key: [float(value) for value in values] for key, values in stats.items()}


def write_dashboard_snapshot(
    path: str | Path,
    env: GridWorld,
    policy: nn.Module,
    stats: dict[str, list[float]],
    episode_number: int,
    episode: Episode | None = None,
    status: str = "training",
) -> dict:
    """Write the current training state as a JSON file for the dashboard."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    latest_trajectory: list[list[int]] = []
    latest_actions: list[str] = []
    latest_return = None
    latest_length = None
    latest_success = None
    if episode is not None and len(episode["states"]) > 0:
        trajectory = episode["states"] + episode["next_states"][-1:]
        latest_trajectory = [[int(row), int(col)] for row, col in trajectory]
        latest_actions = [ACTION_NAMES[action] for action in episode["actions"]]
        latest_return = float(sum(episode["rewards"]))
        latest_length = int(len(episode["rewards"]))
        latest_success = bool(episode["infos"][-1].reached_goal)

    payload = {
        "status": status,
        "episode": int(episode_number),
        "updated_at": time.time(),
        "policy": {
            "mode": getattr(policy, "policy_mode", policy.__class__.__name__),
            "class_name": policy.__class__.__name__,
            "num_parameters": int(sum(parameter.numel() for parameter in policy.parameters())),
        },
        "env": {
            "n": int(env.n),
            "start": [int(env.start[0]), int(env.start[1])],
            "goal": [int(env.goal[0]), int(env.goal[1])],
            "obstacles": [[int(row), int(col)] for row, col in sorted(env.obstacles)],
            "action_names": ACTION_NAMES,
            "reward_mode": env.reward_mode,
            "reward_mode_description": REWARD_MODES[env.reward_mode],
            "reward_settings": {
                "step_cost": float(env.step_cost),
                "wall_penalty": float(env.wall_penalty),
                "noop_penalty": float(env.noop_penalty),
                "goal_reward": float(env.goal_reward),
            },
        },
        "stats": json_ready_stats(stats),
        "summary": {
            "return_50": recent_mean(stats["returns"], 50),
            "length_50": recent_mean(stats["lengths"], 50),
            "success_50": recent_mean(stats["success"], 50),
            "latest_return": latest_return,
            "latest_length": latest_length,
            "latest_success": latest_success,
        },
        "policy_probs": json_ready_policy_probs(policy, env),
        "latest_trajectory": latest_trajectory,
        "latest_actions": latest_actions,
    }

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)
    return payload
