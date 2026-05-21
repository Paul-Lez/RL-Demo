"""Training and evaluation loops for the REINFORCE notebook.

The notebook keeps these loops in a helper file so learners can focus first on
the experiment. This file is still meant to be readable: the training loop is
just repeated episode collection, policy update, and metric recording.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output
from torch import nn

from .environment import GridWorld
from .plotting import moving_average, plot_policy, policy_action_probs
from .policies import (
    DEFAULT_UPDATE_MODE,
    Episode,
    collect_episode,
    reinforce_update_batch,
    set_seed,
)


def train_reinforce(
    env: GridWorld,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    episodes: int = 900,
    gamma: float = 0.98,
    normalize_returns: bool = True,
    entropy_coef: float = 0.01,
    snapshot_every: int = 150,
    seed: int = 7,
    live: bool = False,
    refresh_every: int = 50,
    dashboard_path: str | None = None,
    dashboard_every: int = 10,
    update_mode: str = DEFAULT_UPDATE_MODE,
    batch_size: int = 1,
) -> tuple[dict[str, list[float]], dict[int, np.ndarray]]:
    """Train a policy with REINFORCE and return metrics plus snapshots.

    `stats` stores one value per episode for plotting. `snapshots` stores full
    policy probabilities at selected episodes so the notebook can show how the
    arrows changed over time. `batch_size` controls how many sampled episodes
    are averaged into each policy-gradient update. `update_mode` selects raw
    vanilla REINFORCE or the stabilized advantage-style update.
    """

    set_seed(seed)
    dashboard_every = max(1, int(dashboard_every))
    snapshot_every = max(1, int(snapshot_every))
    batch_size = max(1, int(batch_size))
    stats = {
        "returns": [],
        "lengths": [],
        "success": [],
        "losses": [],
        "grad_norms": [],
    }
    snapshots = {0: policy_action_probs(policy, env)}
    last_episode: Episode | None = None

    if dashboard_path is not None:
        from .dashboard import write_dashboard_snapshot

        write_dashboard_snapshot(dashboard_path, env, policy, stats, 0, status="starting")

    episode_number = 0
    next_snapshot = snapshot_every
    while episode_number < episodes:
        current_batch_size = min(batch_size, episodes - episode_number)

        # 1. Try the task several times using the current policy.
        batch = [
            collect_episode(env, policy, greedy=False, track_grad=True)
            for _ in range(current_batch_size)
        ]
        last_episode = batch[-1]

        # 2. Average those sampled episodes into one policy-gradient update.
        metrics = reinforce_update_batch(
            policy,
            optimizer,
            batch,
            gamma=gamma,
            normalize_returns=normalize_returns,
            entropy_coef=entropy_coef,
            update_mode=update_mode,
        )

        # 3. Record per-episode outcomes so we can inspect learning afterward.
        for episode in batch:
            episode_number += 1
            reached_goal = bool(episode["infos"][-1].reached_goal)
            stats["returns"].append(float(sum(episode["rewards"])))
            stats["lengths"].append(len(episode["rewards"]))
            stats["success"].append(float(reached_goal))
            stats["losses"].append(metrics["loss"])
            stats["grad_norms"].append(metrics["grad_norm"])

        while episode_number >= next_snapshot:
            snapshots[next_snapshot] = policy_action_probs(policy, env)
            next_snapshot += snapshot_every
        if episode_number == episodes:
            snapshots[episode_number] = policy_action_probs(policy, env)

        should_write_dashboard = (
            episode_number == current_batch_size
            or episode_number % dashboard_every == 0
            or episode_number == episodes
        )
        if dashboard_path is not None and should_write_dashboard:
            from .dashboard import write_dashboard_snapshot

            write_dashboard_snapshot(
                dashboard_path,
                env,
                policy,
                stats,
                episode_number,
                episode=episode,
                status="training",
            )

        if live and (episode_number == current_batch_size or episode_number % refresh_every == 0):
            clear_output(wait=True)
            fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
            xs = np.arange(1, len(stats["returns"]) + 1)
            axes[0].plot(xs, stats["returns"], alpha=0.25, color="#64748b")
            axes[0].plot(xs, moving_average(stats["returns"], 40), color="#0f766e")
            axes[0].set_title(f"training return through episode {episode_number}")
            axes[0].set_xlabel("episode")
            plot_policy(env, policy, ax=axes[1], title=f"policy after {episode_number} episodes")
            plt.show()

    if dashboard_path is not None:
        from .dashboard import write_dashboard_snapshot

        write_dashboard_snapshot(
            dashboard_path,
            env,
            policy,
            stats,
            episodes,
            episode=last_episode,
            status="complete",
        )

    return stats, snapshots


def evaluate_policy(env: GridWorld, policy: nn.Module, episodes: int = 100, greedy: bool = True) -> dict[str, float]:
    """Run several evaluation episodes and summarize performance."""

    returns, lengths, successes = [], [], []
    for _ in range(episodes):
        episode = collect_episode(env, policy, greedy=greedy, track_grad=False)
        returns.append(sum(episode["rewards"]))
        lengths.append(len(episode["rewards"]))
        successes.append(float(episode["infos"][-1].reached_goal))
    return {
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
    }


def sample_policy_trajectories(
    env: GridWorld,
    policy: nn.Module,
    num_trajectories: int = 5,
    seed: int | None = None,
) -> list[Episode]:
    """Collect several sampled rollouts for visualization."""

    if seed is not None:
        set_seed(seed)
    return [collect_episode(env, policy, greedy=False, track_grad=False) for _ in range(num_trajectories)]
