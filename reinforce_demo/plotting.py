"""Plotting helpers for the REINFORCE notebook.

These functions do not change the RL algorithm. They turn the grid, policy,
training curves, and trajectories into pictures so the notebook is easier to
reason about.
"""

from __future__ import annotations

import math
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML
from matplotlib import colors
from matplotlib.animation import FuncAnimation
from torch import nn

from .environment import ACTION_NAMES, GridWorld, State
from .policies import (
    DEFAULT_UPDATE_MODE,
    Episode,
    discounted_returns,
    policy_device,
    resolve_update_mode,
    returns_and_advantages,
)

GRID_CMAP = colors.ListedColormap(["#f8fafc", "#111827", "#bae6fd", "#bbf7d0"])
GRID_NORM = colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], GRID_CMAP.N)
ACTION_COLORS = ["#2563eb", "#16a34a", "#f97316", "#7c3aed"]
DISPLAY_DELTAS = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)], dtype=float)


def configure_plots() -> None:
    """Set a few Matplotlib defaults so figures are readable in notebooks."""

    plt.rcParams.update(
        {
            "figure.figsize": (6, 5),
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "font.size": 10,
        }
    )


def grid_array(env: GridWorld) -> np.ndarray:
    """Encode the grid as numbers that Matplotlib can color."""

    arr = np.zeros((env.n, env.n), dtype=int)
    for row, col in env.obstacles:
        arr[row, col] = 1
    arr[env.goal] = 2
    arr[env.start] = 3
    return arr


def decorate_grid_axes(ax, env: GridWorld) -> None:
    """Add row/column ticks and grid lines to an axes."""

    ax.set_xticks(np.arange(env.n))
    ax.set_yticks(np.arange(env.n))
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_xticks(np.arange(-0.5, env.n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.n, 1), minor=True)
    ax.grid(which="minor", color="#cbd5e1", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlim(-0.5, env.n - 0.5)
    ax.set_ylim(env.n - 0.5, -0.5)
    ax.set_aspect("equal")


def plot_grid(
    env: GridWorld,
    ax=None,
    title: str | None = None,
    trajectory: Sequence[State] | None = None,
):
    """Draw the grid, optionally with a trajectory path overlaid."""

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid_array(env), cmap=GRID_CMAP, norm=GRID_NORM)
    decorate_grid_axes(ax, env)
    ax.text(env.start[1], env.start[0], "S", ha="center", va="center", fontweight="bold")
    ax.text(env.goal[1], env.goal[0], "G", ha="center", va="center", fontweight="bold")

    if trajectory:
        xs = [state[1] for state in trajectory]
        ys = [state[0] for state in trajectory]
        ax.plot(xs, ys, color="#ef4444", linewidth=2.0, alpha=0.85, zorder=4)
        ax.scatter(xs, ys, color="#ef4444", s=22, zorder=5)

    if title:
        ax.set_title(title)
    return ax


def policy_action_probs(policy: nn.Module, env: GridWorld) -> np.ndarray:
    """Return action probabilities for every non-obstacle grid cell."""

    probs = np.full((env.n, env.n, env.num_actions), np.nan, dtype=float)
    device = policy_device(policy)
    with torch.no_grad():
        for state in env.free_states(include_goal=False):
            index = torch.tensor([env.state_to_index(state)], dtype=torch.long, device=device)
            state_probs = policy.dist(index).probs.squeeze(0).cpu().numpy()
            probs[state[0], state[1]] = state_probs
    return probs


def plot_policy_from_probs(
    env: GridWorld,
    probs: np.ndarray,
    ax=None,
    title: str | None = None,
    min_prob: float = 0.04,
):
    """Draw arrows whose lengths show action probabilities."""

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    plot_grid(env, ax=ax, title=title)

    for row, col in env.free_states(include_goal=False):
        state_probs = probs[row, col]
        if not np.all(np.isfinite(state_probs)):
            continue
        for action, prob in enumerate(state_probs):
            if prob < min_prob:
                continue
            dx, dy = DISPLAY_DELTAS[action] * (0.42 * float(prob))
            ax.arrow(
                col,
                row,
                dx,
                dy,
                width=0.012,
                head_width=0.09,
                head_length=0.08,
                length_includes_head=True,
                color=ACTION_COLORS[action],
                alpha=0.85,
                zorder=6,
            )
    return ax


def plot_policy(env: GridWorld, policy: nn.Module, ax=None, title: str | None = None):
    """Compute and draw the policy arrows for the current policy."""

    return plot_policy_from_probs(env, policy_action_probs(policy, env), ax=ax, title=title)


def moving_average(values: Sequence[float], window: int = 50) -> np.ndarray:
    """Smooth noisy episode-by-episode values for easier reading."""

    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    smoothed = np.empty_like(values)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed[i] = values[start : i + 1].mean()
    return smoothed


def plot_training(stats: dict[str, list[float]], window: int = 50):
    """Plot returns, episode lengths, and success rate during training."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    episodes = np.arange(1, len(stats["returns"]) + 1)

    axes[0].plot(episodes, stats["returns"], alpha=0.25, color="#64748b")
    axes[0].plot(episodes, moving_average(stats["returns"], window), color="#0f766e")
    axes[0].set_title("Episode return")
    axes[0].set_xlabel("episode")

    axes[1].plot(episodes, stats["lengths"], alpha=0.25, color="#64748b")
    axes[1].plot(episodes, moving_average(stats["lengths"], window), color="#b45309")
    axes[1].set_title("Episode length")
    axes[1].set_xlabel("episode")

    axes[2].plot(episodes, moving_average(stats["success"], window), color="#2563eb")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_title(f"Success rate, last {window}")
    axes[2].set_xlabel("episode")
    return fig, axes


def plot_policy_snapshots(env: GridWorld, snapshots: dict[int, np.ndarray], episodes_to_show=None):
    """Draw saved policy snapshots from different training episodes."""

    keys = sorted(snapshots) if episodes_to_show is None else list(episodes_to_show)
    ncols = min(3, len(keys))
    nrows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, episode_number in zip(axes, keys):
        plot_policy_from_probs(env, snapshots[episode_number], ax=ax, title=f"episode {episode_number}")
    for ax in axes[len(keys) :]:
        ax.axis("off")
    return fig, axes


def plot_probability_deltas(env: GridWorld, before: np.ndarray, after: np.ndarray):
    """Show how each action probability changed between two policies."""

    delta = after - before
    max_abs = np.nanmax(np.abs(delta))
    max_abs = max(float(max_abs), 1e-6)
    fig, axes = plt.subplots(1, env.num_actions, figsize=(4 * env.num_actions, 4), constrained_layout=True)
    for action, ax in enumerate(axes):
        im = ax.imshow(delta[:, :, action], cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
        decorate_grid_axes(ax, env)
        ax.set_title(f"delta P({ACTION_NAMES[action]})")
        ax.text(env.start[1], env.start[0], "S", ha="center", va="center", fontweight="bold")
        ax.text(env.goal[1], env.goal[0], "G", ha="center", va="center", fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig, axes


def summarize_episode_update(
    env: GridWorld,
    episode: Episode,
    gamma: float = 0.98,
    normalize_returns: bool = True,
    max_rows: int = 18,
    update_mode: str = DEFAULT_UPDATE_MODE,
) -> None:
    """Print the quantities used by the selected REINFORCE update."""

    update_mode = resolve_update_mode(update_mode)
    returns, advantages = returns_and_advantages(
        episode["rewards"],
        gamma,
        normalize_returns=normalize_returns,
        update_mode=update_mode,
    )
    print(f"update mode: {update_mode}")
    print(
        f"{'t':>3}  {'state':>8}  {'action':>6}  "
        f"{'reward':>8}  {'return':>8}  {'signal':>8}  effect"
    )
    print("-" * 67)
    for t, (state, action, reward, ret, adv) in enumerate(
        zip(episode["states"], episode["actions"], episode["rewards"], returns, advantages)
    ):
        if t >= max_rows:
            print(f"... {len(episode['states']) - max_rows} more steps")
            break
        if adv.item() > 1e-8:
            effect = "raise sampled action"
        elif adv.item() < -1e-8:
            effect = "lower sampled action"
        else:
            effect = "little change"
        print(
            f"{t:>3}  {str(state):>8}  {ACTION_NAMES[action]:>6}  "
            f"{reward:>8.3f}  {ret.item():>8.3f}  {adv.item():>8.3f}  {effect}"
        )


def plot_episode_credit(
    env: GridWorld,
    episode: Episode,
    gamma: float = 0.98,
    ax=None,
    title: str = "Episode trajectory and first-visit returns",
):
    """Draw an episode path and label states with first-visit returns."""

    trajectory = episode["states"] + episode["next_states"][-1:]
    plot_grid(env, ax=ax, title=title, trajectory=trajectory)
    returns = discounted_returns(episode["rewards"], gamma).numpy()
    first_visit_return = {}
    for state, ret in zip(episode["states"], returns):
        first_visit_return.setdefault(state, ret)
    if ax is None:
        ax = plt.gca()
    for (row, col), ret in first_visit_return.items():
        ax.text(
            col,
            row,
            f"{ret:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.75, edgecolor="none"),
            zorder=8,
        )
    return ax


def animate_episode(env: GridWorld, episode: Episode, interval: int = 350):
    """Animate a single rollout inside the notebook."""

    trajectory = episode["states"] + episode["next_states"][-1:]
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_grid(env, ax=ax, title="Greedy rollout")
    (agent,) = ax.plot([], [], marker="o", markersize=14, color="#ef4444", zorder=9)

    def init():
        agent.set_data([], [])
        return (agent,)

    def update(frame):
        row, col = trajectory[frame]
        agent.set_data([col], [row])
        return (agent,)

    anim = FuncAnimation(fig, update, init_func=init, frames=len(trajectory), interval=interval, blit=True)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def plot_sampled_trajectories(env: GridWorld, episodes: Sequence[Episode], ax=None):
    """Draw several stochastic rollouts on the same grid."""

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    plot_grid(env, ax=ax, title=f"{len(episodes)} sampled trajectories from the learned policy")
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(episodes))))

    for i, (episode, color) in enumerate(zip(episodes, colors), start=1):
        trajectory = episode["states"] + episode["next_states"][-1:]
        xs = [state[1] for state in trajectory]
        ys = [state[0] for state in trajectory]
        success = episode["infos"][-1].reached_goal
        label = f"rollout {i}: {len(episode['rewards'])} steps, {'success' if success else 'timeout'}"
        ax.plot(xs, ys, color=color, linewidth=2.0, alpha=0.75, label=label, zorder=5)
        ax.scatter(xs[0], ys[0], color=color, s=30, marker="o", zorder=6)
        ax.scatter(xs[-1], ys[-1], color=color, s=44, marker="x", zorder=7)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=9)
    return ax
