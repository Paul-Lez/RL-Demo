"""Grid-world environment used by the REINFORCE notebook.

The environment is intentionally small and plain Python. It follows the same
basic idea as many RL environments:

1. `reset()` starts a new episode.
2. `step(action)` applies one action and returns the new state, reward, done
   flag, and some extra information.

States are `(row, col)` grid coordinates. The top-left cell is `(0, 0)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

State = tuple[int, int]

# Actions are changes in `(row, col)`: up, right, down, left.
ACTIONS: list[State] = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ACTION_NAMES = ["up", "right", "down", "left"]

REWARD_MODES = {
    "default": "Dense reward: step cost, small blocked-move penalty, goal bonus.",
    "sparse": "Delayed reward: zero until the goal, then a fixed goal bonus.",
    "sparse_length": "Delayed reward: zero until the goal, then goal bonus divided by episode length.",
    "dense_noop_penalty": "Dense reward with a stronger penalty for blocked/no-op moves.",
}
REWARD_MODE_ALIASES = {
    "1": "default",
    "2": "sparse",
    "3": "sparse_length",
    "4": "dense_noop_penalty",
}


@dataclass
class StepInfo:
    """Extra facts about the most recent environment step."""

    bumped: bool
    reached_goal: bool
    timeout: bool


class GridWorld:
    """A tiny grid world with obstacles, a start cell, and a goal cell."""

    def __init__(
        self,
        n: int = 6,
        obstacles: Iterable[State] | None = None,
        start: State | None = None,
        goal: State | None = None,
        step_cost: float = -0.01,
        wall_penalty: float = -0.04,
        noop_penalty: float = -0.12,
        goal_reward: float = 1.0,
        reward_mode: str = "default",
        max_steps: int | None = None,
    ):
        reward_mode = REWARD_MODE_ALIASES.get(str(reward_mode), str(reward_mode))
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {sorted(REWARD_MODES)}")

        self.n = int(n)
        self.start = start if start is not None else (self.n - 1, 0)
        self.goal = goal if goal is not None else (0, self.n - 1)
        self.obstacles = set(obstacles or [])
        self.step_cost = float(step_cost)
        self.wall_penalty = float(wall_penalty)
        self.noop_penalty = float(noop_penalty)
        self.goal_reward = float(goal_reward)
        self.reward_mode = reward_mode
        self.max_steps = int(max_steps or 4 * self.n * self.n)

        self._validate_layout()
        self.state = self.start
        self.steps = 0

    @property
    def num_states(self) -> int:
        return self.n * self.n

    @property
    def num_actions(self) -> int:
        return len(ACTIONS)

    def reset(self) -> State:
        """Start a fresh episode and return the start state."""

        self.state = self.start
        self.steps = 0
        return self.state

    def in_bounds(self, state: State) -> bool:
        """Return whether a state is inside the square grid."""

        row, col = state
        return 0 <= row < self.n and 0 <= col < self.n

    def is_blocked(self, state: State) -> bool:
        """Return whether a move would hit a wall or obstacle."""

        return (not self.in_bounds(state)) or state in self.obstacles

    def state_to_index(self, state: State) -> int:
        """Convert `(row, col)` into one integer for the policy network."""

        row, col = state
        return row * self.n + col

    def index_to_state(self, index: int) -> State:
        """Convert one integer back into `(row, col)` coordinates."""

        return divmod(int(index), self.n)

    def free_states(self, include_goal: bool = False) -> list[State]:
        """List cells where the agent is allowed to stand."""

        states = []
        for row in range(self.n):
            for col in range(self.n):
                state = (row, col)
                if state in self.obstacles:
                    continue
                if state == self.goal and not include_goal:
                    continue
                states.append(state)
        return states

    def step(self, action: int) -> tuple[State, float, bool, StepInfo]:
        """Apply one action.

        Returns:
            `(next_state, reward, done, info)`, where `done` says whether the
            episode has finished.
        """

        if not 0 <= action < self.num_actions:
            raise ValueError(f"action must be in [0, {self.num_actions})")

        self.steps += 1

        row, col = self.state
        d_row, d_col = ACTIONS[action]
        candidate = (row + d_row, col + d_col)
        bumped = self.is_blocked(candidate)
        next_state = self.state if bumped else candidate

        self.state = next_state
        reached_goal = next_state == self.goal
        timeout = self.steps >= self.max_steps
        reward = self.reward(bumped=bumped, reached_goal=reached_goal)

        done = reached_goal or timeout
        return next_state, float(reward), done, StepInfo(bumped, reached_goal, timeout)

    def reward(self, bumped: bool, reached_goal: bool) -> float:
        """Compute the reward for the current step under the chosen mode."""

        if self.reward_mode == "default":
            reward = self.step_cost + (self.wall_penalty if bumped else 0.0)
            if reached_goal:
                reward += self.goal_reward
            return reward

        if self.reward_mode == "sparse":
            return self.goal_reward if reached_goal else 0.0

        if self.reward_mode == "sparse_length":
            return self.goal_reward / max(1, self.steps) if reached_goal else 0.0

        if self.reward_mode == "dense_noop_penalty":
            reward = self.step_cost + (self.noop_penalty if bumped else 0.0)
            if reached_goal:
                reward += self.goal_reward
            return reward

        raise ValueError(f"Unknown reward mode: {self.reward_mode}")

    def _validate_layout(self) -> None:
        if not self.in_bounds(self.start):
            raise ValueError("start must be inside the grid")
        if not self.in_bounds(self.goal):
            raise ValueError("goal must be inside the grid")
        if self.start in self.obstacles:
            raise ValueError("start cannot be an obstacle")
        if self.goal in self.obstacles:
            raise ValueError("goal cannot be an obstacle")
