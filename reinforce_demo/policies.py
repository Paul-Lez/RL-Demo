"""Policy models and REINFORCE update helpers.

A policy maps a state to action scores called logits. PyTorch's
`Categorical` distribution turns those logits into probabilities, samples
actions, and gives us the log-probabilities needed for REINFORCE.
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .environment import GridWorld

Episode = dict[str, list]

POLICY_MODES = {
    "tabular": "Independent learnable action logits for every grid cell.",
    "mlp": "Small MLP from one-hot state to action logits.",
    "large_mlp": "Large over-parameterized MLP; useful for showing slow neural-policy learning.",
}
POLICY_MODE_ALIASES = {
    "1": "tabular",
    "2": "mlp",
    "3": "large_mlp",
}

DEFAULT_POLICY_MODE = "tabular"
MLP_HIDDEN_SIZE = 32
LARGE_MLP_HIDDEN_SIZE = 256
LARGE_MLP_HIDDEN_LAYERS = 4


def set_seed(seed: int = 0) -> None:
    """Make random choices repeatable across Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TabularSoftmaxPolicy(nn.Module):
    """One independent action-score vector for each grid cell."""

    def __init__(self, num_states: int, num_actions: int):
        super().__init__()
        self.policy_mode = "tabular"
        self.logits = nn.Embedding(num_states, num_actions)
        # Start with all action scores equal, so the first policy is uniform.
        nn.init.zeros_(self.logits.weight)

    def forward(self, state_indices: torch.Tensor) -> torch.Tensor:
        """Return action logits for each state index."""

        return self.logits(state_indices.long())

    def dist(self, state_indices: torch.Tensor) -> Categorical:
        """Return a distribution over actions for each state index."""

        return Categorical(logits=self.forward(state_indices))


class MLPSoftmaxPolicy(nn.Module):
    """Small neural policy that reads a one-hot state vector."""

    def __init__(self, num_states: int, num_actions: int, hidden_size: int = MLP_HIDDEN_SIZE):
        super().__init__()
        self.policy_mode = "mlp"
        self.num_states = int(num_states)
        self.hidden_size = int(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(self.num_states, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, num_actions),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state_indices: torch.Tensor) -> torch.Tensor:
        """Convert state indices to one-hot vectors, then return logits."""

        state_indices = state_indices.long().view(-1, 1)
        one_hot = torch.zeros(state_indices.shape[0], self.num_states, device=state_indices.device)
        one_hot.scatter_(1, state_indices, 1.0)
        return self.net(one_hot)

    def dist(self, state_indices: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(state_indices))


class LargeMLPSoftmaxPolicy(nn.Module):
    """Larger neural policy used to compare capacity and optimization speed."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        hidden_size: int = LARGE_MLP_HIDDEN_SIZE,
        hidden_layers: int = LARGE_MLP_HIDDEN_LAYERS,
    ):
        super().__init__()
        self.policy_mode = "large_mlp"
        self.num_states = int(num_states)
        self.hidden_size = int(hidden_size)
        self.hidden_layers = int(hidden_layers)

        layers = []
        input_size = self.num_states
        for _ in range(self.hidden_layers):
            layers.append(nn.Linear(input_size, self.hidden_size))
            layers.append(nn.Tanh())
            input_size = self.hidden_size
        layers.append(nn.Linear(input_size, num_actions))
        self.net = nn.Sequential(*layers)

        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state_indices: torch.Tensor) -> torch.Tensor:
        """Convert state indices to one-hot vectors, then return logits."""

        state_indices = state_indices.long().view(-1, 1)
        one_hot = torch.zeros(state_indices.shape[0], self.num_states, device=state_indices.device)
        one_hot.scatter_(1, state_indices, 1.0)
        return self.net(one_hot)

    def dist(self, state_indices: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(state_indices))


def resolve_policy_mode(policy_mode: str) -> str:
    """Allow numbered shortcuts, then check that the policy mode exists."""

    policy_mode = POLICY_MODE_ALIASES.get(str(policy_mode), str(policy_mode))
    if policy_mode not in POLICY_MODES:
        raise ValueError(f"policy_mode must be one of {sorted(POLICY_MODES)}")
    return policy_mode


def make_policy(
    num_states: int,
    num_actions: int,
    policy_mode: str = DEFAULT_POLICY_MODE,
    mlp_hidden_size: int = MLP_HIDDEN_SIZE,
    large_mlp_hidden_size: int = LARGE_MLP_HIDDEN_SIZE,
    large_mlp_hidden_layers: int = LARGE_MLP_HIDDEN_LAYERS,
) -> nn.Module:
    """Construct the selected policy class."""

    policy_mode = resolve_policy_mode(policy_mode)
    if policy_mode == "tabular":
        return TabularSoftmaxPolicy(num_states, num_actions)
    if policy_mode == "mlp":
        return MLPSoftmaxPolicy(num_states, num_actions, hidden_size=mlp_hidden_size)
    if policy_mode == "large_mlp":
        return LargeMLPSoftmaxPolicy(
            num_states,
            num_actions,
            hidden_size=large_mlp_hidden_size,
            hidden_layers=large_mlp_hidden_layers,
        )
    raise ValueError(f"Unknown policy mode: {policy_mode}")


def default_learning_rate(policy_mode: str) -> float:
    """Choose a conservative learning rate for each policy type."""

    policy_mode = resolve_policy_mode(policy_mode)
    if policy_mode == "tabular":
        return 0.05
    if policy_mode == "mlp":
        return 0.01
    if policy_mode == "large_mlp":
        return 0.002
    raise ValueError(f"Unknown policy mode: {policy_mode}")


def collect_episode(
    env: GridWorld,
    policy: nn.Module,
    greedy: bool = False,
    track_grad: bool = True,
) -> Episode:
    """Sample one complete episode from the current policy.

    When `greedy=False`, actions are sampled from the policy probabilities.
    When `greedy=True`, the most likely action is chosen every time.

    `track_grad=True` keeps the log-probabilities connected to PyTorch's
    computation graph, which is needed for the policy-gradient update.
    """

    state = env.reset()
    states, actions, rewards = [], [], []
    log_probs, entropies, next_states, infos = [], [], [], []

    for _ in range(env.max_steps):
        state_index = torch.tensor([env.state_to_index(state)], dtype=torch.long)
        with torch.set_grad_enabled(track_grad):
            dist = policy.dist(state_index)
            if greedy:
                action_tensor = torch.argmax(dist.probs, dim=-1)
            else:
                action_tensor = dist.sample()
            log_prob = dist.log_prob(action_tensor).squeeze(0)
            entropy = dist.entropy().squeeze(0)

        action = int(action_tensor.item())
        next_state, reward, done, info = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        entropies.append(entropy)
        next_states.append(next_state)
        infos.append(info)

        state = next_state
        if done:
            break

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "log_probs": log_probs,
        "entropies": entropies,
        "next_states": next_states,
        "infos": infos,
    }


def discounted_returns(rewards: Sequence[float], gamma: float) -> torch.Tensor:
    """Compute future reward totals `G_t` for every step in an episode.

    `gamma` discounts later rewards. A value near 1 means future rewards still
    matter a lot; a smaller value makes the agent focus more on near rewards.
    """

    returns = []
    running_return = 0.0
    for reward in reversed(rewards):
        running_return = float(reward) + gamma * running_return
        returns.append(running_return)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def returns_and_advantages(
    rewards: Sequence[float],
    gamma: float,
    normalize_returns: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute returns and normalized advantages.

    In this simple notebook we use the return itself as the advantage, then
    optionally normalize it so updates are less sensitive to scale.
    """

    returns = discounted_returns(rewards, gamma)
    advantages = returns.clone()
    if normalize_returns and len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    return returns, advantages


def reinforce_update(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    episode: Episode,
    gamma: float = 0.98,
    normalize_returns: bool = True,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """Apply one REINFORCE update to the policy.

    The loss is the negative policy-gradient objective:
    sampled actions with positive advantage are made more likely, and sampled
    actions with negative advantage are made less likely.
    """

    returns, advantages = returns_and_advantages(episode["rewards"], gamma, normalize_returns)
    log_probs = torch.stack(episode["log_probs"])
    entropies = torch.stack(episode["entropies"])

    policy_loss = -(log_probs * advantages.detach()).sum()
    entropy_bonus = entropies.sum()
    loss = policy_loss - entropy_coef * entropy_bonus

    optimizer.zero_grad()
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "entropy": float(entropies.mean().item()),
        "grad_norm": float(grad_norm),
        "episode_return": float(sum(episode["rewards"])),
    }
