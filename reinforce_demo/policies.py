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

UPDATE_MODES = {
    "vanilla": "Raw discounted returns; no normalization, entropy bonus, or gradient clipping.",
    "advantage": "Return-as-advantage signal with optional normalization and entropy bonus.",
}
UPDATE_MODE_ALIASES = {
    "1": "vanilla",
    "2": "advantage",
}

DEFAULT_POLICY_MODE = "tabular"
DEFAULT_UPDATE_MODE = "advantage"
MLP_HIDDEN_SIZE = 32
LARGE_MLP_HIDDEN_SIZE = 256
LARGE_MLP_HIDDEN_LAYERS = 4


def set_seed(seed: int = 0) -> None:
    """Make random choices repeatable across Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(use_gpu: bool | str = False) -> torch.device:
    """Choose the PyTorch device used by the policy network.

    `use_gpu=False` always returns CPU. `use_gpu=True` tries CUDA first, then
    Apple Silicon MPS, and falls back to CPU if no GPU backend is available.
    String values such as `"cuda"`, `"mps"`, `"cpu"`, and `"auto"` are accepted
    so notebooks can expose a simple tweakable setting.
    """

    if isinstance(use_gpu, str):
        requested = use_gpu.strip().lower()
        if requested in {"false", "0", "no", "cpu"}:
            return torch.device("cpu")
        if requested == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "mps":
            has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            return torch.device("mps" if has_mps else "cpu")
        if requested not in {"true", "1", "yes", "gpu", "auto"}:
            raise ValueError("use_gpu must be a bool or one of: cpu, cuda, mps, auto")
    elif not use_gpu:
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def policy_device(policy: nn.Module) -> torch.device:
    """Return the device where a policy's parameters live."""

    try:
        return next(policy.parameters()).device
    except StopIteration:
        return torch.device("cpu")


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


def resolve_update_mode(update_mode: str) -> str:
    """Allow numbered shortcuts, then check that the update mode exists."""

    update_mode = UPDATE_MODE_ALIASES.get(str(update_mode), str(update_mode))
    if update_mode not in UPDATE_MODES:
        raise ValueError(f"update_mode must be one of {sorted(UPDATE_MODES)}")
    return update_mode


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
    device = policy_device(policy)

    for _ in range(env.max_steps):
        state_index = torch.tensor([env.state_to_index(state)], dtype=torch.long, device=device)
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


def discounted_returns(
    rewards: Sequence[float],
    gamma: float,
    device: torch.device | str | None = None,
) -> torch.Tensor:
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
    return torch.tensor(returns, dtype=torch.float32, device=device)


def returns_and_advantages(
    rewards: Sequence[float],
    gamma: float,
    normalize_returns: bool = True,
    update_mode: str = DEFAULT_UPDATE_MODE,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute returns and the learning signal used by the policy update.

    `vanilla` mode uses the raw discounted returns. `advantage` mode uses the
    return itself as an advantage-style signal, then optionally normalizes it so
    updates are less sensitive to scale.
    """

    update_mode = resolve_update_mode(update_mode)
    returns = discounted_returns(rewards, gamma, device=device)
    advantages = returns.clone()
    if update_mode == "advantage" and normalize_returns and len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    return returns, advantages


def _to_float(value: float | torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _gradient_norm(parameters) -> torch.Tensor:
    """Measure the current gradient norm without changing gradients."""

    norms = [parameter.grad.detach().norm(2) for parameter in parameters if parameter.grad is not None]
    if not norms:
        return torch.tensor(0.0)
    return torch.norm(torch.stack(norms), p=2)


def reinforce_update_batch(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    episodes: Sequence[Episode],
    gamma: float = 0.98,
    normalize_returns: bool = True,
    entropy_coef: float = 0.01,
    max_grad_norm: float | None = 1.0,
    update_mode: str = DEFAULT_UPDATE_MODE,
) -> dict[str, float]:
    """Apply one REINFORCE update from a batch of sampled episodes.

    `vanilla` mode uses raw discounted returns in the policy-gradient loss.
    `advantage` mode keeps the existing normalized return-as-advantage signal
    and optional entropy bonus. The batch loss averages the per-episode policy
    gradients, matching the usual sampled-trajectory estimate of REINFORCE.
    """

    episodes = list(episodes)
    if len(episodes) == 0:
        raise ValueError("episodes must contain at least one sampled episode")

    update_mode = resolve_update_mode(update_mode)
    if update_mode == "vanilla":
        normalize_returns = False
        entropy_coef = 0.0
        max_grad_norm = None

    prepared = []
    all_advantages = []
    for episode in episodes:
        log_probs = torch.stack(episode["log_probs"])
        entropies = torch.stack(episode["entropies"])
        returns = discounted_returns(episode["rewards"], gamma, device=log_probs.device)
        advantages = returns.clone()
        prepared.append((episode, log_probs, entropies, advantages))
        all_advantages.append(advantages)

    if update_mode == "advantage" and normalize_returns:
        joined_advantages = torch.cat(all_advantages)
        if len(joined_advantages) > 1:
            mean = joined_advantages.mean()
            std = joined_advantages.std(unbiased=False) + 1e-8
            prepared = [
                (episode, log_probs, entropies, (advantages - mean) / std)
                for episode, log_probs, entropies, advantages in prepared
            ]

    policy_losses = []
    entropy_bonuses = []
    all_entropies = []
    for _, log_probs, entropies, advantages in prepared:
        policy_losses.append(-(log_probs * advantages.detach()).sum())
        entropy_bonuses.append(entropies.sum())
        all_entropies.append(entropies.detach())

    policy_loss = torch.stack(policy_losses).mean()
    entropy_bonus = torch.stack(entropy_bonuses).mean()
    loss = policy_loss - entropy_coef * entropy_bonus

    optimizer.zero_grad()
    loss.backward()
    if max_grad_norm is None:
        grad_norm = _gradient_norm(policy.parameters())
    else:
        grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()

    return {
        "loss": _to_float(loss),
        "policy_loss": _to_float(policy_loss),
        "entropy": _to_float(torch.cat(all_entropies).mean()),
        "grad_norm": _to_float(grad_norm),
        "episode_return": float(np.mean([sum(episode["rewards"]) for episode in episodes])),
        "mean_episode_return": float(np.mean([sum(episode["rewards"]) for episode in episodes])),
        "mean_length": float(np.mean([len(episode["rewards"]) for episode in episodes])),
        "batch_size": float(len(episodes)),
    }


def reinforce_update(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    episode: Episode,
    gamma: float = 0.98,
    normalize_returns: bool = True,
    entropy_coef: float = 0.01,
    max_grad_norm: float | None = 1.0,
    update_mode: str = DEFAULT_UPDATE_MODE,
) -> dict[str, float]:
    """Apply one REINFORCE update from a single sampled episode."""

    return reinforce_update_batch(
        policy,
        optimizer,
        [episode],
        gamma=gamma,
        normalize_returns=normalize_returns,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        update_mode=update_mode,
    )
