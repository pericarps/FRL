import math
import random
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, int] = (256, 256)):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.val = nn.Linear(hidden_dims[1], 1)
        self.adv = nn.Linear(hidden_dims[1], action_dim)
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))
        nn.init.uniform_(self.val.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.adv.weight, -1e-3, 1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        V = self.val(x)
        A = self.adv(x)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.frame = 1

    def push(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def beta_by_frame(self):
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames))

    def sample(self, batch_size: int):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.alpha
        probs /= probs.sum() + 1e-8
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        self.frame += 1
        beta = self.beta_by_frame()
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max() + 1e-8
        weights = np.array(weights, dtype=np.float32)
        return samples, indices, torch.tensor(weights)

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = float(prio + 1e-6)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        device: str = "cpu",
        hidden_dims: Tuple[int, int] = (256, 256),
    ):
        self.device = torch.device(device)
        self.q_net = DuelingQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_q_net = DuelingQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        for g in self.optimizer.param_groups:
            g['initial_lr'] = g['lr']
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

    def act(self, state: np.ndarray, eps: float = 0.1) -> int:
        if random.random() < eps:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q_net(s)
            return int(torch.argmax(q, dim=1).item())

    def soft_update(self):
        with torch.no_grad():
            for p, tp in zip(self.q_net.parameters(), self.target_q_net.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def compute_td_loss(self, batch, weights: torch.Tensor):
        states = torch.tensor(np.stack([b[0] for b in batch], axis=0), dtype=torch.float32, device=self.device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device).unsqueeze(-1)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states = torch.tensor(np.stack([b[3] for b in batch], axis=0), dtype=torch.float32, device=self.device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(-1)
        weights = weights.to(self.device).unsqueeze(-1)

        q_vals = self.q_net(states).gather(1, actions)
        # Double DQN target
        with torch.no_grad():
            next_q_online = self.q_net(next_states)
            next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
            next_q_target = self.target_q_net(next_states).gather(1, next_actions)
            target = rewards + (1.0 - dones) * self.gamma * next_q_target

        td_errors = target - q_vals
        loss = (weights * td_errors.pow(2)).mean()
        return loss, td_errors.detach().abs().cpu().numpy()

    def step(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

    def add_dp_noise_to_grads(self, sigma: float, clip_C: float):
        # Per-step global norm clip then Gaussian noise
        total_norm_sq = 0.0
        for p in self.q_net.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.data.norm(2).item() ** 2
        total_norm = math.sqrt(total_norm_sq) + 1e-12
        clip_coef = min(1.0, clip_C / total_norm)
        for p in self.q_net.parameters():
            if p.grad is None:
                continue
            p.grad.data.mul_(clip_coef)
            if sigma > 0.0:
                p.grad.data.add_(torch.randn_like(p.grad.data) * sigma)

    def get_state_dict(self):
        return self.q_net.state_dict()

    def load_state_dict(self, sd):
        self.q_net.load_state_dict(sd)
        self.target_q_net.load_state_dict(sd)

    def to(self, device: str):
        self.device = torch.device(device)
        self.q_net.to(self.device)
        self.target_q_net.to(self.device)