import math
import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from DAG import DAGTasks
from model import DQNAgent


def channel_capacity(B: float, theta: float, P: float, G: float, N0: float, interference: float = 0.0) -> float:
    denom = N0 * B + interference
    snr = (P * G) / max(denom, 1e-12)
    return B * theta * math.log2(1.0 + snr + 1e-12)


def T_local(c_i: int, f_local: float) -> float:
    return c_i / max(1e-12, f_local)


def T_rsu(c_i: int, d_i: int, R_vr: float, f_r: float, eta_vr: float) -> float:
    return (d_i / max(1e-12, R_vr)) + (c_i / max(1e-12, (f_r * eta_vr)))


def T_bs(c_i: int, d_i: int, R_vb: float, f_b: float, eta_vb: float) -> float:
    return (d_i / max(1e-12, R_vb)) + (c_i / max(1e-12, (f_b * eta_vb)))


def dp_sigma_from_epsilon_alpha(alpha: float, delta_v: float, eps_alpha: float) -> float:
    return math.sqrt(max(1e-12, alpha * (delta_v ** 2) / (2.0 * max(eps_alpha, 1e-12))))


class RDPAccountant:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.eps_alpha_total = 0.0

    def add(self, eps_alpha_once: float):
        self.eps_alpha_total += float(eps_alpha_once)

    def to_epsilon(self, delta: float) -> float:
        return self.eps_alpha_total + (math.log(1.0 / max(delta, 1e-12)) / (self.alpha - 1.0))


def aggregate_gradients(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    assert len(state_dicts) > 0
    agg = {}
    for k in state_dicts[0].keys():
        agg[k] = sum([sd[k] for sd in state_dicts]) / float(len(state_dicts))
    return agg


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.cnt = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1):
        self.sum += float(val) * n
        self.cnt += n
        self.avg = self.sum / max(1, self.cnt)


class OffloadEnv:
    def __init__(
        self,
        dag: DAGTasks,
        num_vehicles: int,
        num_rsus: int,
        f_local: float = 5e8,
        f_rsu: float = 2e9,
        f_bs: float = 1e10,
        B_r: float = 10e6,
        B_b: float = 20e6,
        theta_r: float = 0.5,
        theta_b: float = 0.5,
        P_v: float = 0.5,
        G_vr: float = 1e-3,
        G_vb: float = 1e-3,
        N0: float = 1e-9,
        I_vr: float = 1e-9,
        eta_vr: float = 0.2,
        eta_vb: float = 0.1,
        lambda_priv: float = 0.01,
        mu_deadline: float = 2.0,
        gamma: float = 0.99,
        seed: int = 42
    ):
        self.dag = dag
        self.num_vehicles = num_vehicles
        self.num_rsus = num_rsus
        self.f_local = f_local
        self.f_rsu = f_rsu
        self.f_bs = f_bs
        self.B_r = B_r
        self.B_b = B_b
        self.theta_r = theta_r
        self.theta_b = theta_b
        self.P_v = P_v
        self.G_vr = G_vr
        self.G_vb = G_vb
        self.N0 = N0
        self.I_vr = I_vr
        self.eta_vr = eta_vr
        self.eta_vb = eta_vb
        self.lambda_priv = lambda_priv
        self.mu_deadline = mu_deadline
        self.gamma = gamma
        self.seed = seed
        np.random.seed(seed)

        # State dims: [c, d, l, p, layer, f_local, R_vr, R_vb, eps_remain, slack]
        self.state_dim = 10
        self.action_dim = 3

        # scales for normalization
        self._prepare_scales()
        self.reset()

    def _prepare_scales(self):
        self.c_max = max([n.c for n in self.dag.nodes.values()]) if self.dag.nodes else 1.0
        self.d_max = max([n.d for n in self.dag.nodes.values()]) if self.dag.nodes else 1.0
        self.l_max = max([n.l for n in self.dag.nodes.values()]) if self.dag.nodes else 1.0
        self.layer_max = max(1, self.dag.max_layers - 1)
        self.f_local_scale = max(1.0, self.f_local)  # constant
        self.R_vr_max_est = channel_capacity(self.B_r, self.theta_r, self.P_v, self.G_vr * 1.2, self.N0, self.I_vr)
        self.R_vb_max_est = channel_capacity(self.B_b, self.theta_b, self.P_v, self.G_vb * 1.2, self.N0, 0.0)
        self.eps_max_seen = 1.0  # will be updated when set_per_task_eps_remaining is called

    def reset(self):
        self.completed = set()
        self.time = 0.0
        self.topo = self.dag.get_topo_order()
        self.arrival_time = {i: 0.0 for i in self.topo}
        self.deadlines = {i: self.dag.nodes[i].l for i in self.topo}
        self.per_task_eps_remaining = {i: 0.0 for i in self.topo}
        # randomize channel gains per episode
        self.G_vr_ep = self.G_vr * np.random.uniform(0.8, 1.2)
        self.G_vb_ep = self.G_vb * np.random.uniform(0.8, 1.2)
        return self.get_observation_for_next_ready()

    def set_per_task_eps_remaining(self, eps_map: Dict[int, float]):
        self.per_task_eps_remaining.update(eps_map)
        mx = max(1e-6, max(eps_map.values()) if eps_map else 1.0)
        self.eps_max_seen = max(self.eps_max_seen, mx)

    def get_ready_list(self) -> List[int]:
        ready = []
        for i in self.topo:
            if i in self.completed:
                continue
            if self.dag.nodes[i].pre.issubset(self.completed):
                ready.append(i)
        return ready

    def _norm01(self, x: float, m: float) -> float:
        return float(np.clip(x / max(m, 1e-12), 0.0, 1.0))

    def get_observation_for_next_ready(self):
        ready = self.get_ready_list()
        if not ready:
            return None
        t = ready[0]
        node = self.dag.nodes[t]
        R_vr = channel_capacity(self.B_r, self.theta_r, self.P_v, self.G_vr_ep, self.N0, self.I_vr)
        R_vb = channel_capacity(self.B_b, self.theta_b, self.P_v, self.G_vb_ep, self.N0, 0.0)
        slack = node.l - max(0.0, self.time - self.arrival_time[t])
        eps_remain = self.per_task_eps_remaining.get(t, 0.0)

        s = np.array([
            self._norm01(node.c, self.c_max),
            self._norm01(node.d, self.d_max),
            self._norm01(node.l, self.l_max),
            float(np.clip(node.p, 0.0, 1.0)),
            self._norm01(node.layer, self.layer_max),
            self._norm01(self.f_local, self.f_local_scale),
            self._norm01(R_vr, self.R_vr_max_est),
            self._norm01(R_vb, self.R_vb_max_est),
            self._norm01(eps_remain, self.eps_max_seen),
            float(np.clip(slack / max(1e-6, self.l_max), -1.0, 1.0)),  # allow negative
        ], dtype=np.float32)
        return s

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        ready = self.get_ready_list()
        if not ready:
            return None, 0.0, True, {}
        t = ready[0]
        node = self.dag.nodes[t]
        R_vr = channel_capacity(self.B_r, self.theta_r, self.P_v, self.G_vr_ep, self.N0, self.I_vr)
        R_vb = channel_capacity(self.B_b, self.theta_b, self.P_v, self.G_vb_ep, self.N0, 0.0)

        if action == 0:
            T = T_local(node.c, self.f_local)
        elif action == 1:
            T = T_rsu(node.c, node.d, R_vr, self.f_rsu, self.eta_vr)
        else:
            T = T_bs(node.c, node.d, R_vb, self.f_bs, self.eta_vb)

        self.time += T
        # continuous lateness
        lateness = max(0.0, self.time - (self.arrival_time[t] + node.l))
        penalty = self.mu_deadline * lateness

        # privacy proxy (for reward shaping only)
        eps_used_proxy = min(0.05, self.per_task_eps_remaining.get(t, 0.0))
        self.per_task_eps_remaining[t] = max(0.0, self.per_task_eps_remaining.get(t, 0.0) - eps_used_proxy)

        reward = -(T + self.lambda_priv * eps_used_proxy + penalty)

        self.completed.add(t)
        done = len(self.completed) == len(self.topo)
        next_state = self.get_observation_for_next_ready()
        info = {
            "task": t,
            "T": T,
            "lateness": lateness,
            "eps_used_proxy": eps_used_proxy
        }
        return next_state, reward, done, info


def make_agents(num_agents: int, state_dim: int, action_dim: int, device: str = "cpu") -> List[DQNAgent]:
    agents = []
    for _ in range(num_agents):
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=5e-4,
            gamma=0.99,
            tau=0.01,
            device=device,
            hidden_dims=(256, 256)
        )
        agents.append(agent)
    return agents


class ProgressPrinter:
    def __init__(self, total_steps: int):
        self.pbar = tqdm(total=total_steps, dynamic_ncols=True, leave=True)

    def update(self, n: int = 1, postfix: Optional[Dict[str, Any]] = None):
        self.pbar.update(n)
        if postfix:
            self.pbar.set_postfix(postfix)

    def close(self):
        self.pbar.close()


def setup_writer(log_dir: str) -> SummaryWriter:
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)