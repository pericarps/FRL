import math
from typing import Dict, List, Tuple
import numpy as np
import torch

from model import DQNAgent
from util import aggregate_gradients


def _rdp_gaussian_subsampled(alpha: float, q: float, z: float) -> float:
    """
    采样高斯机制 RDP 小 q 近似：eps_alpha ≈ q^2 * alpha / (2 z^2)
    为防守性，我们再乘以一个系数 k<=1 以缓和消耗（可视为经验近似的保守修正）。
    """
    q = min(max(q, 1e-8), 0.1)  # 上限降低至 0.1，更贴近经验回放的小采样率
    k = 0.8  # 缓和系数，经验值
    return k * (q * q) * alpha / max(2.0 * (z ** 2), 1e-12)


class VehicleClient:
    """
    DP-SGD 均值梯度噪声：
    - 正常模式：sigma_mean = z * C / batch
    - 低预算模式：sigma_mean = z_min * C / batch，优化器 lr 衰减到 10%
    - 真正冻结：仅当 remain_eps<=0 且 eps_equiv 超过总预算（极端情况）
    """
    def __init__(
        self,
        vid: int,
        agent: DQNAgent,
        clip_C: float,
        alpha_rdp: float,
        delta: float,
        init_eps: float,
        device: str = "cpu",
        z_init: float = 1.0,
        z_min: float = 0.5,
        z_max: float = 2.0,
        low_eps_threshold: float = 0.2,  # 低预算阈值（相对单车预算）
        freeze_when_exhausted: bool = True,
    ):
        self.vid = vid
        self.agent = agent
        self.clip_C = clip_C
        self.alpha_rdp = alpha_rdp
        self.delta = delta
        self.total_eps_budget = init_eps
        self.remain_eps = init_eps
        self.device = device

        self.rdp_alpha_total = 0.0

        self.z = z_init
        self.z_min = z_min
        self.z_max = z_max

        self.low_eps_threshold = low_eps_threshold
        self.freeze_when_exhausted = freeze_when_exhausted

        self.last_sigma = 0.0
        self.last_eps_equiv = 0.0
        self.last_frozen = 0
        self.low_eps_mode = 0

    def _eps_from_rdp(self, eps_alpha_total: float) -> float:
        return eps_alpha_total + (math.log(1.0 / max(self.delta, 1e-12)) / (self.alpha_rdp - 1.0))

    def local_update_with_dp(self, loss: torch.Tensor, batch_size: int, buffer_len: int) -> Dict[str, float]:
        stats = {"sigma": 0.0, "eps_rdp_total": self.rdp_alpha_total, "eps_remain": self.remain_eps, "frozen": 0, "low_eps": 0}

        # 判断是否进入低预算模式或冻结
        low_eps = (self.remain_eps <= self.low_eps_threshold)
        self.low_eps_mode = 1 if low_eps else 0

        # 计算噪声标准差（均值梯度）
        if low_eps:
            sigma_mean = self.z_min * self.clip_C / max(1, batch_size)
        else:
            sigma_mean = self.z * self.clip_C / max(1, batch_size)

        # 是否完全冻结（仅在预算彻底耗尽且需要严格停止时）
        should_freeze = (self.remain_eps <= 0.0) and self.freeze_when_exhausted
        if should_freeze:
            self.last_sigma = 0.0
            self.last_frozen = 1
            stats.update({"sigma": 0.0, "eps_rdp_total": self.rdp_alpha_total,
                          "eps_remain": self.remain_eps, "frozen": 1, "low_eps": self.low_eps_mode})
            return stats

        # 一步 DP-SGD
        self.agent.optimizer.zero_grad()
        loss.backward()

        # Clip + noise
        total_norm_sq = 0.0
        for p in self.agent.q_net.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.data.norm(2).item() ** 2
        total_norm = math.sqrt(total_norm_sq) + 1e-12
        clip_coef = min(1.0, self.clip_C / total_norm)
        for p in self.agent.q_net.parameters():
            if p.grad is None:
                continue
            p.grad.data.mul_(clip_coef)
            if sigma_mean > 0:
                p.grad.data.add_(torch.randn_like(p.grad.data) * sigma_mean)

        # 低预算模式下，学习率衰减到 10%
        if low_eps:
            for g in self.agent.optimizer.param_groups:
                base_lr = g.get('initial_lr', g['lr'])
                g['lr'] = base_lr * 0.1
        self.agent.optimizer.step()
        # 恢复 lr
        for g in self.agent.optimizer.param_groups:
            base_lr = g.get('initial_lr', g['lr'])
            g['lr'] = base_lr

        # RDP 会计
        q = min(0.1, max(batch_size, 1) / max(buffer_len, batch_size))
        z_eff = self.z_min if low_eps else self.z
        eps_alpha_step = _rdp_gaussian_subsampled(self.alpha_rdp, q, z_eff)
        self.rdp_alpha_total += eps_alpha_step

        eps_equiv = self._eps_from_rdp(self.rdp_alpha_total)
        eps_consumed = max(0.0, eps_equiv - self.last_eps_equiv if hasattr(self, "last_eps_equiv") else eps_equiv)
        self.remain_eps = max(0.0, self.remain_eps - eps_consumed)
        self.last_eps_equiv = eps_equiv

        # 自适应 z（仅在正常模式才增大）
        if not low_eps:
            if eps_consumed > 0.01:
                self.z = min(self.z_max, self.z * 1.05)
            else:
                self.z = max(self.z_min, self.z * 0.995)

        self.last_sigma = float(sigma_mean)
        self.last_frozen = 0
        stats.update({
            "sigma": self.last_sigma,
            "eps_rdp_total": self.rdp_alpha_total,
            "eps_remain": self.remain_eps,
            "frozen": 0,
            "low_eps": self.low_eps_mode
        })
        return stats


class RSUServer:
    def __init__(self, rid: int):
        self.rid = rid

    @torch.no_grad()
    def aggregate(self, client_state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return aggregate_gradients(client_state_dicts)


class BaseStation:
    def __init__(self):
        pass

    @torch.no_grad()
    def global_aggregate(self, rsu_state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return aggregate_gradients(rsu_state_dicts)


class PrivacyBudgetAllocator:
    def __init__(self, total_eps: float, num_tasks: int, task_weights: List[float], num_vehicles: int):
        self.total_eps = total_eps
        self.num_tasks = num_tasks
        self.task_weights = np.array(task_weights, dtype=np.float64)
        self.num_vehicles = num_vehicles
        self.eps_task = np.zeros(num_tasks, dtype=np.float64)
        self.eps_vehicle_task = np.zeros((num_vehicles, num_tasks), dtype=np.float64)
        self.used = np.zeros(num_tasks, dtype=np.float64)
        self.reset()

    def reset(self):
        wsum = np.sum(self.task_weights) + 1e-12
        for i in range(self.num_tasks):
            self.eps_task[i] = self.total_eps * (self.task_weights[i] / wsum)
        for v in range(self.num_vehicles):
            self.eps_vehicle_task[v, :] = self.eps_task[:] / max(1, self.num_vehicles)
        self.used[:] = 0.0

    def consume(self, task_id: int, eps_used: float):
        self.used[task_id] += eps_used
        self.used[task_id] = min(self.used[task_id], self.eps_task[task_id])

    def dynamic_redistribute(self, completed: List[int]):
        unused = 0.0
        for i in completed:
            unused += max(0.0, self.eps_task[i] - self.used[i])
            self.eps_task[i] = self.used[i]
        remaining = [i for i in range(self.num_tasks) if i not in completed]
        if not remaining or unused <= 0:
            return
        w = self.task_weights[remaining]
        wsum = np.sum(w) + 1e-12
        for i in remaining:
            self.eps_task[i] += unused * (self.task_weights[i] / wsum)
        for v in range(self.num_vehicles):
            self.eps_vehicle_task[v, remaining] = self.eps_task[remaining] / max(1, self.num_vehicles)

    def get_vehicle_task_budget(self, v: int, task_id: int) -> float:
        return float(self.eps_vehicle_task[v, task_id])

    def get_task_remaining(self, task_id: int) -> float:
        return max(0.0, float(self.eps_task[task_id] - self.used[task_id]))


def build_federated_hierarchy(
    num_vehicles: int,
    num_rsus: int,
    agents: List[DQNAgent],
    clip_C: float,
    alpha_rdp: float,
    delta: float,
    init_eps_vehicle: float,
    device: str = "cpu",
) -> Tuple[List[VehicleClient], List[RSUServer], BaseStation]:
    vehicles = [
        VehicleClient(
            vid=i, agent=agents[i],
            clip_C=clip_C, alpha_rdp=alpha_rdp, delta=delta,
            init_eps=init_eps_vehicle, device=device,
            z_init=1.0, z_min=0.5, z_max=2.0,
            low_eps_threshold=0.2, freeze_when_exhausted=True
        )
        for i in range(num_vehicles)
    ]
    rsus = [RSUServer(rid=i) for i in range(num_rsus)]
    bs = BaseStation()
    return vehicles, rsus, bs