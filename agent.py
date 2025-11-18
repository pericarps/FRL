# agent.py

import random
from collections import defaultdict, deque, namedtuple
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from network import FusedQNetwork
from noise_config import CLIP_NORM
# Opacus fast path
try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    raise ImportError("Opacus is required. Please install opacus>=1.4.0")

ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8


def _to_float(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


Experience = namedtuple(
    "Experience",
    ["experience_id", "state", "action", "reward", "next_state", "done", "aux", "task_id", "sigma"],
)


class ReplayBuffer:
    def __init__(self, capacity: int, device: str = "cpu") -> None:
        self.capacity = int(capacity)
        self.memory: deque[Experience] = deque(maxlen=self.capacity)
        self.device = torch.device(device)

        self.per_task_store: Dict[int, deque] = defaultdict(deque)
        self.per_task_live: Dict[int, set] = defaultdict(set)

        self._expid_to_task: Dict[int, int] = {}
        self._next_experience_id = 0

    def _make_experience(
        self, state, action, reward, next_state, done, aux, task_id, sigma
    ) -> Experience:
        eid = self._next_experience_id
        self._next_experience_id += 1
        return Experience(
            eid, state, action, reward, next_state, done, aux,
            task_id if task_id is not None else 0, _to_float(sigma, 0.0),
        )

    @staticmethod
    def _is_valid_aux(aux: Optional[Dict]) -> bool:
        if aux is None or "task_features" not in aux:
            return False
        tf = aux["task_features"]
        return isinstance(tf, torch.Tensor) and tf.dim() == 3

    def _collate_aux(self, aux_list: List[Optional[Dict]]) -> Optional[Dict]:
        valid_aux = [aux for aux in aux_list if self._is_valid_aux(aux)]
        if not valid_aux:
            return None
        try:
            task_lengths = [aux["task_features"].shape[1] for aux in valid_aux]
            max_len = max(task_lengths)
            first = valid_aux[0]
            tf_dim = first["task_features"].shape[2]
            pf_dim = first["privacy_features"].shape[2]
            df_dim = first["dag_features"].shape[2]
            batch_size = len(valid_aux)
            device = self.device

            tf_padded = torch.zeros(batch_size, max_len, tf_dim, dtype=torch.float32, device=device)
            pf_padded = torch.zeros(batch_size, max_len, pf_dim, dtype=torch.float32, device=device)
            df_padded = torch.zeros(batch_size, max_len, df_dim, dtype=torch.float32, device=device)
            adj_padded = torch.zeros(batch_size, max_len, max_len, dtype=torch.float32, device=device)
            attn_mask = torch.zeros(batch_size, max_len, max_len, dtype=torch.bool, device=device)

            for row_idx, aux in enumerate(valid_aux):
                length = aux["task_features"].shape[1]
                tf_padded[row_idx, :length] = aux["task_features"].to(device)
                pf_padded[row_idx, :length] = aux["privacy_features"].to(device)
                df_padded[row_idx, :length] = aux["dag_features"].to(device)
                adj_padded[row_idx, :length, :length] = aux["dag_adjacency"].to(device)
                if "attention_mask" in aux and isinstance(aux["attention_mask"], torch.Tensor):
                    am = aux["attention_mask"].to(device)
                    if am.dim() == 3 and am.shape[0] == 1:
                        am = am[0]
                    if am.dim() == 2 and am.shape[0] == am.shape[1] == length:
                        attn_mask[row_idx, :length, :length] = (am > 0.5)
                    elif am.dim() == 3 and am.shape[1] == am.shape[2] == length:
                        attn_mask[row_idx, :length, :length] = (am[0] > 0.5) if am.shape[0] == 1 else (am[row_idx] > 0.5)

            return {
                "task_features": tf_padded,
                "privacy_features": pf_padded,
                "dag_features": df_padded,
                "dag_adjacency": adj_padded,
                "attention_mask": attn_mask,  # [B, L, L]
                "total_budget": float(valid_aux[0].get("total_budget", 5.0)),
            }
        except Exception:
            return None

    def push(self, state, action, reward, next_state, done, aux, task_id, sigma) -> None:
        exp = self._make_experience(state, action, reward, next_state, done, aux, task_id, sigma)
        will_evict = (len(self.memory) == self.capacity)
        evicted = self.memory[0] if will_evict else None

        self.memory.append(exp)
        tid = exp.task_id
        self.per_task_store[tid].append(exp)
        self.per_task_live[tid].add(exp.experience_id)
        self._expid_to_task[exp.experience_id] = tid

        if will_evict and evicted is not None:
            old_tid = self._expid_to_task.pop(evicted.experience_id, None)
            if old_tid is not None and evicted.experience_id in self.per_task_live.get(old_tid, set()):
                self.per_task_live[old_tid].remove(evicted.experience_id)

    def _purge_dead_front(self, task_id: int) -> None:
        dq = self.per_task_store.get(task_id)
        live = self.per_task_live.get(task_id)
        if dq is None or live is None:
            return
        while dq and dq[0].experience_id not in live:
            dq.popleft()
        if not dq and (not live or len(live) == 0):
            self.per_task_store.pop(task_id, None)
            self.per_task_live.pop(task_id, None)

    def sample_for_task(self, task_id: int, batch_size: int) -> Optional[Tuple[torch.Tensor, ...]]:
        self._purge_dead_front(task_id)
        dq = self.per_task_store.get(task_id)
        live = self.per_task_live.get(task_id)
        if not dq or not live or len(live) < batch_size:
            return None

        chosen_ids = set(random.sample(list(live), batch_size))
        batch_exps: List[Experience] = []
        for e in dq:
            if e.experience_id in chosen_ids:
                batch_exps.append(e)
                if len(batch_exps) == batch_size:
                    break
        if len(batch_exps) < batch_size:
            self._purge_dead_front(task_id)
            dq2 = self.per_task_store.get(task_id, deque())
            batch_exps = [e for e in dq2 if e.experience_id in chosen_ids][:batch_size]
        if len(batch_exps) < batch_size:
            return None

        states = torch.from_numpy(np.stack([e.state for e in batch_exps])).float()
        actions = torch.tensor([int(e.action) for e in batch_exps], dtype=torch.long)
        rewards = torch.from_numpy(np.stack([e.reward for e in batch_exps])).float()
        next_states = torch.from_numpy(np.stack([e.next_state for e in batch_exps])).float()
        dones = torch.tensor([bool(e.done) for e in batch_exps], dtype=torch.bool)
        aux_batch = self._collate_aux([e.aux for e in batch_exps])
        task_ids = torch.full((len(batch_exps),), task_id, dtype=torch.long)
        sigmas = torch.tensor([_to_float(e.sigma, 0.0) for e in batch_exps], dtype=torch.float)
        return (states, actions, rewards, next_states, dones, aux_batch, task_ids, sigmas)

    def __len__(self) -> int:
        return len(self.memory)

    def current_sample_rate_for_task(self, task_id: int, batch_size: int) -> float:
        self._purge_dead_front(task_id)
        live = self.per_task_live.get(task_id, set())
        pool_size = len(live)
        if pool_size <= 0:
            return 1.0
        return float(min(1.0, batch_size / pool_size))

    def pool_size_for_task(self, task_id: int) -> int:
        self._purge_dead_front(task_id)
        return len(self.per_task_live.get(task_id, set()))


# ---- Dummy dataset/loader to satisfy PrivacyEngine.make_private on older Opacus ----
class _DummyDataset(Dataset):
    def __init__(self, length: int = 1) -> None:
        self._length = max(1, int(length))
    def __len__(self) -> int:
        return self._length
    def __getitem__(self, idx: int):
        return (torch.zeros(1, dtype=torch.float32), torch.zeros(1, dtype=torch.long))


def _make_dummy_loader(size_hint: int, batch_size: int) -> DataLoader:
    ds_len = max(batch_size, int(size_hint))
    ds = _DummyDataset(ds_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


class DoubleDQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        params: Dict,
        device: str = "cpu",
        per_vehicle_budget: Optional[Dict[str, float]] = None,
        vehicle_id: int = 0,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(device)
        self.params = dict(params)
        self.vehicle_id = int(vehicle_id)
        self.per_vehicle_budget = dict(per_vehicle_budget or {})

        try:
            self.dp_dataset_size = int(params["dp_dataset_size"])
        except KeyError:
            # 如果 train.py 忘记传入，则抛出致命错误
            raise KeyError("DoubleDQNAgent params 缺少关键参数: 'dp_dataset_size' (N_total)")
        if self.dp_dataset_size <= 0:
            raise ValueError(f"dp_dataset_size 必须为正, 得到 {self.dp_dataset_size}")
        # ---------------------------------------------------------------------

        # Critic (DDQN)
        q_network_kwargs = dict(
            state_size=state_size,
            action_size=action_size,
            hidden_size=params.get("hidden_size", 256),
            use_gace=params.get("use_gace", False),
            d_model=params.get("d_model", 128),
            nhead=params.get("nhead", 4),
            num_layers=params.get("num_layers", 2),
            device=self.device,
            enable_budget_scaling=params.get("enable_budget_scaling", False),
        )
        self.q_network = FusedQNetwork(**q_network_kwargs).to(self.device)
        self.target_network = FusedQNetwork(**q_network_kwargs).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.gamma = float(params.get("gamma", 0.99))
        self.epsilon = float(params.get("epsilon", 1.0))
        self.epsilon_min = float(params.get("epsilon_min", 0.01))
        self.epsilon_decay = float(params.get("epsilon_decay", 0.995))
        self.max_grad_norm_default = float(params.get("max_grad_norm", 1.0))

        self.memory = ReplayBuffer(params.get("buffer_size", 50000), device=self.device)
        self.default_batch_size = int(params.get("batch_size", 64))

        self.q_optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=params.get("lr", 1e-5),
            betas=ADAM_BETAS,
            eps=ADAM_EPS,
            weight_decay=params.get("weight_decay", 1e-6),
        )

        # Privacy
        self.critic_privacy_engine: Optional[PrivacyEngine] = None
        self.critic_noise_multiplier = float(params.get("critic_noise_multiplier", 1.0))
        self.critic_max_grad_norm = float(params.get("max_grad_norm", 1.0))
        self.secure_mode = bool(params.get("secure_mode", False))

        # 从 per_vehicle_budget 获取 delta 
        self.delta_critic = float(self.per_vehicle_budget.get("delta_critic", params.get("delta_critic", 1e-6)))

        # Steps for stats
        self.critic_steps_done = 0
        self._task_type_map: Dict[int, str] = {}

        if not OPACUS_AVAILABLE:
            raise RuntimeError("Opacus is required for DP-SGD fast path.")

        critic_batch = int(self.params.get("batch_size", 64))
        self._critic_dummy_loader = _make_dummy_loader(size_hint=self.dp_dataset_size, batch_size=critic_batch)
        # ---------------------------------------------------------------------

        self.critic_privacy_engine = PrivacyEngine(secure_mode=self.secure_mode)
        self.q_network, self.q_optimizer, _ = self.critic_privacy_engine.make_private(
            module=self.q_network,
            optimizer=self.q_optimizer,
            data_loader=self._critic_dummy_loader, # <-- 使用了基于 N_total 的 loader
            noise_multiplier=self.critic_noise_multiplier,
            max_grad_norm=self.critic_max_grad_norm,
        )

    def act(self, state: np.ndarray, aux: Optional[Dict] = None, sigma: Optional[float] = None) -> int:
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        state_tensor = torch.nan_to_num(state_tensor, nan=0.0).clamp(-10.0, 10.0)
        aux_device = None
        if aux is not None:
            aux_device = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in aux.items()}
        with torch.no_grad():
            q_values, _ = self.q_network(state_tensor, aux=aux_device)
        return int(torch.argmax(q_values, dim=1).item())

    def remember(self, state, action, reward, next_state, done, aux, task_id, sigma) -> None:
        self.memory.push(state, action, reward, next_state, done, aux, task_id, _to_float(sigma, 0.0))


    def _compute_losses(self, states, actions, rewards, next_states, dones, aux):
        device = self.device
        states = torch.nan_to_num(states, nan=0.0).clamp(-10.0, 10.0).to(device)
        rewards = torch.nan_to_num(rewards, nan=0.0).clamp(-10.0, 10.0).to(device)
        next_states = torch.nan_to_num(next_states, nan=0.0).clamp(-10.0, 10.0).to(device)
        actions = actions.to(device)
        dones = dones.to(device)
        aux_device = None
        if aux is not None:
            aux_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in aux.items()}
        q_all, _ = self.q_network(states, aux=aux_device)
        q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next_online, _ = self.q_network(next_states, aux=aux_device)
            next_actions = torch.argmax(q_next_online, dim=1, keepdim=True)
            q_next_target, _ = self.target_network(next_states, aux=aux_device)
            q_next = q_next_target.gather(1, next_actions).squeeze(1)
            targets = rewards + self.gamma * q_next * (~dones)
            targets = targets.clamp(-10.0, 10.0)
        td_errors = targets - q_sa
        loss_elements = F.smooth_l1_loss(q_sa, targets, reduction="none")
        loss_q = loss_elements.mean()
        info = {
            "total_loss": float(loss_q.item()),
            "q_loss": float(loss_q.item()),
            "td_signal_mean": float(torch.relu(td_errors).mean().item()),
        }
        return loss_q, info, loss_elements

    def train_critic_step(self, task_id: int, noise_multiplier: float) -> Tuple[float, Dict]:
        batch_size = self.default_batch_size
        sample = self.memory.sample_for_task(task_id, batch_size)
        if sample is None:
            return 0.0, {"note": f"insufficient samples for task {task_id}"}
        (states, actions, rewards, next_states, dones, aux, _, sigmas) = sample
        loss, info, loss_elements = self._compute_losses(states, actions, rewards, next_states, dones, aux)

        self.q_optimizer.zero_grad(set_to_none=True)
        loss_elements.mean().backward()
        self.q_optimizer.step()

        self.critic_steps_done += 1

        info.update({
            "dp/critic_q": float(self.memory.current_sample_rate_for_task(task_id, batch_size)),
            "dp/critic_z": float(self.critic_noise_multiplier),
        })

        return float(loss.item()), info

    def update_target_network(self, hard: bool = False, tau: float = 0.001) -> None:
        if hard:
            q_sd = self.get_state_dict()
            self.target_network.load_state_dict(q_sd)
            return
        with torch.no_grad():
            for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.mul_(1.0 - tau).add_(tau * param.data)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_state_dict(self) -> Dict:
        if hasattr(self.q_network, "_module"):
            return self.q_network._module.state_dict()
        return self.q_network.state_dict()

    def set_state_dict(self, state_dict: Dict) -> None:
        if hasattr(self.q_network, "_module"):
            self.q_network._module.load_state_dict(state_dict)
            self.target_network.load_state_dict(state_dict)
        else:
            self.q_network.load_state_dict(state_dict)
            self.target_network.load_state_dict(state_dict)

    def update_task_type_info(self, task_type_map: Dict[int, str]) -> None:
        self._task_type_map = dict(task_type_map)

    def get_all_privacy_status(self) -> Dict[int, Dict]:
        return {}

    def get_type_privacy_status(self) -> Dict[str, Dict]:
        return {}

    def get_current_epsilons(self) -> Dict[str, float]:
        # 仅保留 critic 的占位符（真实 epsilon 由外部 DP 账户逻辑维持）
        return {"critic": float("nan")}

    def get_effective_sample_size(self, task_id: int) -> Dict[str, float]:
        return {"critic": self.memory.pool_size_for_task(task_id)}

    def get_privacy_stats(self) -> Dict[str, float]:
        return {
            "critic_noise_multiplier": self.critic_noise_multiplier,
            "epsilon": self.get_current_epsilons(),
        }

    def get_training_step(self) -> int:
        # 只跟踪 critic 的训练步数
        return self.critic_steps_done

    def set_training_step(self, step: int) -> None:
        self.critic_steps_done = step

    def save(self, path: str) -> None:
        torch.save(self.get_state_dict(), path)

    def load(self, path: str) -> None:
        self.set_state_dict(torch.load(path, map_location=self.device))
