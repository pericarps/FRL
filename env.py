# env.py
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Set
from dag import DAGTasks
from noise_config import CLIP_NORM

EPS = 1e-8

class OffloadEnv:
    def __init__(self, dag: DAGTasks, params: Dict, device='cpu'):
        self.dag = dag
        self.num_tasks = dag.num_tasks
        self.device = device
        self.action_size = 3  # Local, RSU, BS

        # 通信参数
        self.B_r = max(float(params.get("B_r", 1e6)), EPS)        # RSU 带宽
        self.B_b = max(float(params.get("B_b", 2e6)), EPS)        # 云带宽
        self.theta_vr = float(params.get("theta_vr", 0.5))        # 车辆到RSU资源分配比例
        self.theta_vb = float(params.get("theta_vb", 0.5))        # 车辆到云资源分配比例
        self.P_v = max(float(params.get("P_v", 0.5)), EPS)        # 车辆发射功率
        self.G_vr = max(float(params.get("G_vr", 1e-5)), EPS)     # 车辆到RSU信道增益
        self.G_vb = max(float(params.get("G_vb", 1e-6)), EPS)     # 车辆到云信道增益
        self.N_0 = max(float(params.get("N_0", 1e-20)), EPS)      # 噪声功率谱密度
        self.I_vr = max(float(params.get("I_vr", 1e-9)), EPS)     # RSU干扰功率
        self.backbone_bw = max(float(params.get("backbone_bw", 1e9)), EPS)  # RSU-云骨干网带宽

        # 计算参数
        self.f_l = max(float(params.get("f_l", 1.5e9)), EPS)      # 本地计算频率
        self.f_r = max(float(params.get("f_r", 3e10)), EPS)       # RSU 计算频率
        self.f_b = max(float(params.get("f_b", 2e11)), EPS)       # 云计算频率
        self.eta_vr = float(params.get("eta_vr", 0.2))            # 车辆到RSU计算效率
        self.eta_vb = float(params.get("eta_vb", 0.1))            # 车辆到云计算效率

        # 环境内部默认 DP 参数（sigma removed - should be set by step() from actor）
        self.dp_defaults = (
            max(float(params.get("q_default", 0.01)), 0.001),
            max(float(params.get("c_default", 1.0)), EPS),
            0.0  # sigma placeholder - will be overridden by step() dp_params
        )

        # 在线本地模型（用于注入本底噪声的占位）
        # 这与 Agent 的 Critic/Actor 完全解耦，仅模拟一次局部训练步的加噪更新
        self.local_model = nn.Linear(15, 1, bias=True).to(self.device)
        self.local_lr = float(params.get("local_lr", 1e-3))
        self.enable_local_bg_noise_update = bool(params.get("enable_local_bg_noise_update", True))
    
        self.local_noise_scale = float(params.get("local_noise_scale", 1.0))
        # 可选：每个任务进行 K 次本地更新，默认 1
        self.local_updates_per_task = int(params.get("local_updates_per_task", 1))

        self.task_dp = {}
        self.completed: Set[int] = set()
        self.device_assign: Dict[int, str] = {}
        self.task_finish_time: Dict[int, float] = {}
        self.earliest_start_time: Dict[int, float] = {}
        self.deadline_violation_penalty = max(float(params.get("deadline_violation_penalty", 10.0)), 0.0)
        self.use_gace = params.get("use_gace", False)
        self.gace_optimizer = None

        self.reset()

    def reset(self):
        self.current_time = 0.0
        self.completed.clear()
        self.device_assign.clear()
        self.task_finish_time.clear()
        self.earliest_start_time.clear()
        self.task_dp = {}
        for i in range(self.num_tasks):
            self.task_dp[i] = {
                "q": float(self.dp_defaults[0]),
                "c": float(self.dp_defaults[1]),
                "sigma": float(self.dp_defaults[2])
            }
        return self.get_state()

    def get_ready_tasks(self) -> List[int]:
        """获取就绪任务列表，使用 DAG 的排序方法（按层级和优先级）"""
        return self.dag.get_ready_sorted(self.completed)

    def get_state(self) -> np.ndarray:
        ready = self.get_ready_tasks()
        if not ready:
            return np.zeros(15, dtype=np.float32)
        return self.build_state(ready[0])

    def build_state(self, task_id: int) -> np.ndarray:
        try:
            node = self.dag.nodes[task_id]
            C, d, l = node.C, node.d, node.l
            C = max(float(C), EPS)
            d = max(float(d), EPS)
            l = max(float(l), EPS)

            log_C = np.log(C + 1.0)
            log_d = np.log(d + 1.0)

            progress = len(self.completed) / max(self.num_tasks, 1)
            progress = float(np.clip(progress, 0.0, 1.0))

            time_to_deadline = max(0.0, l - self.current_time)
            deadline_pressure = 1.0 - (time_to_deadline / l)
            deadline_pressure = float(np.clip(deadline_pressure, 0.0, 1.0))

            priority = float(getattr(node, 'priority', getattr(node, 'p', 1.0)))
            priority = float(np.clip(priority, 0.0, 10.0))

            layer = int(node.layer)
            max_layers = max(int(self.dag.max_layers), 1)
            normalized_layer = float(np.clip(float(layer) / float(max_layers), 0.0, 1.0))

            # sigma must be provided by task_dp (from actor), no hardcoded default
            sigma = float(self.task_dp[task_id].get("sigma", 0.0))
            sigma = float(np.clip(sigma, 0.01, 1000.0))  # safety bounds, but no hardcoded default
            q = float(self.task_dp[task_id].get("q", 0.01))
            q = float(np.clip(q, 0.0001, 1.0))

            # 信道容量
            R_vr = self._compute_channel_capacity_rsu(d)
            R_vb = self._compute_channel_capacity_cloud(d)

            log_R_vr = np.log(R_vr + 1.0)
            log_R_vb = np.log(R_vb + 1.0)
            log_f_l = np.log(self.f_l + 1.0)
            log_f_r = np.log(self.f_r + 1.0)
            log_f_b = np.log(self.f_b + 1.0)

            # 任务类型特征
            accuracy_req = float(getattr(node, 'accuracy_requirement', 0.5))
            privacy_sens = float(getattr(node, 'privacy_sensitivity', 0.5))

            state = np.array([
                log_C, log_d, deadline_pressure, priority, normalized_layer, progress, sigma, q,
                log_R_vr, log_R_vb, log_f_l, log_f_r, log_f_b,
                accuracy_req, privacy_sens
            ], dtype=np.float32)
            state = np.nan_to_num(state, nan=0.0, posinf=100.0, neginf=-100.0)
            state = np.clip(state, -100.0, 100.0)
            if np.isnan(state).any() or np.isinf(state).any():
                state = np.zeros(15, dtype=np.float32)
            return state
        except Exception:
            return np.zeros(15, dtype=np.float32)

    def _generate_attention_mask(self, ready_tasks: List[int]) -> torch.Tensor:
        L = len(ready_tasks)
        progress = len(self.completed) / max(self.num_tasks, 1)
        if progress < 0.3:
            return torch.zeros(L, L, dtype=torch.bool, device=self.device)
        elif progress < 0.7:
            attn_mask = torch.ones(L, L, dtype=torch.bool, device=self.device)
            for i, tid_i in enumerate(ready_tasks):
                layer_i = self.dag.nodes[tid_i].layer
                preds_i = set(self.dag.get_predecessors(tid_i))
                succs_i = set(self.dag.get_successors(tid_i))
                for j, tid_j in enumerate(ready_tasks):
                    layer_j = self.dag.nodes[tid_j].layer
                    if (tid_i == tid_j or abs(layer_i - layer_j) <= 1 or tid_j in preds_i or tid_j in succs_i):
                        attn_mask[i, j] = False
            return attn_mask
        else:
            attn_mask = torch.ones(L, L, dtype=torch.bool, device=self.device)
            for i, tid_i in enumerate(ready_tasks):
                preds_i = set(self.dag.get_predecessors(tid_i))
                succs_i = set(self.dag.get_successors(tid_i))
                for j, tid_j in enumerate(ready_tasks):
                    if (tid_i == tid_j or tid_j in preds_i or tid_j in succs_i):
                        attn_mask[i, j] = False
            return attn_mask

    def prepare_transformer_inputs(self, ready_tasks: List[int]) -> Optional[Tuple]:
        if not ready_tasks:
            return None
        try:
            L = len(ready_tasks)
            # 15维任务特征
            tf = torch.zeros(1, L, 15, dtype=torch.float32, device=self.device)
            # 6维隐私特征
            pf = torch.zeros(1, L, 6, dtype=torch.float32, device=self.device)
            df = torch.zeros(1, L, 4, dtype=torch.float32, device=self.device)

            for col, tid in enumerate(ready_tasks):
                node = self.dag.nodes[tid]
                C = max(float(node.C), EPS)
                d = max(float(node.d), EPS)
                l = max(float(node.l), EPS)

                log_C = np.log(C + 1.0)
                log_d = np.log(d + 1.0)

                progress = len(self.completed) / max(self.num_tasks, 1)
                progress = float(np.clip(progress, 0.0, 1.0))

                time_to_deadline = max(0.0, l - self.current_time)
                deadline_pressure = 1.0 - (time_to_deadline / l)
                deadline_pressure = float(np.clip(deadline_pressure, 0.0, 1.0))

                priority = float(getattr(node, 'priority', getattr(node, 'p', 1.0)))
                priority = float(np.clip(priority, 0.0, 10.0))

                layer = int(node.layer)
                max_layers = max(int(self.dag.max_layers), 1)
                normalized_layer = float(layer) / float(max_layers)
                normalized_layer = float(np.clip(normalized_layer, 0.0, 1.0))

                # 信道容量
                R_vr = self._compute_channel_capacity_rsu(d)
                R_vb = self._compute_channel_capacity_cloud(d)

                log_R_vr = np.log(R_vr + 1.0)
                log_R_vb = np.log(R_vb + 1.0)
                log_f_l = np.log(self.f_l + 1.0)
                log_f_r = np.log(self.f_r + 1.0)
                log_f_b = np.log(self.f_b + 1.0)

                accuracy_req = float(getattr(node, 'accuracy_requirement', 0.5))
                privacy_sens = float(getattr(node, 'privacy_sensitivity', 0.5))

                task_feat = [
                    log_C, log_d, deadline_pressure, priority,
                    normalized_layer, progress,
                    log_R_vr, log_R_vb, log_f_l, log_f_r, log_f_b,
                    0.0, 0.0,
                    accuracy_req,
                    privacy_sens
                ]
                task_feat = [float(np.nan_to_num(x, nan=0.0)) for x in task_feat]
                task_feat = [float(np.clip(x, -100.0, 100.0)) for x in task_feat]
                tf[0, col, :] = torch.tensor(task_feat, dtype=torch.float32)

                # 隐私特征 - sigma from task_dp (no hardcoded default)
                sigma = float(np.clip(self.task_dp[tid].get("sigma", 0.0), 0.01, 1000.0))
                pf[0, col, 0] = np.log(sigma + 1.0)            # log_sigma
                pf[0, col, 1] = privacy_sens                   # privacy_sens
                pf[0, col, 2] = accuracy_req                   # accuracy_req
                pf[0, col, 3] = privacy_sens * accuracy_req    # 交互项
                task_type = getattr(node, 'task_type', 'unknown')
                type_mapping = {
                    'computation': 0.2,
                    'communication': 0.4,
                    'data_processing': 0.6,
                    'sensing': 0.8,
                    'unknown': 0.5
                }
                pf[0, col, 4] = type_mapping.get(task_type, 0.5)
                pf[0, col, 5] = float(len(self.completed)) / float(max(self.num_tasks, 1))  # 预算压力估计

                preds = self.dag.get_predecessors(tid)
                succs = self.dag.get_successors(tid)
                df[0, col, 0] = float(len(preds))
                df[0, col, 1] = float(len(succs))
                df[0, col, 2] = normalized_layer
                df[0, col, 3] = priority

            adj = torch.zeros(1, L, L, dtype=torch.float32, device=self.device)
            for i, tid_i in enumerate(ready_tasks):
                for j, tid_j in enumerate(ready_tasks):
                    if tid_j in self.dag.get_successors(tid_i):
                        adj[0, i, j] = 1.0

            attn_mask = self._generate_attention_mask(ready_tasks)
            for i in range(L):
                if attn_mask[i, :].all():
                    attn_mask[i, i] = False

            total_positions = L * L
            masked_positions = attn_mask.sum().item()
            mask_ratio = masked_positions / total_positions
            if mask_ratio > 0.95:
                attn_mask = torch.zeros(L, L, dtype=torch.bool, device=self.device)

            total_budget = 1.0  # placeholder
            
            # 全面 NaN/Inf 清洗所有输入
            if torch.isnan(tf).any() or torch.isinf(tf).any():
                tf = torch.nan_to_num(tf, nan=0.0, posinf=100.0, neginf=-100.0)
                tf = torch.clamp(tf, -100.0, 100.0)
            if torch.isnan(pf).any() or torch.isinf(pf).any():
                pf = torch.nan_to_num(pf, nan=0.0, posinf=100.0, neginf=-100.0)
                pf = torch.clamp(pf, -100.0, 100.0)
            if torch.isnan(df).any() or torch.isinf(df).any():
                df = torch.nan_to_num(df, nan=0.0, posinf=100.0, neginf=-100.0)
                df = torch.clamp(df, -100.0, 100.0)
            if torch.isnan(adj).any() or torch.isinf(adj).any():
                adj = torch.nan_to_num(adj, nan=0.0, posinf=1.0, neginf=0.0)
                adj = torch.clamp(adj, 0.0, 1.0)
            
            return tf, pf, df, adj, attn_mask, total_budget
        except Exception:
            return None

    def build_aux_features(self, ready_tasks: List[int]) -> Dict:
        result = self.prepare_transformer_inputs(ready_tasks)
        if result is None:
            L = len(ready_tasks) if ready_tasks else 1
            pf_default = torch.zeros(1, L, 6, device=self.device)
            pf_default[:, :, 1] = 0.5  # 默认敏感度
            pf_default[:, :, 2] = 0.5  # 默认精度需求
            pf_default[:, :, 4] = 0.5  # 默认任务类型
            return {
                'task_features': torch.zeros(1, L, 15, device=self.device),
                'privacy_features': pf_default,
                'dag_features': torch.zeros(1, L, 4, device=self.device),
                'dag_adjacency': torch.zeros(1, L, L, device=self.device),
                'attention_mask': torch.zeros(L, L, dtype=torch.bool, device=self.device),
                'total_budget': 1.0
            }
        tf, pf, df, adj, attn_mask, total_budget = result
        return {
            'task_features': tf,
            'privacy_features': pf,
            'dag_features': df,
            'dag_adjacency': adj,
            'attention_mask': attn_mask,
            'total_budget': total_budget
        }

    def step(self, task_id: int, action: int, dp_params: Optional[Dict] = None):
        try:
            node = self.dag.nodes[task_id]
            C = max(float(node.C), EPS)
            d = max(float(node.d), EPS)
            l = max(float(node.l), EPS)

            # 应用外部（Transformer/Actor）给出的 dp_params（σ, q）
            if dp_params is not None:
                sigma_new = float(dp_params.get('sigma', self.task_dp[task_id]["sigma"]))
                q_new = float(dp_params.get('q', self.task_dp[task_id]["q"]))
                # NaN safety - but no hardcoded fallback, use existing value
                sigma_new = float(np.nan_to_num(sigma_new, nan=self.task_dp[task_id]["sigma"]))
                q_new = float(np.nan_to_num(q_new, nan=0.01))
                # 完全信任 Transformer 的 σ（其范围已在外部保证）
                self.task_dp[task_id]["sigma"] = sigma_new
                self.task_dp[task_id]["q"] = float(np.clip(q_new, 0.001, 0.1))

            # 在线本地模型：按 σ 注入本底噪声进行一次占位更新（与 DP-SGD 解耦）
            if self.enable_local_bg_noise_update:
                try:
                    for _ in range(max(1, self.local_updates_per_task)):
                        grads = self._compute_local_gradient(task_id)
                        sigma_eff = float(self.task_dp[task_id]["sigma"]) * self.local_noise_scale
                        self._apply_background_noise_update(sigma_eff, grads)
                except Exception:
                    pass

            # 选择动作与执行时间
            action = int(action) % 3
            offload_choices = ['Local', 'RSU', 'BS']
            offload_choice = offload_choices[action]

            if offload_choice == "Local":
                exec_t = self.time_local(C)
            elif offload_choice == "RSU":
                exec_t = self.time_rsu(d, C)
            else:
                exec_t = self.time_bs(d, C)

            exec_t = float(np.nan_to_num(exec_t, nan=1.0))
            exec_t = float(np.clip(exec_t, EPS, 1e6))

            # 计算最早开始时间（考虑前驱任务结果在不同设备间的数据传输）
            if task_id not in self.earliest_start_time:
                preds = self.dag.get_predecessors(task_id)
                if preds:
                    pred_ready_times = []
                    for pid in preds:
                        pred_finish_time = self.task_finish_time.get(pid, 0.0)
                        pred_device = self.device_assign.get(pid, 'Local')
                        data_transfer_time = self._compute_data_transfer_time(
                            pred_device, offload_choice, d
                        )
                        pred_ready_times.append(pred_finish_time + data_transfer_time)
                    max_pred_ready = max(pred_ready_times)
                    self.earliest_start_time[task_id] = max(self.current_time, max_pred_ready)
                else:
                    self.earliest_start_time[task_id] = self.current_time

            actual_start_time = self.earliest_start_time[task_id]
            finish_time = actual_start_time + exec_t
            self.current_time = finish_time
            self.completed.add(task_id)
            self.device_assign[task_id] = offload_choice
            self.task_finish_time[task_id] = finish_time

            # 计算端到端时延（包括等待时间）
            # 任务到达时间设为episode开始时刻（reset时current_time=0）
            # 端到端时延 = 完成时间 - 任务到达时间（假设所有任务在episode开始时到达）
            end_to_end_delay = finish_time - 0.0  # 从episode开始到任务完成的总时延
            
            # 等待时延 = 实际开始时间 - 任务到达时间
            waiting_delay = actual_start_time - 0.0

            reward, reward_breakdown = self._calculate_constrained_reward(task_id, action, exec_t, finish_time)
            reward = float(np.nan_to_num(reward, nan=-10.0))
            reward = float(np.clip(reward, -100.0, 100.0))
            done = (len(self.completed) == self.num_tasks)

            info = {
                **reward_breakdown,
                'offload_choice': offload_choice,
                'exec_time': float(exec_t),
                'finish_time': float(finish_time),
                'end_to_end_delay': float(end_to_end_delay),  # 新增：端到端时延
                'waiting_delay': float(waiting_delay),        # 新增：等待时延
                'dp_sigma': float(self.task_dp[task_id]["sigma"]),
                'dp_q': float(self.task_dp[task_id]["q"]),
                'bg_sigma': float(self.task_dp[task_id]["sigma"]) * float(self.local_noise_scale),
                'clip_norm': float(CLIP_NORM),
            }
            next_state = self.get_state()
            return next_state, reward, done, info
        except Exception as e:
            print(f"[ERROR] step异常: {e}, task_id={task_id}, action={action}")
            next_state = np.zeros(15, dtype=np.float32)
            reward = -10.0
            done = False
            info = {
                'reward_delay': -1.0,
                'penalty_deadline': -1.0,
                'bonus_deadline': 0.0,
                'total_reward': -10.0,
                'deadline_violated': True,
                'offload_choice': 'Local',
                'exec_time': 0.0,
                'finish_time': 0.0,
                'end_to_end_delay': 0.0,  # 新增
                'waiting_delay': 0.0,      # 新增
                'dp_sigma': 1.0,
                'dp_q': 0.01,
                'bg_sigma': 1.0,
                'clip_norm': float(CLIP_NORM),
            }
            return next_state, reward, done, info

    def _compute_local_gradient(self, task_id: int) -> Dict[str, torch.Tensor]:
        """
        以当前任务构造一个占位训练样本，计算本地模型的梯度。
        - 输入：build_state(task_id) 15 维
        - 目标：使用任务的 accuracy_requirement 作为回归目标（占位）
        """
        self.local_model.zero_grad(set_to_none=True)
        state_vec = torch.tensor(self.build_state(task_id), dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, 15]
        pred = self.local_model(state_vec).squeeze(0)  # [1] or [1,1] -> [1]

        node = self.dag.nodes[task_id]
        target_val = float(getattr(node, 'accuracy_requirement', 0.5))
        target = torch.tensor([target_val], dtype=torch.float32, device=self.device)

        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()

        grad_dict: Dict[str, torch.Tensor] = {}
        for name, p in self.local_model.named_parameters():
            if p.grad is not None:
                grad_dict[name] = p.grad.detach().clone()
        return grad_dict

    def _apply_background_noise_update(self, sigma: float, grad_dict: Dict[str, torch.Tensor]) -> None:
        """
        将梯度按全局范数裁剪到 CLIP_NORM，并加上 N(0, σ^2 I) 噪声；随后做一次 SGD 更新。
        注意：这里的 σ 来自 Transformer（每任务），与 DP-SGD 会计解耦。
        """
        if sigma is None or sigma <= 0.0 or not grad_dict:
            return

        with torch.no_grad():
            # 计算全局 L2 范数
            flats = [g.view(-1) for g in grad_dict.values()]
            if not flats:
                return
            all_flat = torch.cat(flats)
            norm = torch.norm(all_flat, p=2).item()

            scale = 1.0
            if norm > 1e-12 and norm > CLIP_NORM:
                scale = CLIP_NORM / norm

            # 逐参数：裁剪 + 加噪 + 更新
            for name, p in self.local_model.named_parameters():
                if name not in grad_dict or p is None:
                    continue
                g = grad_dict[name] * scale
                noise = torch.randn_like(g) * float(sigma)
                g_noisy = g + noise
                p.add_(-self.local_lr * g_noisy)

    def _calculate_constrained_reward(self, task_id: int, action: int, exec_t: float, finish_t: float) -> Tuple[float, Dict]:
        try:
            node = self.dag.nodes[task_id]
            C = max(float(node.C), EPS)
            d = max(float(node.d), EPS)
            l = max(float(node.l), EPS)

            baseline_t = min(self.time_local(C), self.time_rsu(d, C), self.time_bs(d, C))
            baseline_t = max(float(baseline_t), EPS)
            exec_t = max(float(exec_t), EPS)

            normalized_delay = exec_t / baseline_t
            normalized_delay = float(np.clip(normalized_delay, 0.0, 10.0))
            reward_delay = 1.0 - min(normalized_delay, 2.0)
            reward_delay = float(np.clip(reward_delay, -1.0, 1.0))

            earliest_start = float(self.earliest_start_time.get(task_id, 0.0))
            actual_duration = finish_t - earliest_start
            actual_duration = max(float(actual_duration), 0.0)

            if actual_duration <= l:
                penalty_deadline = 0.0
                time_margin = l - actual_duration
                bonus_deadline = 0.1 * (time_margin / l)
                bonus_deadline = float(np.clip(bonus_deadline, 0.0, 1.0))
            else:
                violation_ratio = (actual_duration - l) / l
                violation_ratio = float(np.clip(violation_ratio, 0.0, 10.0))
                penalty_deadline = -self.deadline_violation_penalty * min(violation_ratio, 1.0)
                bonus_deadline = 0.0

            # 隐私惩罚：normalized by dynamic sigma range (no hardcoded sigma_ref)
            sigma = float(self.task_dp[task_id].get("sigma", 0.0))
            sigma = max(sigma, 0.01)
            # Use a relative measure instead of hardcoded reference
            # Privacy leakage inversely proportional to sigma (lower sigma = higher leakage)
            privacy_leakage = 1.0 / (1.0 + sigma)  # normalized: higher sigma -> lower leakage
            privacy_leakage = float(np.clip(privacy_leakage, 0.0, 1.0))

            alpha = 0.3
            privacy_penalty = -alpha * privacy_leakage

            total_reward = reward_delay + penalty_deadline + bonus_deadline 

            reward_delay = float(np.nan_to_num(reward_delay, nan=0.0))
            penalty_deadline = float(np.nan_to_num(penalty_deadline, nan=0.0))
            bonus_deadline = float(np.nan_to_num(bonus_deadline, nan=0.0))
            privacy_penalty = float(np.nan_to_num(privacy_penalty, nan=0.0))
            total_reward = float(np.nan_to_num(total_reward, nan=-10.0))
            total_reward = float(np.clip(total_reward, -100.0, 100.0))

            breakdown = {
                'reward_delay': reward_delay,
                'penalty_deadline': penalty_deadline,
                'bonus_deadline': bonus_deadline,
                'privacy_penalty': privacy_penalty,
                'privacy_leakage': privacy_leakage,
                'sigma': sigma,
                'total_reward': total_reward,
                'deadline_violated': bool(actual_duration > l),
            }
            return total_reward, breakdown
        except Exception:
            breakdown = {
                'reward_delay': -1.0,
                'penalty_deadline': -1.0,
                'bonus_deadline': 0.0,
                'privacy_penalty': 0.0,
                'privacy_leakage': 1.0,
                'sigma': 1.0,
                'total_reward': -10.0,
                'deadline_violated': True,
            }
            return -10.0, breakdown

    def _compute_data_transfer_time(self, from_device: str, to_device: str, data_size: float) -> float:
        """
        计算 DAG 任务间数据传输时间
        """
        data_size = max(float(data_size), EPS)

        if from_device == to_device:
            return 0.0

        if (from_device == 'Local' and to_device == 'RSU') or \
           (from_device == 'RSU' and to_device == 'Local'):
            R_vr = self._compute_channel_capacity_rsu(data_size)
            return data_size / R_vr

        elif (from_device == 'Local' and to_device == 'BS') or \
             (from_device == 'BS' and to_device == 'Local'):
            R_vb = self._compute_channel_capacity_cloud(data_size)
            return data_size / R_vb

        elif (from_device == 'RSU' and to_device == 'BS') or \
             (from_device == 'BS' and to_device == 'RSU'):
            return data_size / self.backbone_bw

        return 0.0

    def _compute_channel_capacity_rsu(self, data_size: float) -> float:
        """
        R_{v,r} = B_r * θ_{v,r} * log_2(1 + (P_v * G_{v,r}) / (N_0 * B_r + I_{v,r}))
        """
        numerator = self.P_v * self.G_vr
        denominator = self.N_0 * self.B_r + self.I_vr
        snr = numerator / max(denominator, EPS)
        capacity = self.B_r * self.theta_vr * np.log2(1.0 + snr)
        return max(float(capacity), EPS)

    def _compute_channel_capacity_cloud(self, data_size: float) -> float:
        """
        R_{v,b} = B_b * θ_{v,b} * log_2(1 + (P_v * G_{v,b}) / (N_0 * B_b))
        """
        numerator = self.P_v * self.G_vb
        denominator = self.N_0 * self.B_b
        snr = numerator / max(denominator, EPS)
        capacity = self.B_b * self.theta_vb * np.log2(1.0 + snr)
        return max(float(capacity), EPS)

    def time_local(self, C: float) -> float:
        """本地执行时间：T_local = C / f_l"""
        C = max(float(C), EPS)
        return C / self.f_l

    def time_rsu(self, d: float, C: float) -> float:
        """
        RSU执行时间：T_rsu = T_trans + T_comp
        T_trans = d / R_{v,r}
        T_comp = C / (η_{v,r} * f_r)
        """
        d = max(float(d), EPS)
        C = max(float(C), EPS)
        R_vr = self._compute_channel_capacity_rsu(d)
        t_trans = d / R_vr
        t_comp = C / (self.eta_vr * self.f_r)
        return t_trans + t_comp

    def time_bs(self, d: float, C: float) -> float:
        """
        云执行时间：T_cloud = T_trans + T_comp
        T_trans = d / R_{v,b}
        T_comp = C / (η_{v,b} * f_b)
        """
        d = max(float(d), EPS)
        C = max(float(C), EPS)
        R_vb = self._compute_channel_capacity_cloud(d)
        t_trans = d / R_vb
        t_comp = C / (self.eta_vb * self.f_b)
        return t_trans + t_comp