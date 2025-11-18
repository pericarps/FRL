# train.py
import torch
import torch.nn.functional as F
import numpy as np
import random
import copy
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime
from dag import DAGTasks
from env import OffloadEnv
from agent import DoubleDQNAgent
from federated_learning import HierFL
from noise_config import (
    DELTA_SENS, EPSILON_TOTAL, DELTA_TOTAL,
    compute_level0_budgets, compute_sigma_range_for_episode,
    compute_dpsgd_noise_multiplier
)

_ALPHAS_DENSE = [1.0 + 0.1 * i for i in range(2, 100)]
_ALPHAS_WIDE = [12, 14, 16, 20, 32, 64, 128, 256, 512]
_ALPHAS_EXT = [float(a) for a in (_ALPHAS_DENSE + _ALPHAS_WIDE)]


class DPSigmaPredictor:
    """
    差分隐私Sigma预测器包装类（基于标准 DP-SGD）
    
    功能：
    1. 加载预训练的Sigma预测器
    2. 使用DP-SGD进行在线微调（batch_size=1）
    3. 保留规则基线作为正则化约束
    4. 分阶段调整微调强度
    
    """
    
    def __init__(self, 
                 pretrained_model,
                 epsilon: float,
                 delta: float,
                 total_steps: int,
                 learning_rate: float = 1e-6,
                 device: str = 'cpu',
                 params: Dict = None):
        """
        初始化DP Sigma预测器
        
        Args:
            pretrained_model: 预训练的Sigma预测器模型
            epsilon: DP-SGD隐私预算
            delta: DP-SGD delta参数
            total_steps: 总训练步数
            learning_rate: 学习率
            device: 设备
            params: 其他参数
        """
        self.device = device
        self.params = params or {}
        
        # 模型
        self.model = pretrained_model.to(device)
        
        # 保存预训练参数（用于L2正则化，保持预训练特性）
        self.pretrained_params = {}
        for name, param in self.model.named_parameters():
            self.pretrained_params[name] = param.data.clone().detach()
        
        # DP-SGD参数
        self.epsilon = epsilon
        self.delta = delta
        self.total_steps = total_steps
        self.current_step = 0
        self.max_grad_norm = 1.0  # 梯度裁剪范数（必须在_compute_noise_multiplier之前定义）
        
        # 计算噪声倍数（依赖max_grad_norm）
        self.noise_multiplier = self._compute_noise_multiplier()
        
        # 优化器（添加 weight_decay 作为隐式 L2 正则）
        # weight_decay 会自动对所有参数施加 L2 惩罚，更高效且不影响梯度计算
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5  # 轻量级 L2 正则，防止参数爆炸
        )
        
        # 训练阶段配置
        self.phase_config = {
            'phase1': {
                'name': 'Frozen',
                'beta_l2': 1.0,
                'lr_scale': 0.0,  # 完全冻结
            },
            'phase2': {
                'name': 'Conservative',
                'beta_l2': 0.8,  # 强L2约束
                'lr_scale': 0.3,
            },
            'phase3': {
                'name': 'Moderate',
                'beta_l2': 0.5,  # 中等约束
                'lr_scale': 0.7,
            },
            'phase4': {
                'name': 'Aggressive',
                'beta_l2': 0.2,  # 弱约束
                'lr_scale': 1.0,
            }
        }
        
        print(f"\n{'='*60}")
        print(f"DP Sigma预测器初始化")
        print(f"{'='*60}")
        print(f"隐私预算: ε={epsilon:.2f}, δ={delta:.2e}")
        print(f"噪声倍数: {self.noise_multiplier:.4f}")
        print(f"梯度裁剪: C={self.max_grad_norm}")
        print(f"学习率: {learning_rate:.2e}")
        print(f"总步数: {total_steps}")
        print(f"{'='*60}\n")
    
    def _compute_noise_multiplier(self) -> float:
        """
        计算满足(ε,δ)-DP的噪声倍数（理论保证）
        
        使用与 Critic 相同的 RDP 会计方法：
        - 通过 RDPAccountant 累积隐私损失
        - 二分搜索找到满足目标 ε 的最小噪声倍数
        - batch_size=1（在线学习）
        
        这确保了与 Critic 一致的理论保证。
        """
        # 使用与 Critic 相同的方法
        dp_result = compute_dpsgd_noise_multiplier(
            epsilon_target=self.epsilon,
            delta_target=self.delta,
            num_steps=self.total_steps,
            batch_size=1,  # Sigma预测器是在线学习（每个任务独立更新）
            dataset_size=self.total_steps  # 假设每步都是新样本
        )
        
        noise_multiplier = dp_result['noise_multiplier']
        
        print(f"\n[DP-SGD 理论参数 - Sigma预测器]")
        print(f"  总步数 T: {self.total_steps}")
        print(f"  隐私预算 ε: {self.epsilon:.2f}")
        print(f"  失败概率 δ: {self.delta:.2e}")
        print(f"  Batch Size: 1 (在线学习)")
        print(f"  采样率 q: {dp_result['sampling_rate']:.6f}")
        print(f"  梯度裁剪 C: {self.max_grad_norm}")
        print(f"  噪声倍数 σ/C: {noise_multiplier:.4f}")
        print(f"  噪声标准差 σ: {noise_multiplier * self.max_grad_norm:.4f}")
        print(f"  理论保证: ({self.epsilon:.2f}, {self.delta:.2e})-DP (RDP会计)")
        print(f"  方法: 与Critic相同的RDP会计\n")
        
        return noise_multiplier
    
    def get_current_phase(self, episode: int, total_episodes: int) -> Dict:
        """获取当前训练阶段配置"""
        progress = episode / max(total_episodes, 1)
        
        if progress < 0.001:  # 0-25%
            phase = 'phase1'
        elif progress < 0.50:  # 25-50%
            phase = 'phase2'
        elif progress < 0.75:  # 50-75%
            phase = 'phase3'
        else:  # 75-100%
            phase = 'phase4'
        
        return self.phase_config[phase]
    
    def predict(self, features: torch.Tensor, requires_grad: bool = False) -> torch.Tensor:
        """
        预测Sigma值
        
        Args:
            features: 输入特征 [B, D]
            requires_grad: 是否需要梯度（微调时为True）
        
        Returns:
            raw_logits: 原始输出 [B, L]
        """
        if requires_grad:
            self.model.train()
            return self.model(features)
        else:
            self.model.eval()
            with torch.no_grad():
                return self.model(features)
    
    def compute_rule_based_sigma(self, 
                                 privacy_sensitivity: float,
                                 accuracy_requirement: float,
                                 sigma_min: float,
                                 sigma_max: float,
                                 task_node = None) -> float:
        """
        基于规则的Sigma计算（用作正则化基线）
        
        📌 与数据生成器的 _compute_comprehensive_target_sigma 完全一致
        这是预训练时使用的规则，保持作为约束
        
        核心设计：
        - Sigma 与隐私敏感度强正相关（主导因素，60%）
        - Sigma 与精度需求强负相关（主导因素，40%）
        - 其他因素作为微调（任务类型、DAG结构等，合计~15%）
        
        Args:
            privacy_sensitivity: 隐私敏感度 [0, 1]
            accuracy_requirement: 精度需求 [0, 1]
            sigma_min: sigma下界
            sigma_max: sigma上界
            task_node: 可选的任务节点对象（包含更多元数据）
        """
        # 超参数（与数据生成器一致）
        lambda_privacy = 5.0
        lambda_utility = 5.0
        temperature = 10.0
        target_scale_temp = 8.0
        epsilon = 1e-9
        
        # 归一化输入
        privacy_sensitivity = float(np.clip(privacy_sensitivity, 0.0, 1.0))
        accuracy_requirement = float(np.clip(accuracy_requirement, 0.0, 1.0))
        
        # ========== 核心权重计算 ==========
        # 5.1 隐私权重（核心 - 占60%影响力）
        privacy_weight = 1.0 + 4.0 * privacy_sensitivity  # [1.0, 5.0]
        
        # 5.2 精度权重（核心 - 占40%影响力）
        utility_weight = 1.0 + 3.0 * accuracy_requirement  # [1.0, 4.0]
        
        # ========== 多因素微调（如果提供了task_node）==========
        if task_node is not None:
            # 5.3 任务类型调整（微调 - 10%影响）
            task_type = getattr(task_node, 'task_type', 'unknown')
            task_type_map = {
                'computation': 0.2,
                'communication': 0.4,
                'data_processing': 0.6,
                'sensing': 0.8,
                'unknown': 0.5
            }
            task_type_encoded = task_type_map.get(task_type, 0.5)
            type_privacy_bonus = task_type_encoded * 0.1  # [0, 0.08]
            privacy_weight = privacy_weight + type_privacy_bonus
            
            # 5.4 截止时间压力调整（微调 - 减弱影响）
            # 从任务特征中提取（如果环境有提供）
            deadline_pressure = getattr(task_node, 'deadline_pressure', 0.5)
            deadline_pressure = float(np.clip(deadline_pressure, 0.0, 1.0))
            deadline_factor = np.exp(-1.0 * deadline_pressure)
            privacy_weight = privacy_weight * (0.9 + 0.1 * deadline_factor)
            
            # 5.5 优先级调整（微调）
            priority = getattr(task_node, 'priority', 5.0)
            priority_normalized = float(np.clip(priority / 10.0, 0.0, 1.0))
            utility_weight = utility_weight * (1.0 + 0.1 * priority_normalized)
            
            # 5.6 DAG结构调整（微调）
            num_preds = len(getattr(task_node, 'pre', []))
            num_succs = len(getattr(task_node, 'suc', []))
            dag_complexity = (num_preds + num_succs) / 10.0
            dag_complexity = float(np.clip(dag_complexity, 0.0, 1.0))
            utility_weight = utility_weight * (1.0 + 0.05 * dag_complexity)
            
            # 5.7 层级/进度调整（如果有进度信息）
            # 注意：训练时可能没有全局进度信息，使用层级作为代理
            layer = getattr(task_node, 'layer', 0)
            max_layers = 5  # 假设最大层数
            progress_proxy = layer / max(max_layers, 1)
            progress_proxy = float(np.clip(progress_proxy, 0.0, 1.0))
            progress_factor = 1.0 - 0.1 * (1.0 - progress_proxy)
            utility_weight = utility_weight * progress_factor
        
        # ========== 计算最优Sigma ==========
        # 综合权重
        A = lambda_privacy * privacy_weight
        B = lambda_utility * utility_weight
        
        # 计算比率
        ratio = A / (B + epsilon)
        ratio = float(np.clip(ratio, epsilon, 1e6))
        
        # 对数变换
        s_optimal_raw = temperature * np.log(ratio)
        
        # 映射到 sigma 范围
        s_mid = (sigma_max + sigma_min) / 2.0
        s_range = (sigma_max - sigma_min) / 2.0
        target_sigma = s_mid + s_range * np.tanh(s_optimal_raw / target_scale_temp)
        
        return float(np.clip(target_sigma, sigma_min, sigma_max))
    
    def dp_update_step(self,
                      features: torch.Tensor,
                      predicted_sigma: torch.Tensor,
                      reward: float,
                      phase_config: Dict,
                      rule_sigma: float = None) -> Dict:
        """
        差分私有的单步更新（标准 DP-SGD，batch_size=1）
        
        Args:
            features: 输入特征 [1, D]（单样本）
            predicted_sigma: 预测的sigma值（标量tensor）
            reward: 实际奖励（标量）
            phase_config: 当前阶段配置
            rule_sigma: 规则基线sigma（用于正则化）
        
        Returns:
            损失信息字典
        
        DP-SGD 步骤（理论保证）：
        1. 计算单样本梯度: ∇L(θ; x)
        2. 裁剪梯度: g' = g / max(1, ||g||₂/C)
        3. 添加高斯噪声: g̃ = g' + N(0, σ²C²I)
        4. 更新参数: θ ← θ - η·g̃
        
        注意：batch_size=1 时，步骤2的全局裁剪等价于per-sample裁剪
        """
 
        
        beta_l2 = phase_config['beta_l2']  # L2参数正则化权重（随阶段动态调整）
        lr_scale = phase_config['lr_scale']
        
        # 如果是冻结阶段，直接返回
        if lr_scale == 0.0:
            return {
                'total_loss': 0.0,
                'rl_loss': 0.0,
                'l2_loss': 0.0,
                'rule_loss': 0.0,
                'phase': phase_config['name']
            }
        
        self.model.train()
        
        # 1. RL损失（负奖励）
        rl_loss = -torch.tensor(reward, dtype=torch.float32, device=self.device)
        
        # 2. L2参数正则化（相对预训练参数，带时间衰减）
        # 📌 使用时间衰减：早期强约束（保持预训练知识），后期弱约束（允许适应）
        # 📌 防止在线学习中 L2 损失无限增长
        time_decay = max(0.0, 1.0 - self.current_step / self.total_steps)  # 1.0 → 0.0
        l2_loss = torch.tensor(0.0, device=self.device)
        if time_decay > 0.01:  # 只在前99%的训练中使用 L2 正则
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.pretrained_params:
                    l2_loss += torch.sum((param - self.pretrained_params[name]) ** 2)
        
        # 3. 规则正则化损失（保持个性化策略）
        rule_loss = torch.tensor(0.0, device=self.device)
        if rule_sigma is not None:
            rule_target = torch.tensor(rule_sigma, dtype=torch.float32, device=self.device)
            # 软约束：不要求完全等于规则，但不能偏离太远
            rule_loss = F.smooth_l1_loss(predicted_sigma.squeeze(), rule_target)
        
        # 4. 组合损失（动态权重平衡）
        # 
        # 权重设计原则：
        # 1. RL损失提供学习信号（主导）
        # 2. L2损失防止灾难性遗忘（中等）
        # 3. Rule损失提供软约束（辅助）
        #
        # 基于观察到的损失数量级：
        # - RL loss: ~2-8（奖励的负值）
        # - L2 loss: ~3e5-3e6（参数平方和）
        # - Rule loss: ~0.8-1.0（smooth_l1_loss）
        #
        # 目标：让三者的加权贡献在同一数量级
        
        # 动态权重（随训练阶段调整）
        # Early: 强约束（保持预训练特性）
        # Late: 弱约束（允许适应新任务）
        # 
        # 权重修正：基于实际观察到的数值
        # - RL loss (原始): ~2-8 (负奖励)
        # - L2 loss (原始): ~0.001-0.002 (参数平方和)
        # - Rule loss (原始): ~0.8-1.0 (smooth_l1)
        #
        # 目标贡献比例: RL主导(70%) > L2防遗忘(20%) > Rule引导(10%)
        # 
        # 时间衰减策略：
        # - L2 权重随时间线性衰减到 0（避免在线学习中无限增长）
        # - Rule 权重保持稳定（持续提供软约束）
        weight_rl = 1.0                                          # RL信号（主导）: 贡献 2-8
        weight_l2 = beta_l2 * 1000.0 * time_decay               # L2正则（时间衰减）: 贡献 0.8→0
        weight_rule = 0.3 * beta_l2                              # 规则约束（软引导）: 贡献 0.24-0.3
        
        total_loss = (
            weight_rl * rl_loss                   # 主导: ~2-8 (70-90%)
            + weight_l2 * l2_loss                 # 辅助: ~0.8→0 (10-20%→0)
            + weight_rule * rule_loss             # 引导: ~0.24-0.3 (5-10%)
        )
        
        # 5. 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 6. 梯度裁剪（DP-SGD关键步骤）
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.max_grad_norm
        )
        
        # 7. 添加高斯噪声（DP-SGD关键步骤）
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    mean=0.0,
                    std=self.noise_multiplier * self.max_grad_norm,
                    size=param.grad.shape,
                    device=param.grad.device,
                    dtype=param.grad.dtype
                )
                param.grad.add_(noise)
        
        # 8. 更新参数（缩放学习率）
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_scale
        
        self.optimizer.step()
        
        # 恢复原始学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / lr_scale
        
        # 更新步数
        self.current_step += 1
        
        return {
            'total_loss': total_loss.item(),
            'rl_loss': rl_loss.item(),
            'l2_loss': l2_loss.item(),
            'rule_loss': rule_loss.item(),
            'grad_norm': total_norm.item(),
            'phase': phase_config['name'],
            'beta_l2': beta_l2,
            'lr_scale': lr_scale,
            # 添加权重信息用于监控
            'weight_rl': weight_rl,
            'weight_l2': weight_l2,
            'weight_rule': weight_rule,
            'time_decay': time_decay  # 监控时间衰减
        }
    
    def save(self, path: str):
        """保存微调后的模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_step': self.current_step,
            'epsilon': self.epsilon,
            'delta': self.delta,
        }, path)
        print(f"DP Sigma预测器已保存到: {path}")
    
    def load(self, path: str):
        """加载微调后的模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_step = checkpoint['current_step']
        print(f"DP Sigma预测器已从 {path} 加载")


def _epsilon_single_gaussian_scan(sigma: float, sensitivity: float, delta: float) -> float:
    import math
    if sigma <= 0 or sensitivity <= 0:
        return float('inf')
    best = float('inf')
    s2 = sensitivity * sensitivity
    sig2 = sigma * sigma
    for a in _ALPHAS_EXT:
        if a <= 1.0:
            continue
        rho = a * s2 / (2.0 * sig2)
        eps = rho + math.log(1.0 / delta) / (a - 1.0)
        if eps < best:
            best = eps
    return float(best)


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def create_dag(num_tasks: int = 5, max_layers: int = 3, seed: int = None) -> DAGTasks:
    task_config = {
        "cycles_range": (1e7, 1e8),
        "data_range": (1e5, 1e6),
        "deadline_base_range": (0.1, 0.5),
        "deadline_layer_offset": 0.015,
        "alpha_weights": (0.3, 0.3, 0.4),
    }
    dag = DAGTasks(num_tasks=num_tasks, max_layers=max_layers, seed=seed, task_config=task_config)
    return dag

def select_task_with_sigma(env: OffloadEnv, dp_sigma_predictor, params: Dict,
                           sigma_min: float, sigma_max: float,
                           enable_finetune: bool = False) -> Tuple[Optional[int], Optional[float], List[int], Optional[torch.Tensor], Optional[Dict]]:
    """
    使用DP Sigma预测器或静态均分策略选择任务并生成sigma。
    
    Args:
        env: 环境
        dp_sigma_predictor: DPSigmaPredictor实例（可为None）
        params: 参数字典
        sigma_min: sigma最小值
        sigma_max: sigma最大值
        enable_finetune: 是否启用微调（需要返回额外信息）
    
    Returns:
        task_id: 选择的任务ID
        sigma: 该任务的噪声参数
        ready_tasks: 当前就绪的任务列表
        features: 输入特征（仅enable_finetune=True时返回）
        task_metadata: 任务元数据（仅enable_finetune=True时返回）
    """
    try:
        ready_tasks = env.get_ready_tasks()
        if not ready_tasks:
            return None, None, [], None, None
        
        task_id = ready_tasks[0]  # 选择第一个就绪任务
        node = env.dag.nodes.get(task_id)
        
        # 提取任务元数据
        privacy_sens = float(getattr(node, 'privacy_sensitivity', 0.5))
        accuracy_req = float(getattr(node, 'accuracy_requirement', 0.5))
        
        # 计算规则基线sigma（用于正则化）
        rule_sigma = None
        if dp_sigma_predictor is not None:
            rule_sigma = dp_sigma_predictor.compute_rule_based_sigma(
                privacy_sens, accuracy_req, sigma_min, sigma_max,
                task_node=node  # 传入完整的任务节点以计算多因素权重
            )
        
        # 尝试使用Sigma预测器
        if dp_sigma_predictor is not None and params.get("use_gace", False):
            try:
                inputs = env.prepare_transformer_inputs(ready_tasks)
                if inputs is not None:
                    tf, pf, df, adj, attn_mask, total_budget = inputs
                    
                    from transformer_alignment import ensure_self_loops_in_mask
                    device = dp_sigma_predictor.device
                    
                    tf = tf.to(device)
                    pf = pf.to(device)
                    df = df.to(device)
                    adj = adj.to(device)
                    attn_mask = ensure_self_loops_in_mask(attn_mask.to(device))
                    
                    # 构造输入特征
                    B, L = tf.shape[0], tf.shape[1]
                    base_features = torch.cat([tf, pf, df], dim=-1)  # [B, L, 25]
                    base_flat = base_features.reshape(B, -1)  # [B, L*25]
                    adj_flat = adj.reshape(B, -1)  # [B, L*L]
                    mask_flat = attn_mask.to(torch.float32).reshape(B, -1)  # [B, L*L]
                    
                    # 添加元特征
                    meta_L = torch.full((B, 1), float(L), device=device, dtype=torch.float32)
                    meta_smin = torch.full((B, 1), sigma_min, device=device, dtype=torch.float32)
                    meta_smax = torch.full((B, 1), sigma_max, device=device, dtype=torch.float32)
                    meta_ps = torch.full((B, 1), privacy_sens, device=device, dtype=torch.float32)
                    meta_ar = torch.full((B, 1), accuracy_req, device=device, dtype=torch.float32)
                    
                    meta = torch.cat([meta_L, meta_smin, meta_smax, meta_ps, meta_ar], dim=1)
                    concatenated = torch.cat([base_flat, adj_flat, mask_flat, meta], dim=1)
                    
                    # 预测（根据是否微调决定是否需要梯度）
                    raw_logits = dp_sigma_predictor.predict(
                        concatenated, 
                        requires_grad=enable_finetune
                    )
                    
                    # 找到task_id在ready_tasks中的索引
                    try:
                        task_idx = ready_tasks.index(task_id)
                    except ValueError:
                        task_idx = 0
                    
                    # Tanh映射到sigma范围
                    s_mid = (sigma_max + sigma_min) / 2.0
                    s_range = (sigma_max - sigma_min) / 2.0
                    sigma_pred_tensor = s_mid + s_range * torch.tanh(raw_logits[0, task_idx])
                    sigma_pred_clamped = torch.clamp(sigma_pred_tensor, min=sigma_min, max=sigma_max)
                    
                    # 转为float
                    sigma = float(sigma_pred_clamped.item()) if not enable_finetune else sigma_pred_clamped
                    
                    # 如果启用微调，返回额外信息
                    if enable_finetune:
                        task_metadata = {
                            'privacy_sensitivity': privacy_sens,
                            'accuracy_requirement': accuracy_req,
                            'rule_sigma': rule_sigma,
                            'task_id': task_id,
                            'task_idx': task_idx
                        }
                        return task_id, sigma, ready_tasks, concatenated, task_metadata
                    else:
                        return task_id, sigma, ready_tasks, None, None
                        
            except Exception as e:
                print(f"[WARNING] Sigma预测器推断失败: {e}，使用规则基线")
                import traceback
                traceback.print_exc()
                
                # 回退到规则基线
                if rule_sigma is not None:
                    return task_id, rule_sigma, ready_tasks, None, None
        
        # 静态均分策略（默认）
        sigma = (sigma_min + sigma_max) / 2.0
        
        # 如果没有预测器但有规则，优先使用规则
        if rule_sigma is not None and dp_sigma_predictor is not None:
            sigma = rule_sigma
        
        return task_id, sigma, ready_tasks, None, None
        
    except Exception as e:
        print(f"[ERROR] select_task_with_sigma failed: {e}")
        ready_tasks = env.get_ready_tasks()
        if not ready_tasks:
            return None, None, [], None, None
        return ready_tasks[0], (sigma_min + sigma_max) / 2.0, ready_tasks, None, None


def prefill_replay_buffer(agents: List[DoubleDQNAgent], envs: List[OffloadEnv], params: Dict,
                          dp_sigma_predictor, min_samples: int, sigma_min: float, sigma_max: float):
    
    total_samples = 0
    for env in envs:
        env.reset()
    from tqdm import trange as _trange
    
    print(f"开始预填充 {min_samples} 条经验（仅Critic）...")
    
    with _trange(min_samples, desc="Prefill Buffer") as pbar:
        while total_samples < min_samples:
            for agent, env in zip(agents, envs):
                done = False
                step_count = 0
                max_steps = 200 # 限制单次 episode 的最大步数
                
                while not done and step_count < max_steps and total_samples < min_samples:
                    # 使用新的sigma生成函数（预填充时不微调）
                    task_id, sigma, ready_tasks, _, _ = select_task_with_sigma(
                        env, dp_sigma_predictor, params, sigma_min, sigma_max,
                        enable_finetune=False  # 预填充时不微调
                    )
                    
                    if task_id is None:
                        env.reset()
                        break # DAG 完成或出错，重置环境
                        
                    state = env.build_state(task_id)
                    
                    # 如果使用transformer，准备aux数据
                    aux = None
                    if params.get("use_gace", False):
                        inputs = env.prepare_transformer_inputs(ready_tasks)
                        if inputs is not None:
                            tf, pf, df, adj, attn_mask, total_budget = inputs
                            aux = {
                                'task_features': tf, 'privacy_features': pf, 'dag_features': df,
                                'dag_adjacency': adj, 'attention_mask': attn_mask,
                                'total_budget': total_budget
                            }


                    # 预填充时使用随机动作
                    action = random.randint(0, 2)
                    dp_params = {"sigma": sigma, "q": 0.01}
                    next_state, reward, done, info = env.step(task_id, action, dp_params)

                    # 存储 Critic (DQN) 的经验
                    agent.remember(state, action, reward, next_state, done, aux, task_id, sigma)
                    
                    total_samples += 1
                    step_count += 1
                    pbar.update(1)
                    
                    if total_samples >= min_samples: 
                        break
                        
                if done:
                    env.reset()
                    
                if total_samples >= min_samples: 
                    break
    
    print(f"Prefill done. Total Critic samples={total_samples}")

def main(override_params=None, exp_name=None):
    SEED = 42
    if override_params and 'seed' in override_params: SEED = int(override_params['seed'])
    set_seed(SEED)
    print(f"Seed set: {SEED}")

    params = {
        # ========== 环境配置 ==========
        "num_vehicles": 10,
        "num_tasks": 10,
        "min_tasks_per_episode": 5,
        "max_tasks_per_episode": 50,
        "max_layers": 3,

        # ========== 物理参数 ==========
        "B_r": 1e6, "B_b": 2e6, "theta_vr": 0.5, "theta_vb": 0.5,
        "P_v": 0.5, "G_vr": 1e-5, "G_vb": 1e-6, "N_0": 1e-20, "I_vr": 1e-9, "backbone_bw": 1e9,
        "f_l": 1.5e9, "f_r": 3e10, "f_b": 2e11, "eta_vr": 0.2, "eta_vb": 0.1,

        # ========== 任务默认参数 ==========
        "q_default": 0.01, "c_default": 1.0,
        "deadline_violation_penalty": 10.0,

        # ========== 预训练Sigma预测器配置 ==========
        "use_gace": True,  # 是否使用预训练的sigma预测器（基于图感知上下文编码）
        "pretrained_sigma_predictor_path": "transformer_sigma_allocator_for_rl.pth",  # 预训练模型路径，None则使用静态均分策略

        # ========== DDQN配置 ==========
        "hidden_size": 256,
        "lr": 1e-5,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.99,
        "buffer_size": 50000,
        "batch_size": 64,

        # ========== 训练配置 ==========
        "episodes": 800,
        "train_frequency": 1,  # 每N步训练一次critic
        "sync_target_every_steps": 100,
        "prefill_steps": 7000,
        "max_grad_norm": 1.0,

        # ========== 联邦学习配置 ==========
        "num_rsus": 5,
        "fl_aggregate_every_episodes": 10,

        # ========== 损失平滑配置 ==========
        "loss_smoothing_window": 10,
        "use_ema_loss": True,
        "ema_beta": 0.9,

        # ========== 差分隐私配置 ==========
        "use_opacus": True,  # 使用Opacus进行DP-SGD
        "enable_budget_scaling": False,

        # ========== 系统配置 ==========
        "seed": SEED,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    if override_params:
        params.update({k: v for k, v in override_params.items()
                       if k not in ['exp_name', 'exp_description', 'output_dir', 'model_dir']})
        print(f"Applied override params")


    N_TOTAL_THEORETICAL = int(params["episodes"] * params["max_tasks_per_episode"])
    if N_TOTAL_THEORETICAL <= 0:
        raise ValueError("N_TOTAL_THEORETICAL must be positive. Check 'episodes' and 'max_tasks_per_episode'.")
    print(f"Privacy Theory: Theoretical Dataset Size (N_total) = {N_TOTAL_THEORETICAL}")
    # ---------------------------------------------------------------------

    # 生成实验名称
    exp_name = f"{params['episodes']}iters_{EPSILON_TOTAL}eps_{params['num_vehicles']}vehicles_{params['max_tasks_per_episode']}tasks_fullmodel"
    exp_description = override_params.get('exp_description', '') if override_params else ''
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 统一目录结构：runs/experiments 和 model/experiments
    log_dir = os.path.join('runs/experiments', exp_name)
    checkpoint_dir = os.path.join('model/experiments', exp_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 写入配置文件
    with open(os.path.join(log_dir, 'config.txt'), 'w', encoding='utf-8') as f:
        f.write(f"训练配置 - {timestamp}\n")
        f.write(f"实验名称: {exp_name}\n")
        if exp_description:
            f.write(f"实验描述: {exp_description}\n")
        f.write("=" * 60 + "\n")
        for k, v in params.items():
            f.write(f"{k}: {v}\n")
        f.write(f"dp_theoretical_dataset_size_N_total: {N_TOTAL_THEORETICAL}\n")

    print("=" * 60)
    if exp_name: print(f"实验名称: {exp_name}")
    if exp_description: print(f"实验描述: {exp_description}")
    print(f"日志目录: {log_dir}\n模型目录: {checkpoint_dir}")
    print(f"设备: {params['device']}")
    print(f"图感知上下文编码器(GACE): {'启用' if params['use_gace'] else '禁用'}")
    print(f"单车总预算: ε_total={EPSILON_TOTAL}, δ_total={DELTA_TOTAL}")
    print("=" * 60)

    dags = [create_dag(params["num_tasks"], params["max_layers"], seed=SEED+i) for i in range(params["num_vehicles"])]
    envs = [OffloadEnv(dag, params, device=params["device"]) for dag in dags]

    # 计算预算分配
    budgets = compute_level0_budgets(EPSILON_TOTAL, DELTA_TOTAL)
    eps_local = budgets["epsilon_local"]
    del_local = budgets["delta_local"]
    eps_critic = budgets["epsilon_critic"]
    del_critic = budgets["delta_critic"]
    eps_sigma_predictor = budgets["epsilon_sigma_predictor"]
    del_sigma_predictor = budgets["delta_sigma_predictor"]

    total_env_steps = params["episodes"] * params["max_tasks_per_episode"]* params["num_vehicles"]
    critic_real_steps = total_env_steps // params["train_frequency"]

    # Critic DP-SGD噪声计算
    critic_dp = compute_dpsgd_noise_multiplier(
        epsilon_target=eps_critic, delta_target=del_critic, num_steps=critic_real_steps,
        batch_size=params["batch_size"],
        dataset_size=N_TOTAL_THEORETICAL
    )
    print(f"[DP-SGD] Critic: ε={eps_critic:.2f}, δ={del_critic:.2e}, z={critic_dp['noise_multiplier']:.4f}, q={critic_dp['sampling_rate']:.6f}")

    # 加载并初始化DP Sigma预测器
    dp_sigma_predictor = None
    pretrained_model_path = params.get("pretrained_sigma_predictor_path", None)
    
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"\n{'='*60}")
        print(f"加载预训练Sigma预测器: {pretrained_model_path}")
        print(f"{'='*60}")
        try:
            # 加载预训练模型（使用weights_only=False因为需要加载自定义模型对象）
            # 注意：仅在信任模型来源时使用
            pretrained_model = torch.load(
                pretrained_model_path, 
                map_location=params['device'],
                weights_only=False  # 明确指定，消除FutureWarning
            )
            
            # 计算sigma预测器的总更新步数
            # 假设每个episode平均执行一定数量的任务，每个任务都可能更新sigma预测器
            avg_tasks_per_episode = (params["min_tasks_per_episode"] + params["max_tasks_per_episode"]) / 2.0
            sigma_predictor_steps = int(params["episodes"] * avg_tasks_per_episode * params["num_vehicles"])
            
            # 初始化DP Sigma预测器
            dp_sigma_predictor = DPSigmaPredictor(
                pretrained_model=pretrained_model,
                epsilon=eps_sigma_predictor,
                delta=del_sigma_predictor,
                total_steps=sigma_predictor_steps,
                learning_rate=params.get("sigma_predictor_lr", 1e-6),
                device=params['device'],
                params=params
            )
            print(f"✓ DP Sigma预测器初始化成功")
            print(f"  预算: ε={eps_sigma_predictor:.2f}, δ={del_sigma_predictor:.2e}")
            print(f"  预估更新步数: {sigma_predictor_steps}")
            
        except Exception as e:
            print(f"✗ Sigma预测器加载失败: {e}")
            print(f"  将使用静态均分策略")
            import traceback
            traceback.print_exc()
            dp_sigma_predictor = None
    else:
        print("\n⚠ 未指定预训练Sigma预测器，使用静态均分策略")

    agents = []
    for i in range(params["num_vehicles"]):
        
        # [!! DP-FIX 3 !!]
        # ---------------------------------------------------------------------
        # 准备要传递给 Agent 的参数（不再包含actor）
        agent_params = {
            **params, # 复制所有基础参数
            # 传入计算好的critic噪声
            "critic_noise_multiplier": critic_dp["noise_multiplier"],
            # 传入理论上的数据集大小 N_total
            "dp_dataset_size": N_TOTAL_THEORETICAL
        }
        # 准备要传递给 Agent 的预算 (用于日志记录或 delta，仅critic)
        agent_budget = {
            "epsilon_critic": eps_critic,
            "delta_critic": del_critic,
        }

        agent = DoubleDQNAgent(
            state_size=15, 
            action_size=3, 
            params=agent_params, # <-- 传入修正后的参数
            device=params["device"], 
            per_vehicle_budget=agent_budget, # <-- 传入预算
            vehicle_id=i
        )
        # ---------------------------------------------------------------------
        agents.append(agent)

    fl = HierFL(num_vehicles=params["num_vehicles"], device=params["device"],
                num_rsus=params.get("num_rsus", 2))
    fl.set_global_model(agents[0])

    writer = SummaryWriter(log_dir=log_dir)

    for agent, env in zip(agents, envs):
        task_type_map = {task_idx: getattr(env.dag.nodes[task_idx], 'task_type', 'unknown') for task_idx in range(env.num_tasks)}
        agent.update_task_type_info(task_type_map)

    sigma_min_init, sigma_max_init = compute_sigma_range_for_episode(
        num_episodes=params["episodes"],
        tasks_in_episode=params["num_tasks"],
        epsilon_local=eps_local,
        delta_local=del_local,
        sensitivity=DELTA_SENS,
        max_tasks_per_episode=params["max_tasks_per_episode"] # 确保使用 M_max
    )
    print(f"初始噪声范围 (基于 M_max={params['max_tasks_per_episode']}): σ ∈ [{sigma_min_init:.4f}, {sigma_max_init:.4f}]")

    prefill_replay_buffer(agents=agents, envs=envs, params=params,
                          dp_sigma_predictor=dp_sigma_predictor,
                          min_samples=params["prefill_steps"],
                          sigma_min=sigma_min_init, sigma_max=sigma_max_init)

    global_step = 0
    critic_actual_steps = 0

    for ep in trange(params["episodes"], desc="Training"):
        for i, env in enumerate(envs):
            new_seed = SEED + ep * 1000 + i
            num_new_tasks = random.randint(params["min_tasks_per_episode"], params["max_tasks_per_episode"])
            new_dag = create_dag(num_tasks=num_new_tasks, max_layers=params["max_layers"], seed=new_seed)
            env.dag = new_dag; env.num_tasks = new_dag.num_tasks; env.reset()


        tasks_in_episode_M_max = params["max_tasks_per_episode"]
        sigma_min_ep, sigma_max_ep = compute_sigma_range_for_episode(
            num_episodes=params["episodes"],
            tasks_in_episode=tasks_in_episode_M_max, # <-- 修复：使用 M_max
            epsilon_local=eps_local, delta_local=del_local,
            sensitivity=DELTA_SENS,
            max_tasks_per_episode=tasks_in_episode_M_max # 确保 M_upper 也是 M_max
        )
        # ---------------------------------------------------------------------
        
        writer.add_scalar("Episode/Sigma_Min_Target", sigma_min_ep, ep)
        writer.add_scalar("Episode/Sigma_Max_Target", sigma_max_ep, ep)

        episode_rewards = [0.0] * params["num_vehicles"]
        episode_losses = [[] for _ in range(params["num_vehicles"])]
        episode_sigma_losses = []  # Sigma预测器微调损失
        
        # 获取当前训练阶段
        if dp_sigma_predictor is not None:
            phase_config = dp_sigma_predictor.get_current_phase(ep, params["episodes"])
            enable_sigma_finetune = (phase_config['lr_scale'] > 0.0)
        else:
            phase_config = None
            enable_sigma_finetune = False
        
        # 每车任务完成率和时延统计
        episode_task_stats = {
            "total_tasks": [0] * params["num_vehicles"],
            "task_rewards": [[] for _ in range(params["num_vehicles"])],      # 新增：每任务奖励
            "end_to_end_delays": [[] for _ in range(params["num_vehicles"])], # 端到端时延（含等待）
            "exec_times": [[] for _ in range(params["num_vehicles"])],        # 执行时延
            "waiting_delays": [[] for _ in range(params["num_vehicles"])],    # 等待时延
            "deadline_violations": [0] * params["num_vehicles"],
            "sigmas": [[] for _ in range(params["num_vehicles"])],            # 新增：记录实际使用的sigma
            "rule_sigmas": [[] for _ in range(params["num_vehicles"])],       # 新增：记录规则基线sigma
            "sigma_deviations": [[] for _ in range(params["num_vehicles"])]   # 新增：预测与规则的偏差
        }

        for vehicle_id, (agent, env) in enumerate(zip(agents, envs)):
            done = False; step_count = 0; max_steps = 200
            while not done and step_count < max_steps:
                # 使用DP Sigma预测器生成sigma（支持微调）
                task_id, sigma, ready_tasks, features, task_metadata = select_task_with_sigma(
                    env, dp_sigma_predictor, params, sigma_min_ep, sigma_max_ep,
                    enable_finetune=enable_sigma_finetune
                )
                if task_id is None: break
                state = env.build_state(task_id)
                
                # 准备aux数据
                aux = None
                if params.get("use_gace", False):
                    inputs = env.prepare_transformer_inputs(ready_tasks)
                    if inputs is not None:
                        tf, pf, df, adj, attn_mask, total_budget = inputs
                        aux = {
                            'task_features': tf, 'privacy_features': pf, 'dag_features': df,
                            'dag_adjacency': adj, 'attention_mask': attn_mask,
                            'total_budget': total_budget
                        }

                action = agent.act(state, aux, sigma=sigma if not enable_sigma_finetune else float(sigma.item()))
                next_state, reward, done, info = env.step(task_id, action, {"sigma": sigma if not enable_sigma_finetune else float(sigma.item()), "q": 0.01})
                
                # 📊 Sigma预测器DP微调（仅在允许的阶段）
                if enable_sigma_finetune and features is not None and task_metadata is not None:
                    try:
                        loss_info = dp_sigma_predictor.dp_update_step(
                            features=features,
                            predicted_sigma=sigma,
                            reward=reward,
                            phase_config=phase_config,
                            rule_sigma=task_metadata.get('rule_sigma')
                        )
                        episode_sigma_losses.append(loss_info)
                        
                    except Exception as e:
                        print(f"[WARNING] Sigma微调失败: {e}")
                
                # 收集任务执行统计（基于环境实际返回的字段）
                episode_task_stats["total_tasks"][vehicle_id] += 1
                episode_task_stats["task_rewards"][vehicle_id].append(reward)  # 记录每任务奖励
                
                # 记录sigma相关统计
                sigma_val = sigma if not enable_sigma_finetune else float(sigma.item())
                episode_task_stats["sigmas"][vehicle_id].append(sigma_val)
                
                # 如果有规则基线，记录偏差
                if task_metadata is not None and 'rule_sigma' in task_metadata:
                    rule_sigma_val = task_metadata['rule_sigma']
                    episode_task_stats["rule_sigmas"][vehicle_id].append(rule_sigma_val)
                    episode_task_stats["sigma_deviations"][vehicle_id].append(abs(sigma_val - rule_sigma_val))
                
                if "end_to_end_delay" in info:
                    episode_task_stats["end_to_end_delays"][vehicle_id].append(info["end_to_end_delay"])
                if "exec_time" in info:
                    episode_task_stats["exec_times"][vehicle_id].append(info["exec_time"])
                if "waiting_delay" in info:
                    episode_task_stats["waiting_delays"][vehicle_id].append(info["waiting_delay"])
                if info.get("deadline_violated", False):
                    episode_task_stats["deadline_violations"][vehicle_id] += 1

                agent.remember(state, action, reward, next_state, done, aux, task_id, sigma)

                if global_step % params["train_frequency"] == 0 and len(agent.memory) >= params["batch_size"]:
                    loss, loss_info = agent.train_critic_step(task_id, noise_multiplier=agent.params["critic_noise_multiplier"])
                    episode_losses[vehicle_id].append(loss)
                    critic_actual_steps += 1
                    pool_sz = agent.memory.pool_size_for_task(task_id)
                    writer.add_scalar(f"Critic/PoolSize_Task{task_id}_V{vehicle_id}", pool_sz, global_step)
                    writer.add_scalar(f"DP/Critic_q_V{vehicle_id}_Task{task_id}", loss_info.get("dp/critic_q", 0.0), global_step)
                    writer.add_scalar(f"DP/Critic_z_V{vehicle_id}", loss_info.get("dp/critic_z", 0.0), global_step)

                if global_step % params["sync_target_every_steps"] == 0:
                    agent.update_target_network(hard=False, tau=0.001)

                episode_rewards[vehicle_id] += reward
                global_step += 1; step_count += 1

        # 联邦学习聚合（仅critic）
        if (ep + 1) % 5 == 0:
            fl.aggregate_models(agents)
            fl.distribute_model(agents)
        if (ep + 1) % params["fl_aggregate_every_episodes"] == 0:
            fl.aggregate_models(agents)
            fl.distribute_model(agents)

        # 计算每车每任务的平均指标
        # 任务完成率 = (总任务数 - 截止时间违反数) / 总任务数
        total_tasks_all = sum(episode_task_stats["total_tasks"])
        total_violations = sum(episode_task_stats["deadline_violations"])
        
        # 全局平均任务完成率
        global_completion_rate = ((total_tasks_all - total_violations) / max(1, total_tasks_all)) * 100.0
        
        # 平均每车每任务完成率
        per_vehicle_completion_rates = []
        for v_id in range(params["num_vehicles"]):
            v_tasks = episode_task_stats["total_tasks"][v_id]
            v_violations = episode_task_stats["deadline_violations"][v_id]
            if v_tasks > 0:
                v_rate = ((v_tasks - v_violations) / v_tasks) * 100.0
                per_vehicle_completion_rates.append(v_rate)
        avg_per_vehicle_completion_rate = np.mean(per_vehicle_completion_rates) if per_vehicle_completion_rates else 0.0
        
        # 时延指标计算
        # 1. 端到端时延（包括等待时间）
        all_e2e_delays = [d for delays in episode_task_stats["end_to_end_delays"] for d in delays]
        global_avg_e2e_delay = np.mean(all_e2e_delays) if all_e2e_delays else 0.0
        
        per_vehicle_avg_e2e_delays = []
        for v_id in range(params["num_vehicles"]):
            v_delays = episode_task_stats["end_to_end_delays"][v_id]
            if v_delays:
                per_vehicle_avg_e2e_delays.append(np.mean(v_delays))
        avg_per_vehicle_e2e_delay = np.mean(per_vehicle_avg_e2e_delays) if per_vehicle_avg_e2e_delays else 0.0
        
        # 2. 执行时延（仅任务执行时间）
        all_exec_times = [t for times in episode_task_stats["exec_times"] for t in times]
        global_avg_exec_time = np.mean(all_exec_times) if all_exec_times else 0.0
        
        per_vehicle_avg_exec_times = []
        for v_id in range(params["num_vehicles"]):
            v_times = episode_task_stats["exec_times"][v_id]
            if v_times:
                per_vehicle_avg_exec_times.append(np.mean(v_times))
        avg_per_vehicle_exec_time = np.mean(per_vehicle_avg_exec_times) if per_vehicle_avg_exec_times else 0.0
        
        # 3. 等待时延
        all_waiting_delays = [d for delays in episode_task_stats["waiting_delays"] for d in delays]
        global_avg_waiting_delay = np.mean(all_waiting_delays) if all_waiting_delays else 0.0
        
        per_vehicle_avg_waiting_delays = []
        for v_id in range(params["num_vehicles"]):
            v_delays = episode_task_stats["waiting_delays"][v_id]
            if v_delays:
                per_vehicle_avg_waiting_delays.append(np.mean(v_delays))
        avg_per_vehicle_waiting_delay = np.mean(per_vehicle_avg_waiting_delays) if per_vehicle_avg_waiting_delays else 0.0
        
        # 4. Sigma统计
        all_sigmas = [s for sigmas in episode_task_stats["sigmas"] for s in sigmas]
        global_avg_sigma = np.mean(all_sigmas) if all_sigmas else 0.0
        global_std_sigma = np.std(all_sigmas) if all_sigmas else 0.0
        global_min_sigma = np.min(all_sigmas) if all_sigmas else 0.0
        global_max_sigma = np.max(all_sigmas) if all_sigmas else 0.0
        
        per_vehicle_avg_sigmas = []
        for v_id in range(params["num_vehicles"]):
            v_sigmas = episode_task_stats["sigmas"][v_id]
            if v_sigmas:
                per_vehicle_avg_sigmas.append(np.mean(v_sigmas))
        avg_per_vehicle_sigma = np.mean(per_vehicle_avg_sigmas) if per_vehicle_avg_sigmas else 0.0
        
        # 每任务平均奖励
        all_task_rewards = [r for rewards in episode_task_stats["task_rewards"] for r in rewards]
        global_avg_task_reward = np.mean(all_task_rewards) if all_task_rewards else 0.0
        
        per_vehicle_avg_task_rewards = []
        for v_id in range(params["num_vehicles"]):
            v_rewards = episode_task_stats["task_rewards"][v_id]
            if v_rewards:
                per_vehicle_avg_task_rewards.append(np.mean(v_rewards))
        avg_per_vehicle_task_reward = np.mean(per_vehicle_avg_task_rewards) if per_vehicle_avg_task_rewards else 0.0

        avg_reward = np.mean(episode_rewards)
        avg_loss = np.mean([np.mean(losses) if losses else 0 for losses in episode_losses])
        
        # 记录episode级别奖励（每车累计）
        writer.add_scalar("reward/avg_episode_reward", avg_reward, ep)
        
        # 记录任务级别奖励（每任务平均）
        writer.add_scalar("reward/global_avg_task_reward", global_avg_task_reward, ep)
        writer.add_scalar("reward/avg_per_vehicle_task_reward", avg_per_vehicle_task_reward, ep)
        
        # 记录任务完成率指标
        writer.add_scalar("performance/global_task_completion_rate", global_completion_rate, ep)
        writer.add_scalar("performance/avg_per_vehicle_completion_rate", avg_per_vehicle_completion_rate, ep)
        writer.add_scalar("performance/deadline_violation_count", total_violations, ep)
        
        # 记录端到端时延指标（含等待时间）
        writer.add_scalar("delay/global_avg_end_to_end_delay", global_avg_e2e_delay, ep)
        writer.add_scalar("delay/avg_per_vehicle_end_to_end_delay", avg_per_vehicle_e2e_delay, ep)
        
        # 记录执行时延指标（仅执行时间）
        writer.add_scalar("delay/global_avg_exec_time", global_avg_exec_time, ep)
        writer.add_scalar("delay/avg_per_vehicle_exec_time", avg_per_vehicle_exec_time, ep)
        
        # 记录等待时延指标
        writer.add_scalar("delay/global_avg_waiting_delay", global_avg_waiting_delay, ep)
        writer.add_scalar("delay/avg_per_vehicle_waiting_delay", avg_per_vehicle_waiting_delay, ep)
        
        # 记录Sigma统计指标（实际使用值）
        writer.add_scalar("sigma/global_avg", global_avg_sigma, ep)
        writer.add_scalar("sigma/global_std", global_std_sigma, ep)
        writer.add_scalar("sigma/global_min", global_min_sigma, ep)
        writer.add_scalar("sigma/global_max", global_max_sigma, ep)
        writer.add_scalar("sigma/avg_per_vehicle", avg_per_vehicle_sigma, ep)
        writer.add_scalar("sigma/range_utilization", 
                         (global_max_sigma - global_min_sigma) / max(0.01, sigma_max_ep - sigma_min_ep) if all_sigmas else 0.0, 
                         ep)
        
        # 📊 Sigma预测器 vs 规则基线对比分析
        all_rule_sigmas = [s for rule_sigmas in episode_task_stats["rule_sigmas"] for s in rule_sigmas]
        all_sigma_deviations = [d for deviations in episode_task_stats["sigma_deviations"] for d in deviations]
        
        if all_rule_sigmas:
            global_avg_rule_sigma = np.mean(all_rule_sigmas)
            global_std_rule_sigma = np.std(all_rule_sigmas)
            
            writer.add_scalar("sigma_comparison/avg_predicted_sigma", global_avg_sigma, ep)
            writer.add_scalar("sigma_comparison/avg_rule_sigma", global_avg_rule_sigma, ep)
            writer.add_scalar("sigma_comparison/sigma_diff", global_avg_sigma - global_avg_rule_sigma, ep)
        
        if all_sigma_deviations:
            global_avg_deviation = np.mean(all_sigma_deviations)
            global_max_deviation = np.max(all_sigma_deviations)
            global_std_deviation = np.std(all_sigma_deviations)
            
            writer.add_scalar("sigma_comparison/avg_deviation", global_avg_deviation, ep)
            writer.add_scalar("sigma_comparison/max_deviation", global_max_deviation, ep)
            writer.add_scalar("sigma_comparison/deviation_ratio", 
                            global_avg_deviation / max(global_avg_sigma, 1e-9), ep)
        
        # 📊 Sigma预测器对隐私-效用权衡的影响
        if all_sigmas and all_task_rewards:
            # 计算sigma与奖励的相关性（Pearson相关系数）
            if len(all_sigmas) == len(all_task_rewards) and len(all_sigmas) > 1:
                try:
                    # 检查标准差是否为0（避免除以0警告）
                    sigma_std = np.std(all_sigmas)
                    reward_std = np.std(all_task_rewards)
                    
                    if sigma_std > 1e-9 and reward_std > 1e-9:
                        # 标准差非零，可以安全计算相关系数
                        sigma_reward_corr = np.corrcoef(all_sigmas, all_task_rewards)[0, 1]
                        # 检查是否是NaN
                        if not np.isnan(sigma_reward_corr):
                            writer.add_scalar("sigma_predictor/analysis/sigma_reward_correlation", sigma_reward_corr, ep)
                    else:
                        # 标准差为0，相关系数无意义，记录为0
                        writer.add_scalar("sigma_predictor/analysis/sigma_reward_correlation", 0.0, ep)
                except Exception as e:
                    # 计算失败，静默处理
                    pass
            
            # 高sigma vs 低sigma的奖励对比
            median_sigma = np.median(all_sigmas)
            high_sigma_rewards = [r for s, r in zip(all_sigmas, all_task_rewards) if s >= median_sigma]
            low_sigma_rewards = [r for s, r in zip(all_sigmas, all_task_rewards) if s < median_sigma]
            
            if high_sigma_rewards and low_sigma_rewards:
                writer.add_scalar("sigma_predictor/analysis/high_sigma_avg_reward", np.mean(high_sigma_rewards), ep)
                writer.add_scalar("sigma_predictor/analysis/low_sigma_avg_reward", np.mean(low_sigma_rewards), ep)
                writer.add_scalar("sigma_predictor/analysis/reward_gap_by_sigma", 
                                np.mean(high_sigma_rewards) - np.mean(low_sigma_rewards), ep)
        
        writer.add_scalar("loss/critic_avg", avg_loss, ep)
        writer.add_scalar("epsilon/value", agents[0].epsilon, ep)
        
        # 📊 Sigma预测器微调日志（详细观测）
        if episode_sigma_losses:
            # === 基础损失统计 ===
            avg_sigma_total_loss = np.mean([x['total_loss'] for x in episode_sigma_losses])
            avg_sigma_rl_loss = np.mean([x['rl_loss'] for x in episode_sigma_losses])
            avg_sigma_l2_loss = np.mean([x['l2_loss'] for x in episode_sigma_losses])
            avg_sigma_rule_loss = np.mean([x['rule_loss'] for x in episode_sigma_losses])


            # === 训练阶段信息 ===
            current_phase = episode_sigma_losses[0]['phase']
            current_beta_l2 = episode_sigma_losses[0]['beta_l2']
            
            # 获取实际使用的权重（新权重方案）
            weight_rl = episode_sigma_losses[0].get('weight_rl', 1.0)
            weight_l2 = episode_sigma_losses[0].get('weight_l2', current_beta_l2 * 1e-6)
            weight_rule = episode_sigma_losses[0].get('weight_rule', 0.5 * current_beta_l2)
            
            phase_idx = ['frozen', 'conservative', 'moderate', 'aggressive'].index(current_phase.lower())
            
            # === 组合损失比例分析（使用实际权重）===
            total_rl_contribution = avg_sigma_rl_loss * weight_rl
            total_l2_contribution = avg_sigma_l2_loss * weight_l2
            total_rule_contribution = avg_sigma_rule_loss * weight_rule
            total_weighted = total_rl_contribution + total_l2_contribution + total_rule_contribution
            
            rl_ratio = total_rl_contribution / max(total_weighted, 1e-9)
            l2_ratio = total_l2_contribution / max(total_weighted, 1e-9)
            rule_ratio = total_rule_contribution / max(total_weighted, 1e-9)
            
            # === 损失变化率（与上一episode对比）===
            # 需要全局变量存储上一episode的损失，这里先计算当前值
            
            # 1. 损失均值
            writer.add_scalar("sigma_predictor/loss/total_loss", avg_sigma_total_loss, ep)
            writer.add_scalar("sigma_predictor/loss/rl_loss", avg_sigma_rl_loss, ep)
            writer.add_scalar("sigma_predictor/loss/l2_loss", avg_sigma_l2_loss, ep)
            writer.add_scalar("sigma_predictor/loss/rule_loss", avg_sigma_rule_loss, ep)
            
            # 6. 损失组成比例（诊断哪个损失项占主导）
            writer.add_scalar("sigma_composition/composition/rl_contribution", total_rl_contribution, ep)
            writer.add_scalar("sigma_composition/composition/l2_contribution", total_l2_contribution, ep)
            writer.add_scalar("sigma_composition/composition/rule_contribution", total_rule_contribution, ep)
            writer.add_scalar("sigma_composition/composition/rl_ratio", rl_ratio, ep)
            writer.add_scalar("sigma_composition/composition/l2_ratio", l2_ratio, ep)
            writer.add_scalar("sigma_composition/composition/rule_ratio", rule_ratio, ep)
            

            
            # 8. 加权损失对比（验证权重设计 - 使用实际权重）
            writer.add_scalar("sigma_predictor/weighted_loss/weighted_rl", weight_rl * avg_sigma_rl_loss, ep)
            writer.add_scalar("sigma_predictor/weighted_loss/weighted_l2", weight_l2 * avg_sigma_l2_loss, ep)
            writer.add_scalar("sigma_predictor/weighted_loss/weighted_rule", weight_rule * avg_sigma_rule_loss, ep)
            
            # 9. 权重本身的监控（观察动态变化）
            writer.add_scalar("sigma_predictor/weights/weight_rl", weight_rl, ep)
            writer.add_scalar("sigma_predictor/weights/weight_l2", weight_l2, ep)
            writer.add_scalar("sigma_predictor/weights/weight_rule", weight_rule, ep)


        for agent in agents: agent.decay_epsilon()
        
    print(f"\n{'='*60}")
    print(f"训练完成 - DP-SGD 步数验证:")
    print(f"{'='*60}")
    print(f"Critic (所有DPSGD预算):")
    print(f"  预估步数: {critic_real_steps}")
    print(f"  实际步数: {critic_actual_steps}")
    print(f"  差异率: {abs(critic_actual_steps - critic_real_steps) / max(1, critic_real_steps) * 100:.2f}%")
    if critic_actual_steps > critic_real_steps:
        print(f"\n⚠️  警告: Critic实际步数 ({critic_actual_steps}) 超过预估 ({critic_real_steps})! 隐私预算已超支!")
    print(f"{'='*60}\n")

    os.makedirs(checkpoint_dir, exist_ok=True)
    for i, agent in enumerate(agents):
        if hasattr(agent.q_network, "_module"):
            torch.save(agent.q_network._module.state_dict(), os.path.join(checkpoint_dir, f'agent_critic_{i}_final.pth'))
        else:
            torch.save(agent.q_network.state_dict(), os.path.join(checkpoint_dir, f'agent_critic_{i}_final.pth'))
    
    # 💾 保存微调后的Sigma预测器
    if dp_sigma_predictor is not None:
        sigma_save_path = os.path.join(checkpoint_dir, 'sigma_predictor_finetuned.pth')
        dp_sigma_predictor.save(sigma_save_path)
        print(f"\n✅ Sigma预测器微调模型已保存: {sigma_save_path}")
        print(f"   隐私预算: ε={budgets['epsilon_sigma_predictor']:.2f}, δ={budgets['delta_sigma_predictor']:.2e}")
        print(f"   总微调步数: {sigma_predictor_steps}")
        print(f"   噪声乘数: {dp_sigma_predictor.noise_multiplier:.4f}\n")

    writer.close()
    return 0.0

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='训练层次化联邦强化学习模型')
    parser.add_argument('--config', type=str, default=None, help='YAML配置文件路径')
    parser.add_argument('--exp_name', type=str, default='default', help='实验名称')
    parser.add_argument('--exp_description', type=str, default='', help='实验描述')
    parser.add_argument('--output_dir', type=str, default='runs/new', help='输出目录')
    parser.add_argument('--model_dir', type=str, default='model/new', help='模型保存目录')
    parser.add_argument('--use_gace', type=lambda x: x.lower() == 'true', default=None, help='是否使用图感知上下文编码器')
    parser.add_argument('--episodes', type=int, default=None)
    parser.add_argument('--num_tasks', type=int, default=None)
    parser.add_argument('--num_rsus', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    override_params = {}
    if args.config:
        import yaml
        print(f"加载配置文件: {args.config}")
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        override_params.update(config)
    if args.use_gace is not None: override_params['use_gace'] = args.use_gace
    if args.episodes is not None: override_params['episodes'] = args.episodes
    if args.num_tasks is not None: override_params['num_tasks'] = args.num_tasks
    if args.num_rsus is not None: override_params['num_rsus'] = args.num_rsus
    if args.seed is not None: override_params['seed'] = args.seed
    override_params['exp_name'] = args.exp_name if args.exp_name != 'default' else override_params.get('exp_name', 'default')
    override_params['exp_description'] = args.exp_description or override_params.get('exp_description', '')
    override_params['output_dir'] = args.output_dir if args.output_dir != 'runs/new' else override_params.get('output_dir', 'runs/new')
    override_params['model_dir'] = args.model_dir if args.model_dir != 'model/new' else override_params.get('model_dir', 'model/new')
    main(override_params)