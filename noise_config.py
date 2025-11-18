import numpy as np
from typing import Tuple, Dict, Optional

# ---------------------------
# 训练时每辆车都会独立派生自己的预算
# ---------------------------
EPSILON_TOTAL = 10.0
DELTA_TOTAL = 1e-5

BUDGET_SPLIT = {
    "episode_local": 0.7,   # -> ε_local = 7.0 
    "dpsgd_total": 0.3,     # -> ε_dpsgd = 3.0 (critic + sigma_predictor)
}

# DP-SGD预算细分：critic占0.2, sigma_predictor占0.1
DPSGD_SPLIT = {
    "critic": 0.2 / 0.3,         # critic占总预算的0.2 (即dpsgd的2/3)
    "sigma_predictor": 0.1 / 0.3  # sigma_predictor占总预算的0.1 (即dpsgd的1/3)
}

DELTA_SPLIT = {
    "episode_local": 0.7 * DELTA_TOTAL,
    "critic": 0.2 * DELTA_TOTAL,          # critic占0.2的delta
    "sigma_predictor": 0.1 * DELTA_TOTAL,  # sigma_predictor占0.1的delta
}

# Task-level 敏感度和裁剪
DELTA_SENS = 0.0447     
CLIP_NORM = DELTA_SENS / 2.0  

# RDP 基本公式
ALPHA_DEFAULT = 20

def rdp_epsilon_gaussian(alpha: float, sigma: float, sensitivity: float = 1.0) -> float:
    # ρ(α) = α Δ^2 / (2 σ^2)
    return float(alpha * (sensitivity ** 2) / (2.0 * (sigma ** 2)))

def rdp_to_dp_epsilon(rho: float, alpha: int, delta: float) -> float:
    # ε(α, δ) = ρ + log(1/δ) / (α - 1)
    return float(rho + np.log(1.0 / delta) / (alpha - 1))

def dp_to_rdp_epsilon(epsilon: float, alpha: int, delta: float) -> float:
    # 从 ε(α, δ) ≈ ρ + log(1/δ)/(α-1) 推回 ρ
    return float(max(0.0, epsilon - np.log(1.0 / delta) / (alpha - 1)))

def compute_level0_budgets(epsilon_total: float = EPSILON_TOTAL,
                           delta_total: float = DELTA_TOTAL) -> Dict[str, float]:
    """
    返回一个"单车"的 Level-0 预算切分。
    训练中应当为每辆车独立调用本函数，得到 per-vehicle 的预算。
    
    预算分配：
    - episode_local: 0.7 (7.0)
    - dpsgd_total: 0.3 (3.0)
      - critic: 0.2 (2.0)
      - sigma_predictor: 0.1 (1.0)
    """
    eps_local = float(epsilon_total * BUDGET_SPLIT["episode_local"])
    eps_dpsgd = float(epsilon_total * BUDGET_SPLIT["dpsgd_total"])

    # DP-SGD预算细分
    eps_critic = float(eps_dpsgd * DPSGD_SPLIT["critic"])
    eps_sigma_predictor = float(eps_dpsgd * DPSGD_SPLIT["sigma_predictor"])

    del_local = float(DELTA_SPLIT["episode_local"])
    del_critic = float(DELTA_SPLIT["critic"])
    del_sigma_predictor = float(DELTA_SPLIT["sigma_predictor"])

    return {
        "epsilon_local": eps_local,
        "epsilon_dpsgd": eps_dpsgd,
        "epsilon_critic": eps_critic,
        "epsilon_sigma_predictor": eps_sigma_predictor,
        "delta_local": del_local,
        "delta_critic": del_critic,
        "delta_sigma_predictor": del_sigma_predictor,
    }

def compute_episode_task_rho(num_episodes: int,
                             max_tasks_per_episode: int,
                             epsilon_local: float,
                             delta_local: float,
                             alpha: int = ALPHA_DEFAULT) -> Tuple[float, float]:
    """
    用 (ε_local, δ_local) 在给定 α 下得到 ρ_total，并保守均分到 E * M_max。
    返回 (ρ_episode, ρ_task)。
    """
    rho_total = dp_to_rdp_epsilon(epsilon_local, alpha, delta_local)
    rho_episode = rho_total / max(1, num_episodes)
    rho_task = rho_episode / max(1, max_tasks_per_episode)
    return float(rho_episode), float(rho_task)

def sigma_from_rho(rho: float,
                   alpha: int = ALPHA_DEFAULT,
                   sensitivity: float = DELTA_SENS) -> float:
    """
    ρ = α Δ^2 / (2 σ^2) → σ = Δ sqrt(α / (2 ρ))
    """
    if rho <= 0:
        return 1e9
    return float(sensitivity * np.sqrt(alpha / (2.0 * rho)))

def find_optimal_alpha_for_sigma_min(
    num_episodes: int,
    max_tasks_per_episode: int,
    epsilon_local: float,
    delta_local: float,
    sensitivity: float = DELTA_SENS,
    alpha_search_range: list = list(range(2, 101)),
) -> Tuple[int, float]:
    """
    搜索 α 使得 σ_min 最小：σ(α) = Δ sqrt( α / (2 ρ_task(α)) )，
    其中 ρ_task(α) 由 (ε_local, δ_local) 反推 ρ_total(α)，再除以 N_total=E*M_max。
    """
    best_alpha = alpha_search_range[0]
    min_sigma = np.inf
    N_total = max(1, num_episodes) * max(1, max_tasks_per_episode)

    # α 必须满足：ε > log(1/δ)/(α - 1) → α > 1 + log(1/δ)/ε
    alpha_min_valid = 1.0 + np.log(1.0 / delta_local) / (max(1e-12, epsilon_local))

    for alpha in alpha_search_range:
        if alpha <= alpha_min_valid:
            continue
        rho_total = dp_to_rdp_epsilon(epsilon_local, alpha, delta_local)
        if rho_total <= 1e-12:
            continue
        rho_task = rho_total / float(N_total)
        sigma = sigma_from_rho(rho_task, alpha, sensitivity)
        if sigma < min_sigma:
            min_sigma = sigma
            best_alpha = alpha

    if not np.isfinite(min_sigma):
        # 回退策略：取 just-above alpha_min_valid 的整数
        safe_alpha = int(np.ceil(alpha_min_valid)) + 1
        rho_total = dp_to_rdp_epsilon(epsilon_local, safe_alpha, delta_local)
        rho_task = rho_total / float(max(1, N_total))
        min_sigma = sigma_from_rho(rho_task, safe_alpha, sensitivity)
        best_alpha = safe_alpha

    return int(best_alpha), float(min_sigma)

def compute_sigma_range_for_episode(num_episodes: int,
                                    tasks_in_episode: int,
                                    epsilon_local: float,
                                    delta_local: float,
                                    sensitivity: float = DELTA_SENS,
                                    sigma_max_mult: float = 5.0,
                                    max_tasks_per_episode: Optional[int] = None,
                                    alpha_search_range: list = list(range(2, 101))
                                ) -> Tuple[float, float]:
    """
    用基于 task-level 预算的 σ_min，设 σ_max = mult × σ_min。
    """
    M_upper = max(tasks_in_episode, max_tasks_per_episode or tasks_in_episode)
    best_alpha, sigma_min = find_optimal_alpha_for_sigma_min(
        num_episodes=num_episodes,
        max_tasks_per_episode=M_upper,
        epsilon_local=epsilon_local,
        delta_local=delta_local,
        sensitivity=sensitivity,
        alpha_search_range=alpha_search_range,
    )
    sigma_max = float(sigma_min * sigma_max_mult)
    return float(sigma_min), float(sigma_max)

# ========== DP-SGD 噪声倍数计算（用 RDPAccountant 扫描） ==========
from opacus.accountants.rdp import RDPAccountant

def compute_dpsgd_noise_multiplier(
    epsilon_target: float,
    delta_target: float,
    num_steps: int,
    batch_size: int,
    dataset_size: int,
) -> Dict[str, float]:
    """
    二分搜索噪声倍数 z，使得在给定 num_steps、采样率 q 下，ε ≈ epsilon_target。
    其中 z = σ_dpsgd / C，C = CLIP_NORM。
    """
    q = min(1.0, batch_size / max(1, dataset_size))
    if q <= 0:
        q = 1.0

    z_low, z_high = 0.05, 200.0
    tol = 0.01
    for _ in range(60):
        z_mid = 0.5 * (z_low + z_high)
        acc = RDPAccountant()
        for _ in range(max(1, num_steps)):
            acc.step(noise_multiplier=z_mid, sample_rate=q)
        eps_mid = acc.get_epsilon(delta=delta_target)
        if eps_mid > epsilon_target:
            # 噪声不足（ε偏大），提高 z
            z_low = z_mid
        else:
            z_high = z_mid
        if abs(z_high - z_low) < tol:
            break

    z = 0.5 * (z_low + z_high)
    sigma_dpsgd = z * CLIP_NORM
    return {
        "noise_multiplier": float(z),
        "sigma_dpsgd": float(sigma_dpsgd),
        "sampling_rate": float(q),
        "steps": int(num_steps),
        "clip_norm": float(CLIP_NORM),
    }