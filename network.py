# network.py
# 主要功能：FusedQNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math

# ========= 辅助：注意力掩码规范化 =========

def ensure_self_loops_in_mask(mask_bool: torch.Tensor) -> torch.Tensor:
    """
    确保 [B, L, L] 的布尔掩码中，每个查询行至少能看到自身（自环）。
    仅修补“整行全 False”的行：将该行的对角元素设为 True，其余不变。
    """
    if mask_bool is None:
        return None
    if mask_bool.dtype is not torch.bool:
        mask_bool = mask_bool.bool()
    if mask_bool.dim() == 2:
        mask_bool = mask_bool.unsqueeze(0)
    if mask_bool.dim() != 3:
        raise ValueError(f"mask_bool must be [B, L, L] or [L, L], got {tuple(mask_bool.shape)}")
    B, L, _ = mask_bool.shape
    row_has_any = mask_bool.any(dim=-1)  # [B, L]
    need_fix = ~row_has_any
    if need_fix.any():
        idx_b, idx_l = torch.where(need_fix)
        mask_bool = mask_bool.clone()
        mask_bool[idx_b, idx_l, idx_l] = True
    return mask_bool

def normalize_attn_mask_any(am: Optional[torch.Tensor], B: int, L: int, device: torch.device) -> Optional[torch.Tensor]:
    """
    把任意 0D/1D/2D/3D 的注意力掩码规范为严格 [B,L,L] 的 bool 张量，并修补自环。
    若 am 为 None，返回 None。
    """
    if am is None:
        return None
    if not torch.is_tensor(am):
        raise TypeError(f"attn_mask must be a torch.Tensor or None, got {type(am)}")
    am = am.to(device)
    if am.dim() == 0:
        m = (am > 0.5).view(1, 1, 1).expand(B, L, L).contiguous()
    elif am.dim() == 1:
        if am.numel() == 1:
            m = (am > 0.5).view(1, 1, 1).expand(B, L, L).contiguous()
        else:
            raise ValueError(f"1D attn_mask unsupported unless single element, got {tuple(am.shape)}")
    elif am.dim() == 2:
        if am.shape != (L, L):
            raise ValueError(f"2D attn_mask shape mismatch: got {tuple(am.shape)}, expected ({L},{L})")
        m = (am > 0.5).unsqueeze(0).expand(B, L, L).contiguous()
    elif am.dim() == 3:
        if am.shape == (B, L, L):
            m = (am > 0.5).contiguous()
        elif am.shape[0] == 1 and am.shape[1:] == (L, L):
            m = (am > 0.5).expand(B, L, L).contiguous()
        else:
            raise ValueError(f"3D attn_mask shape mismatch: got {tuple(am.shape)}, expected ({B},{L},{L}) or (1,{L},{L})")
    else:
        raise ValueError(f"attn_mask must be 0D/1D/2D/3D, got dim={am.dim()}")
    m = ensure_self_loops_in_mask(m.to(torch.bool))
    return m.to(device)

# ========= 模型定义 =========

class ResidualScaler(nn.Module):
    def __init__(self, init: float = 0.1):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(init))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logit) * x

class _MultiHeadSelfAttentionDP(nn.Module):
    """
    自定义自注意力，支持布尔或浮点加性掩码，掩码输入形状推荐 [B, L, L]。
    内部会广播到 [B, H, L, L]。
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        x = x.view(B, L, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, L, D = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B, L, H * D)
        return x

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L, L]

        # 处理注意力掩码
        if attn_mask is not None:
            # 接受 [B,L,L] 或 [1,L,L] 或 [L,L]，但优先推荐上游规范为 [B,L,L]
            if attn_mask.dim() == 2 and attn_mask.shape == (L, L):
                am = attn_mask
                am_B = 1
            elif attn_mask.dim() == 3 and attn_mask.shape[-2:] == (L, L):
                am = attn_mask
                am_B = attn_mask.shape[0]
            else:
                raise ValueError(f"attn_mask must be [L,L] or [B,L,L], got {tuple(attn_mask.shape)}")
            # 布尔掩码：True=屏蔽；浮点掩码：加性偏置
            if attn_mask.dtype == torch.bool:
                # 广播到 [B,1,L,L] 再到 [B,H,L,L]
                if am_B == 1 and B > 1:
                    am = am.expand(B, L, L)
                am = am.view(B, 1, L, L)
                attn_scores = attn_scores.masked_fill(am, float('-inf'))
            else:
                # 浮点加性掩码（例如 -inf/0/正偏置）
                if am_B == 1 and B > 1:
                    am = am.expand(B, L, L)
                am = am.view(B, 1, L, L).to(attn_scores.dtype)
                attn_scores = attn_scores + am

            # 检测并修复全遮蔽行（所有值都是 -inf）
            max_scores = attn_scores.max(dim=-1, keepdim=True)[0]
            fully_masked = torch.isinf(max_scores) & (max_scores < 0)
            if fully_masked.any():
                diag = torch.arange(L, device=attn_scores.device)
                # 对 fully_masked 的行，解除对角线遮挡 => 0
                attn_scores[:, :, diag, diag] = torch.where(
                    fully_masked[:, :, diag, 0],
                    torch.zeros_like(attn_scores[:, :, diag, diag]),
                    attn_scores[:, :, diag, diag]
                )

        attn_weights = torch.softmax(attn_scores, dim=-1)
        if torch.isnan(attn_weights).any():
            attn_weights = torch.nan_to_num(attn_weights, nan=1.0 / max(L, 1))
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)  # [B, H, L, D]
        out = self.out_proj(self._merge_heads(out))  # [B, L, d_model]
        return out

class _DPTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.self_attn = _MultiHeadSelfAttentionDP(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        sa = self.self_attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.dropout(sa)
        ff = self.ffn(self.norm2(x))
        x = x + self.dropout(ff)
        return x

class GraphAwareContextEncoder(nn.Module):
    """
    Graph-Aware Context Encoder (GACE)：
    图结构感知的上下文编码器，通过多头自注意力机制融合任务特征、隐私特征和DAG结构信息。
    在进入编码层前，先将 attention_mask 规范为 [B,L,L] 布尔掩码，并修补自环；
    然后转换为加性浮点掩码（-inf 屏蔽，0 保留）。
    """
    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 2, device: str = 'cpu'):
        super().__init__()
        self.d_model = d_model
        self.device = torch.device(device)
        self.task_proj = nn.Linear(15, d_model)
        self.privacy_proj = nn.Linear(6, d_model)
        self.dag_proj = nn.Linear(4, d_model)
        self.layers = nn.ModuleList([
            _DPTransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=0.2)
            for _ in range(num_layers)
        ])
        self.pre_norm = nn.LayerNorm(d_model)
        self.post_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, local_global: torch.Tensor, aux: Optional[Dict] = None, 
                return_intermediates: bool = False) -> Tuple[torch.Tensor, Dict]:
        if aux is None or 'task_features' not in aux:
            return local_global, {}
        try:
            device = self.device
            B = local_global.shape[0]
            tf = aux['task_features'].to(device)
            pf = aux['privacy_features'].to(device)
            df = aux['dag_features'].to(device)
            # 对单样本上下文在更大 batch 的场景，允许扩展到 B
            if tf.shape[0] == 1 and B > 1:
                tf = tf.expand(B, -1, -1)
                pf = pf.expand(B, -1, -1)
                df = df.expand(B, -1, -1)
            L = int(tf.shape[1])

            tf_emb = self.task_proj(tf)
            pf_emb = self.privacy_proj(pf)
            df_emb = self.dag_proj(df)
            context = tf_emb + pf_emb + df_emb
            context = self.pre_norm(context)

            # 规范化注意力掩码到 [B,L,L] 布尔，并修补自环
            attn_mask_bool = aux.get('attention_mask', None)
            attn_mask_3d_bool = normalize_attn_mask_any(attn_mask_bool, B=B, L=L, device=device) if attn_mask_bool is not None else None

            # 转为加性浮点掩码 [B,L,L]：True -> -inf，False -> 0.0
            attn_mask_float = None
            if attn_mask_3d_bool is not None:
                attn_mask_float = torch.zeros((B, L, L), dtype=torch.float32, device=device)
                attn_mask_float = attn_mask_float.masked_fill(attn_mask_3d_bool, float('-inf'))

            x = context
            for layer in self.layers:
                x = layer(x, attn_mask=attn_mask_float)
            context_encoded = self.post_norm(x)

            global_context = context_encoded.mean(dim=1)  # [B, d_model]
            fused_context = global_context + 0.1 * local_global.to(device)
            aligned = self.output_proj(fused_context)
            
            # 返回中间激活用于评估
            intermediates = {}
            if return_intermediates:
                intermediates = {
                    'context_encoded': context_encoded,  # [B, L, d_model]
                    'global_context': global_context,    # [B, d_model]
                    'tf_emb': tf_emb,
                    'pf_emb': pf_emb,
                    'df_emb': df_emb,
                    'task_features': tf,
                    'privacy_features': pf,
                    'dag_features': df,
                }
            
            return aligned, intermediates
        except Exception:
            return local_global, {}

class FusedQNetwork(nn.Module):
    def __init__(self,
                 state_size: int = 15,
                 action_size: int = 3,
                 hidden_size: int = 256,
                 use_gace: bool = False,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 device: str = 'cpu',
                 enable_budget_scaling: bool = False):
        super().__init__()
        self.enable_budget_scaling = enable_budget_scaling
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        q_head_input_dim = hidden_size + (1 if enable_budget_scaling else 0)
        self.q_head = nn.Sequential(
            nn.Linear(q_head_input_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, action_size)
        )
        self.use_gace = use_gace
        self.device = torch.device(device)
        if use_gace:
            self.context_encoder = GraphAwareContextEncoder(d_model=d_model, nhead=nhead, num_layers=num_layers, device=device)
            self.local_to_global = nn.Linear(hidden_size, d_model)
            self.global_to_local = nn.Linear(d_model, hidden_size)
            self.residual = ResidualScaler(init=0.1)
        else:
            self.context_encoder = None
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        for m in self.q_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.001, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        if self.use_gace:
            nn.init.xavier_uniform_(self.local_to_global.weight, gain=1.0); nn.init.constant_(self.local_to_global.bias, 0.0)
            nn.init.xavier_uniform_(self.global_to_local.weight, gain=1.0); nn.init.constant_(self.global_to_local.bias, 0.0)

    def forward(self, state: torch.Tensor, aux: Optional[Dict] = None) -> Tuple[torch.Tensor, Optional[Dict]]:
        state = torch.nan_to_num(state, nan=0.0).clamp(-10.0, 10.0)
        local_feat = self.state_encoder(state).clamp(-10.0, 10.0)
        aligned_feat = local_feat
        if self.use_gace and self.context_encoder is not None and aux is not None:
            try:
                local_global = self.local_to_global(local_feat).clamp(-10.0, 10.0)
                aligned_global, _ = self.context_encoder(local_global, aux=aux)
                aligned_local = self.global_to_local(aligned_global).clamp(-10.0, 10.0)
                aligned_feat = (local_feat + self.residual(aligned_local)).clamp(-10.0, 10.0)
            except Exception:
                pass
        q_head_input = aligned_feat
        if self.enable_budget_scaling:
            B = aligned_feat.shape[0]
            budget_sufficiency = torch.full((B, 1), 0.5, device=aligned_feat.device, dtype=aligned_feat.dtype)
            q_head_input = torch.cat([aligned_feat, budget_sufficiency], dim=1)
        q_values = self.q_head(q_head_input)
        q_values = torch.nan_to_num(q_values, nan=0.0).clamp(-10.0, 10.0)
        return q_values, None