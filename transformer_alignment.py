# transformer_alignment.py
from typing import Optional, List

import torch
import torch.nn as nn


def ensure_self_loops_in_mask(mask_bool: torch.Tensor) -> torch.Tensor:
    """确保注意力掩码中至少有自环，防止节点被完全隔离。"""
    if mask_bool is None:
        return None
    if mask_bool.dtype is not torch.bool:
        mask_bool = mask_bool.bool()
    if mask_bool.dim() == 2:
        mask_bool = mask_bool.unsqueeze(0)
    if mask_bool.dim() != 3:
        raise ValueError(f"mask_bool must be [B, L, L] or [L, L], got {mask_bool.shape}")

    row_is_all_masked = mask_bool.all(dim=-1)  # [B, L]
    
    if row_is_all_masked.any():
        idx_b, idx_l = torch.where(row_is_all_masked)
        mask_bool = mask_bool.clone()
        mask_bool[idx_b, idx_l, idx_l] = False
    return mask_bool


def _build_attention_bias(
    attn_bool: Optional[torch.Tensor],
    adj: Optional[torch.Tensor],
    bias_strength: float,
    large_neg: float = -1e4,
) -> Optional[torch.Tensor]:
    """构建浮点型的加性注意力偏置"""
    bias = None
    if attn_bool is not None:
        # True (屏蔽) -> -1e4, False (保留) -> 0.0
        bias = torch.where(
            attn_bool,
            torch.full_like(attn_bool, large_neg, dtype=torch.float32),
            torch.zeros_like(attn_bool, dtype=torch.float32),
        )

    if adj is not None:
        bias_adj = bias_strength * torch.clamp(adj, min=0.0, max=1.0)
        
        if bias is None:
            bias = bias_adj
        else:

            bias = bias + bias_adj
            
    return bias




class SingleInputFusionLayer(nn.Module):
    """
    一个标准的 Pre-Norm Transformer 编码器层，
    但注入了图偏置（Graph Bias）。
    """
    def __init__(
        self,
        in_dim: int,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        use_graph_conv: bool = True,  # 此参数保留，但不再使用
        adj_bias_strength: float = 1.0,
    ):
        super().__init__()
        
        
        self.fuse = nn.Linear(in_dim, d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 可学习的图偏置强度
        self.adj_bias_log_scale = nn.Parameter(torch.log(torch.tensor(adj_bias_strength, dtype=torch.float32)))

    def forward(
        self,
        x: torch.Tensor,                      # [B, L, in_dim or d_model]
        attention_mask: Optional[torch.Tensor],   # [B, L, L] (bool, True=屏蔽)
        adjacency_matrix: Optional[torch.Tensor]  # [B, L, L] (float)
    ) -> torch.Tensor:
        
        if x.shape[-1] != self.attn.embed_dim:
            x = self.fuse(x)  # [B, L, d_model]


        B, L = x.shape[0], x.shape[1]

        # 2. 构建加性注意力偏置 (Additive Attention Bias)
        attn_bool = None
        if attention_mask is not None:
            # 确保掩码是 [B, L, L] 
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0)
            if attention_mask.shape[0] == 1 and B > 1:
                attention_mask = attention_mask.expand(B, L, L).contiguous()
            
            # 确保设备一致
            if attention_mask.device != x.device:
                attention_mask = attention_mask.to(x.device)
            
            # 确保被屏蔽的行至少可以关注自己
            attn_bool = ensure_self_loops_in_mask(attention_mask.bool())

        adj_bias = adjacency_matrix if adjacency_matrix is not None else None
        if adj_bias is not None and adj_bias.device != x.device:
            adj_bias = adj_bias.to(x.device)
            
        bias_strength = torch.exp(self.adj_bias_log_scale) # 学习到的强度
        attn_bias = _build_attention_bias(attn_bool, adj_bias, bias_strength=bias_strength)

        x_res = x
        x_norm = self.norm1(x) # 归一化 *在* 注意力之前
        
        # 将偏置 [B, L, L] 扩展到 [B*nhead, L, L]
        if attn_bias is not None:
            attn_bias = attn_bias.unsqueeze(1).expand(B, self.attn.num_heads, L, L)
            attn_bias = attn_bias.reshape(B * self.attn.num_heads, L, L)
            
        x_attn, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_bias, need_weights=False)
        x = x_res + self.dropout1(x_attn) # 残差连接

        x_res = x
        x_norm = self.norm2(x) # 归一化 *在* FFN 之前
        x_ff = self.ffn(x_norm)
        x = x_res + self.dropout2(x_ff) # 残差连接

        return x


class TransformerPrivacyAligner(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_graph_conv: bool = True,
        adj_bias_strength: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_graph_conv = use_graph_conv
        self.adj_bias_strength = adj_bias_strength
        self.layers: Optional[nn.ModuleList] = None
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        
        self.final_norm = nn.LayerNorm(d_model)

    def _lazy_init_layers(self, in_dim: int):
        """懒加载模型层，在第一次前向传播时根据输入维度构建"""
        if self.layers is not None:
            return
            
        layers: List[SingleInputFusionLayer] = []
        for i in range(self.num_layers):
            # 第一层的 in_dim 是特征维度，之后是 d_model
            layer_in = in_dim if i == 0 else self.d_model
            layers.append(
                SingleInputFusionLayer(
                    in_dim=layer_in,
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dropout=self.dropout,
                    use_graph_conv=self.use_graph_conv,
                    adj_bias_strength=self.adj_bias_strength,
                )
            )
        self.layers = nn.ModuleList(layers)
        
        # 确保新层被移动到正确的设备
        try:
            dev = next(self.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        self.layers.to(dev)
        self.final_norm.to(dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        x 是一个 [B, D] 的 *扁平* 张量，由 _pack_transformer_input 打包。
        """
        model_device = next(self.parameters()).device
        if x.device != model_device:
            x = x.to(model_device)

        B, D = x.shape
        
        L_meta = x[:, -5].round().long()
        if L_meta.numel() == 0:
            raise RuntimeError(f"无法从元数据中提取 L，x.shape={x.shape}")
        L = L_meta.max().item()
        # [!! 结束修复 !!]

        if L <= 0:
             raise RuntimeError(f"解析出的 L={L} 必须为正。输入 x 的元数据可能已损坏。")

        num_meta_features = 5 
        base_total = D - (L * L + L * L + num_meta_features)
        if base_total % L != 0 or base_total <= 0:
            raise RuntimeError(f"无法解析打包的张量: D={D}, L={L}, base_total={base_total}")
        feat_total = base_total // L

        idx = 0
        base = x[:, idx : idx + L * feat_total]; idx += L * feat_total
        adj_flat = x[:, idx : idx + L * L]; idx += L * L
        mask_flat = x[:, idx : idx + L * L]; idx += L * L

        base = base.view(B, L, feat_total)  # [B, L, F]
        adj = adj_flat.view(B, L, L)        # [B, L, L]
        attn_mask_bool = (mask_flat.view(B, L, L) > 0.5) # [B, L, L]

        device = x.device
        if adj.device != device: adj = adj.to(device)
        if attn_mask_bool.device != device: attn_mask_bool = attn_mask_bool.to(device)

        # 2. 懒加载（如果需要）
        self._lazy_init_layers(in_dim=feat_total)

        # 3. 运行 Transformer 层
        h = base
        for layer in self.layers:
            h = layer(h, attention_mask=attn_mask_bool, adjacency_matrix=adj)

        # 4. [!! 修改：Pre-Norm !!]
        #    在送入 head 之前，应用最后一次归一化
        h = self.final_norm(h) 

        # 5. 输出 Head
        out = self.head(h).squeeze(-1) # [B, L]
        return out