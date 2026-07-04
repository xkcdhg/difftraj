"""
Shared building blocks for the LIM (leap initializer) and the core denoising
Transformer. These mirror the architecture actually used by LED (Mao et al.,
CVPR'23) -- the codebase DiffTrajectory explicitly extends (paper: "Unlike
LED [29], we not only directly predict the variance and mean, but also
impose additional constraints...", Section 4.2) -- because LED's publicly
reported NBA numbers match the "LED" baseline column in DiffTrajectory's
Table 3 almost to the decimal, which is strong evidence the two papers share
this backbone.

Correspondence to DiffTrajectory's Eq. (13)-(15):

    e_spatio = softmax(f_q(X) f_k(X_N)^T / sqrt(d)) f_v(X_N)      (Eq. 13)
    e_social = f_GRU(f_1D-CNN(X))                                  (Eq. 14)
    mu_t     = MLP([e_spatio ; e_social])                          (Eq. 15)

maps onto the code below as:

    e_spatio  <-> SocialTransformer   (attention *across agents in a scene*,
                                        implemented as a masked
                                        TransformerEncoder rather than literal
                                        ego-query / neighbor-key cross
                                        attention -- functionally the same
                                        "who-influences-whom" aggregation,
                                        differently parameterized. See
                                        docs/PLAN.md for the nuance.)
    e_social  <-> TrajectoryEncoder    (1D-Conv -> GRU over the ego agent's
                                        own history, matches Eq. 14 exactly)
    mu_t      <-> MLP(...)             (concatenate + feed through an MLP)
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al.), used inside
    the core denoising Transformer's self-attention over the future-horizon
    time axis."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class ConcatSquashLinear(nn.Module):
    """
    FiLM-style conditioning layer: y = Linear(x) * sigmoid(gate(ctx)) + bias(ctx).

    This is how the diffusion time step and social context get injected into
    the denoising Transformer at every stage, instead of simple concatenation.
    Used by the core denoiser (not by LIM).
    """

    def __init__(self, dim_in: int, dim_out: int, dim_ctx: int):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        return self._layer(x) * gate + bias

    def batch_generate(self, ctx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Same computation, kept as a separate entry point for the
        multi-sample-in-parallel accelerated sampling path (K predictions at
        once), matching LED's `generate_accelerate`."""
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        return self._layer(x) * gate + bias


class MLP(nn.Module):
    """Simple feed-forward MLP with configurable hidden layer widths,
    matching the Mean/Variance/Scale decoders described in Section 5.2."""

    def __init__(self, in_feat: int, out_feat: int, hid_feat=(1024, 512),
                 activation: nn.Module = None, dropout: float = -1):
        super().__init__()
        dims = (in_feat,) + tuple(hid_feat) + (out_feat,)
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )
        self.activation = activation if activation is not None else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout != -1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.dropout(self.activation(x))
            x = layer(x)
        return x


class TrajectoryEncoder(nn.Module):
    """
    1D-Conv + GRU encoder of a single agent's own trajectory -- Eq. (14):
        e_social = f_GRU(f_1D-CNN(X))

    (Named `TrajectoryEncoder` rather than the paper's "e_social" because
    what it actually encodes is the *ego* agent's own kinematic history;
    the inter-agent "social" aggregation happens in SocialTransformer below.
    We keep both names in the docstrings to avoid ambiguity when
    cross-referencing the paper.)
    """

    def __init__(self, in_channels: int = 6, conv_out_channels: int = 32,
                 kernel_size: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spatial_conv = nn.Conv1d(in_channels, conv_out_channels, kernel_size,
                                       stride=1, padding=kernel_size // 2)
        self.temporal_encoder = nn.GRU(conv_out_channels, hidden_dim, 1, batch_first=True)
        self.relu = nn.ReLU()
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_normal_(self.spatial_conv.weight)
        nn.init.kaiming_normal_(self.temporal_encoder.weight_ih_l0)
        nn.init.kaiming_normal_(self.temporal_encoder.weight_hh_l0)
        nn.init.zeros_(self.spatial_conv.bias)
        nn.init.zeros_(self.temporal_encoder.bias_ih_l0)
        nn.init.zeros_(self.temporal_encoder.bias_hh_l0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, T, in_channels) -> (batch, hidden_dim)"""
        x_t = x.transpose(1, 2)                      # (B, C, T)
        x_conv = self.relu(self.spatial_conv(x_t))    # (B, 32, T)
        x_conv = x_conv.transpose(1, 2)               # (B, T, 32)
        _, h_n = self.temporal_encoder(x_conv)
        return h_n.squeeze(0)                          # (B, hidden_dim)


class SocialTransformer(nn.Module):
    """
    Attention-based aggregation across agents present in the same scene --
    plays the role of Eq. (13)'s e_spatio. Every agent's flattened past
    trajectory is projected to a 256-dim token; a small TransformerEncoder
    then attends *across the agent dimension* (using `mask` to prevent
    attending across different scenes batched together), which lets each
    agent's representation absorb information from the others -- the same
    goal as literal query(ego)/key,value(neighbors) cross-attention, just
    parameterized as full self-attention with masking (this is what LED
    actually implements and what produced its published numbers).
    """

    def __init__(self, past_len: int, in_dim: int = 6, d_model: int = 256,
                 nhead: int = 2, num_layers: int = 2):
        super().__init__()
        self.encode_past = nn.Linear(past_len * in_dim, d_model, bias=False)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                            dim_feedforward=d_model)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        h   : (n_agents_in_batch, T, in_dim)
        mask: additive attention mask, shape (n_agents, n_agents), with
              -inf between agents from different scenes and 0 within the
              same scene (see data_preprocess in the training script).
        """
        h_feat = self.encode_past(h.reshape(h.size(0), -1)).unsqueeze(1)  # (N, 1, d_model)
        h_feat_attn = self.transformer_encoder(h_feat, mask)
        return h_feat + h_feat_attn
