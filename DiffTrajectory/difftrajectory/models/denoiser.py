"""
Core denoising network: this is eps_theta(x, t) from paper Eq. (2)-(3), the
noise-prediction network whose recursive evaluation is what RK4/ADSS are
trying to call as few times as possible.

Architecture mirrors LED's TransformerDenoisingModel (models/model_diffusion.py
in MediaBrain-SJTU/LED), which DiffTrajectory reuses as-is -- the paper's only
changes are *how* this network gets called (RK4 + ADSS instead of fixed-step
DDIM) and *what* initializes the chain (LIM instead of LED's leapfrog
initializer), not the network's own architecture. Section 5.2 confirms this:
it specifies LIM's decoder dimensions in detail but only ever says "the
hidden layer size of the core denoising module was also set to 256" --
i.e. it's treated as an inherited/frozen backbone, not a novel contribution.

Two forward paths are provided, matching LED:
  * forward()             : one (context, x) pair per call -- used for
                             standard training of this backbone via the
                             denoising score-matching loss (Eq. 3).
  * generate_accelerate()  : K samples processed in parallel per scene,
                             batched into a single forward pass -- used
                             during RK4/ADSS sampling so that evaluating
                             the network for e.g. K=20 candidate future
                             trajectories costs one batched call, not 20
                             sequential ones.
"""
import torch
import torch.nn as nn

from .layers import PositionalEncoding, ConcatSquashLinear, SocialTransformer


class TransformerDenoisingModel(nn.Module):
    def __init__(self, past_len: int = 10, motion_dim: int = 2,
                 context_dim: int = 256, tf_layer: int = 2, future_len: int = 20):
        super().__init__()
        self.encoder_context = SocialTransformer(past_len=past_len, in_dim=6,
                                                   d_model=context_dim)
        self.pos_emb = PositionalEncoding(d_model=2 * context_dim, dropout=0.1,
                                           max_len=future_len + 4)
        # +3 context channels = [beta, sin(beta), cos(beta)] time embedding
        self.concat1 = ConcatSquashLinear(motion_dim, 2 * context_dim, context_dim + 3)
        layer = nn.TransformerEncoderLayer(d_model=2 * context_dim, nhead=2,
                                            dim_feedforward=2 * context_dim)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=tf_layer)
        self.concat3 = ConcatSquashLinear(2 * context_dim, context_dim, context_dim + 3)
        self.concat4 = ConcatSquashLinear(context_dim, context_dim // 2, context_dim + 3)
        self.linear = ConcatSquashLinear(context_dim // 2, motion_dim, context_dim + 3)

    def _time_context_embedding(self, beta: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        beta = beta.view(beta.size(0), 1, 1)
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B,1,3)
        return torch.cat([time_emb, context], dim=-1)                          # (B,1,F+3)

    def forward(self, x: torch.Tensor, beta: torch.Tensor, context: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, T_future, motion_dim) -- the noisy trajectory y_t
        beta    : (B,) or (B,1) -- the diffusion beta_t for this noise level
        context : (B, T_past, 6) -- augmented past trajectory (abs, rel, vel)
        mask    : additive attention mask across agents, see SocialTransformer

        Returns eps_theta(x, t): predicted noise, same shape as x.
        """
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
        ctx_feat = self.encoder_context(context, mask)
        ctx_emb = self._time_context_embedding(beta, ctx_feat)

        h = self.concat1(ctx_emb, x)
        h = h.permute(1, 0, 2)
        h = self.pos_emb(h)
        h = self.transformer_encoder(h).permute(1, 0, 2)
        h = self.concat3(ctx_emb, h)
        h = self.concat4(ctx_emb, h)
        return self.linear(ctx_emb, h)

    def generate_accelerate(self, x: torch.Tensor, beta: torch.Tensor,
                             context: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Batched-K variant: x is (B, K, T_future, motion_dim) -- K candidate
        futures per scene, evaluated in one forward pass. Used inside the
        RK4/ADSS sampler so a "step" costs one network call regardless of K.
        """
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
        ctx_feat = self.encoder_context(context, mask)
        ctx_emb = self._time_context_embedding(beta, ctx_feat)

        K, T = x.size(1), x.size(2)
        ctx_emb_rep = ctx_emb.repeat(1, K, 1).unsqueeze(2)             # (B,K,1,F+3)
        h = self.concat1.batch_generate(ctx_emb_rep, x)                 # (B,K,T,2*ctx)
        h = h.contiguous().view(-1, T, h.size(-1))                      # (B*K,T,2*ctx)
        h = h.permute(1, 0, 2)
        h = self.pos_emb(h)
        h = self.transformer_encoder(h).permute(1, 0, 2)
        h = h.contiguous().view(x.size(0), K, T, -1)                    # (B,K,T,2*ctx)
        h = self.concat3.batch_generate(ctx_emb_rep, h)
        h = self.concat4.batch_generate(ctx_emb_rep, h)
        return self.linear.batch_generate(ctx_emb_rep, h)
