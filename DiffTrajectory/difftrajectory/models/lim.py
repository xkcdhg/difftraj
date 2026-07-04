"""
LIM: Leap Initializer Module (paper Section 4.2, Eq. 13-15, Fig. 3).

Purpose: instead of starting the reverse diffusion process from pure noise
at t=Gamma (=100), LIM directly *predicts* a distribution (mean mu_tau,
log-variance, and a set of K normalized sample offsets S_tau) for the state
at an intermediate step tau (=10), so RK4/ADSS only need to denoise the
remaining tau steps instead of all Gamma of them. This is what LED calls
the "leapfrog initializer"; the paper's LIM keeps LED's architecture (which
is why the two share the exact same encoder/decoder shapes, Section 5.2)
and adds one thing LED didn't have: a *regularization* pulling (mu, sigma^2)
toward priors (mu_prior, sigma^2_prior) estimated from real data, so the
predicted distribution can't drift into a degenerate/overconfident or
wildly-uncertain shape:

    L_reg = lambda_1 * || mu - mu_prior ||^2 + lambda_2 * || sigma^2 - sigma_prior^2 ||^2

Architecture (Section 5.2, exact dims):
    social_encoder      : Transformer, 2 heads, 2 layers, d_model=256
    ego_{var,mean,scale}_encoder : three independent TrajectoryEncoder
                            (1D-Conv k=3 -> 32ch -> GRU hidden=256) instances
    scale_encoder        : small MLP mapping the scalar scale guess -> 32-d
    variance_decoder     : MLP(544 -> 1024 -> 1024 -> K*T_f*d_f), ReLU
    mean_decoder          : MLP(512 -> 256 -> 128 -> T_f*d_f), ReLU
    scale_decoder         : MLP(512 -> 256 -> 128 -> 1), ReLU

Final reparameterization (paper, end of Section 4.2):
    X_tau = mu_tau + sigma_tau * S_tau
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .layers import MLP, SocialTransformer, TrajectoryEncoder


class LeapInitializerModule(nn.Module):
    def __init__(self, t_h: int = 10, d_h: int = 6, t_f: int = 20, d_f: int = 2,
                 k_pred: int = 20, context_dim: int = 256, scale_embed_dim: int = 32):
        """
        Parameters
        ----------
        t_h : history length (frames)
        d_h : per-frame history feature dim (paper's "augmented input":
              absolute pos (2) + relative pos (2) + velocity (2) = 6)
        t_f : future length to predict (frames)
        d_f : per-frame future dim (2 for x,y)
        k_pred : number of stochastic samples K (best-of-K evaluation)
        """
        super().__init__()
        self.k_pred = k_pred
        self.t_f = t_f
        self.d_f = d_f

        self.social_encoder = SocialTransformer(past_len=t_h, in_dim=d_h, d_model=context_dim)
        self.ego_var_encoder = TrajectoryEncoder(in_channels=d_h, hidden_dim=context_dim)
        self.ego_mean_encoder = TrajectoryEncoder(in_channels=d_h, hidden_dim=context_dim)
        self.ego_scale_encoder = TrajectoryEncoder(in_channels=d_h, hidden_dim=context_dim)

        self.scale_encoder = MLP(1, scale_embed_dim, hid_feat=(4, 16), activation=nn.ReLU())

        self.var_decoder = MLP(context_dim * 2 + scale_embed_dim, k_pred * t_f * d_f,
                                hid_feat=(1024, 1024), activation=nn.ReLU())
        self.mean_decoder = MLP(context_dim * 2, t_f * d_f,
                                 hid_feat=(256, 128), activation=nn.ReLU())
        self.scale_decoder = MLP(context_dim * 2, 1,
                                  hid_feat=(256, 128), activation=nn.ReLU())

    def forward(self, past_traj: torch.Tensor, mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        past_traj : (B, t_h, d_h) augmented past trajectory
        mask      : additive-attention agent mask, see SocialTransformer

        Returns
        -------
        sample_offsets : (B, K, t_f, d_f) -- S_tau (normalized sample positions)
        mean            : (B, t_f, d_f)    -- mu_tau
        log_scale       : (B, 1)           -- a learned scalar "spread" used
                                              to derive sigma_tau (see
                                              diffusion.py for how this
                                              combines with sample_offsets:
                                              X_tau = mu_tau + sigma_tau * S_tau)
        """
        social_embed = self.social_encoder(past_traj, mask).squeeze(1)   # (B, 256)

        ego_var_embed = self.ego_var_encoder(past_traj)
        ego_mean_embed = self.ego_mean_encoder(past_traj)
        ego_scale_embed = self.ego_scale_encoder(past_traj)

        mean_total = torch.cat((ego_mean_embed, social_embed), dim=-1)
        mean = self.mean_decoder(mean_total).view(-1, self.t_f, self.d_f)

        scale_total = torch.cat((ego_scale_embed, social_embed), dim=-1)
        log_scale = self.scale_decoder(scale_total)

        scale_feat = self.scale_encoder(log_scale)
        var_total = torch.cat((ego_var_embed, social_embed, scale_feat), dim=-1)
        sample_offsets = self.var_decoder(var_total).view(
            past_traj.size(0), self.k_pred, self.t_f, self.d_f
        )
        return sample_offsets, mean, log_scale

    @staticmethod
    def prior_regularization(mean: torch.Tensor, log_scale: torch.Tensor,
                              fut_traj: torch.Tensor,
                              lambda_1: float = 0.05, lambda_2: float = 0.1
                              ) -> torch.Tensor:
        """
        Paper's smooth-weighted-loss regularizer:
            lambda_1 * || mu - mu_prior ||^2 + lambda_2 * || sigma^2 - sigma_prior^2 ||^2
        (Section 4.2 / Eq. 16's "weighted_smoothing" term; lambda_1=0.05,
        lambda_2=0.1 per Section 5.2.)

        `mu_prior`/`sigma_prior^2` are described only as "calculated from
        real multi trajectory samples" -- we take the natural reading: the
        empirical mean/variance of the *ground-truth* future trajectories
        within the current training batch, i.e. a population statistic the
        model's per-scene prediction shouldn't stray arbitrarily far from.
        This is a documented interpretation, not something stated
        unambiguously in the paper -- flagged in docs/PLAN.md for
        validation once real training is possible.

        mean      : (B, t_f, d_f)         -- predicted mu_tau
        log_scale : (B, 1)                -- predicted scalar log-scale
        fut_traj  : (B, t_f, d_f)         -- ground-truth future trajectories
                                              for this batch, used to build
                                              the empirical prior
        """
        with torch.no_grad():
            mu_prior = fut_traj.mean(dim=0, keepdim=True)               # (1, t_f, d_f)
            sigma2_prior = fut_traj.var(dim=0, unbiased=False).mean()   # scalar

        sigma2_pred = torch.exp(log_scale).pow(2).mean()  # scalar, matches sigma_tau's role
        term_mu = (mean - mu_prior).pow(2).mean()
        term_sigma = (sigma2_pred - sigma2_prior).pow(2)
        return lambda_1 * term_mu + lambda_2 * term_sigma

    @staticmethod
    def reparameterize(mean: torch.Tensor, log_scale: torch.Tensor,
                        sample_offsets: torch.Tensor) -> torch.Tensor:
        """
        X_tau = mu_tau + sigma_tau * S_tau  (end of Section 4.2), with the
        offsets additionally re-normalized the way LED does it (divide by
        their own std so `log_scale` alone controls the spread) -- this
        detail isn't spelled out in the paper's equation but is needed for
        the reparameterization to be numerically well-behaved, and is how
        LED's released code implements the same formula.
        """
        sigma = torch.exp(log_scale / 2.0)[..., None, None]                  # (B,1,1,1)
        offsets_norm = sample_offsets / sample_offsets.std(dim=1, keepdim=True).mean(
            dim=(2, 3), keepdim=True
        )
        return sigma * offsets_norm + mean[:, None]
