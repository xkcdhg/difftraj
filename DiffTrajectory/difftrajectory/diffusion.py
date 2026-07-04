"""
Connects the trained noise-prediction network eps_theta to the ODE solvers.

Implements:
  * the forward noising process / discrete beta schedule used to *train*
    eps_theta (paper Eq. 3), matching LED's config (linear schedule,
    Gamma=100 steps, beta in [1e-4, 5e-2] -- see configs/nba.yaml)
  * a continuous-time extension beta(t)/alpha_bar(t) of that discrete
    schedule, needed because RK4/ADSS take *fractional* step sizes (Section
    4.3's delta is not restricted to integers), so the drift function F(x,t)
    must be evaluable at non-integer t
  * ProbabilityFlowDrift: F(x, t) = f(x,t) - 0.5 g(t)^2 eps_theta(x,t)
    (Eq. 9), the callable that ode/solvers.py and ode/adss.py treat as a
    black box

A note on Eq. (2)'s score convention
-------------------------------------
Eq. (2) substitutes eps_theta directly in place of the score term
-0.5 g(t)^2 grad_x log p_t(x). The mathematically-standard connection
(Song et al. 2021, cited as ref [7]) between a *noise-prediction* network
and the score is  grad_x log p_t(x) = -eps_theta(x,t) / sigma(t)  -- i.e.
there's usually a 1/sigma(t) rescaling that Eq. (2) doesn't show. We
implement the paper's literal equation (no extra rescaling) as the default,
since that's what's actually written, but expose `use_sigma_rescaling` to
switch to the textbook VP-SDE convention -- this is exactly the kind of
detail that should be checked empirically (does the sampler actually
denoise sensibly / do the loss curves look right?) once real training is
possible, so we surface it as a constructor flag rather than a silent
choice buried in the math.
"""
import math
from dataclasses import dataclass

import torch


@dataclass
class NoiseSchedule:
    steps: int = 100          # Gamma
    beta_start: float = 1e-4
    beta_end: float = 5e-2
    schedule: str = "linear"

    def __post_init__(self):
        if self.schedule != "linear":
            raise NotImplementedError(
                "Only the linear schedule is implemented; the paper/LED "
                "config both use 'linear' for every dataset."
            )
        # discrete schedule, indices 0..steps-1, used to train eps_theta
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    # ---- discrete-time helpers (standard DDPM forward process) ----------
    def q_sample(self, y0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """
        Forward noising q(y_t | y_0) for training eps_theta (Eq. 3).
        t: (B,) integer indices in [0, steps).
        """
        if noise is None:
            noise = torch.randn_like(y0)
        a = self.sqrt_alphas_cumprod.to(y0.device)[t].view(-1, *[1] * (y0.dim() - 1))
        am1 = self.sqrt_one_minus_alphas_cumprod.to(y0.device)[t].view(-1, *[1] * (y0.dim() - 1))
        return y0 * a + noise * am1, noise

    def beta_at_index(self, t: torch.Tensor) -> torch.Tensor:
        return self.betas.to(t.device)[t]

    # ---- continuous-time extension, needed for fractional RK4/ADSS steps
    def beta(self, t: float) -> float:
        """beta(t) via linear interpolation of the discrete schedule,
        exact at integer t, smooth in between. t in [0, steps]."""
        t = min(max(t, 0.0), float(self.steps))
        return self.beta_start + (self.beta_end - self.beta_start) * (t / self.steps)

    def alpha_bar(self, t: float) -> float:
        """
        Continuous-time alpha_bar(t) = exp(-integral_0^t beta(s) ds), the
        standard VP-SDE relation (d log(alpha_bar)/dt = -beta(t)).
        Closed form since beta(t) is linear in t.
        """
        t = min(max(t, 0.0), float(self.steps))
        integral = self.beta_start * t + 0.5 * (self.beta_end - self.beta_start) * (t ** 2) / self.steps
        return math.exp(-integral)

    def sigma(self, t: float) -> float:
        return math.sqrt(max(1.0 - self.alpha_bar(t), 1e-12))


class ProbabilityFlowDrift:
    """
    F(x, t) = f(x,t) - 0.5 * g(t)^2 * eps_theta(x, t)     (Eq. 9)
    with the VP-SDE choice f(x,t) = -0.5 beta(t) x, g(t)^2 = beta(t), so
        F(x, t) = -0.5 * beta(t) * (x + eps_theta(x, t))          [default,
                                                                    literal Eq. 2]
        F(x, t) = -0.5 * beta(t) * (x + eps_theta(x, t) / sigma(t))  [if
                                                                    use_sigma_rescaling]

    This is what gets handed to `difftrajectory.ode.solvers` / `ode.adss` as
    the drift callable -- they just see a function of (x, t) and don't know
    or care that it's wrapping a neural network.
    """

    def __init__(self, network: torch.nn.Module, schedule: NoiseSchedule,
                 context: torch.Tensor, mask: torch.Tensor,
                 use_sigma_rescaling: bool = False, batched_k: bool = False):
        self.network = network
        self.schedule = schedule
        self.context = context
        self.mask = mask
        self.use_sigma_rescaling = use_sigma_rescaling
        self.batched_k = batched_k  # True -> use generate_accelerate (x is B,K,T,2)

    def __call__(self, x: torch.Tensor, t: float) -> torch.Tensor:
        beta_t = self.schedule.beta(t)
        beta_tensor = torch.full((x.size(0),), beta_t, device=x.device, dtype=x.dtype)

        if self.batched_k:
            eps = self.network.generate_accelerate(x, beta_tensor, self.context, self.mask)
        else:
            eps = self.network(x, beta_tensor, self.context, self.mask)

        if self.use_sigma_rescaling:
            eps = eps / self.schedule.sigma(t)

        return -0.5 * beta_t * (x + eps)
