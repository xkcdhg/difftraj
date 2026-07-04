# DiffTrajectory reimplementation -- plan & status

Source: Li, Gong, Xu, Wang. "DiffTrajectory: Mitigating cumulative errors and
enhancing inference efficiency in diffusion-based trajectory prediction."
Pattern Recognition 172 (2026) 112339.

## 1. What the paper actually adds on top of prior work

DiffTrajectory extends **LED** (Leapfrog Diffusion Model, Mao et al., CVPR'23,
[github.com/MediaBrain-SJTU/LED](https://github.com/MediaBrain-SJTU/LED)) with
three pieces, and reuses LED's backbone (data pipeline, core denoising
Transformer, overall two-stage training recipe) otherwise. We know this with
high confidence, not just because the paper says "Unlike LED [29], we not
only directly predict..." (Sec 4.2), but because **LED's own reported NBA
numbers (ADE/FDE 0.18/0.27 -> 0.81/1.10 across 1-4s) match the "LED" baseline
column in DiffTrajectory's Table 3 almost to the decimal** -- i.e. the same
codebase produced both.

| Component | Paper section | Status |
|---|---|---|
| RK4 ODE solver | 4.1, Eq 9-12, Table 1 | **Built + tested** |
| ADSS (adaptive step size) | 4.3, Algorithm 1 | **Built + tested** |
| LIM (leap initializer) | 4.2, Eq 13-15 | **Built + shape-tested** |
| Core denoising Transformer | inherited from LED | **Built + shape-tested** |
| Training loss | 4.4, Eq 16 | **Built** |
| Data pipeline (any dataset) | 5.1-5.2 | **Not started** -- see Sec 4 below |
| Core-denoiser pretraining | implied, not detailed | **Not started** |
| Two-stage training script | 4.4 | **Not started** |
| Eval (ADE/FDE) script | 5.4 | **Not started** |

## 2. Code layout

```
DiffTrajectory/
  difftrajectory/
    ode/
      solvers.py    Euler / Heun / RK4 fixed-step integrators, framework-
                     agnostic (works on numpy floats or torch tensors)
      adss.py        Algorithm 1: embedded Euler/Heun error estimate ->
                     accept/reject -> grow/shrink step, RK4 as the actual
                     accepted step
    models/
      layers.py       shared blocks: TrajectoryEncoder (1D-Conv+GRU, Eq 14),
                       SocialTransformer (Eq 13), MLP, ConcatSquashLinear,
                       PositionalEncoding
      denoiser.py      TransformerDenoisingModel = eps_theta(x,t) (Eq 2-3)
      lim.py            LeapInitializerModule (Eq 13-15) + prior_regularization
                        + reparameterize
    diffusion.py       NoiseSchedule (discrete + continuous-time beta/alpha_bar)
                       + ProbabilityFlowDrift = F(x,t) (Eq 9), the adapter
                       ode/solvers.py and ode/adss.py call
    losses.py          Eq 16: best-of-K distance + uncertainty + LIM's prior reg
  tests/
    test_ode_solvers.py    pure-numpy: verifies Euler/Heun/RK4 hit their
                            theoretical O(h)/O(h^2)/O(h^4) convergence orders
    test_adss.py            verifies ADSS terminates, adapts to problem
                            stiffness, and respects step-size bounds
    test_integration.py      full LIM -> denoiser -> RK4/ADSS -> loss ->
                            backward wiring check on dummy NBA-shaped tensors
  configs/, scripts/, difftrajectory/data/   empty -- per-dataset, see Sec 4
```

All tests pass in this sandbox right now (CPU, Python 3.12, torch 2.12):
```
$ python3 tests/test_ode_solvers.py     # RK4 empirical order = 4.01-4.06 across 4 halvings
$ python3 tests/test_adss.py            # terminates, adapts step count to stiffness
$ python3 tests/test_integration.py     # 67/67 LIM params get gradients; frozen
                                         # denoiser correctly gets none
```

## 3. Judgment calls made where the paper is ambiguous or internally inconsistent

The PDF has several typos/inconsistencies (a duplicated word in Sec 2.1,
`lambda` used in Algorithm 1 where prose clearly means `xi`, an OCR-mangled
argument in Algorithm 1 step 5). Where the *math* itself is genuinely
ambiguous, rather than silently picking an interpretation, each one is
flagged in code comments at the relevant spot and summarized here. **These
are exactly the things worth double-checking once real training against a
real dataset is possible**, since right now nothing here can be validated
against the paper's actual reported numbers.

1. **RK4 sign convention (Eq 10-11)**: literally typeset, the intermediate
   stages step forward (`x + k/2`, `t + delta/2`) but the final combination
   steps backward (`x_tau - (...)`.) That's not a self-consistent RK4
   recurrence for backward integration. `ode/solvers.py` implements the
   standard, self-consistent backward-stepping recurrence instead (verified
   via the forward-then-backward round-trip test).
2. **Score/noise-prediction convention (Eq 2)**: the paper substitutes
   `eps_theta` directly for the score term with no `1/sigma(t)` rescaling,
   where the textbook VP-SDE connection (Song et al., their own ref [7])
   usually includes one. `diffusion.py`'s `ProbabilityFlowDrift` implements
   the literal paper equation by default, with `use_sigma_rescaling=True`
   available as the textbook alternative.
3. **ADSS Algorithm 1 box vs. prose**: implemented the prose (self-consistent,
   standard embedded predictor-corrector step control), not the typeset
   pseudocode (uses an undefined symbol and a garbled formula). Detailed in
   `ode/adss.py`'s module docstring.
4. **LIM's `mu_prior` / `sigma_prior^2`** ("calculated from real multi
   trajectory samples", Sec 4.2): implemented as the empirical mean/variance
   of ground-truth future trajectories *within the current training batch*.
   Reasonable, but the paper doesn't specify the reference set precisely
   (could instead mean: global training-set statistics, or statistics over
   the K samples for a single scene). Easy to swap in `lim.py`'s
   `prior_regularization`.
5. **Whether ADSS is used during training at all**: found empirically, not
   just inferred from text -- differentiating through the full variable-length
   ADSS loop caused an autograd-graph blowup (OOM) in `test_integration.py`'s
   first draft. LED's own released training code only ever backprops through
   a small **fixed** number of steps (`NUM_Tau = 5`). We now train through a
   fixed small step count and treat ADSS's adaptive behavior as an
   **inference-time-only** optimization, matching both LED's precedent and
   the paper's own framing ("substantially shortening *inference* time").
6. **Mean-decoder output dims (Sec 5.2)**: text says "the output layer
   produces a 240-dimensional vector corresponding to 6 future steps with 40
   features per step," which doesn't square with NBA's stated 20-frame,
   2D (x,y) future (=40-dim, not 240). LED's actual code (which we mirror)
   uses `t_f * d_f` generically -- 40 for NBA. Flagging the paper's number as
   likely a copy-paste artifact from a different config, not implementing it
   literally.

## 4. Data availability -- the real bottleneck

This sandbox's network is allow-listed to a small set of domains (GitHub,
PyPI, a few package registries -- no Google Drive, no direct dataset mirrors).
That matters a lot here because of how each dataset is actually distributed:

| Dataset | Used in paper's | Obtainable from this sandbox? |
|---|---|---|
| **ETH/UCY** | Table 2 | **Yes.** Raw, already train/val/test-split, bundled directly inside `github.com/Gutianpei/MID` (`raw_data/`, ~23MB, standard `frame,id,x,y` format). Confirmed by cloning it here. |
| **SDD** (Stanford Drone) | Table 5 | **Yes**, same MID repo, pre-packaged as `train_trajnet.pkl`/`test_trajnet.pkl`. |
| **NBA SportVU** | Table 3, 6-10 (most of the paper's ablations) | **No.** LED's own README points to a Google Drive folder for the preprocessed `.npy` files; not reachable from here. |
| **NFL** | Table 4 | **No**, same issue -- no public direct-download mirror found on GitHub. |

So: I can fetch, preprocess, and smoke-test against **real ETH/UCY (and SDD)
data right here**, on CPU, right now. NBA and NFL -- which is where most of
the paper's headline numbers and every ablation table live -- need data
supplied from your side (either the Google Drive links from LED's README, or
whatever source you already have) and will need to run on your GPU regardless
(this sandbox has 1 CPU core / ~4GB RAM / no GPU, so real training was never
going to happen here either way).

## 5. Recommended path forward

1. Pick a first dataset to wire up end-to-end (dataloader + config +
   core-denoiser pretraining + two-stage LIM/ADSS training + ADE/FDE eval).
2. If it's ETH/UCY or SDD: I can build *and run* a small-scale version of
   the whole pipeline right here to shake out bugs before you spend GPU time.
3. If it's NBA (where the paper's main results live) or NFL: I build the
   complete, ready-to-run code here (can't execute it without the data), you
   run it on your machine.
4. Either way, the RK4/ADSS/LIM core built so far is dataset-agnostic and
   carries over unchanged -- only the dataloader + config + the "augmented
   input" feature construction (abs/rel/velocity, Sec 4's `data_preprocess`)
   are dataset-specific.
5. Once real training is possible, revisit the six judgment calls in Sec 3 --
   these are the most likely spots for a real discrepancy against the paper's
   reported numbers, and each is a small, isolated change if a different
   interpretation turns out to be correct.
