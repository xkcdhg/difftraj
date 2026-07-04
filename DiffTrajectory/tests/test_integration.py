"""
End-to-end integration smoke test on random dummy tensors, shaped like the
NBA config (t_h=10, t_f=20, d_h=6, d_f=2, k_pred=20 -- Section 5.2). This
can't validate *numerical correctness* against the paper's reported numbers
(that needs real data + real training on a GPU), but it does validate that:

  1. every module actually constructs and runs with the paper's stated
     dimensions (Section 5.2) without shape errors,
  2. the full pipeline -- LIM produces (mean, log_scale, sample_offsets) ->
     reparameterize -> ADSS+RK4 denoises the remainder -> loss (Eq. 16) ->
     backward() -- is wired together correctly and gradients actually reach
     every trainable parameter (a very common class of silent bug: a
     detached tensor or a `.item()` called too early quietly kills
     gradient flow without raising any error),
  3. the "core denoiser is frozen / only LIM trains" two-stage-training
     setup (Section 4.4) behaves as intended.

Run with: python3 tests/test_integration.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from difftrajectory.models.denoiser import TransformerDenoisingModel  # noqa: E402
from difftrajectory.models.lim import LeapInitializerModule  # noqa: E402
from difftrajectory.diffusion import NoiseSchedule, ProbabilityFlowDrift  # noqa: E402
from difftrajectory.ode.adss import ADSSConfig, adss_denoise  # noqa: E402
from difftrajectory.losses import total_loss, LossWeights  # noqa: E402

torch.manual_seed(0)

# ---- NBA-shaped config, Section 5.2 -----------------------------------
B_SCENES = 2          # scenes (mini-batch of games/clips)
N_AGENTS = 11          # NBA: 10 players + ball
T_H, D_H = 10, 6        # past frames, augmented feature dim
T_F, D_F = 20, 2        # future frames, xy
K_PRED = 20             # best-of-K
TAU = 10.0               # remaining denoising steps after the LIM's leap


def make_agent_mask(n_scenes: int, n_agents: int) -> torch.Tensor:
    """Block-diagonal mask: agents attend within their own scene only."""
    total = n_scenes * n_agents
    mask = torch.zeros(total, total)
    for i in range(n_scenes):
        mask[i * n_agents:(i + 1) * n_agents, i * n_agents:(i + 1) * n_agents] = 1.0
    return mask


def test_lim_forward_shapes():
    lim = LeapInitializerModule(t_h=T_H, d_h=D_H, t_f=T_F, d_f=D_F, k_pred=K_PRED)
    n = B_SCENES * N_AGENTS
    past = torch.randn(n, T_H, D_H)
    mask = make_agent_mask(B_SCENES, N_AGENTS)

    sample_offsets, mean, log_scale = lim(past, mask)
    assert sample_offsets.shape == (n, K_PRED, T_F, D_F)
    assert mean.shape == (n, T_F, D_F)
    assert log_scale.shape == (n, 1)
    print(f"LIM forward OK: sample_offsets={tuple(sample_offsets.shape)}, "
          f"mean={tuple(mean.shape)}, log_scale={tuple(log_scale.shape)}")

    x_tau = LeapInitializerModule.reparameterize(mean, log_scale, sample_offsets)
    assert x_tau.shape == (n, K_PRED, T_F, D_F)
    print(f"Reparameterize OK: X_tau={tuple(x_tau.shape)}")
    return lim, past, mask, x_tau


def test_denoiser_forward_shapes():
    net = TransformerDenoisingModel(past_len=T_H, motion_dim=D_F, future_len=T_F)
    n = B_SCENES * N_AGENTS
    past = torch.randn(n, T_H, D_H)
    mask = make_agent_mask(B_SCENES, N_AGENTS)

    x = torch.randn(n, T_F, D_F)
    beta = torch.rand(n)
    out = net(x, beta, past, mask)
    assert out.shape == x.shape
    print(f"Denoiser forward() OK: out={tuple(out.shape)}")

    x_batched = torch.randn(n, K_PRED, T_F, D_F)
    out_batched = net.generate_accelerate(x_batched, beta, past, mask)
    assert out_batched.shape == x_batched.shape
    print(f"Denoiser generate_accelerate() OK: out={tuple(out_batched.shape)}")
    return net


def test_adss_inference_sampling():
    """ADSS/RK4 is an *inference-time* accelerator (paper abstract:
    "substantially shortening inference time"): run it under no_grad, the
    way it's actually used at test/eval time. This safely exercises the
    full variable-length ADSS loop without growing an autograd graph across
    however many accept/reject iterations a randomly-initialized network
    happens to trigger."""
    lim = LeapInitializerModule(t_h=T_H, d_h=D_H, t_f=T_F, d_f=D_F, k_pred=K_PRED)
    net = TransformerDenoisingModel(past_len=T_H, motion_dim=D_F, future_len=T_F)
    lim.eval()
    net.eval()

    n = B_SCENES * N_AGENTS
    past = torch.randn(n, T_H, D_H)
    mask = make_agent_mask(B_SCENES, N_AGENTS)
    schedule = NoiseSchedule(steps=100, beta_start=1e-4, beta_end=5e-2)

    with torch.no_grad():
        sample_offsets, mean, log_scale = lim(past, mask)
        x_tau = LeapInitializerModule.reparameterize(mean, log_scale, sample_offsets)

        drift = ProbabilityFlowDrift(net, schedule, context=past, mask=mask, batched_k=True)
        adss_cfg = ADSSConfig(delta_init=3.0, delta_min=0.5, delta_max=10.0,
                                xi=0.05, gamma=0.6, max_total_steps=50)
        x_final, trace = adss_denoise(drift, x_tau, TAU, adss_cfg)

    assert x_final.shape == (n, K_PRED, T_F, D_F)
    assert torch.isfinite(x_final).all()
    print(f"\nInference-mode ADSS+RK4 sampling OK: {len(trace)} steps taken "
          f"(tau={TAU} -> 0) on a randomly-initialized network, "
          f"final shape={tuple(x_final.shape)}")
    return x_final


def test_full_pipeline_forward_and_backward():
    """
    LIM -> a small FIXED number of RK4 denoising steps (through the *frozen*
    core denoiser, Section 4.4 Stage 2) -> Eq. 16 loss -> backward.

    Why fixed steps and not the full adaptive ADSS loop: LED's own released
    training code (trainer/train_led_trajectory_augment_input.py) trains its
    initializer through a small constant number of steps
    (`NUM_Tau = 5`, see `p_sample_loop_accelerate`), not a variable-length
    loop -- almost certainly for exactly the reason this test file
    originally hit: differentiating through a variable-length,
    retry-until-accepted loop makes the autograd graph size unpredictable
    and, with a poorly-calibrated (here: randomly initialized) network, it
    can explode. DiffTrajectory's Section 4.4 similarly describes Stage 2
    as training only the LIM "while the remaining denoising process still
    uses the traditional denoising method" -- i.e. ADSS's *variable* step
    count reads as an inference-time optimization layered on top of a
    conventionally-trained backbone, not something literally backpropagated
    through during training. We implement that reading here; see
    docs/PLAN.md for the flag on this being an interpretation, not a
    literal statement in the paper.

    Note: this uses a much smaller B/N_AGENTS/K than the shape-check tests
    above. Retaining an autograd graph across several sequential Transformer
    forward passes (RK4's 4 evals/step) is memory-hungry, and this sandbox
    has only ~4GB RAM / 1 CPU core -- a real GPU training run has no trouble
    with this at full batch size, but the point of *this* smoke test is only
    to prove gradients flow correctly end-to-end, not to simulate realistic
    training throughput.
    """
    b_scenes, n_agents, k_pred = 1, 4, 4
    lim = LeapInitializerModule(t_h=T_H, d_h=D_H, t_f=T_F, d_f=D_F, k_pred=k_pred)
    net = TransformerDenoisingModel(past_len=T_H, motion_dim=D_F, future_len=T_F)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)   # Stage 2 of Section 4.4: denoiser frozen

    n = b_scenes * n_agents
    past = torch.randn(n, T_H, D_H)
    fut = torch.randn(n, T_F, D_F)
    mask = make_agent_mask(b_scenes, n_agents)
    schedule = NoiseSchedule(steps=100, beta_start=1e-4, beta_end=5e-2)

    sample_offsets, mean, log_scale = lim(past, mask)
    x_tau = LeapInitializerModule.reparameterize(mean, log_scale, sample_offsets)

    drift = ProbabilityFlowDrift(net, schedule, context=past, mask=mask, batched_k=True)

    # Fixed-step differentiable RK4 denoising for training (mirrors LED's
    # NUM_Tau=5 pattern): a handful of equal-sized RK4 steps from TAU to 0.
    from difftrajectory.ode.solvers import rk4_step
    n_fixed_steps = 5
    step = TAU / n_fixed_steps
    x = x_tau
    t = TAU
    for _ in range(n_fixed_steps):
        x = rk4_step(drift, x, t, -step)
        t -= step
    x_final = x

    assert x_final.shape == (n, k_pred, T_F, D_F)
    print(f"\nFixed-{n_fixed_steps}-step differentiable RK4 sampling OK: "
          f"final shape={tuple(x_final.shape)}")

    loss, components = total_loss(x_final, fut, mean, log_scale,
                                    weights=LossWeights())
    print(f"Loss components: {components}")
    assert torch.isfinite(loss), "loss is not finite -- numerical blow-up somewhere"

    loss.backward()
    grad_norms = [p.grad.norm().item() for p in lim.parameters() if p.grad is not None]
    n_params = sum(1 for _ in lim.parameters())
    print(f"Backward OK: {len(grad_norms)}/{n_params} LIM parameter tensors "
          f"received gradients; mean grad norm={sum(grad_norms)/len(grad_norms):.4e}")
    assert len(grad_norms) == n_params, "some LIM parameters got NO gradient -- broken graph"
    assert all(g == g for g in grad_norms), "NaN gradient detected"

    for p in net.parameters():
        assert p.grad is None, "denoiser should be frozen (Stage 2) but received a gradient"
    print("Confirmed: frozen denoiser received zero gradients, as required by Stage 2.")


if __name__ == "__main__":
    test_lim_forward_shapes()
    test_denoiser_forward_shapes()
    test_adss_inference_sampling()
    test_full_pipeline_forward_and_backward()
    print("\nAll integration tests passed.")
