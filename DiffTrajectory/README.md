# DiffTrajectory (reimplementation)

Reimplementation of *DiffTrajectory: Mitigating cumulative errors and
enhancing inference efficiency in diffusion-based trajectory prediction*
(Li, Gong, Xu, Wang, Pattern Recognition 172 (2026) 112339).

**Start here:** [`docs/PLAN.md`](docs/PLAN.md) -- architecture-to-code
mapping, what's built vs. not, data availability per dataset, and every
place the paper is ambiguous and what judgment call was made.

## Quickstart (verify the core, no data or GPU needed)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python3 tests/test_ode_solvers.py   # RK4/Heun/Euler convergence-order checks
python3 tests/test_adss.py          # adaptive step-size controller checks
python3 tests/test_integration.py   # LIM + denoiser + RK4/ADSS + loss, wired
                                     # together end-to-end on dummy tensors
```

## Status

Dataset-agnostic core (RK4/Heun/Euler solvers, ADSS, LIM, core denoising
Transformer, training loss) is implemented and passing its tests. Per-dataset
pieces (dataloaders, configs, training/eval scripts) are not yet built --
see `docs/PLAN.md` Section 5 for the plan.
