"""
DiffTrajectory: reimplementation of

    Li, Gong, Xu, Wang. "DiffTrajectory: Mitigating cumulative errors and
    enhancing inference efficiency in diffusion-based trajectory prediction."
    Pattern Recognition 172 (2026) 112339.

This package is organized dataset-agnostic-first:

    difftrajectory/ode/        RK4 / Heun / Euler ODE steppers + ADSS
                                (Sections 3.2, 4.1, 4.3 -- fully specified,
                                 no dependency on any particular dataset)
    difftrajectory/models/     LIM (leap initializer) + core denoising
                                Transformer (Section 4.2, 5.2)
    difftrajectory/diffusion.py  noise schedule + score-network adapter that
                                connects the two above into a full sampler
    difftrajectory/losses.py   training objective (Eq. 16)

Dataset-specific pieces (dataloaders, configs, training/eval scripts) live
under data/, configs/, scripts/ and are added per-dataset.
"""
