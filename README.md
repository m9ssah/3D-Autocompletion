# 3D-Autocompletion

Monitors, 48x48x48

## Day 1-2 Task Progress:
- [X] Repo structure **(after conversion is finished, create ModelNet40/sdf_conversion/[data_splits])**
- [X] Train/val/test split 
- [X] SDF voxelization conversion pipeline
- [X] pass data through sdf conversion pipeline
- [X] Masking function built, test-set masks pre-generated and saved, visually spot-checked
- [X] GMM/EM implemented from scratch and validated on synthetic data
- [X] Evaluation utilities (masked IoU/SDF error, timer, result recorder, render helper) built and unit-tested on dummy arrays

## Day 3-4 Task Progress:
- [X] unify k value
- [X] PCA: SVD computed via economy method, k-sweep reconstruction chart, artifacts saved
- [X] AE: SDF value range checked (truncated if needed), trained, reconstruction evaluated via the shared (all-ones-mask) utility, artifacts saved
    - [X] Encoder: a few Conv3d layers downsampling 48³ → roughly 6³ (stride-2 convs, doubling channels each step: 1→16→32→64), flattened and projected to your k=32 latent vector via a final linear layer
    - [X] Decoder: mirror this with ConvTranspose3d layers back up to 48³
    - [X] Output activation: linear, b/c regressing continuous SDF values now, not occupancy probabilities
    - [X] Loss: MSE, not BCE, for the same reason
- [X] Both looked at at least one rendered reconstruction (marching cubes) per branch, not just the numeric error

## Day 5-6 Task Progress:
- [ ] AE completion: mask a shape, optimize z through frozen decoder + single-Gaussian prior (skip full GMM for now)
- [ ] PCA closed-form completion, single-Gaussian prior version
- [ ] draft lit review + problem formulation


## Task Split:

| Days | Richard | Massah |
|---|---|---|
|July 31| **Joint:** masking function, generic GMM/EM utility (tested on synthetic codes, not real ones yet), generic evaluation utility (masked IoU/SDF error, solve-time timer, marching-cubes render helper). begin full-category batch voxelize/SDF conversion in the background using the existing scripts. | Same |
| Aug 1 | PCA fit (SVD), reconstruction eval using the shared utility | Finish/train conv autoencoder, reconstruction eval |
| Aug 2 | PCA closed-form completion (single-Gaussian case first) | AE gradient-based completion (Adam optimization through frozen decoder) |
| Aug 3 | Extend to full GMM-weighted closed-form completion; draft your Problem Formulation subsection (PCA half) and Lit Review (classical papers) whenever you have downtime | Refine/tune AE completion; draft your Problem Formulation subsection (AE half) and Lit Review (modern papers) whenever you have downtime |
| Aug 4 | **Joint** fit GMM/EM on both branches' *actual* latent codes using the shared utility from day 1-2 | Same |
| Aug 5 | **Joint** full evaluation: masked IoU/SDF error both branches, prior ablation (λ=0 vs λ>0), solve-time comparison, qualitative renders | Same |
| Aug 5 | **Joint** Results, Limitations, Conclusion sections (need metrics from both branches). Merge with earlier-drafted Intro/Lit Review/Problem Formulation | Same |
| Aug 6 | Final polish, formatting, contributions section, llm.pdf, submit | Same |
