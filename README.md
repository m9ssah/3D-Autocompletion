# 3D-Autocompletion

## Task Split:


| Days | Richard | Massah |
|---|---|---|
| 1-2 | **Joint:** masking function, generic GMM/EM utility (tested on synthetic codes, not real ones yet), generic evaluation utility (masked IoU/SDF error, solve-time timer, marching-cubes render helper). begin full-category batch voxelize/SDF conversion in the background using the existing scripts. | Same |
| 3-4 | PCA fit (SVD), reconstruction eval using the shared utility | Finish/train conv autoencoder, reconstruction eval |
| 5-6 | PCA closed-form completion (single-Gaussian case first) | AE gradient-based completion (Adam optimization through frozen decoder) |
| 7-8 | Extend to full GMM-weighted closed-form completion; draft your Problem Formulation subsection (PCA half) and Lit Review (classical papers) whenever you have downtime | Refine/tune AE completion; draft your Problem Formulation subsection (AE half) and Lit Review (modern papers) whenever you have downtime |
| 9 | **Joint** fit GMM/EM on both branches' *actual* latent codes using the shared utility from day 1-2 | Same |
| 10-11 | **Joint** full evaluation: masked IoU/SDF error both branches, prior ablation (λ=0 vs λ>0), solve-time comparison, qualitative renders | Same |
| 12-13 | **Joint** Results, Limitations, Conclusion sections (need metrics from both branches). Merge with earlier-drafted Intro/Lit Review/Problem Formulation | Same |
| 14 | Final polish, formatting, contributions section, llm.pdf, submit | Same |
