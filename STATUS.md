# USplat4D Project Status

## Objective
Implement USplat4D as a **post-processing refinement module** that takes the output
of a pretrained Shape-of-Motion (SoM) model and refines it with uncertainty-aware
graph optimization.

---

## Paper References
- **USplat4D** (ICLR 2026): `~/Yun/USplat4D/USplat4D_paper.pdf`
  - arXiv: https://arxiv.org/abs/2510.12768
  - GitHub: https://github.com/TAMU-Visual-AI/usplat4d (repo says "Coming Soon" — no code yet)
- **Shape-of-Motion (SoM)** (ICCV 2025): `~/Yun/USplat4D/SoM_paper.pdf`
  - arXiv: https://arxiv.org/abs/2407.13764
  - Code: `~/Yun/shape-of-motion/`

---

## SoM Output Format (verified from source code)

**Checkpoint path:** `{work_dir}/checkpoints/last.ckpt`

**Key data structures** (from `flow3d/params.py`, `flow3d/scene_model.py`):

| Name | Shape | Description |
|------|-------|-------------|
| `fg.means` | (G, 3) | Canonical 3D positions |
| `fg.quats` | (G, 4) | Canonical quaternion rotations (wxyz) |
| `fg.scales` | (G, 3) | Log-scales |
| `fg.colors` | (G, 3) | Pre-sigmoid RGB |
| `fg.opacities` | (G,) | Pre-sigmoid opacities |
| `fg.motion_coefs` | (G, K) | Pre-softmax motion basis coefficients |
| `motion_bases.rots` | (K, T, 6) | Rotation bases (6D continuous) |
| `motion_bases.transls` | (K, T, 3) | Translation bases |
| `bg.means` | (Bg, 3) | Background Gaussian positions |
| `Ks` | (T, 3, 3) | Camera intrinsics per frame |
| `w2cs` | (T, 4, 4) | World-to-camera extrinsics per frame |

**How to get Gaussian position at time t** (from `scene_model.compute_poses_fg`):
```python
coefs = softmax(motion_coefs)           # (G, K)
transls = sum_k coefs[g,k] * transls[k,t]   # (G, 3)
rots    = sum_k coefs[g,k] * rots[k,t]      # (G, 6)
R = cont_6d_to_rmat(rots)               # (G, 3, 3)
means_t[g] = R[g] @ means[g] + transls[g]   # (G, 3)
```

---

## USplat4D Method Summary (from paper)

### 4.1 Dynamic Uncertainty Estimation
- Scalar uncertainty of Gaussian i at frame t:
  - `u_{i,t} = (sum_h (T_i^h * alpha_i)^2)^{-1}` — inverse sum of squared alpha-blending weights
  - If not all pixels covered by i are converged (|C_gt - C_pred|_1 < eta_c), set u_{i,t} = phi (large constant)
- Anisotropic 3D uncertainty matrix:
  - `U_{i,t} = R_wc * diag(rx*u, ry*u, rz*u) * R_wc^T`
  - Default: [rx, ry, rz] = [1, 1, 0.01] (depth axis down-weighted)
- Hyperparameters: eta_c = 0.5, phi = large constant, [rx,ry,rz] = [1,1,0.01]

### 4.2 Uncertainty-Encoded Graph Construction
- **Key node selection** (2 stages):
  1. Voxel grid: discard voxels with only high-uncertainty Gaussians; randomly pick 1 low-u Gaussian per voxel
  2. Significant period threshold (SPT=5): keep only candidates with uncertainty < threshold for ≥5 frames
  - Key ratio: top 2% of all Gaussians
- **Edge construction**:
  - Key-key: UA-kNN using Mahalanobis distance `||p_i - p_j||_{U_i + U_j}` at best frame t_hat = argmin_t u_{i,t}
  - Non-key to key: assign to closest key node aggregated over all T frames

### 4.3 Uncertainty-Aware Optimization
- **Key node loss**: Mahalanobis deviation from pretrained positions + ℒ_motion (iso, rigid, rot, vel, acc)
- **Non-key node loss**: Mahalanobis deviation from pretrained + Mahalanobis deviation from DQB interpolation + ℒ_motion
- **DQB**: Dual Quaternion Blending from key neighbors
- **Total loss**: ℒ_rgb + ℒ_key + ℒ_non-key

### Appendix B — Implementation Details
- Extract `{G_i}` from SoM checkpoint; reformat to unified structure
- Normalize spatial volume for selecting key Gaussians
- SoM base: train for **400 extra epochs**, batch size 8
- First 10% and last 20% of training: disable density control & opacity reset
- Motion loss weights: lambda_iso=1, lambda_rigid=1, lambda_rot=0.01, lambda_vel=0.01, lambda_acc=0.01

---

## Data & Output Paths (SoM)

From training command:
```
python run_training.py \
  --work-dir /media/ee904/DATA1/Yun/Outputs/shape-of-motion/nvidia/Skating \
  data:nvidia \
  --data.data-dir /media/ee904/DATA1/Yun/Datasets/shape-of-motion/nvidia/Skating
```

| | Path pattern |
|---|---|
| **Datasets** | `/media/ee904/DATA1/Yun/Datasets/shape-of-motion/{dataset}/{scene}/` |
| **Outputs / checkpoints** | `/media/ee904/DATA1/Yun/Outputs/shape-of-motion/{dataset}/{scene}/` |
| **Checkpoint file** | `{work_dir}/checkpoints/last.ckpt` |
| **Train config** | `{work_dir}/cfg.yaml` |

---

## Instruction from User
> Implement USplat4D as a module that refines 3D Gaussians, using SoM output as input.
> - Always talk based on reliable source. No made-up code.
> - Doubt and verify assumptions.
> - Speak easy, concise, clear.

---

## Current Status (2026-03-18)

### ✅ Completed
- [x] Project directory created: `/home/ee904/Yun/USplat4D/`
- [x] Papers downloaded: `USplat4D_paper.pdf`, `SoM_paper.pdf`
- [x] SoM codebase explored and output format documented
- [x] USplat4D GitHub repo checked — code not yet released ("Coming Soon")
- [x] Implementation plan approved by user
- [x] All 6 modules implemented and syntax/import-checked
- [x] USplat4D training run completed on `iphone/backpack`
- [x] Rendering pipeline for USplat4D outputs validated (viewer works after prep)
- [x] Added helper script: `prepare_for_rendering.sh`
- [x] Baseline-ablation run setup validated (`--lambda-key 0 --lambda-non-key 0` CLI usage fixed)

### Module Inventory

| File | Purpose | Status |
|------|---------|--------|
| `usplat4d/__init__.py` | Package exports | ✅ Complete |
| `usplat4d/uncertainty.py` | Per-Gaussian uncertainty estimation (Section 4.1, Eq. 3/6-8) | ✅ Complete |
| `usplat4d/graph.py` | Uncertainty-encoded spatio-temporal graph (Section 4.2) | ✅ Complete |
| `usplat4d/dqb.py` | Dual Quaternion Blending — Kavan et al. 2007 (Eq. 12) | ✅ Complete |
| `usplat4d/losses.py` | All loss functions: L_iso, L_rigid, L_rot, L_vel, L_acc, L_key, L_non_key (Eqs. 11/13, S8-S13) | ✅ Complete |
| `usplat4d/trainer.py` | `USplat4DTrainer` wrapping SoM's `Trainer`; injects graph losses | ✅ Complete |
| `run_usplat4d.py` | CLI entry-point (tyro-based, mirrors SoM's `run_training.py`) | ✅ Complete |

### Bugs Found and Fixed
1. **`losses.py` / `rigidity_loss` + `rotation_loss`**: weight tensor ndim mismatch — used `unsqueeze(1).unsqueeze(-1)` where only `unsqueeze(1)` was needed to broadcast against a 3D `diff_sq` tensor.
2. **`losses.py` / `rotation_loss`**: neighbor quaternions were indexed from the full `quats_t` (B frames) instead of the `q_curr` slice (B−Δ frames), causing a shape mismatch at runtime.
3. **`trainer.py` / training loop**: batch device transfer only handled plain `Tensor` values. SoM's `train_collate_fn` keeps `target_ts`, `target_w2cs`, etc. as **lists of tensors**; those stayed on CPU, causing a device mismatch in the `einsum` inside `scene_model.render`. Fixed to also iterate over list-of-tensor batch fields.

Both bugs 1–2 were caught by smoke tests with random tensors (verified forward pass + backward pass).

### Smoke Test Results (as of 2026-03-04)
```
iso:   1.18    rigid: 1.38    rot: 0.51    vel: 4.97    acc: 7.20
motion_loss_key: 2.70   grad OK: True
All losses OK
```

---

## Known Issues & Workarounds

### USplat4D quality regression on `iphone/backpack`
**Observation:** USplat4D output is renderable, but quality is worse than SoM baseline. The backpack deformation during rotation is not improved, and overall scene quality degrades.

**Interpretation:** code path is functioning end-to-end (training + checkpoint + rendering), but optimization behavior is likely incorrect (loss design/weighting, graph construction, or uncertainty/anchor selection), not a crash/integration issue.

**Current debug direction:**
- Run A/B ablation with graph losses disabled (`--lambda-key 0 --lambda-non-key 0`) to isolate whether regression comes from graph losses.
- Compare SoM baseline vs USplat4D outputs under the same rendering setup.
- Inspect graph statistics and loss scale balance if regression persists.

### SoM iphone training — OOM kill during `save_train_videos`
**Root cause:** `save_train_videos` renders all training frames and stacks them fully in RAM before writing to disk. For long iphone sequences (722 frames) this spikes memory by ~15–20 GB, exhausting RAM + swap and triggering the OOM killer.

**Immediate fix:** disable in-training video saving by raising `--save-videos-every` to equal `--num-epochs`:
```bash
python run_training.py \
  --work-dir /media/ee904/DATA1/Yun/Outputs/shape-of-motion/iphone/spin \
  --save-videos-every 500 \
  data:iphone \
  --data.data-dir /media/ee904/DATA1/Yun/Datasets/shape-of-motion/iphone/spin
```

### Visualizing USplat4D result with `run_rendering.py`
`run_rendering.py` expects `{work_dir}/checkpoints/last.ckpt` and `{work_dir}/cfg.yaml` (with `use_2dgs` key from SoM). USplat4D writes its checkpoint to `{out_dir}/usplat4d/final.ckpt` and its own `cfg.yaml` (different schema). Prep steps before running the viewer:

```bash
OUT=/media/ee904/DATA1/Yun/Outputs/usplat4d/iphone/backpack  # example
SOM=/media/ee904/DATA1/Yun/Outputs/shape-of-motion/iphone/backpack

mkdir -p $OUT/checkpoints
ln -sf $OUT/usplat4d/final.ckpt $OUT/checkpoints/last.ckpt
cp $SOM/cfg.yaml $OUT/cfg.yaml   # use SoM's cfg.yaml for the use_2dgs flag

cd /home/ee904/Yun/shape-of-motion
python run_rendering.py --work-dir $OUT --port 8890
# then open http://localhost:8890
```

### Rendering prep helper script
To avoid repeating manual prep commands for each new USplat4D output folder:

```bash
cd /home/ee904/Yun/USplat4D
./prepare_for_rendering.sh <usplat4d_out_dir> <som_dir>
```

Example:

```bash
./prepare_for_rendering.sh \
  /media/ee904/DATA1/Yun/Outputs/usplat4d/iphone/backpack_graph_check \
  /media/ee904/DATA1/Yun/Outputs/shape-of-motion/iphone/backpack
```

---

## Next Steps

### 🔲 Quantitative Evaluation
- Render with final USplat4D checkpoint using SoM's existing `Validator`
- Compare PSNR / SSIM / LPIPS against SoM baseline on iphone/backpack
