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
4. **`losses.py` / neighbor index semantics** (2026-03-18): Non-key neighbors are KEY nodes, but old code used key-space indices to index non-key tensors. **FIXED**: Added optional neighbor array parameters to `isometry_loss()`, `rigidity_loss()`, `rotation_loss()` and updated call chain to pass correct key node arrays.
5. **`run_usplat4d.py` / checkpoint corruption** (2026-03-18): Line 157 set `work_dir=cfg.som_dir`, causing USplat4D training to overwrite the SoM checkpoint. **FIXED**: Changed to `work_dir=cfg.out_dir` to preserve SoM checkpoint independence.

All bugs 1–4 were caught by smoke tests or static code analysis; bug 5 discovered during multi-run testing.

### Smoke Test Results (as of 2026-03-04)
```
iso:   1.18    rigid: 1.38    rot: 0.51    vel: 4.97    acc: 7.20
motion_loss_key: 2.70   grad OK: True
All losses OK
```

---

## Known Issues & Workarounds

### ✅ RESOLVED (2026-03-19): USplat4D quality regression on `iphone/backpack` — root cause identified & fixed

**Initial Observation:** USplat4D output is renderable but degrades quality; deformation during backpack rotation is not improved.

**Deep Dive Diagnosis:**
After adding detailed telemetry to `uncertainty.py`, we found:
- **Median uncertainty = 1e+06** (`phi`)
- **Median color error = 0.53** (massive for a `[0, 1]` image)
- Due to the high error, **~60% of Key Nodes** were falsely classified as "unconverged" and had their graph rigidity constraints disabled, leading to severe deformation tearing.

**The Real Root Cause:**
In `uncertainty.py`, the convergence check compared the ground truth `gt_img` against a temporarily rendered image padded with `colors=torch.zeros(G, 3)`. The model was effectively rendering **pitch-black silhouettes** of the tracked objects and penalizing them against full RGB ground truth. This guaranteed massive color errors, breaking the threshold `|C_gt - C_pred|_1 < eta_c` completely regardless of true performance. 

**The Fix:**
Patched `uncertainty.py` to render the actual scene using `model.render(...)` including both background and valid RGB foreground.
- **After Fix:** Median color error dropped to **0.17**. Median uncertainty is **6.09**.
- The default `eta_c = 0.5` works perfectly again. Workarounds like `--eta-c 5.0` are no longer needed.



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

### Rendering USplat4D results
Simple one-command rendering (no manual prep needed):

```bash
python render_usplat4d.py \
  --work-dir /media/ee904/DATA1/Yun/Outputs/usplat4d/iphone/backpack_debug_eta \
  --port 8890
```

Then open `http://localhost:8890` in your browser.

**`render_usplat4d.py` automatically:**
- Finds USplat4D's `final.ckpt`
- Reads `cfg.yaml` to extract SoM directory
- Sets up symlinks and config files
- Launches the SoM renderer

No more manual prep needed! (Legacy prep steps removed.)

---

## Debug Tools & Process (Added 2026-03-18)

### Diagnostic Logging in `trainer.py`
- Added `_first_batch_logged` flag to log statistics on first training batch
- On `global_step==0`, prints:
  - `l_key` and `l_non_key` loss values
  - `pos_dev_key` / `pos_dev_nk`: mean position deviation from pretrained
  - Uncertainty shape and range for key/non-key nodes
- Helps quickly identify:
  - If losses are non-zero (constraints are active)
  - If deviations are non-trivial (model has room to improve)
  - If uncertainty distribution is sane

### Quick Debug Commands
**Test with relaxed convergence threshold:**
```bash
python run_usplat4d.py --som-dir ... --out-dir ... --eta-c 5.0 --extra-epochs 5 data:iphone ...
```

**Test ablations:**
```bash
# Key loss only
python run_usplat4d.py ... --lambda-key 1 --lambda-non-key 0 ...

# Non-key loss only
python run_usplat4d.py ... --lambda-key 0 --lambda-non-key 1 ...

# No graph losses (baseline)
python run_usplat4d.py ... --lambda-key 0 --lambda-non-key 0 ...
```

### Guiding Principles
- **Don't blindly tune hyperparameters.** Observe the output, analyze the intermediate data (losses, gradients, uncertainty), and form a hypothesis about the root cause before changing parameters. Find the problem, don't just tweak the knobs.

### Enhanced Diagnostic Logging (2026-03-20)
Added comprehensive training diagnostics to identify optimization issues and prevent endless hyperparameter tweaking:

**Features:**
- **CSV diagnostic log** written to `{work_dir}/usplat4d/diagnostics.csv`
- **Tracked metrics:**
  - *Loss breakdown:* Individual motion components (iso, rigid, rot, vel, acc) for both key and non-key nodes
  - *Uncertainty stats:* Min/median/max for all Gaussians and per-node-type (key/non-key), unconverged count
  - *Position deviations:* Mean/max distance from pretrained positions (key/non-key)
- **Logging frequency:** Every N steps (default 10) during training

**Analysis of backpack_log_report run (2026-03-24):**
🔴 **CRITICAL FINDING (Now Updated 2026-04-02):** 
- The convergence check works correctly - only 5.7% fail convergence (not 100%)
- Color error median: 0.1656 (well below eta_c=0.5)
- **Actual bug: Key node selection is picking MOSTLY UNCONVERGED Gaussians**

**Root Cause - Graph Building Bug:**
The key node selection algorithm uses voxelization + SPT (significant period) filtering, then ranks by average uncertainty (ascending = lowest first). However:
- Tau threshold (20th percentile) set too low, filtering out good candidates
- Voxel gridization selects only 1 candidate per voxel with any low-u Gaussian
- SPT filter (5 frames minimum) may over-filter reliable nodes
- Result: By the time we select top 2%, we're picking from a depleted pool dominated by HIGH-u Gaussians

**Evidence from fresh 1-epoch run (2026-04-02):**
- Final u==phi ratio: 38.1% (across all frames)
- Convergence failure rate: 5.7% (acceptable)
- But u_key_nodes: [min=6.96e-05, median=1.00e+06, max=1.00e+06] ← **BUG: median is phi!**

**Solution Needed:**
Adjust graph building parameters (tau_percentile, spt_threshold) or rethink key node selection to avoid picking unconverged Gaussians as key nodes.

---

## Deep Dive: Key Node Selection Bug Investigation (2026-04-02)

### Investigation Technique: "Don't Blindly Tune"
User requested: **Before changing code, check what the paper says about the algorithm and understand what tau means.**

### What is `tau`?
**Definition:** `tau` = uncertainty threshold that separates "reliable" from "unreliable" Gaussians
- Computed as: `tau = quantile(all_uncertainties, u_tau_percentile)`
- Current implementation: `u_tau_percentile = 0.20` (20th percentile)
- This means: only bottom 20% of Gaussians qualify as "low-u", rest are "high-u"

**How tau is used:**
1. Voxelization stage: voxels containing ONLY high-u Gaussians (all u ≥ tau) are discarded
2. SPT filtering: counts frames where u < tau (must have ≥5 frames to survive)
3. Final ranking: selects by lowest average uncertainty

### Paper Quote on Graph Construction
From Section 4.2 of USplat4D paper:
> "For each Gaussian candidate, we compute its significant period, defined as the number of frames where its uncertainty stays **below a threshold**. We retain only those with a significant period of at least 5 frames."

**Key finding:** Paper does NOT specify what the threshold should be — it's left as an implementation choice.

### Test 1: Change tau from 20th to 50th percentile

**Before (tau = 0.0236, 20th percentile):**
- Candidates after voxelization: 449
- Candidate pool: median mean_u = 5.33e+05 (very high)
- Final u_key_nodes median: **1e6 (phi)** ← BUG

**After (tau = 5.41, 50th percentile):**
- Candidates after voxelization: 513 (slightly more)
- Candidate pool: median mean_u = 5.39e+05 (still very high!)
- Final u_key_nodes median: **Still 1e6 (phi)** ← BUG PERSISTS

**Conclusion:** Changing tau percentile alone doesn't solve it. Problem is in the ranking algorithm, not the threshold.

### Improved Diagnostics: Understanding Key Node Quality

**Added to trainer.py** (2026-04-02):
- Per-key-node best-frame uncertainty: min across all T frames
- Per-key-node average uncertainty: mean across all T frames  
- Ratio: percentage of (key_node, frame) pairs with u < phi

**Output from test with tau=50th percentile:**
```
u_key (best frame):   [6.79e-05, 1.66e-02, 5.31e+00]     ← best frame of each key node
u_key (avg time):     [1.67e+04, 5.28e+05, 9.72e+05]     ← average across all frames
{key_frames < phi:    48.2%}                             ← only 48% unconverged, 52% are phi
```

**Interpretation:**
- ✅ Good: Each key node HAS at least one good frame (best = 0.0166)
- ❌ Bad: But on average they're still very uncertain (5.28e+05)
- ❌ Bad: **51.8% of all key-node-frame pairs are unconverged (u == phi)**

### Root Cause: Ranking by Average Uncertainty

**Current algorithm in graph.py line 182:**
```python
mean_u = u_scalar[candidates].mean(dim=1)  # Average over ALL frames  
top = mean_u.topk(max_key, largest=False).indices  # Pick lowest average
```

**The problem:** A Gaussian that fails convergence in even 50% of frames will have average u heavily pulled toward phi (1e6), making it uncompetitive for key node selection despite being good in the other 50% of frames.

**Calculated ratio (48.2% < phi):**
- N_k = 505 key nodes
- T ≈ 400 frames
- Total pairs = 202,000
- Only 48.2% have u < phi = 97,464 pairs
- The other 104,536 pairs are phi (unconverged)

### Recommendation for Next Session

**Option A: Rank by "best frame" instead of average**
- Current: `mean_u = u_scalar[candidates].mean(dim=1)`
- Try: `best_u = u_scalar[candidates].min(dim=1).values`
- Rationale: Prioritize Gaussians that are reliable in at least their best frame

**Option B: Filter by convergence consistency**
- Require: ≥ 80–90% of frames have u < phi (not just "u < tau")
- Rationale: Key nodes should be consistently converged, not flaky

**Option C: Both - rank by best-frame AND filter by consistency**
- Would ensure key nodes are both reliably observed AND frequently converged

---


### ✅ Completed: Critical bugs fixed (2026-03-18)
1. **Secondary index bug** ✓ — Key nodes now correctly indexed in motion losses
2. **Checkpoint corruption bug** ✓ — SoM checkpoint no longer overwritten by USplat4D training
3. **Rendering prep simplified** ✓ — New `render_usplat4d.py` eliminates manual setup

### 🔲 Immediate Action: Run standard USplat4D full pipeline
With the color rendering bug fixed, run the full 400-epoch training loop using default parameters:
```bash
python run_usplat4d.py \
  --som-dir /media/ee904/DATA1/Yun/Outputs/shape-of-motion/iphone/backpack \
  --out-dir /media/ee904/DATA1/Yun/Outputs/usplat4d/iphone/backpack_fixed \
  --lambda-key 1 --lambda-non-key 1 --extra-epochs 400 \
  data:iphone \
  --data.data-dir /media/ee904/DATA1/Yun/Datasets/shape-of-motion/iphone/backpack
```

Once completed, **Render and compare** quality vs SoM baseline:
```bash
python render_usplat4d.py --work-dir /media/ee904/DATA1/Yun/Outputs/usplat4d/iphone/backpack_fixed
```

### 🔲 After pipeline validation
- Assess visual fidelity of the backpack sequence (verify the topology tearing is resolved)
- Test on other datasets (nvidia, davis) for robustness
- Quantitative metrics: PSNR/SSIM/LPIPS once quality baseline established
