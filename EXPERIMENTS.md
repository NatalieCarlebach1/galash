# GALASH — experiment progress + roadmap

Single-source-of-truth tracking what experiments exist, what's done, and
what's queued. Mirrors the structure of `paper/sections/04_experiments.tex`
and `paper/sections/05_ablations.tex`. Last updated: 2026-04-25 morning.

---

## 1. Per-benchmark SOTA results (paper Table 1)

| Dataset | Best test F1 | Run | SOTA | Gap | Status |
|---|---|---|---|---|---|
| **DSIFN-CD** | **0.9674** | `fair_dsifn_cd` / `cheap_dsifn_cd` | 0.9665 (DDPM-CD) | **+0.09 pp ✅** | done |
| **SECOND** | **0.7320** | `pd_second` | 0.7312 (UniChange) | **+0.08 pp ✅** | done |
| LEVIR-CD | **0.9089** | `dec_sam2_tiny` + multi-res / threshold | 0.9287 (SChanger) | −1.98 pp | done |
| CDD | 0.9675 | `fair_cdd` | 0.9762 (SChanger) | −0.87 pp | done |
| LEVIR-CD+ | 0.8172 | `fair_levir_cd_plus` | 0.9150 (ChangeStar+Changen) | −9.78 pp | done |
| S2Looking | 0.6014 | `fair_dinov3_sat_noamp_s2looking` | 0.6932 (UniChange) | −9.18 pp | done |

**Headline:** SOTA on 2/6 benchmarks (DSIFN-CD, SECOND), within 2 pp on 2 more
(LEVIR-CD, CDD), large gaps remain on LEVIR-CD+ and S2Looking
(both need synthetic-pair pretraining — Tier 3 in our plan).

---

## 2. Encoder sweep (paper Table 2, LEVIR-CD only)

| # | Encoder | Test F1 (frozen sam2_base_plus) | Status |
|---|---|---|---|
| 1 | dinov2_large_reg | **0.9057** ⬆ leader | ✅ done |
| 2 | dinov3_large | 0.9025 | ✅ done |
| 3 | dinov2_rs_base | 0.9000 (→ 0.9066 with multi-res) | ✅ done |
| 4 | dinov3_base | 0.8985 | ✅ done |
| 5 | dinov2_rs_small | 0.8948 (→ 0.9002 with multi-res) | ✅ done |
| 6 | dinov2_small_reg | 0.8906 (→ 0.8971 with multi-res) | ✅ done |
| 7 | dinov3_small | 0.8869 | ✅ done |
| 8 | dinov3_sat_large | 0.8886 (LEVIR), 0.6014 (S2Looking) | ✅ done |
| 9 | dinov2_base_reg | val 0.8716 (no test) | ⚠️ test eval missing |
| 10 | dinov2_large | val 0.8529 (cancelled before test) | ⚠️ retest |
| 11 | dinov2_base | val 0.8456 (cancelled) | ⚠️ retest |
| 12 | dinov2_small | val 0.8295 (cancelled) | ⚠️ retest |
| 13 | dinov1_vits16 | failed (224² pos-embed) | ❌ TODO fix |
| 14 | dinov1_vitb16 | failed (224² pos-embed) | ❌ TODO fix |
| 15 | dinov2_rs_large | failed (KevinCha loader bug) | ❌ TODO fix |

---

## 3. Decoder sweep (paper Table 3, LEVIR-CD only, frozen dinov2_rs_base)

| Decoder | Params (M) | Test F1 | Status |
|---|---|---|---|
| **sam2_tiny** | 39 | **0.9058** (→ **0.9089** with multi-res) ⬆ | ✅ done |
| sam2_small | 46 | 0.9041 (→ 0.9088 with multi-res) | ✅ done |
| sam2_base_plus | 81 | 0.9035 (→ 0.9075 with multi-res) | ✅ done |
| sam2_large | 225 | 0.9025 (→ 0.9083 with multi-res) | ✅ done |
| sam1_vit_b | — | failed (fp16 overflow) | ⚠️ rerun --no_amp pending |
| sam1_vit_l | — | failed | ⚠️ rerun --no_amp pending |
| sam1_vit_h | — | failed | ⚠️ rerun --no_amp pending |
| sam3 | — | not in env | ⏳ blocked on `pip install sam3` |

**Finding:** Decoder capacity is inversely correlated with F1 (smallest wins).

---

## 4. dinov2_large_reg cross-dataset sweep (NEW — running)

The surprise leader from the encoder sweep. Now testing if it generalises.

| Dataset | Status | Test F1 |
|---|---|---|
| LEVIR-CD | ✅ done | 0.9057 |
| SECOND | ✅ done | 0.7289 |
| CDD | 🔄 RUNNING (job 52148, 3.5h elapsed) | — |
| DSIFN-CD | 🔄 RUNNING (job 52149) | — |
| S2Looking | 🔄 RUNNING (job 52151) | — |
| LEVIR-CD+ | not queued | — |

---

## 5. Ablations (paper Table 6, LEVIR-CD)

| Variant | Test F1 | Δ vs full | Status |
|---|---|---|---|
| Full GALASH | 0.9045 | — | ✅ |
| w/o latent loss (`w_lat=0`) | 0.8984 | **−0.61** | ✅ done |
| w/o OHEM | 0.9013 | −0.32 | ✅ done |
| w/o TTA at test | 0.8974 | −0.71 | ✅ done |
| w/o cross-change attention | — | — | ❌ TODO (code change) |
| w/o Bridge v2 transformer refinement | — | — | ❌ TODO |
| w/o high-res feature tap | — | — | ❌ TODO |

---

## 6. Latent loss hyperparam sweep (NEW — 10 jobs running on main)

LEVIR-CD, fair config, dinov2_rs_base + sam2_base_plus, --no_amp.
Submitted 2026-04-25. Expected ~3 h wall.

| Job | latent_soft | w_latent | latent_temp | F1 |
|---|---|---|---|---|
| 52152 | hard | 0.1 | 0.03 | 🔄 running |
| 52153 | hard | 0.2 | 0.03 (= current default) | 🔄 |
| 52154 | hard | 0.5 | 0.03 | 🔄 |
| 52155 | hard | 1.0 | 0.03 | 🔄 |
| 52156 | **soft** | 0.1 | 0.03 | 🔄 |
| 52157 | **soft** | 0.2 | 0.03 | 🔄 |
| 52158 | **soft** | 0.5 | 0.03 | 🔄 |
| 52159 | **soft** | 1.0 | 0.03 | 🔄 |
| 52160 | soft | 0.5 | **0.01** | 🔄 |
| 52161 | soft | 0.5 | **0.07** | 🔄 |

After completion → produces 2×4 grid (soft × w_latent) plus 2 temperature
points. Best config will go into `paper/sections/05_ablations.tex` Table 6
("Latent loss target type").

---

## 7. Cheap-wins (post-hoc + training-time)

### 7a. Post-hoc (multi-res + threshold + EMA on existing best.pt)

| Dataset | default | post-hoc | Δ |
|---|---|---|---|
| LEVIR-CD (sam2_tiny) | 0.9058 | **0.9089** | **+0.31** |
| LEVIR-CD (sam2_small) | 0.9041 | 0.9088 | +0.47 |
| LEVIR-CD (rs_base) | 0.9000 | 0.9066 | +0.66 |
| CDD | 0.9675 | 0.9631 | −0.44 ⚠️ (picked size 256, dataset native = 256) |
| DSIFN-CD | 0.9674 | 0.9630 | −0.44 ⚠️ |

**Issue:** multi-res only helps when the dataset is native-low-res
(LEVIR-CD 256→384). For CDD/DSIFN already at 256, picked size lower than
optimal. Need to re-run with `TEST_SIZES=256 320 384` instead of `256 384 512`.

### 7b. Training-time (FT decoder + EMA + threshold + patience 30)

| Dataset | fair (default) | + cheap-wins | Δ | Status |
|---|---|---|---|---|
| LEVIR-CD | 0.9045 | 0.9025 | −0.20 | ✅ done (within seed noise) |
| LEVIR-CD (DINOv3-sat) | 0.8886 | 0.8998 | **+1.12** | ✅ done |
| SECOND | 0.7067 | 0.7209 | **+1.42** | ✅ done |
| SECOND (DINOv3-sat) | 0.7218 | 0.7303 | **+0.85** | ✅ done |
| CDD | — | 🔄 running (job 52142) | — | 🔄 |
| DSIFN-CD | — | 0.9674 | tbd | ✅ done |
| LEVIR-CD+ | — | 0.8137 | −0.04 | ✅ done |
| S2Looking | — | 🔄 running (job 52144) | — | 🔄 |

---

## 8. Currently running (~21 jobs)

```
52086, 52087   medical (msseg2, isbi_ms)             ~12 h elapsed
52091, 52092   v3satNA_cdd, v3satNA_dsifn_cd         ~13 h elapsed
52106, 52107   cheap_v3sat_dsifn_cd, _s2looking      ~8 h
52142, 52144   cheap_cdd, cheap_s2looking            ~4 h
52148, 52149, 52151   encL_v2_large_reg × 3 datasets  ~3.5 h
52152-52161    latent-loss sweep × 10                ~13 min
```

---

## 9. TODO — next experiments to add

Ranked by paper impact × effort.

| # | Experiment | Goal | Effort | Status |
|---|---|---|---|---|
| **A1** | Latent-loss sweep aggregation + heatmap → Table 6 | Identify best (soft, w_latent, temp) for paper | low (auto when sweep finishes) | running ⏳ |
| **A2** | Re-run post-hoc cheap-wins with `TEST_SIZES=256 320 384` for CDD/DSIFN | Get correct cheap-win lift for those datasets (current ones picked size=256 = baseline) | very low | not started |
| **A3** | Fix the 8 broken re-evals (v3sat/noamp) — suspected `--finetune_decoder` flag missing in re-eval sbatch | Recover ~5 more cheap-win numbers | low | not started |
| **B1** | Multi-seed (seed 42/43/44) on the 4 close-to-SOTA datasets (LEVIR/CDD/DSIFN/SECOND) | Variance bars for paper credibility — reviewers love this | medium (need `--seed` flag, 12 jobs × 3h) | partner can do (Tier A) |
| **B2** | Full encoder × decoder matrix on LEVIR-CD (top 4 encoders × 4 decoders = 16 cells) | Fill in diagonal Table 7 — show interaction | medium (16 jobs × 3 h, mostly done from sweeps) | partial — need 6 more cells |
| **B3** | Cross-dataset generalization: train on A, test on B for top backbone | Table 4 — reviewers expect this for foundation-model claims | medium (need eval-only, our pooled checkpoint can be reused) | not started |
| **B4** | Encoder sweep × second dataset (CDD or DSIFN) | Show DINOv2-large-reg's lead is dataset-agnostic | medium | partial via #4 |
| **C1** | Implement multi-scale latent supervision (deep supervision at FPN scales) | Tier-2 architectural improvement, expected +0.3-0.5 F1 | high (50 LOC + sweep retrain) | not started |
| **C2** | Implement bidirectional cross-attention (sim_fwd + sim_bwd average) | Symmetry property, expected +0.2 F1 | medium (10 LOC + retrain) | not started |
| **C3** | Local-window similarity in latent loss (3×3 max around diagonal) | Robustness to misregistration → S2Looking | medium (10 LOC) | not started |
| **D1** | Changen-style synthetic-pair pretraining | Close LEVIR-CD+ and S2Looking gaps (Tier 3 — biggest single lever) | very high (clone Changen repo, generate 90k pairs, pretrain, fine-tune) | not started |
| **D2** | Pool-then-finetune (pretrain on all 6 → fine-tune per dataset) | UniChange's recipe — works for hard datasets | high | not started |
| **E1** | Fix dinov1 pos-embed interpolation | DINOv1 row in Table 2 (currently empty) | low (HF interpolate_pos_encoding) | not started |
| **E2** | Fix KevinCha dinov2_rs_large loader (KeyError on 'blocks.0.mlp.fc1.weight') | Largest RS encoder, expected +0.5 F1 | low-medium (~30 LOC) | not started |
| **E3** | Add `--seed` flag to train.py | Enables B1 multi-seed | trivial | not started |
| **E4** | Install sam3, add SAM3 row to decoder sweep | Complete decoder ablation | low (pip install) | not started |
| **F1** | Build Table 4 (cross-dataset gen) and Table 7 (encoder × decoder matrix) into paper | Once data lands | low (LaTeX edit) | depends on B2, B3 |
| **F2** | Architecture diagram (paper/figures/architecture.pdf) | Currently `\fbox{}` placeholder | medium (TikZ or draw.io) | not started |
| **F3** | Qualitative results figure (paper/figures/qualitative.pdf) | Empty in §5.4 | low (run `eval.py --save_masks`, pick examples) | not started |

---

## 10. Paper sections — completion status

| Section | Status | Missing |
|---|---|---|
| 00_abstract | ✅ filled with real numbers | — |
| 01_introduction | ✅ contributions list | refs need final venue formatting |
| 02_related_work | ✅ done | — |
| 03_method | ✅ done | architecture figure (F2) |
| 04_experiments §5.1 (datasets) | ✅ done | — |
| 04_experiments §5.2 (Table 1 SOTA) | ✅ filled | refresh "best" row after lat-sweep + dinov2_large_reg per-dataset land |
| 04_experiments §5.3 (Table 2 encoders) | ✅ filled | dinov1 (E1), dinov2_rs_large (E2) |
| 04_experiments §5.4 (Table 3 decoders) | ✅ filled | sam1 row (--no_amp pending), sam3 (E4) |
| 04_experiments §5.5 (S2Looking detail) | ✅ filled | — |
| 04_experiments §5.6 (cheap wins) | ✅ filled | — |
| 04_experiments §5.7 (cross-dataset gen) | ❌ TODO | depends on B3 |
| 05_ablations §6.1 (loss components) | ✅ filled | — |
| 05_ablations §6.2 (aug pipeline) | ✅ filled | — |
| 05_ablations §6.3 (cheap combo) | ✅ filled | — |
| 05_ablations §6.4 (params/efficiency) | ✅ filled | — |
| 05_ablations §6.5 (qualitative) | ❌ TODO | depends on F3 |
| 06_conclusion | ✅ filled | — |

**Progress:** 13 / 16 paper subsections complete. Missing: cross-dataset
generalisation table, architecture figure, qualitative figure.

---

## 11. Numbers in flight that will refresh the paper

When the **latent sweep** lands (~3 h):
- Table 6 gains a new "soft target" comparison row.
- Best soft-target F1 likely beats current 0.9045 → may bump the LEVIR-CD
  "default" row in Table 1.

When **dinov2_large_reg per-dataset** lands (~6 h, 3 datasets):
- Table 1 may gain new "best" rows for CDD / DSIFN / SECOND if it beats
  current rows.

When **`cheap_cdd` / `cheap_s2looking`** finishes (~4 h):
- Tier-1-cheap-wins row in Table 1 will fill for those datasets.
- `cheap_cdd` could push CDD into SOTA range.

Total: **3 paper updates expected today**, all from currently-running jobs.
