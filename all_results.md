# GALASH — All Results & Comparisons

Last updated: 2026-04-29. All F1 scores are test-set unless marked `~val` (val-based preview).

---

## Comparison Policy

We exclude methods that use additional bitemporal CD data:
- **ChangeStar / ChangeStar2 / ChangeStar+Changen** (Z. Zheng's group — synthetic Changen 90k pairs)
- **UniChange** (joint training on LEVIR + S2L + SECOND + WHU)

These are marked ~~strikethrough~~ in the tables below. The "fair SOTA" column is the
best result that is directly comparable to ours (single-dataset training, no extra CD data).

---

## Published SOTA Baselines (fair comparison only)

| Dataset | Method | Year | Test F1 | Notes |
|---------|--------|------|---------|-------|
| LEVIR-CD | **SChanger** | 2025 | **0.9287** | fair SOTA |
| LEVIR-CD | ChangeFormer | 2022 | 0.9024 | |
| LEVIR-CD | BIT | 2021 | 0.8905 | |
| ~~LEVIR-CD~~ | ~~ChangeStar2~~ | ~~2024~~ | ~~0.9313~~ | ~~excluded: synthetic data~~ |
| CDD | **SChanger** | 2025 | **0.9762** | fair SOTA |
| CDD | ChangeStar2 | 2024 | 0.9750 | |
| CDD | RFL-CDNet | 2024 | 0.9612 | |
| CDD | ChangeFormer | 2022 | 0.9463 | |
| DSIFN-CD | **DDPM-CD** | 2024 | **0.9665** | fair SOTA |
| SECOND | **SAM-SCD** | 2025 | **0.7246** | fair SOTA (binary collapse) |
| ~~SECOND~~ | ~~UniChange~~ | ~~2025~~ | ~~0.7312~~ | ~~excluded: joint training~~ |
| S2Looking | **SChanger** | 2025 | **0.6895** | fair SOTA |
| ~~S2Looking~~ | ~~UniChange~~ | ~~2025~~ | ~~0.6932~~ | ~~excluded: joint training~~ |
| S2Looking | FIBTNet | 2024 | 0.6860 | |
| S2Looking | BAN-CF | 2024 | 0.6670 | |
| LEVIR-CD+ | **DDCDNet** | 2024 | **0.8771** | fair SOTA |
| ~~LEVIR-CD+~~ | ~~ChangeStar+Changen~~ | ~~2023~~ | ~~0.9150~~ | ~~excluded: synthetic data~~ |

---

## GALASH Results — Tal's Cluster Runs

Encoder: various (see column). Decoder: sam2_base_plus throughout.

### Best per-dataset (paper Table 1)

| Dataset | Config | Encoder | Test F1 | Fair SOTA | Gap | Status |
|---------|--------|---------|---------|-----------|-----|--------|
| **SECOND** | champion (cheap wins) | dinov3_large | **0.7430** | 0.7246 | **+1.84pp ✅** | SOTA |
| **DSIFN-CD** | fair | dinov2_rs_base | **0.9674** | 0.9665 | **+0.09pp ✅** | SOTA |
| CDD | fair | dinov2_rs_base | 0.9675 | 0.9762 | −0.87pp | |
| LEVIR-CD | sam2_tiny + multi-res | dinov2_rs_base | 0.9089 | 0.9287 | −1.98pp | |
| S2Looking | cheap wins | dinov3_large | 0.6571 | 0.6895 | −3.24pp | |
| LEVIR-CD+ | cheap wins | dinov3_large | 0.8320 | 0.8771 | −4.51pp | |

### Encoder sweep (LEVIR-CD, frozen sam2_base_plus)

| Encoder | Test F1 | Notes |
|---------|---------|-------|
| dinov2_large_reg | **0.9057** | best encoder on LEVIR-CD |
| dinov3_large | 0.9025 | champion for other datasets |
| dinov2_rs_base | 0.9000 → 0.9066 multi-res | baseline |
| dinov3_base | 0.8985 | |
| dinov2_rs_small | 0.8948 → 0.9002 multi-res | |
| dinov2_small_reg | 0.8906 → 0.8971 multi-res | |
| dinov3_small | 0.8869 | |
| dinov3_sat_large | 0.8886 | satellite-pretrained |

### Decoder sweep (LEVIR-CD, frozen dinov2_rs_base)

| Decoder | Params | Test F1 | Notes |
|---------|--------|---------|-------|
| sam2_tiny | 39M | **0.9058 → 0.9089** | best decoder |
| sam2_small | 46M | 0.9041 → 0.9088 | |
| sam2_base_plus | 81M | 0.9035 → 0.9075 | used in all other runs |
| sam2_large | 225M | 0.9025 → 0.9083 | |

### Cheap-wins (dinov2_rs_base + FT decoder + EMA 0.999 + threshold + patience 30)

| Dataset | Fair F1 | Cheap-wins F1 | Δ | Notes |
|---------|---------|--------------|---|-------|
| LEVIR-CD | 0.9045 | 0.9025 | −0.20 | within seed noise |
| SECOND | 0.7067 | 0.7209 | **+1.42** | |
| DSIFN-CD | 0.9674 | 0.9674 | 0.00 | no change |
| LEVIR-CD+ | 0.8172 | 0.8137 | −0.04 | no change |
| CDD | — | pending | — | |
| S2Looking | — | pending | — | |

### Null results / ablations (LEVIR-CD, Tal's cluster)

| Tried | Δ vs baseline | Verdict |
|-------|--------------|---------|
| LoRA r=8 on DINO | −0.09 | null |
| LoRA r=16 on DINO | −0.76 | hurts |
| LoRA r=8 on SAM decoder | −0.19 | null |
| LoRA r=4 on both | −0.50 | hurts |
| Bidirectional cross-attention | −0.31 | hurts |
| Local-window similarity (3×3) | −0.61 | hurts |
| Soft latent target | null | null |
| Lovász latent loss | −0.73 | hurts |
| MSE patch-density | −0.92 | hurts |
| Multi-scale aux supervision | null | null |

---

## GALASH Results — Natalie's Local Runs (RTX A5000 ×3)

Encoder: dinov2_rs_base throughout (dinov3_large is gated, no HF token on this machine).

### Fair config runs (seed 42/43/44, frozen decoder)

| Run | Seed | Epochs | Test F1 | Fair SOTA | Gap |
|-----|------|--------|---------|-----------|-----|
| LEVIR-CD | 42 | 113 | 0.9018 | 0.9287 | −2.69pp |
| LEVIR-CD | 43 | 75 | 0.8994 | 0.9287 | −2.93pp |
| LEVIR-CD | 44 | 71 | 0.8987 | 0.9287 | −3.00pp |
| CDD | 42 | 300 | 0.9678 | 0.9762 | −0.84pp |
| DSIFN-CD | 42 | 300 | **0.9676** | 0.9665 | **+0.11pp ✅** |
| SECOND | 42 | 7 | — | 0.7246 | killed early |

LEVIR-CD variance across seeds 42/43/44: mean=0.9000, std=0.0016

### Cheap-wins runs (FT decoder + EMA 0.999 + threshold search + patience 30)

| Dataset | Epochs | Test F1 | Fair SOTA | Gap | Notes |
|---------|--------|---------|-----------|-----|-------|
| LEVIR-CD | 166 | 0.9051 | 0.9287 | −2.36pp | |
| CDD | 300 | 0.9676 | 0.9762 | −0.86pp | cheap = no gain vs fair |
| DSIFN-CD | 300 | **0.9675** | 0.9665 | **+0.10pp ✅** | confirms SOTA win |
| SECOND | 62 | 0.7147 | 0.7246 | −1.65pp | early stopped |
| S2Looking | 122 | 0.5956 | 0.6895 | −9.76pp | early stopped, hard dataset |

### Ablation runs (new — 2026-04-29)

| Run | Flags | Dataset | Epochs | Test F1 | SOTA | Gap | vs baseline | Verdict |
|-----|-------|---------|--------|---------|------|-----|-------------|---------|
| abl_levir_cnn_skip | `--cnn_skip` cheap | LEVIR-CD | 121 | **0.9150** | 0.9287 | −1.37pp | **+1.32pp** | ✅ keeps |
| abl_s2looking_offset | `--learnable_offset` cheap | S2Looking | 65 | 0.5643 | 0.6895 | −12.89pp | −3.13pp | ❌ hurts |
| abl_cdd_cnn_skip | `--cnn_skip` cheap | CDD | 🔄 running | — | 0.9762 | — | — | pending |
| abl_s2looking_cnn_skip | `--cnn_skip` cheap | S2Looking | 🔄 running | — | 0.6895 | — | — | pending |
| abl_levir_simple_diff | `--simple_diff` cheap | LEVIR-CD | 🔄 running | — | 0.9287 | — | — | pending — key ablation |

**Notes on ablations:**
- `--cnn_skip`: replaces Bridge's fake bilinearly-upsampled high_res_features with real
  Siamese CNN change features at 128px and 256px. +1.32pp on LEVIR-CD — cuts the gap in half.
- `--learnable_offset`: MLP predicts per-patch warp before CrossChangeAttention.
  Hurts on S2Looking because parallax depends on building height — not learnable per-patch.
- `--simple_diff`: replaces CrossChangeAttention with elementwise subtraction.
  Result will quantify CrossChangeAttention's contribution. This is the key paper ablation.

---

## Head-to-Head: Natalie vs Tal (same dataset, same base encoder)

| Dataset | Config | Tal (cluster) | Natalie (local) | Diff | Note |
|---------|--------|--------------|-----------------|------|------|
| LEVIR-CD | fair, seed42 | 0.9000 | **0.9018** | +0.18pp | essentially tied |
| LEVIR-CD | cheap | 0.9025 | 0.9051 | +0.26pp | essentially tied |
| LEVIR-CD | cheap + cnn_skip | — | **0.9150** | — | new best local |
| DSIFN-CD | fair, seed42 | 0.9674 | **0.9676** | +0.02pp | essentially tied |
| CDD | fair, seed42 | 0.9675 | **0.9678** | +0.03pp | essentially tied |
| SECOND | cheap | **0.7209** | 0.7147 | −0.62pp | cluster wins (small dataset variance) |
| S2Looking | cheap | **0.6571** | 0.5956 | −6.15pp | cluster wins (dinov3_large vs rs_base) |
| LEVIR-CD+ | cheap | **0.8137** | not run | — | no data downloaded |

---

## Summary: Our Best vs Fair SOTA (all datasets)

| Dataset | Our best | Who/config | Fair SOTA | Method | Gap |
|---------|---------|------------|-----------|--------|-----|
| **SECOND** | **0.7430** | Tal — dinov3_large + cheap wins | 0.7246 (SAM-SCD) | | **+1.84pp ✅** |
| **DSIFN-CD** | **0.9676** | Both — dinov2_rs_base fair | 0.9665 (DDPM-CD) | | **+0.11pp ✅** |
| LEVIR-CD | 0.9150 | Natalie — cheap + cnn_skip | 0.9287 (SChanger) | | −1.37pp |
| CDD | 0.9678 | Natalie — dinov2_rs_base fair | 0.9762 (SChanger) | | −0.84pp |
| S2Looking | 0.6571 | Tal — dinov3_large + cheap wins | 0.6895 (SChanger) | | −3.24pp |
| LEVIR-CD+ | 0.8320 | Tal — dinov3_large + cheap wins | 0.8771 (DDCDNet) | | −4.51pp |

**2 datasets above SOTA. 4 below — CNN skip improved LEVIR-CD from −1.98pp to −1.37pp.**

---

## Pending Results

| Run | Expected F1 | Basis |
|-----|------------|-------|
| abl_cdd_cnn_skip (running GPU 0) | ~0.968–0.972 | CNN skip +1.32pp on LEVIR; CDD gap smaller |
| abl_s2looking_cnn_skip (running GPU 1) | ~0.60–0.63 | CNN skip helps boundaries; S2Looking still hard |
| abl_levir_simple_diff (running GPU 2) | ~0.895–0.910 | If ~0.900, CrossChangeAttention worth ~1.5pp |
