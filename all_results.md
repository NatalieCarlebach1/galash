# GALASH — All Results & Comparisons

Last updated: 2026-04-28. All F1 scores are test-set unless marked `~val` (val-based preview).

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
| S2Looking | fair | dinov3_large | 0.6571 | 0.6895 | −3.24pp | |
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
| CDD | — | 🔄 cluster result pending | — | |
| S2Looking | — | 🔄 cluster result pending | — | |

### Null results / ablations (LEVIR-CD)

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
| SECOND | 42 | 7 | — | 0.7312 | killed early |

LEVIR-CD variance across seeds 42/43/44: mean=0.9000, std=0.0016

### Cheap-wins runs (FT decoder + EMA 0.999 + threshold search + patience 30)

| Dataset | Epochs | Test F1 | Fair SOTA | Gap | Notes |
|---------|--------|---------|-----------|-----|-------|
| SECOND | 62 | 0.7147 | 0.7246 | −0.99pp | early stopped |
| S2Looking | 122 | 0.5956 | 0.6895 | −9.39pp | early stopped, hard dataset |
| CDD | 208 | ~val 0.9623 | 0.9762 | ~−1.39pp | 🔄 still running |
| DSIFN-CD | 201 | ~val 0.9619 | 0.9665 | ~−0.46pp | 🔄 still running |

---

## Head-to-Head: Natalie vs Tal (same dataset, same base encoder)

| Dataset | Config | Tal (cluster) | Natalie (local) | Diff | Note |
|---------|--------|--------------|-----------------|------|------|
| LEVIR-CD | fair, seed42 | 0.9000 | **0.9018** | +0.18pp | Natalie slightly better |
| DSIFN-CD | fair, seed42 | 0.9674 | **0.9676** | +0.02pp | essentially tied |
| CDD | fair, seed42 | 0.9675 | **0.9678** | +0.03pp | essentially tied |
| SECOND | cheap wins | **0.7209** | 0.7147 | −0.62pp | cluster wins (small dataset variance) |
| S2Looking | cheap wins | 🔄 pending | 0.5956 | — | Tal's cluster result not yet known locally |
| LEVIR-CD+ | cheap wins | **0.8137** | not run | — | |

Fair-config results are essentially identical between cluster and local — the randomness
from the small SECOND dataset explains the 0.62pp gap there.

---

## Summary: Our Best vs Fair SOTA (all datasets)

| Dataset | Our best | Who/config | Fair SOTA | Method | Gap |
|---------|---------|------------|-----------|--------|-----|
| **SECOND** | **0.7430** | Tal — dinov3_large + cheap wins | 0.7246 (SAM-SCD) | | **+1.84pp ✅** |
| **DSIFN-CD** | **0.9676** | Both — dinov2_rs_base fair | 0.9665 (DDPM-CD) | | **+0.11pp ✅** |
| CDD | 0.9678 | Natalie — dinov2_rs_base fair | 0.9762 (SChanger) | | −0.84pp |
| LEVIR-CD | 0.9089 | Tal — sam2_tiny + multi-res | 0.9287 (SChanger) | | −1.98pp |
| S2Looking | 0.6571 | Tal — dinov3_large + cheap wins | 0.6895 (SChanger) | | −3.24pp |
| LEVIR-CD+ | 0.8320 | Tal — dinov3_large + cheap wins | 0.8771 (DDCDNet) | | −4.51pp |

**2 datasets above SOTA. 4 datasets below — closest gap is CDD at −0.84pp.**

---

## Pending / Expected Results

| Run | Expected test F1 | Basis |
|-----|-----------------|-------|
| cheap_cdd (Natalie, running) | ~0.967–0.969 | val=0.9623 at ep208, seed42 fair got 0.9678 |
| cheap_dsifn_cd (Natalie, running) | ~0.966–0.968 | val=0.9619 at ep201, may extend SOTA win |
| Tal cluster: A_v3l_cheap_cdd | ~0.97+ | val 0.9513 at convergence on cluster |
| Tal cluster: A_v3l_cheap_s2looking | ~0.66+ | val 0.6945 already > UniChange val |
