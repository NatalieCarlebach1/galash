# GALASH — BMVC paper source

LaTeX source for the paper. Target: BMVC 2026 (or next suitable venue).

## Layout

```
paper/
├── main.tex              # entry point — compile this
├── bmvc2k.sty            # style placeholder (SWAP for official BMVC template)
├── references.bib        # bibliography
├── figures/              # figures (architecture.pdf, qualitative.pdf, ...)
└── sections/
    ├── 00_abstract.tex
    ├── 01_introduction.tex
    ├── 02_related_work.tex
    ├── 03_method.tex
    ├── 04_experiments.tex
    ├── 05_ablations.tex
    └── 06_conclusion.tex
```

## Compiling

Requires a working LaTeX installation (texlive-full or MacTeX).

```bash
cd paper
latexmk -pdf main.tex           # or: pdflatex + bibtex + pdflatex + pdflatex
```

Clean build artifacts:
```bash
latexmk -C
```

## Before submission

1. **Swap `bmvc2k.sty` for the official BMVC template.** Download from
   https://bmvc2026.org/info/authorguide/ (or the year you submit to).
   The current file is a lightweight placeholder that approximates the format
   but does **not** include paper-ID handling or blind-review infrastructure.
2. **Fill in all TODO placeholders** in the tables and prose — these are
   numbers that come out of the experiment sweep.
3. **Render figures:** `figures/architecture.pdf` and `figures/qualitative.pdf`
   need to be created (the architecture diagram can be drawn in TikZ or
   a vector tool; qualitative results come from `eval.py --save_masks`).
4. **De-anonymise** authors for camera-ready; leave anonymous for review.

## Where the numbers come from

- `runs/pd_*_20260424_*/test_results.json` — per-dataset baseline F1/IoU
  for the "GALASH (ours, frozen)" row of Table 1.
- `runs/pooled_baseline_*/test_per_dataset_*.json` — pooled model evaluated
  per-dataset (for cross-dataset generalisation Table 4).
- Encoder/decoder sweep runs (launched via
  `./slurm/launch_sweep.sh encoder_sweep decoder_sweep`) populate Table 2.
- Ablation tier (`./slurm/launch_sweep.sh ablations finetune`) populates
  Table 5 component ablations.
- `scripts/monitor.py` gives a live SOTA-gap view to track progress.
