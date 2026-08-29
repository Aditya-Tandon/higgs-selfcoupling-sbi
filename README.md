# higgs-selfcoupling-sbi

This repository estimates the Higgs trilinear self-coupling κ_λ from HH → 4b events recorded by the CMS Phase-2 Level 1 Data Scouting system. The Level 1 Data Scouting system stores L1 trigger primitives for every bunch crossing at 40 MHz, offering statistics several orders of magnitude larger than the standard trigger path. The project asks whether the L1-reconstructed objects are of sufficient quality, and whether modern likelihood-free inference is efficient enough, to constrain κ_λ from this data.

The analysis is structured as a three-stage pipeline: a Particle Transformer trained on L1 jet constituents for b-tagging, an event-level classifier separating HH → 4b signal from QCD multijet background, and a Neural Simulation-Based Inference pipeline for the coupling measurement using analytic LO reweighting and a weighted-CARL ratio estimator with SBC-calibrated intervals.

---

## Motivation

The shape of the Higgs potential is governed by the trilinear self-coupling λ_HHH, which is accessible through Higgs-pair production. The dominant fully-hadronic channel HH → 4b is overwhelmed by QCD multijet background and is normally discarded by the trigger. The Phase-2 Level 1 Data Scouting system circumvents this by storing L1 objects for every bunch crossing, and the newly available L1 tracking enables PUPPI and particle-flow constituents to be built and clustered into jets at Level 1. This project investigates whether those objects carry enough information to constrain κ_λ using simulation-based inference.

---

## Results

### Jet-level b-tagging

A Particle Transformer (ParT) was trained on L1 jet constituents to discriminate b-jets from QCD jets. Two L1 constituent collections were compared: particle-flow (PF) candidates, which carry displaced-track variables (d_xy, z_0) from B-hadron decays, and PUPPI candidates, which suppress these in favour of pileup reduction.

| Model | AUC |
| :-- | :-- |
| ParT on PF constituents | 0.813 |
| ParT on PUPPI constituents | 0.756 |
| L1 Next Gen (existing baseline) | 0.663 |
| L1 Extended (existing baseline) | 0.636 |

The PF model outperforms existing L1 taggers by approximately 0.15 in AUC. The gain over PUPPI is attributable to the displaced-track information that PUPPI suppresses.

### Event-level classification

An event-level ParT was trained directly on all L1 constituents of an event to score HH → 4b signal against QCD multijet background. On the full phase space the classifier reaches AUC ≈ 0.92, but this drops to ≈ 0.84 on the preselected region where easy low-pT QCD has already been removed. The ≈ 0.84 ceiling is set by the L1 observables available to the model, not by the training procedure.

### Neural SBI for κ_λ

No BSM κ_λ Monte Carlo exists, so the forward model is built by analytic leading-order gg → HH reweighting of the SM sample using the generator-level di-Higgs invariant mass and scattering angle. Per-event likelihood ratios are learned with a parameterised weighted-CARL ratio estimator (SNRE-B), and κ_λ intervals are extracted from a binned extended likelihood calibrated with Simulation-Based Calibration (SBC).

The inference method is validated: closure tests recover injected κ_λ values with 4/4 inside the calibrated 68% interval, coverage between 0.68 and 0.71, and |bias| < 0.02.

At true L1-scouting selection, however, the signal-to-background ratio is S/B ≈ 10⁻⁶ and the resulting κ_λ interval spans the entire prior. The data do not yet constrain the coupling. The analysis preselection raises S/√B by a factor of 8, but this remains well below 1. The limiting factor is signal-to-background discrimination, not the inference method itself. Improving the PF and tracking inputs to the classifier, and generating more low-pT QCD Monte Carlo to bound the residual background, are the main routes forward.

---

## Pipeline

```
Stage 1 — L1 object performance and ParT b-tagging
              Compare L1 Data Scouting objects (PUPPI, PF) against HLT references:
              reconstruction efficiency/purity, jet energy scale and resolution,
              and b-tagging discriminant power. Train a ParT b-tagger on L1 jet
              constituents with multi-task loss (classification + pT correction + resolution).
              ↓ (jet scores, energy corrections)
Stage 2 — Event-level HH → 4b vs QCD classifier
              ↓ (event scores, reco m_HH)
Stage 3 — Neural SBI: LO reweighting → weighted-CARL → profile likelihood → κ_λ intervals
```

---

## Repository structure

```
model/           Particle Transformer implementation and LR scheduler
data_pipeline/   ROOT → .npz dataset builders, jet matching, splitting
evaluation/      ROC, efficiency, resolution, di-Higgs reconstruction, feature importance
plotting/        Visualisation
sbi/             Neural SBI pipeline for κ_λ
  kl_reweight.py         Analytic LO gg→HH reweighting (forward model)
  snre.py                Weighted-CARL / SNRE-B ratio estimator
  closure_kl.py          Closure test (unbinned, iteration 1)
  closure_kl_v2.py       Closure test (binned extended likelihood, real observables)
  diagnose_discrimination.py   S/√B decomposition by QCD pT bin
  fisher_info.py         Per-event κ_λ Fisher information
  selection_scan.py      72-point selection grid scan
train_part.py    Jet-level training (single or multi-GPU via torchrun)
train_event.py   Event-level training (DDP, local resume)
```

---

## Setup

```bash
conda env create -f hep-root-ml-clean.yml
conda activate hep-root-ml
```

The project uses PyTorch, uproot/awkward/vector (Scikit-HEP), and Weights & Biases for experiment tracking. GPU training runs via `torchrun` DDP on PBS Pro (Imperial CX3 HPC).

---

## Usage

### Build datasets (ROOT → .npz)
```bash
# Jet-level b-tagging dataset (kinematic + QCD cross-section weights)
python data_pipeline/make_particle_dataset.py   --config config_part.json

# Event-level HH-vs-QCD dataset (trigger emulation via L1Ext jets)
python data_pipeline/make_event_dataset.py       --config config_event.json
```

### Train
```bash
# Jet-level ParT b-tagger
python train_part.py  --config config_part.json  --exp-name ParT_btag

# Event-level HH-vs-QCD classifier (single GPU)
python train_event.py --config config_event.json --exp-name EventParT_HH4b

# Multi-GPU (DDP) event training
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    train_event.py --config config_event_pf.json --exp-name EventParT_HH4b_PF
```

`train_event.py` supports local resume via `"resume_ckpt"` in the config, which restores the model, optimiser, and LR schedule from a checkpoint for continuing a run cut off by a walltime limit.

### Neural SBI for κ_λ
```bash
# 1. Validate the analytic LO reweighting against the sigma(kappa_lambda) quadratic
python sbi/validate_reweight.py --files 'data/hh4b/data_*.root'

# 2. Build the row-aligned observable cache (event-ParT score + reco m_HH + gen m_HH)
python sbi/build_nsbi_cache.py --apply-trigger --out data/event_level/nsbi_cache.npz

# 3. Run the closure (binned extended likelihood + SBC-calibrated intervals)
python sbi/closure_kl_v2.py --score-wp 0.9 --n-cov 300

# Diagnose the discrimination bottleneck (S/sqrt(B) vs score threshold)
python sbi/diagnose_discrimination.py --cache data/event_level/nsbi_cache.npz
```

### Notebooks

Interactive analysis and figures live in `notebooks/` (`b-tagging.ipynb`, `eval_event_classifier.ipynb`, `test_trained_part.ipynb`, `read-data.ipynb`). These should be run with the `hep-root-ml` kernel.

---

## References

1. MSci thesis: *Towards Neural-SBI for Estimating the Higgs' Trilinear Coupling Constant: Physics Object Performance and Jet Tagging for the Phase-2 CMS Level-1 Data Scouting System* (Imperial College London)
2. Cranmer, Brehmer & Louppe, *The frontier of simulation-based inference*, PNAS 117 (2020)
3. Qu, Li & Qian, *Particle Transformer for Jet Tagging*, ICML 2022
