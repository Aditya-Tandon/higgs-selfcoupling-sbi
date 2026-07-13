# root-obj-perf — Neural-SBI for the Higgs Trilinear Coupling from CMS Phase-2 L1 Data Scouting

A research codebase for estimating the Higgs trilinear self-coupling **κλ ≡ λ_HHH / λ_HHH^SM**
from **HH → 4b** events recorded by the **CMS Phase-2 Level-1 Data Scouting** system, using a
**Particle Transformer** for object tagging/classification and **Neural Simulation-Based Inference
(NSBI)** for the coupling measurement.

The repository grew out of an MSci thesis — *"Towards Neural-SBI for Estimating the Higgs' Trilinear
Coupling Constant: Physics Object Performance and Jet Tagging for the Phase-2 CMS Level-1 Data
Scouting System"* (project PART-Maier-Bainbridge-2) — and has since been extended with an event-level
HH-vs-QCD classifier and a working NSBI closure pipeline for κλ.

---

## Why this matters

The shape of the Higgs potential — and therefore the dynamics of Electroweak Symmetry Breaking — is
governed by the Higgs trilinear coupling λ_HHH, accessible through Higgs-pair production (HH). The
dominant fully-hadronic channel HH → 4b is overwhelmed by QCD multijet background and is normally
discarded by the trigger. The **Phase-2 Level-1 Data Scouting** system stores L1 objects for *every*
bunch crossing, offering statistics several orders of magnitude larger than the standard trigger
path. The newly available L1 tracking enables pileup-reduced **PUPPI** and **particle-flow (PF)**
constituents to be built and clustered into jets *at Level-1*. This project asks whether those objects
are good enough — and whether modern likelihood-free inference is efficient enough — to say anything
about κλ from L1-scouting data.

---

## Pipeline overview

The project is a three-stage pipeline. Each stage has its own entrypoints, configs, and evaluation
tooling.

### Stage 1 — L1 object performance & ParT b-tagging (thesis)
- Compare L1 Data Scouting objects (PUPPI, PF) against High-Level-Trigger objects: reconstruction
  efficiency/purity, jet energy scale & resolution (JES/JER), and b-tagging power.
- Train a **Particle Transformer (ParT)** on jet constituents to separate **b-jets from QCD**, and to
  regress the jet pT correction (JES/JER).
- Pair tagged b-jets into two Higgs candidates and reconstruct the di-Higgs system; measure the
  Poisson/Asimov significance in the (m_H1, m_H2) signal region.
- **Headline result:** the ParT trained on **PF constituents reaches AUC ≈ 0.813** vs **PUPPI ≈ 0.756**
  at jet level, both outperforming the existing L1 tagging baselines and narrowing the L1↔HLT gap. PF
  wins because L1 PF carries displaced-track information (dxy, z0) from B-decays that PUPPI suppresses.

### Stage 2 — Event-level HH-vs-QCD classifier
- Train a **single event-level ParT** directly on all L1 constituents of an event to score
  **HH → 4b signal vs QCD multijet**, rather than tagging jets individually.
- Trigger/preselection emulation is applied at dataset-build time (≥4 L1Ext jets over a pT threshold,
  HT > 330 GeV, b-tag working point).
- **Result:** AUC ≈ **0.92 on the full phase space**, dropping to ≈ **0.84 on the preselected
  ("hard") phase space** — the headline number is inflated by easy low-pT QCD that the preselection
  removes anyway. On the surviving mid-pT, genuine multi-b-jet background the classifier discriminates
  far less well; this ~0.84 ceiling is **input-limited** (set by the L1 observables), not
  training-limited.

### Stage 3 — Neural-SBI for κλ
- No BSM κλ Monte Carlo exists, so the forward model is built by **analytic leading-order gg → HH
  reweighting** of the SM sample using generator-level (m_HH, cos θ*): |M|² = a + b·κλ + c·κλ².
- Per-event likelihood ratios are learned with a **parameterized weighted-CARL / SNRE-B ratio
  estimator**, and κλ intervals are extracted from a binned extended likelihood, **calibrated with
  Simulation-Based Calibration (SBC)**.
- **Method is validated:** closure recovers injected κλ (iter-1: 4/4 inside the calibrated 68%
  interval, coverage 0.68; iter-2 on real observables [event-ParT score + reco m_HH]: 4/4 with
  |bias| < 0.02, coverage 0.71, calibrated threshold δ* ≈ 0.45 ≈ the asymptotic 0.5).
- **Honest physics result:** at true L1-scouting selection the signal-to-background is **S/B ≈ 10⁻⁶**,
  so the κλ interval spans the entire prior — **no sensitivity yet**. This is a **discrimination
  ceiling, not an inference failure**. The largest lever found so far is the analysis preselection
  (**8× improvement in S/√B**, 0.0058 → 0.0477); retraining the classifier on the preselected phase
  space gave no further gain. Remaining levers are physics-side: PF/tracking inputs and more low-pT
  QCD Monte Carlo.

---

## Repository structure

```text
model/              ParticleTransformer implementation + LR scheduler
data_pipeline/      Dataset classes, splitting, gen-matching, ROOT loading, ROOT -> .npz conversion
  make_particle_dataset.py   Jet-level dataset: ROOT -> .npz (kinematic + QCD xsec weights)
  make_event_dataset.py      Event-level dataset: ROOT -> .npz (trigger emulation via L1Ext jets)
  datasets.py                StratifiedJetDataset, DataLoader wrappers (mmap-backed)
  root_loading.py            uproot loading, jet collections, load_event_level_data()
evaluation/         ROC, efficiency, resolution, di-Higgs, attention, feature importance, luminosity
  roc.py            roc_from_scores(), working points, 2D-binned trained ROC
  efficiency.py     efficiency_table(), mistag_rate_vs_var(), btag_efficiency_vs_var()
  resolution.py     Gaussian response fits, resolution vs pT/eta
  dihiggs.py        pair_from_4jets(), R_hh, compute_significance_at_luminosity(), reconstruct_dihiggs()
  luminosity.py     luminosity scaling, signal_weight(), QCD xsec-weight conventions
  attention.py      batched attention extraction, pairwise features
plotting/           All visualisation (ROC, resolution, di-Higgs, attention, feature importance)
notebooks/          b-tagging.ipynb, test_trained_part.ipynb, read-data.ipynb, eval_event_classifier.ipynb
sbi/                Neural-SBI pipeline for kappa_lambda
  kl_reweight.py         analytic LO gg->HH kappa_lambda reweighting (forward model)
  validate_reweight.py   reweight validation vs the sigma(kappa_lambda) quadratic
  snre.py                weighted-CARL / SNRE-B parameterized ratio estimator
  build_nsbi_cache.py    row-aligned observable cache (event-ParT score + reco m_HH + gen m_HH)
  closure_kl.py          iteration-1 closure (unbinned, smeared-m_HH proxy)
  closure_kl_v2.py       iteration-2 closure (binned extended likelihood, real observables)
  diagnose_discrimination.py  S/sqrt(B) vs score threshold, QCD pT-bin decomposition
tests/              test_parT.py, test_event_features.py
train_part.py       Jet-level ParT training      (CLI: --config, --set, --exp-name)
train_event.py      Event-level ParT training    (same CLI pattern; supports DDP + local resume)
```

Backward-compatibility shims in the repo root (`parT.py`, `data_loading_helpers.py`, …) re-export from
these canonical locations so older notebooks keep working.

---

## Results at a glance

| Stage | Metric | Value | Notes |
| :-- | :-- | :-- | :-- |
| Jet b-tagging (thesis) | ParT AUC, PF | **0.813** | beats L1 baselines |
| Jet b-tagging (thesis) | ParT AUC, PUPPI | **0.756** | PUPPI better at the loose WP |
| Event classifier | AUC, full phase space | **≈ 0.92** | inflated by easy low-pT QCD |
| Event classifier | AUC, preselected phase space | **≈ 0.84** | input-limited ceiling |
| NSBI closure (iter-1) | κλ recovery / coverage | **4/4 / 0.68** | SBC-calibrated 68% interval |
| NSBI closure (iter-2, real obs) | κλ bias / coverage | **\|bias\|<0.02 / 0.71** | method validated, δ*≈0.45 |
| Discrimination lever | S/√B (preselection) | **0.0058 → 0.0477 (8×)** | still ≪ 1 → no κλ sensitivity yet |
| Honest L1-scouting | S/B | **≈ 10⁻⁶** | discrimination-ceiling limited |

The coupling itself follows from κλ via λ_HHH = κλ · m_H² / (2v²), with λ_HHH^SM ≈ 0.129.

---

## Setup

All work uses the `hep-root-ml` conda environment (the `vector` package and the HEP stack are only
available there):

```bash
conda env create -f hep-root-ml-clean.yml     # creates env "hep-root-ml"
conda activate hep-root-ml
```

Core stack: PyTorch, `uproot` / `awkward` / `vector` (Scikit-HEP), NumPy/SciPy, scikit-learn,
matplotlib, and Weights & Biases for experiment tracking.

The project runs on the Imperial **CX3** HPC cluster under **PBS Pro**; GPU training uses
`torchrun` Distributed Data Parallel. Batch scripts live alongside the code (e.g. `sbi/*.pbs`).

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

# Event-level HH-vs-QCD classifier (single-GPU)
python train_event.py --config config_event.json --exp-name EventParT_HH4b

# Multi-GPU (DDP) event training
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    train_event.py --config config_event_pf.json --exp-name EventParT_HH4b_PF
```
`train_event.py` supports a **local resume** (`"resume_ckpt"` in the config) that restores the model,
optimiser, and LR schedule from a checkpoint — for continuing an offline run cut off by a walltime
limit without re-doing warmup.

### Neural-SBI for κλ
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
Interactive analysis and figures live in `notebooks/` (`b-tagging.ipynb`,
`eval_event_classifier.ipynb`, `test_trained_part.ipynb`, `read-data.ipynb`). Run them with the
`hep-root-ml` kernel. Never redefine functions that already exist in `evaluation/` or `plotting/` —
always import from the canonical module.

---

## Data & configuration conventions

- **Configs** are JSON (`config_part.json`, `config_event.json`, `config_event_pf.json`,
  `hh-bbbb-obj-config.json`), overridable on the CLI via `--set key=value`.
- **QCD cross-section weights** come in three conventions (raw σ_bin, σ_bin/N_gen, ROOT-notebook raw);
  the correct scaling for analysis vs. training is documented in `CLAUDE.md`. For evaluation always use
  the **QCD cross-section weights**, not the kinematic training weights.
- **Luminosity scaling:** expected counts use `w = σ_pb · L_pb / N_gen`, with the physics constants
  (target luminosity, signal cross section, N_gen) in the `"physics"` block of
  `hh-bbbb-obj-config.json`. σ(HH → 4b) ≈ 0.0113 pb (SM NLO, 14 TeV).
- Large artifacts (`data/`, `*.pth`, `wandb/`, `.npz`, logs) are git-ignored; datasets are loaded
  memory-mapped so only metadata sits in RAM.

---

## Status & limitations

- The **NSBI inference machinery is validated** — unbiased κλ recovery with correct interval coverage
  on real observables.
- The **binding constraint is physics discrimination**, not the method. At L1-scouting selection the
  HH → 4b S/B is ~10⁻⁶, and even the 8× S/√B gain from preselection leaves S/√B ≪ 1.
- Background is **QCD-MC-statistics-limited** after preselection: the largest cross-section (low-pT) QCD
  bins have essentially zero surviving simulated events, so the residual background is poorly bounded.

### Directions for future work
- **PF/tracking event classifier:** move the event-level model from PUPPI to PF inputs (dxy, z0) to try
  to break the ~0.84 discrimination ceiling — mirroring the ~0.06 AUC PF advantage seen at jet level.
- **More low-pT QCD Monte Carlo** to bound the surviving background reliably.
- A **κλ-aware / end-to-end optimised** discriminant, and a tighter di-Higgs mass-window selection.

---

## References

1. MSci thesis: *Towards Neural-SBI for Estimating the Higgs' Trilinear Coupling Constant — Physics
   Object Performance and Jet Tagging for the Phase-2 CMS Level-1 Data Scouting System* (this repo).
2. Yang & Li, *Potential of di-Higgs observation via a calibratable jet-free HH → 4b framework*,
   arXiv:2508.15048.
3. Cranmer, Brehmer & Louppe, *The frontier of simulation-based inference*, PNAS 117, 30055 (2020),
   doi:10.1073/pnas.1912789117.
4. Qu, Li & Qian, *Particle Transformer for Jet Tagging*, ICML 2022 (PMLR 162), arXiv:2202.03772.

Developer and agent workflow conventions are documented in `CLAUDE.md`; a running log of issues and
fixes is kept in `agent_problems.md`.
