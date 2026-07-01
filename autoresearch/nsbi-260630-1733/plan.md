# Autoresearch — Neural SBI for κλ (λ_HHH) from L1 scouting HH→4b

mode: orchestrator (optimize-metric) · started 2026-06-30 · branch `sbi-nsbi` · runs on CX3 (PBS Pro, env `hep-root-ml`)

## Goal
Stand up a working Neural-SBI pipeline that infers the Higgs trilinear self-coupling
**κλ ≡ λ_HHH** (as a deviation from SM) from L1 Data-Scouting **HH→4b** event-level data,
and validate it with a closure / coverage test.

## Grounding (from thesis + repo)
- Thesis "Towards Neural-SBI for Estimating the Higgs' Trilinear Coupling" builds the upstream
  pipeline (L1 object study, ParT b-tagger, di-Higgs reco, significance) and names NSBI as the goal.
- Intended method per refs: **[11] Yang & Li, arXiv:2508.15048** — *calibratable, jet-free, event-level*
  HH→4b likelihood-free framework; **[12] Cranmer/Brehmer/Louppe** SBI review; user's review **[8]**.
- Assets on `main`: event-level ParT classifier HH-vs-QCD (`best_event_model_61z973dk.pth`,
  `config_event.json`, `data/event_level/`), di-Higgs reco + luminosity/significance in `evaluation/`.
- Data: **SM-only** HH→4b (50k, unweighted) + ~2M QCD (per-event xsec weights). No BSM/LHE κλ samples.
  Signal ntuples carry full `GenPart` → 2 last-copy Higgs/event → gen (m_HH, cosθ*) recoverable.

## Forward model (the key enabler)
No κλ samples exist, so the simulator family p(x|κλ) is built by **analytic LO gg→HH reweighting**
of the single SM sample: |M|² = a + b·κλ + c·κλ² per event from gen (m_HH, cosθ*).
QCD is κλ-independent (existing xsec weights). Extended likelihood:
`log L(κλ) = −(s(κλ)+b) + Σ_events log[s(κλ)·p_sig(x|κλ) + b·p_bkg(x)]`.

## Method
Parameterized per-event **likelihood-ratio (SNRE-B / calibrated classifier)** — port the
dependency-light from-scratch SNRE from `sbi-test` (`sbi`-pkg not installed), build on `main`,
feed it the event-level ParT score + reconstructed di-Higgs observables. Per-event ratios summed
into L(κλ); posterior via the prior × ratio.

## Prior art (sbi-test branch, stale — concepts only)
- `sbi_models.py`: SNRE-B RatioEstimator + contrastive trainer + MCMC posterior (+1D-Gaussian toy). REUSE.
- `train_hh_sbi.py`: closure workflow (morph→train→recover) + crude m_HH-only `(1+κλr)²` reweighting. REPLACE reweighting.
- `generate_sbi_dataset.py`: computes event m_HH, cosθ* but stores fixed κλ, weights=1. REPLACE.

## Success predicate (verify gate)
Closure: inject κλ_true ∈ {0,1,2.4,5}, run inference on held-out Asimov pseudo-data →
(1) |κλ_recovered − κλ_true| within the 68% interval for ≥ 3/4 points;
(2) SBC/coverage of the 68% interval ∈ [0.6, 0.76] over ≥100 trials.

## Iteration log
- I0 (setup): branch, scaffold, plan, vault note. ✅ in progress
- I1 (reweighting keystone): sbi/kl_reweight.py + validate_reweight.py.
  VALIDATED on 4000 SM events: sigma(kl=0)/SM=2.31, min at kl=2.1, SM=1.0 -> GATE PASS.
  Finding: N_eff/N = 3.1% @kl=2.45, 1.9% @kl=5 -> single-sample reweighting caps the
  usable kl prior; motivates finite-mt ME and/or |kl|<~5 prior. ✅
- I2 (SNRE port + closure build): sbi/snre.py (ported SNRE-B + loglik scan + CI),
  sbi/closure_kl.py (reweight->resample->SNRE->profile scan->recovery+coverage),
  sbi/closure_kl.pbs (v1_medium24 CPU job). Smoke test running. ⏳
  Observable iter-1 = [m_HH_reco(10% smear proxy), cos_theta*]; iter-2 swaps in real
  reco di-Higgs mass + event ParT score (GPU/qsub).

## Iteration 2 design (build only after I1 gate passes)
Real observable x = [reco m_HH, event-ParT score, (cos_theta*)], full S(kl)+B mixture.
Components needed:
- reco m_HH per event: evaluation/dihiggs.py:reconstruct_dihiggs_from_constituents (exists).
- event-ParT score: torch.sigmoid(cls_output) from best_event_model_61z973dk.pth -> GPU forward
  over data/event_level/* -> cache per-event scores (qsub v1_gpu72).
- signal gen (m_HH, cos_theta*) for reweighting: NOT in event_level meta (gen_pt is placeholder)
  -> extend make_event_dataset.py to store gen di-Higgs kinematics for signal, aligned to rows.
- extended likelihood: log L(kl) = -(s(kl)+b) + sum_i log[s(kl) p_sig(x|kl) + b p_bkg(x)];
  QCD uses existing qcd_weights; signal rate s(kl) from cross_section_ratio(kl).

- I3 (closure run #1, resample training): GATE FAIL but informative.
  MLEs TRACK TRUTH: kl 0->0.34, 1->1.16, 2.45->2.53, 5->5.33 (method learns kl).
  BUT 68% intervals too narrow (kl=2.45 -> [2.53,2.53]) + small +bias -> coverage=0.10.
  Cause: (a) unbinned learned-ratio likelihood is OVERCONFIDENT (Wilks/Delta lnL=0.5 invalid
  for a learned ratio); (b) importance-RESAMPLE at low N_eff (125 distinct events @kl2.45)
  starves estimator diversity -> MLE bias. Motivates "calibratable" (ref [11]).
  FIX (I4): (a) CARL-style WEIGHTED training (all 50k events, weights w_i(kl)) -> diversity+less bias;
  (b) SBC-CALIBRATED intervals (calibrate threshold for 68% on held-out trials) -> correct coverage.

- I4 (closure run #2, weighted CARL + SBC-calibrated): *** GATE PASS *** (50k SM events).
  Recovery 4/4 inside calibrated 68%: kl 0->MLE-1.0[-1,1.29], 1->1.02[0.32,1.61],
  2.45->2.77[1.99,3.67], 5->5.22[4.25,6.0]. Coverage=0.682 (target [0.60,0.76]).
  CAVEAT: delta*=203 (vs asymptotic 0.5) -> learned ratio strongly OVERCONFIDENT; partly
  genuine NSBI effect, partly closure artifact (pseudo-data resampled WITH REPLACEMENT at
  low N_eff duplicates high-weight events -> sharpens likelihood). Coverage honest (calib &
  test share structure). Iteration-1 machinery VALIDATED end-to-end.
  NEXT (efficiency, not coverage): smaller delta* via better per-event ratio + distinct-event
  pseudo-data; then iteration-2 real observables (reco m_HH + event ParT score, S(kl)+B, GPU).

## Iteration 2 (real observables) — BUILT + smoke-validated
- sbi/build_nsbi_cache.py (+GPU .pbs): aligned cache (signal score+reco_mhh+gen_mhh+cos*, QCD score+reco_mhh+sigma).
  Smoke OK: classifier discriminates (sig score med 0.47 vs QCD 0.05); reco!=gen (genuine reco); alignment assert passes.
  Full cache job 3155327 (v1_gpu72) running: 50k signal + 9x40k QCD.
- sbi/closure_kl_v2.py: BINNED extended likelihood on x=[event ParT score, reco m_HH].
  Discrete-event unbinned closure INFEASIBLE (B~1e11 at L1 scouting -> OOM); binned draws Poisson
  per-BIN (any B). nu_b(kl)=S(kl) f_sig,b(kl) + B f_bkg,b, signal template morphed by kl reweighting.
  I5 SMOKE GATE PASS (moderate-yield regime, boost 1e7, yield-scale 1e-5): recovery 4/4 (|bias|<0.02),
  coverage 0.645 in [0.60,0.76], delta*=0.40 (~asymptotic 0.5 -> binned likelihood WELL-CALIBRATED,
  vs 203 for unbinned ratio). Real-cache run pending; realistic S/B~1e-9 -> honest sensitivity likely weak
  (matches thesis significance ~0.01) => needs tighter selection / better discriminant.

- I6 (iter-2 real cache, binned extended likelihood): full cache built (50k sig/28728 reco;
  360k QCD/71079 reco; sig score med 0.52 vs QCD 0.05). Two regimes at WP score>=0.9:
  * HONEST full-lumi: S_SM=1270, B=8.09e8, S/B=1.6e-6 -> NO kl sensitivity (68% interval = full
    prior [-1,6]); MLE unbiased, coverage 0.620. Matches thesis significance ~0.01.
  * METHOD-VALIDATION (boost 6e5, yield-scale 1e-5 -> S/B~0.94): recovery 4/4 |bias|<0.02,
    coverage 0.713, delta*=0.45 (~asymptotic 0.5 => binned likelihood WELL-CALIBRATED).
  CONCLUSION: NSBI machinery validated on real observables (unbiased + calibrated); physics
  sensitivity at L1 scouting is negligible with current event classifier + selection. Next
  physics lever: stronger discrimination (tighter WP + mass window / kl-aware net), not method.
