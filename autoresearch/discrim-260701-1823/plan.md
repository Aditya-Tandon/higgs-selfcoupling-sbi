# Autoresearch — improve HH->4b vs QCD discrimination (L1 scouting)

Goal: raise the HH->4b vs QCD discrimination so kappa_lambda gains sensitivity. Bottleneck
confirmed = discrimination + background composition, NOT the NSBI inference (which is validated).

## Bottleneck diagnosis (sbi/diagnose_discrimination.py on nsbi_cache, F1-fixed yields)
- Total @1000/fb: S=12440, B=5.7e14, S/B=2e-11, S/sqrt(B)=5e-4 (no cut).
- Best score-only cut (>=0.9): S/sqrt(B)=0.0058, QCD rej 720x @ eps_s 14%.
- +m_HH window (300,500): S/sqrt(B)=0.023 (~4x) -> mass is a strong UNUSED discriminant.
- B composition @score>=0.9: **81% from qcd_pt_20-50 GeV, from only 5+7 MC events** (sigma~1e8 pb)
  -> low-pt QCD dominates + severe MC-stat limitation.
- ROOT CAUSE: cache built with skip_trigger=True -> NO analysis preselection removes low-pt QCD.

## Hypotheses / levers (priority order)
- H1 (cheap, highest value): apply HH->4b analysis preselection (>=4 L1Ext b-jets over pt
  threshold + HT + b-tag WP = passes_trigger_emulation) -> removes low-pt QCD, raises S/sqrt(B).
- H2: fold di-Higgs kinematics (reco m_HH, per-jet b-tag) into the discriminant (event ParT
  sees constituents only; mass adds ~4x).
- H3: fix training class imbalance (QCD carries 38x the loss weight; pos_weight=1.0) via
  QCD-xsec-aware / balanced / focal loss; retrain event ParT.
- H4 (data): regenerate more low-pt QCD MC to populate the tail (MC-stat).

## Gate: S/sqrt(B) in the di-Higgs mass region at fixed luminosity (higher is better),
with B MC-stat uncertainty tracked (>= N_eff events).

## RESULTS
### H1 (apply preselection / trigger emulation) — CONFIRMED, 8x win
Rebuilt cache with --apply-trigger (>=4 L1Ext jets pt-threshold + HT>330 + b-tag WP).
- Low-pt QCD ELIMINATED: pt_20_30 0/40000, pt_30_50 0/40000, pt_50_80 2, pt_80_120 6.
- Signal 3448/50000 (6.9%) pass; QCD 618 total survive (from 9x40000 loaded).
- Best S/sqrt(B) 0.0058 -> **0.0477 (8x)** @ score>=0.9. B cut from 5.7e14 to 2.1e9.
- IMPLICATION: the iteration-2 "no sensitivity" (S/B=1.6e-6) was largely an ARTIFACT of the
  MISSING preselection (skip_trigger). With preselection B is ~1000x smaller.
- Mass window no longer helps (B now MC-stat-limited: ~5-28 QCD events) -> mass was a proxy
  for "remove low-pt QCD", which the trigger does directly.

### Classifier headroom on the HARD (preselected) phase space
AUC on preselected events: **0.855 unweighted / 0.780 xsec-weighted**, vs 0.918 on full phase
space. The headline AUC is inflated by easy low-pt QCD that preselection removes. On the
relevant background the current model is much weaker -> RETRAIN on preselected phase space
with balanced class weighting (H3).

### Remaining limiter: QCD MC statistics (H4)
Post-preselection B from ~28 MC events; low-pt bins have 0/40000 survivors but sigma so large
that even 1/40000 leakage ~ observed B. Need more low-pt QCD MC to bound trigger leakage.

## Next: H3 retrain (preselected + balanced weighting), then H4 (more low-pt QCD MC).
