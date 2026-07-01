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
