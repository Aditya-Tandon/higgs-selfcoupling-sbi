"""Schema/consistency checks for an extended NSBI cache (see build_nsbi_cache.py).

Usage: python sbi/verify_nsbi_cache.py <cache.npz>
Exits non-zero on the first failed check.
"""
import json
import sys

import numpy as np

path = sys.argv[1]
d = np.load(path)
meta = json.loads(str(d["meta"]))
print("meta:", meta)

required = ["sig_jet_pt", "sig_jet_eta", "sig_jet_btag", "sig_ht", "sig_n_jets",
            "qcd_jet_pt", "qcd_jet_eta", "qcd_jet_btag", "qcd_ht", "qcd_n_jets"]
for k in required:
    assert k in d, f"missing {k}"

ns, nq = len(d["sig_score"]), len(d["qcd_score"])
k = meta["top_k"]
assert d["sig_jet_pt"].shape == (ns, k), d["sig_jet_pt"].shape
assert d["qcd_jet_pt"].shape == (nq, k), d["qcd_jet_pt"].shape
assert d["sig_ht"].shape == (ns,) and d["qcd_ht"].shape == (nq,)

# pT sorted descending where present
jp = d["sig_jet_pt"]
fin = np.isfinite(jp[:, 0]) & np.isfinite(jp[:, 1])
assert (jp[fin, 0] >= jp[fin, 1]).all(), "jets not pT-sorted"

# HT (all jets) >= sum of stored top-k jet pTs
top_sum = np.nansum(jp, axis=1)
assert (d["sig_ht"] >= top_sum - 1e-3).all(), "HT < sum of top-k jet pTs"

# btag within [-1, 1]; -1.0 is the "untagged / no score" sentinel
bt = d["sig_jet_btag"][np.isfinite(d["sig_jet_btag"])]
assert bt.min() >= -1 - 1e-6 and bt.max() <= 1 + 1e-6, (bt.min(), bt.max())

print("ref collections stored: sig", meta["sig_refs"], "| qcd", meta["qcd_refs"])
for pfx in ("sig", "qcd"):
    for short in meta[f"{pfx}_refs"]:
        cols = [c for c in d.keys() if c.startswith(f"{pfx}_{short}_")]
        print(f"  {pfx}_{short}: {cols}")
print("all keys:", sorted(d.keys()))
print(f"VERIFY OK: {path}")
