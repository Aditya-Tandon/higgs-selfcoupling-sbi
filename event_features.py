"""
Event-level feature extraction for HH->4b SBI analysis.

This module extracts event-level observables focused on m_HH sensitivity
for neural simulation-based inference of the Higgs trilinear coupling.
"""

import numpy as np
import awkward as ak
import vector
import torch


def pair_jets_to_higgs(jets):
    """
    Pair 4 leading jets into 2 Higgs candidates using D_HH minimization.

    From README.md and read-data.ipynb:
    D_HH = |m_h1 - (125/120)*m_h2| / sqrt(1 + (125/120)^2)

    This metric ensures consistent pairing across events by minimizing
    the asymmetry in reconstructed Higgs masses.

    Args:
        jets: awkward array of jets with vector field, shape (n_events, n_jets>=4)

    Returns:
        h1, h2: awkward arrays of reconstructed Higgs candidates (n_events,)
                each with vector field (pt, eta, phi, mass)
    """
    # Take 4 leading jets
    jets = jets[:, :4]

    # All possible pairings of 4 jets into 2 pairs
    # Pairing 0: (j0, j1) + (j2, j3)
    # Pairing 1: (j0, j2) + (j1, j3)
    # Pairing 2: (j0, j3) + (j1, j2)

    j0, j1, j2, j3 = jets[:, 0], jets[:, 1], jets[:, 2], jets[:, 3]

    # Compute all 3 pairings
    pair0_h1 = j0.vector + j1.vector
    pair0_h2 = j2.vector + j3.vector

    pair1_h1 = j0.vector + j2.vector
    pair1_h2 = j1.vector + j3.vector

    pair2_h1 = j0.vector + j3.vector
    pair2_h2 = j1.vector + j2.vector

    # Stack all pairing candidates: shape (n_events, 3, 2)
    all_h1_mass = ak.concatenate(
        [pair0_h1.mass[:, None], pair1_h1.mass[:, None], pair2_h1.mass[:, None]], axis=1
    )

    all_h2_mass = ak.concatenate(
        [pair0_h2.mass[:, None], pair1_h2.mass[:, None], pair2_h2.mass[:, None]], axis=1
    )

    # Compute D_HH for all pairings
    d_hh_all_pairs = np.abs(all_h1_mass - (125.0 / 120.0) * all_h2_mass) / np.sqrt(
        1 + (125.0 / 120.0) ** 2
    )

    # Find best pairing
    min_d_hh_pair = ak.argmin(d_hh_all_pairs, axis=1)

    # Select best pairing
    cond_0 = min_d_hh_pair == 0
    cond_1 = min_d_hh_pair == 1
    cond_2 = min_d_hh_pair == 2

    # Build best H1
    best_h1 = ak.where(cond_0, pair0_h1, ak.where(cond_1, pair1_h1, pair2_h1))

    # Build best H2
    best_h2 = ak.where(cond_0, pair0_h2, ak.where(cond_1, pair1_h2, pair2_h2))

    return best_h1, best_h2


def extract_hh_features(events, jets, gen_higgs=None):
    """
    Extract event-level features for HH->4b analysis.

    Prioritizes m_HH as the most sensitive observable to kappa_lambda.

    Args:
        events: awkward array of events
        jets: awkward array of selected jets (b-tagged), shape (n_events, n_jets)
        gen_higgs: optional, gen-level Higgs for validation

    Returns:
        dict with keys:
            - m_hh: di-Higgs invariant mass (GeV)
            - pt_hh: pT of HH system (GeV)
            - eta_hh: eta of HH system
            - phi_hh: phi of HH system
            - m_h1: leading Higgs mass (GeV)
            - m_h2: subleading Higgs mass (GeV)
            - pt_h1: leading Higgs pT (GeV)
            - pt_h2: subleading Higgs pT (GeV)
            - delta_r_hh: Delta R between two Higgs
            - delta_eta_hh: Delta eta between two Higgs
            - delta_phi_hh: Delta phi between two Higgs
            - cos_theta_star: helicity angle in HH rest frame
            - n_jets: number of jets in event
    """
    # Require at least 4 jets
    event_mask = ak.num(jets, axis=1) >= 4
    jets = jets[event_mask]

    if len(jets) == 0:
        # Return empty dict
        return {
            key: np.array([])
            for key in [
                "m_hh",
                "pt_hh",
                "eta_hh",
                "phi_hh",
                "m_h1",
                "m_h2",
                "pt_h1",
                "pt_h2",
                "delta_r_hh",
                "delta_eta_hh",
                "delta_phi_hh",
                "cos_theta_star",
                "n_jets",
            ]
        }

    # Pair jets into Higgs candidates
    h1, h2 = pair_jets_to_higgs(jets)

    # Order by pT (h1 = leading)
    swap_mask = h2.pt > h1.pt
    h1_final = ak.where(swap_mask, h2, h1)
    h2_final = ak.where(swap_mask, h1, h2)

    # Compute HH system
    hh = h1_final + h2_final

    # Angular separations
    delta_r_hh = h1_final.deltaR(h2_final)
    delta_eta_hh = h1_final.eta - h2_final.eta
    delta_phi_hh = h1_final.deltaphi(h2_final)

    # Helicity angle (approximate, assumes massless beam particles)
    # cos(theta*) = (2*pz_h1*pz_h2 - pz_hh*(E_h1 + E_h2)) / (pT_hh * sqrt(m_hh^2 + pT_hh^2))
    # Simplified: just use boost direction
    cos_theta_star = np.tanh((h1_final.eta - h2_final.eta) / 2.0)

    features = {
        "m_hh": ak.to_numpy(hh.mass),
        "pt_hh": ak.to_numpy(hh.pt),
        "eta_hh": ak.to_numpy(hh.eta),
        "phi_hh": ak.to_numpy(hh.phi),
        "m_h1": ak.to_numpy(h1_final.mass),
        "m_h2": ak.to_numpy(h2_final.mass),
        "pt_h1": ak.to_numpy(h1_final.pt),
        "pt_h2": ak.to_numpy(h2_final.pt),
        "delta_r_hh": ak.to_numpy(delta_r_hh),
        "delta_eta_hh": ak.to_numpy(delta_eta_hh),
        "delta_phi_hh": ak.to_numpy(delta_phi_hh),
        "cos_theta_star": ak.to_numpy(cos_theta_star),
        "n_jets": ak.to_numpy(ak.num(jets, axis=1)),
    }

    return features


def load_part_model(checkpoint_path, config_path=None):
    """
    Load trained ParT model for b-tagging.

    Args:
        checkpoint_path: path to .pth file
        config_path: optional path to config JSON

    Returns:
        ParT model in eval mode
    """
    import json
    from parT import ParticleTransformer

    # Load config
    if config_path is None:
        config_path = "config_part.json"

    with open(config_path) as f:
        config = json.load(f)

    # Initialize model
    model = ParticleTransformer(
        input_dim=config["input_dim"],
        embed_dim=config.get("embed_dim", 128),
        num_heads=config.get("num_heads", 8),
        num_layers=config.get("num_layers", 8),
        num_cls_layers=config.get("num_cls_layers", 2),
        dropout=config.get("dropout", 0.1),
        num_classes=config.get("num_classes", 1),
    )

    # Load weights
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    return model


def compute_part_scores(
    jet_constituents, particle_masks, part_model, device="cuda", batch_size=512
):
    """
    Run ParT inference on jet constituents to get b-tagging scores.

    Args:
        jet_constituents: numpy array (N_jets, N_constituents, N_features)
        particle_masks: numpy array (N_jets, N_constituents) boolean mask
        part_model: trained ParT model
        device: 'cuda' or 'cpu'
        batch_size: batch size for inference

    Returns:
        b_scores: numpy array (N_jets,) of b-tagging probabilities [0,1]
    """
    part_model = part_model.to(device)
    part_model.eval()

    n_jets = len(jet_constituents)
    all_scores = []

    with torch.no_grad():
        for i in range(0, n_jets, batch_size):
            batch_x = torch.tensor(
                jet_constituents[i : i + batch_size], dtype=torch.float32, device=device
            )
            batch_mask = torch.tensor(
                particle_masks[i : i + batch_size], dtype=torch.bool, device=device
            )

            logits = part_model(batch_x, batch_mask)
            scores = torch.sigmoid(logits).cpu().numpy().flatten()
            all_scores.append(scores)

    return np.concatenate(all_scores)


def extract_baseline_tagger_features(
    jets, tagger_names=["btagPNetB", "btagUParTAK4probb"]
):
    """
    Extract baseline b-tagging scores from jets.

    Args:
        jets: awkward array of jets with tagger fields
        tagger_names: list of tagger field names

    Returns:
        dict mapping tagger_name -> scores array (N_jets, 4) for top 4 jets
    """
    features = {}

    for tagger in tagger_names:
        if tagger in jets.fields:
            # Take top 4 jets
            scores = ak.to_numpy(ak.pad_none(jets[tagger][:, :4], 4, axis=1, clip=True))
            scores = ak.fill_none(scores, 0.0)
            features[tagger] = ak.to_numpy(scores)

    return features
