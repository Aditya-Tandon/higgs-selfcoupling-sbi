import os
import torch

if not os.getcwd().endswith("root-obj-perf"):
    os.chdir("root-obj-perf")


def extract_wandb_run_id(run_path):
    return run_path[run_path.rfind("/") + 1 :]


def get_wandb_save_path(run_id, wandb_dir=None):
    wandb_dir = os.path.join(os.getcwd(), "wandb") if wandb_dir is None else wandb_dir
    run_dirs = os.listdir(os.path.join(os.getcwd(), "wandb"))
    run_dir_needed = ""
    for run_dir in run_dirs:
        if run_dir.endswith(run_id):
            run_dir_needed = run_dir
            break
    run_dir_path = os.path.join(wandb_dir, run_dir_needed)
    return run_dir_path


"""
Helper functions for ParT model inference.
"""

import numpy as np
import awkward as ak
from data_loading_helpers import one_hot_encode_l1_puppi


def prepare_part_inputs(jets, n_constituents=16):
    """
    Prepare ParT model inputs from jets with constituent information.

    Adapted from make_dataset.py lines 297-362 for SBI event-level processing.

    Args:
        jets: awkward array with 'constituents' field
              jets.constituents shape: (n_events, n_jets, n_const_per_jet)
        n_constituents: number of constituents to pad/clip to

    Returns:
        X: numpy array (n_jets_total, n_constituents, n_features)
        particle_mask: numpy array (n_jets_total, n_constituents) boolean
    """
    # Get constituents from jets
    matched_cands = jets.constituents

    # Get jet-level kinematics for relative features
    j_pt = jets.pt[:, :, None]  # (n_events, n_jets, 1)
    j_eta = jets.eta[:, :, None]
    j_phi = jets.phi[:, :, None]

    # Constituent 4-vector
    m_pt = matched_cands.vector.pt
    m_eta = matched_cands.vector.eta
    m_phi = matched_cands.vector.phi
    m_mass = matched_cands.vector.mass

    # Impact parameters
    m_dxy = matched_cands.dxy
    m_z0 = matched_cands.z0

    # Charge
    m_charge = matched_cands.charge

    # Log pT relative to jet
    log_pt_rel = np.log(np.maximum(m_pt, 1e-3) / np.maximum(j_pt, 1e-3))

    # Delta eta/phi relative to jet axis
    deta = m_eta - j_eta
    dphi = m_phi - j_phi
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]

    # Puppi weight
    m_w = matched_cands.puppiWeight

    # Log DeltaR
    log_dr = np.log(np.maximum(np.sqrt(deta**2 + dphi**2), 1e-3))

    # Particle ID
    m_id = matched_cands.id

    # Pad and fill
    def pad_and_fill(arr, target=n_constituents):
        return ak.fill_none(ak.pad_none(arr, target, axis=2, clip=True), 0.0)

    feature_list = [
        pad_and_fill(m_mass),
        pad_and_fill(m_pt),
        pad_and_fill(m_eta),
        pad_and_fill(m_phi),
        pad_and_fill(m_dxy),
        pad_and_fill(m_z0),
        pad_and_fill(m_charge),
        pad_and_fill(log_pt_rel),
        pad_and_fill(deta),
        pad_and_fill(dphi),
        pad_and_fill(m_w),
        pad_and_fill(log_dr),
        pad_and_fill(m_id),
    ]

    # Stack: (n_events, n_jets, n_constituents, n_features)
    # Need to flatten to (n_jets_total, n_constituents, n_features)
    x_ini = np.stack(
        [ak.to_numpy(ak.flatten(f, axis=1)) for f in feature_list], axis=-1
    )

    # One-hot encode particle ID
    flat_ids = x_ini[..., -1]  # Last feature is ID
    one_hot_ids = one_hot_encode_l1_puppi(flat_ids, n_classes=5)

    # Concatenate: remove ID, add one-hot
    X = np.concatenate([x_ini[..., :-1], one_hot_ids], axis=-1)

    # Generate particle mask
    n_actual_constituents = ak.num(matched_cands, axis=2)  # (n_events, n_jets)
    n_actual_flat = ak.to_numpy(ak.flatten(n_actual_constituents, axis=1))

    # Create mask
    particle_mask = np.zeros((X.shape[0], n_constituents), dtype=bool)
    for i in range(X.shape[0]):
        n_real = min(n_actual_flat[i], n_constituents)
        particle_mask[i, :n_real] = True

    return X, particle_mask


def get_model_ckpt(run_id, ckpt_name="best_part_model.pth", wandb_dir="wandb"):
    run_dir_path = get_wandb_save_path(run_id, wandb_dir=wandb_dir)
    ckpt_path = os.path.join(run_dir_path, "files", ckpt_name)
    ckpt = torch.load(ckpt_path, weights_only=True)
    return ckpt
