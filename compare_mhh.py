"""
Quick comparison of m_HH resolution between ParT and baseline taggers.

This script processes existing data with different taggers and compares
m_HH reconstruction quality.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def compare_mhh_resolution(datasets):
    """
    Compare m_HH resolution across different tagger configurations.

    Args:
        datasets: dict mapping label -> npz file path

    Plots:
        - m_HH distributions
        - Resolution (sigma/mean) comparison
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    stats = {}

    for label, npz_path in datasets.items():
        data = np.load(npz_path)
        m_hh = data["m_hh"]

        # Filter reasonable range
        mask = (m_hh > 200) & (m_hh < 1000)
        m_hh_filtered = m_hh[mask]

        # Compute statistics
        mean = np.mean(m_hh_filtered)
        std = np.std(m_hh_filtered)
        median = np.median(m_hh_filtered)

        stats[label] = {
            "mean": mean,
            "std": std,
            "median": median,
            "resolution": std / mean,
            "n_events": len(m_hh_filtered),
        }

        # Plot distribution
        axes[0, 0].hist(m_hh_filtered, bins=50, alpha=0.5, label=label, density=True)
        axes[0, 1].hist(
            m_hh_filtered,
            bins=50,
            alpha=0.5,
            label=label,
            density=True,
            cumulative=True,
        )

    axes[0, 0].set_xlabel("$m_{HH}$ [GeV]")
    axes[0, 0].set_ylabel("Normalized counts")
    axes[0, 0].legend()
    axes[0, 0].set_title("$m_{HH}$ Distribution")

    axes[0, 1].set_xlabel("$m_{HH}$ [GeV]")
    axes[0, 1].set_ylabel("Cumulative")
    axes[0, 1].legend()
    axes[0, 1].set_title("Cumulative Distribution")

    # Resolution comparison
    labels_list = list(stats.keys())
    resolutions = [stats[l]["resolution"] for l in labels_list]

    axes[1, 0].bar(range(len(labels_list)), resolutions, tick_label=labels_list)
    axes[1, 0].set_ylabel("Resolution ($\\sigma / \\mu$)")
    axes[1, 0].set_title("$m_{HH}$ Resolution Comparison")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # Print stats table
    print("\n=== m_HH Statistics ===")
    print(
        f"{'Tagger':<20} {'Mean (GeV)':<12} {'Std (GeV)':<12} {'Resolution':<12} {'N_events':<10}"
    )
    print("-" * 80)
    for label, stat in stats.items():
        print(
            f"{label:<20} {stat['mean']:>11.1f} {stat['std']:>11.1f} {stat['resolution']:>11.4f} {stat['n_events']:>9}"
        )

    # Individual Higgs mass plot
    ax = axes[1, 1]
    for label, npz_path in datasets.items():
        data = np.load(npz_path)
        m_h1 = data["m_h1"]
        m_h2 = data["m_h2"]

        # 2D scatter
        mask = (m_h1 > 50) & (m_h1 < 200) & (m_h2 > 50) & (m_h2 < 200)
        ax.scatter(m_h1[mask][:1000], m_h2[mask][:1000], alpha=0.3, s=10, label=label)

    ax.axhline(
        125,
        color="red",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="$m_H = 125$ GeV",
    )
    ax.axvline(125, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("$m_{H1}$ [GeV]")
    ax.set_ylabel("$m_{H2}$ [GeV]")
    ax.set_title("Reconstructed Higgs Masses")
    ax.legend()

    plt.tight_layout()
    plt.savefig("mhh_comparison.png", dpi=150)
    print("\nSaved plot to mhh_comparison.png")

    return stats


if __name__ == "__main__":
    # Example usage
    datasets = {
        "PNet (Offline)": "data/sbi_events_pnet.npz",
        "L1 Ext": "data/sbi_events_l1ext.npz",
        # 'ParT': 'data/sbi_events_part.npz',  # Uncomment when available
    }

    # Check which files exist
    import os

    datasets_available = {k: v for k, v in datasets.items() if os.path.exists(v)}

    if len(datasets_available) == 0:
        print("No datasets found. Run generate_sbi_dataset.py first.")
        print("\nExample commands:")
        print("  # Generate with offline PNet taggers")
        print("  python generate_sbi_dataset.py --output data/sbi_events_pnet.npz")
        print("\n  # Generate with L1 taggers")
        print(
            "  python generate_sbi_dataset.py --output data/sbi_events_l1ext.npz --use-l1"
        )
    else:
        print(f"Comparing {len(datasets_available)} datasets...")
        stats = compare_mhh_resolution(datasets_available)
