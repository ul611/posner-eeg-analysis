# -*- coding: utf-8 -*-
"""Obliczanie ERP (evoked) i wizualizacja ipsi/contra."""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import mne

# Kolory
C_VALID, C_INVALID = "#2d2d2d", "#6d6d6d"
C_DIFF1, C_DIFF2 = "#404040", "#5a5a5a"
C_P1, C_N1, C_P3 = "#c4d4c4", "#b8c4d8", "#d4c4b8"

PICKS_OCCIPITAL = ["O1", "O2"]
PICKS_PARIETAL = ["P3", "P4"]
PICKS_CENTRAL = ["C3", "C4"]


def compute_evokeds(epochs, verbose=True):
    """Oblicza evoked dla left_valid, left_invalid, right_valid, right_invalid. Zwraca dict."""
    ev = {
        "left_valid": epochs["left_valid"].average(),
        "left_invalid": epochs["left_invalid"].average(),
        "right_valid": epochs["right_valid"].average(),
        "right_invalid": epochs["right_invalid"].average(),
    }
    ev["valid"] = epochs["left_valid", "right_valid"].average()
    ev["invalid"] = epochs["left_invalid", "right_invalid"].average()
    if verbose:
        print("\n✓ ERP utworzone (epoki po odrzuceniu artefaktów)")
        for k in ["left_valid", "left_invalid", "right_valid", "right_invalid"]:
            print(f"  {k}: {len(epochs[k])} epok")
    return ev


def _style_ax(ax, ylim=None):
    ax.axvline(0, color="#444444", linestyle="--", linewidth=1.2, alpha=0.85)
    ax.axhline(0, color="#444444", linestyle="-", linewidth=1.2, alpha=0.85)
    for t in range(-200, 850, 50):
        ax.axvline(t, color="#b0b0b0", linestyle=":", alpha=0.25, linewidth=0.8)
    ax.axvspan(92, 132, alpha=0.35, color=C_P1)
    ax.axvspan(148, 200, alpha=0.35, color=C_N1)
    ax.axvspan(200, 700, alpha=0.35, color=C_P3)
    ax.set_xlim(-200, 800)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlabel("Czas (ms)", fontsize=11)
    ax.grid(alpha=0.25)
    ax.set_xticks(np.arange(-200, 801, 50))
    p1 = Patch(facecolor=C_P1, alpha=0.6, edgecolor="none", label="P1")
    n1 = Patch(facecolor=C_N1, alpha=0.6, edgecolor="none", label="N1")
    p3 = Patch(facecolor=C_P3, alpha=0.6, edgecolor="none", label="LPD/P3")
    return p1, n1, p3


def _plot_ipsi_contra(evoked_valid, evoked_invalid, picks, ipsi_idx, contra_idx,
                      title_prefix, save_path, times_ms, ylim_erp, ylim_diff):
    data_v = evoked_valid.copy().pick_channels(picks).data * 1e6
    data_i = evoked_invalid.copy().pick_channels(picks).data * 1e6
    diff = data_v - data_i
    ipsi_v, ipsi_i = data_v[ipsi_idx, :], data_i[ipsi_idx, :]
    contra_v, contra_i = data_v[contra_idx, :], data_i[contra_idx, :]
    ipsi_d, contra_d = diff[ipsi_idx, :], diff[contra_idx, :]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes[0, 0].plot(times_ms, ipsi_v, label="Poprawna", color=C_VALID, linewidth=2)
    axes[0, 0].plot(times_ms, ipsi_i, label="Niepoprawna", color=C_INVALID, linestyle="--", linewidth=2)
    axes[0, 0].set_ylabel("Amplituda (µV)")
    axes[0, 0].set_title(f"{title_prefix} ipsi Poprawna vs Niepoprawna")
    p1, n1, p3 = _style_ax(axes[0, 0], ylim=ylim_erp)
    axes[0, 0].legend(axes[0, 0].get_legend_handles_labels()[0] + [p1, n1, p3],
                      axes[0, 0].get_legend_handles_labels()[1] + ["P1", "N1", "LPD/P3"], loc="upper right")
    axes[0, 1].plot(times_ms, contra_v, label="Poprawna", color=C_VALID, linewidth=2)
    axes[0, 1].plot(times_ms, contra_i, label="Niepoprawna", color=C_INVALID, linestyle="--", linewidth=2)
    axes[0, 1].set_ylabel("Amplituda (µV)")
    axes[0, 1].set_title(f"{title_prefix} contra Poprawna vs Niepoprawna")
    _style_ax(axes[0, 1], ylim=ylim_erp)
    axes[0, 1].legend(loc="upper right")
    axes[1, 0].plot(times_ms, ipsi_d, color=C_DIFF1, linewidth=2.5)
    axes[1, 0].set_ylabel("Różnica amplitudy (µV)")
    axes[1, 0].set_title(f"{title_prefix} ipsi Poprawna − Niepoprawna")
    _style_ax(axes[1, 0], ylim=ylim_diff)
    axes[1, 0].legend([Patch(facecolor=C_P1, alpha=0.6), Patch(facecolor=C_N1, alpha=0.6), Patch(facecolor=C_P3, alpha=0.6)], ["P1", "N1", "LPD/P3"])
    axes[1, 1].plot(times_ms, contra_d, color=C_DIFF2, linewidth=2.5)
    axes[1, 1].set_ylabel("Różnica amplitudy (µV)")
    axes[1, 1].set_title(f"{title_prefix} contra Poprawna − Niepoprawna")
    _style_ax(axes[1, 1], ylim=ylim_diff)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def get_global_ylim(evoked_dict):
    """Oblicza wspólne limity Y dla ERP i różnicy z evoked (O1,O2,P3,P4,C3,C4)."""
    picks_list = [PICKS_OCCIPITAL, PICKS_PARIETAL, PICKS_CENTRAL]
    all_erp, all_diff = [], []
    for picks in picks_list:
        for k in ["left_valid", "left_invalid", "right_valid", "right_invalid"]:
            d = evoked_dict[k].copy().pick_channels(picks).data * 1e6
            all_erp.append(d.ravel())
        for side in ["left", "right"]:
            v = evoked_dict[f"{side}_valid"].copy().pick_channels(picks).data * 1e6
            i = evoked_dict[f"{side}_invalid"].copy().pick_channels(picks).data * 1e6
            all_diff.append((v - i).ravel())
    ymax_erp = float(np.ceil(np.max(np.abs(np.concatenate(all_erp))) * 1.05))
    ymax_diff = float(np.ceil(np.max(np.abs(np.concatenate(all_diff))) * 1.05))
    return (-ymax_erp, ymax_erp), (-ymax_diff, ymax_diff)


def plot_all_erp(evoked_dict, save_prefix="ERP", output_dir="results"):
    """Rysuje wszystkie panele ERP (LEFT/RIGHT occipital, parietal, central) i zapisuje PNG do output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    times_ms = evoked_dict["left_valid"].times * 1000
    ylim_erp, ylim_diff = get_global_ylim(evoked_dict)
    _plot_ipsi_contra(
        evoked_dict["left_valid"], evoked_dict["left_invalid"],
        PICKS_OCCIPITAL, 0, 1, "LEWA (O1 ipsi, O2 contra)",
        os.path.join(output_dir, f"{save_prefix}_LEFT_ipsi_contra.png"), times_ms, ylim_erp, ylim_diff,
    )
    _plot_ipsi_contra(
        evoked_dict["right_valid"], evoked_dict["right_invalid"],
        PICKS_OCCIPITAL, 1, 0, "PRAWA (O2 ipsi, O1 contra)",
        os.path.join(output_dir, f"{save_prefix}_RIGHT_ipsi_contra.png"), times_ms, ylim_erp, ylim_diff,
    )
    _plot_ipsi_contra(
        evoked_dict["left_valid"], evoked_dict["left_invalid"],
        PICKS_PARIETAL, 0, 1, "LEWA PARIETAL (P3 ipsi, P4 contra)",
        os.path.join(output_dir, f"{save_prefix}_LEFT_Parietal_ipsi_contra.png"), times_ms, ylim_erp, ylim_diff,
    )
    _plot_ipsi_contra(
        evoked_dict["right_valid"], evoked_dict["right_invalid"],
        PICKS_PARIETAL, 1, 0, "PRAWA PARIETAL (P4 ipsi, P3 contra)",
        os.path.join(output_dir, f"{save_prefix}_RIGHT_Parietal_ipsi_contra.png"), times_ms, ylim_erp, ylim_diff,
    )
    _plot_ipsi_contra(
        evoked_dict["left_valid"], evoked_dict["left_invalid"],
        PICKS_CENTRAL, 0, 1, "LEWA CENTRAL (C3 ipsi, C4 contra)",
        os.path.join(output_dir, f"{save_prefix}_LEFT_Central_ipsi_contra.png"), times_ms, ylim_erp, ylim_diff,
    )
    _plot_ipsi_contra(
        evoked_dict["right_valid"], evoked_dict["right_invalid"],
        PICKS_CENTRAL, 1, 0, "PRAWA CENTRAL (C4 ipsi, C3 contra)",
        os.path.join(output_dir, f"{save_prefix}_RIGHT_Central_ipsi_contra.png"), times_ms, ylim_erp, ylim_diff,
    )
