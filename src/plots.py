# -*- coding: utf-8 -*-
"""Funkcje do wizualizacji analizy Posnera."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.patches import Patch


def plot_posner_effect(
    df_correct,
    ax=None,
    figsize=(5, 5),
    save_path=None,
    show=True,
):
    """
    Wykres pudełkowy: trafna vs nietrafna wskazówka (RT w ms).
    Zaznaczone średnie i różnica z p-value.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    valid_data = df_correct[df_correct["cue_validity"] == "valid"]["rt_clean"] * 1000
    invalid_data = df_correct[df_correct["cue_validity"] == "invalid"]["rt_clean"] * 1000

    bp = ax.boxplot(
        [valid_data, invalid_data],
        labels=["Trafna", "Nietrafna"],
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=2),
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=1.5),
        capprops=dict(color="black", linewidth=1.5),
        flierprops=dict(
            marker="o", markerfacecolor="gray", markersize=4, alpha=0.5
        ),
    )
    ax.scatter(
        [1, 2],
        [valid_data.mean(), invalid_data.mean()],
        color="black",
        s=80,
        zorder=10,
        marker="o",
        linewidth=2,
    )
    effect_size = invalid_data.mean() - valid_data.mean()
    valid_rt = df_correct[df_correct["cue_validity"] == "valid"]["rt_clean"]
    invalid_rt = df_correct[df_correct["cue_validity"] == "invalid"]["rt_clean"]
    _, p_val = stats.ttest_ind(valid_rt, invalid_rt, equal_var=False)
    p_text = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
    ax.text(
        1.5,
        560,
        f"Różnica: {effect_size:.1f} ms\n{p_text}",
        ha="center",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", linewidth=1.5),
    )
    ax.set_ylabel("Czas reakcji (ms)", fontsize=12)
    ax.set_title("Efekt Posnera", fontsize=13, fontweight="bold")
    ax.set_ylim(100, 600)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, axis="y", linestyle="--")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return ax


def plot_block_dynamics(
    df_correct,
    ax=None,
    figsize=(8, 5.5),
    save_path=None,
    show=True,
):
    """Wykres słupkowy: efekt Posnera (ms) w każdym bloku z kolorami istotności."""
    from .statistics import block_effects, p_to_stars

    block_df = block_effects(df_correct, verbose=False)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    colors = ["green" if sig else "lightgray" for sig in block_df["significant"]]
    ax.bar(
        block_df["Blok"],
        block_df["Efekt (ms)"],
        color=colors,
        edgecolor="black",
        linewidth=1.5,
    )
    for _, row in block_df.iterrows():
        y_pos = row["Efekt (ms)"] + 3 if row["Efekt (ms)"] > 0 else row["Efekt (ms)"] - 3
        ax.text(
            row["Blok"],
            y_pos,
            f"{row['Efekt (ms)']:.1f}",
            ha="center",
            va="bottom" if row["Efekt (ms)"] > 0 else "top",
            fontsize=9,
            fontweight="bold",
        )
    for _, row in block_df.iterrows():
        star_y = 60
        ax.text(
            row["Blok"],
            star_y,
            p_to_stars(row["p"]),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_ylim(-20, 75)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=1.5)
    ax.set_xlabel("Blok", fontsize=13)
    ax.set_ylabel("Efekt Posnera (ms)\n(nietrafna - trafna)", fontsize=13)
    ax.set_title("Dynamika efektu Posnera w trakcie eksperymentu", fontsize=14, fontweight="bold")
    ax.set_xticks(range(7))
    ax.set_xticklabels([str(i) for i in range(7)])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, axis="y", linestyle="--")
    legend_elements = [
        Patch(facecolor="green", edgecolor="black", linewidth=1.5, label="p < 0.05"),
        Patch(facecolor="lightgray", edgecolor="black", linewidth=1.5, label="p ≥ 0.05"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10, framealpha=0.95)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return ax


def plot_blocks_violin(
    df_correct,
    ax=None,
    figsize=(15, 5.5),
    save_path=None,
    show=True,
):
    """Violin + boxplot RT według bloków i typu wskazówki (trafna/nietrafna)."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    df_plot = df_correct.copy()
    df_plot["block_str"] = df_plot["block"].astype(int).astype(str)

    sns.violinplot(
        data=df_plot,
        x="block_str",
        y="rt_clean",
        hue="cue_validity",
        ax=ax,
        palette=["lightblue", "salmon"],
        inner=None,
        cut=0,
        split=True,
        linewidth=1.5,
    )
    positions_valid = np.arange(7) - 0.1
    positions_invalid = np.arange(7) + 0.1
    for i, block in enumerate(sorted(df_correct["block"].unique())):
        data_valid = df_correct[
            (df_correct["block"] == block) & (df_correct["cue_validity"] == "valid")
        ]["rt_clean"]
        ax.boxplot(
            [data_valid],
            positions=[positions_valid[i]],
            widths=0.10,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(
                facecolor="white", edgecolor="darkblue", linewidth=1.5, alpha=0.7
            ),
            whiskerprops=dict(color="darkblue", linewidth=1.5),
            capprops=dict(color="darkblue", linewidth=1.5),
            medianprops=dict(color="darkblue", linewidth=2),
        )
        data_invalid = df_correct[
            (df_correct["block"] == block) & (df_correct["cue_validity"] == "invalid")
        ]["rt_clean"]
        ax.boxplot(
            [data_invalid],
            positions=[positions_invalid[i]],
            widths=0.10,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(
                facecolor="white", edgecolor="darkred", linewidth=1.5, alpha=0.7
            ),
            whiskerprops=dict(color="darkred", linewidth=1.5),
            capprops=dict(color="darkred", linewidth=1.5),
            medianprops=dict(color="darkred", linewidth=2),
        )
    ax.set_xticks(range(7))
    ax.set_xticklabels([str(i) for i in range(7)])
    ax.set_xlabel("Blok", fontsize=13)
    ax.set_ylabel("Czas reakcji (s)", fontsize=13)
    ax.set_title("Czas reakcji w poszczególnych blokach", fontsize=14, fontweight="bold")
    blue_patch = Patch(
        color="lightblue", edgecolor="darkblue", linewidth=1.5, label="Trafna"
    )
    red_patch = Patch(
        color="salmon", edgecolor="darkred", linewidth=1.5, label="Nietrafna"
    )
    ax.legend(handles=[blue_patch, red_patch], title="Typ wskazówki", fontsize=10, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, axis="y", linestyle="--")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return ax


def plot_hand_cue_interaction(
    df_correct,
    ax=None,
    figsize=(7, 5.5),
    save_path=None,
    show=True,
):
    """Violin + boxplot: interakcja ręka × typ wskazówki z efektem i p per ręka."""
    from .statistics import p_to_stars

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    sns.violinplot(
        data=df_correct,
        x="response_type",
        y="rt_clean",
        hue="cue_validity",
        ax=ax,
        palette=["lightblue", "salmon"],
        split=True,
        inner=None,
        cut=0,
    )
    positions_map = {
        ("left", "valid"): 0 - 0.2,
        ("left", "invalid"): 0 + 0.2,
        ("right", "valid"): 1 - 0.2,
        ("right", "invalid"): 1 + 0.2,
    }
    for (response_type, validity), pos in positions_map.items():
        data = df_correct[
            (df_correct["response_type"] == response_type)
            & (df_correct["cue_validity"] == validity)
        ]["rt_clean"]
        color = "darkblue" if validity == "valid" else "darkred"
        ax.boxplot(
            [data],
            positions=[pos],
            widths=0.15,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(
                facecolor="white", edgecolor=color, linewidth=1.5, alpha=0.8
            ),
            whiskerprops=dict(color=color, linewidth=1.5),
            capprops=dict(color=color, linewidth=1.5),
            medianprops=dict(color=color, linewidth=2),
        )
    stats_data = (
        df_correct.groupby(["response_type", "cue_validity"])["rt_clean"]
        .mean()
        .reset_index()
    )
    for _, row in stats_data.iterrows():
        pos = positions_map[(row["response_type"], row["cue_validity"])]
        ax.scatter(
            pos,
            row["rt_clean"],
            color="black",
            s=80,
            edgecolor="white",
            linewidth=2,
            zorder=10,
            marker="o",
        )
    hand_effects = {}
    for hand in ["left", "right"]:
        valid = df_correct[
            (df_correct["response_type"] == hand)
            & (df_correct["cue_validity"] == "valid")
        ]["rt_clean"]
        invalid = df_correct[
            (df_correct["response_type"] == hand)
            & (df_correct["cue_validity"] == "invalid")
        ]["rt_clean"]
        effect_ms = (invalid.mean() - valid.mean()) * 1000
        _, p_val = stats.ttest_ind(valid, invalid, equal_var=False)
        hand_effects[hand] = {"effect": effect_ms, "p": p_val}
    x_pos = 0
    for hand in ["left", "right"]:
        effect = hand_effects[hand]["effect"]
        p = hand_effects[hand]["p"]
        p_text = "p < 0,001" if p < 0.001 else f"p = {p:.3f}"
        ax.text(
            x_pos,
            0.55,
            f"Δ = {effect:.0f} ms\n{p_text}",
            ha="center",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8),
        )
        x_pos += 1
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("Ręka odpowiedzi", fontsize=13)
    ax.set_ylabel("Czas reakcji (s)", fontsize=13)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Lewa", "Prawa"])
    ax.set_title("Interakcja: ręka × typ wskazówki", fontsize=14, fontweight="bold")
    blue_patch = Patch(
        color="lightblue", edgecolor="darkblue", linewidth=1.5, label="Trafna"
    )
    red_patch = Patch(
        color="salmon", edgecolor="darkred", linewidth=1.5, label="Nietrafna"
    )
    ax.legend(handles=[blue_patch, red_patch], title="Typ wskazówki", loc="upper right", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, axis="y", linestyle="--")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return ax
