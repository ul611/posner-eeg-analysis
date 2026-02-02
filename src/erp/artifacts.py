# -*- coding: utf-8 -*-
"""Odrzucanie artefaktów ocznych i epok z błędnymi odpowiedziami."""

import numpy as np
import mne
import matplotlib.pyplot as plt
from collections import Counter

from .constants import KANALY_OCZNE, TMIN_ARTEFAKT, TMAX_ARTEFAKT, WRONG_ANS


def ptp_stats(epochs_temp, show_hist=True):
    """Statystyka peak-to-peak dla epok; opcjonalnie histogram."""
    ptp_data = np.ptp(epochs_temp.get_data(), axis=2)
    print("\n=== Statystyka peak-to-peak (BEZ F8) ===")
    print(f"Kanałów w analizie: {ptp_data.shape[1]}")
    print(f"Mediana:         {np.median(ptp_data)*1e6:.1f} µV")
    print(f"90. percentyl:   {np.percentile(ptp_data, 90)*1e6:.1f} µV")
    print(f"95. percentyl:   {np.percentile(ptp_data, 95)*1e6:.1f} µV")
    print(f"99. percentyl:   {np.percentile(ptp_data, 99)*1e6:.1f} µV")
    if show_hist:
        plt.figure(figsize=(10, 4))
        plt.hist(ptp_data.flatten() * 1e6, bins=50, alpha=0.7, edgecolor="black")
        plt.xlabel("Amplituda peak-to-peak (µV)")
        plt.ylabel("Liczba")
        plt.title("Rozkład amplitud peak-to-peak (bez F8)")
        plt.axvline(np.percentile(ptp_data, 95) * 1e6, color="#606060", linestyle="--",
                    label=f"95. percentyl ({np.percentile(ptp_data, 95)*1e6:.1f} µV)")
        plt.axvline(60, color="#707070", linestyle="--", label="Typowy próg (60 µV)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
    return ptp_data


def get_ocular_bad_epochs(epochs_temp, smooth_window=8, verbose=True):
    """
    Zwraca indeksy epok do odrzucenia (artefakty oczne) oraz próg (w V).
    Używa wygładzonego sygnału w oknie TMIN_ARTEFAKT–TMAX_ARTEFAKT.
    """
    idx_ocular = [epochs_temp.ch_names.index(ch) for ch in KANALY_OCZNE if ch in epochs_temp.ch_names]
    times = epochs_temp.times
    mask_t = (times >= TMIN_ARTEFAKT) & (times <= TMAX_ARTEFAKT)
    data_full = epochs_temp.get_data()
    data_win = data_full[:, :, mask_t]
    data_ocular = data_win[:, idx_ocular, :]
    smoothed = np.apply_along_axis(
        lambda x: np.convolve(x, np.ones(smooth_window) / smooth_window, mode="same"),
        2, data_ocular,
    )
    ptp_ocular = np.ptp(smoothed, axis=2)
    max_ptp_ocular = np.max(ptp_ocular, axis=1)
    threshold = np.clip(np.percentile(max_ptp_ocular, 95), 60e-6, 100e-6)
    bad_idx = np.where(max_ptp_ocular > threshold)[0]
    if verbose:
        print(f"\nPróg odrzucania (tylko dla {KANALY_OCZNE}): {threshold*1e6:.1f} µV")
        print(f"  Okno: {TMIN_ARTEFAKT*1000:.0f}–{TMAX_ARTEFAKT*1000:.0f} ms")
        print(f"Epok do odrzucenia (artefakty oczne): {len(bad_idx)}")
    return bad_idx, threshold, data_full, times, mask_t, max_ptp_ocular, idx_ocular


def plot_rejected_epochs(epochs_temp, bad_idx, threshold, data_full, times, max_ptp_ocular, idx_ocular):
    """Wizualizacja odrzuconych epok i histogram zaakceptowane vs odrzucone."""
    if len(bad_idx) == 0:
        return
    KANALY_OCZNE = list(np.array(epochs_temp.ch_names)[idx_ocular]) if idx_ocular else []
    mask_t = (times >= TMIN_ARTEFAKT) & (times <= TMAX_ARTEFAKT)
    n_show = len(bad_idx)
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 3 * n_show))
    if n_show == 1:
        axes = [axes]
    for i, ix in enumerate(bad_idx):
        dat = data_full[ix, :, :] * 1e6
        for ch_i, ch_name in enumerate(epochs_temp.ch_names):
            if ch_name in KANALY_OCZNE:
                axes[i].plot(times * 1000, dat[ch_i, :], label=ch_name, linewidth=2, alpha=0.9)
            else:
                axes[i].plot(times * 1000, dat[ch_i, :], color="gray", alpha=0.3, linewidth=0.5)
        axes[i].axvspan(TMIN_ARTEFAKT*1000, TMAX_ARTEFAKT*1000, alpha=0.1, color="gray")
        axes[i].axhline(threshold*1e6, color="#606060", linestyle="--", linewidth=2)
        axes[i].axhline(-threshold*1e6, color="#606060", linestyle="--", linewidth=2)
        axes[i].set_xlim(-200, 800)
        axes[i].set_xlabel("Czas (ms)")
        axes[i].set_ylabel("Amplituda (µV)")
        axes[i].set_title(f"Odrzucona epoka #{ix} (maks. PtP oczny: {max_ptp_ocular[ix]*1e6:.1f} µV)")
        axes[i].legend(loc="upper right", fontsize=8)
        axes[i].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    good_idx = np.setdiff1d(np.arange(len(epochs_temp.events)), bad_idx)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.hist(max_ptp_ocular[good_idx] * 1e6, bins=30, alpha=0.6, label=f"Zaakceptowane ({len(good_idx)})", color="#909090", edgecolor="#707070")
    ax.hist(max_ptp_ocular[bad_idx] * 1e6, bins=30, alpha=0.6, label=f"Odrzucone ({len(bad_idx)})", color="red", edgecolor="black")
    ax.axvline(threshold*1e6, color="#606060", linestyle="--", linewidth=2, label=f"Próg ({threshold*1e6:.1f} µV)")
    ax.set_xlabel("Maks. Peak-to-Peak w kanałach ocznych (µV)")
    ax.set_ylabel("Liczba")
    ax.set_title("Rozkład artefaktów ocznych: Zaakceptowane vs Odrzucone epoki")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def drop_bad_epochs(epochs_temp, ocular_bad_idx, wrong_ans=None, verbose=True):
    """Łączy indeksy artefaktów ocznych i błędnych odpowiedzi, usuwa epoki. Zwraca epochs (kopia)."""
    if wrong_ans is None:
        wrong_ans = WRONG_ANS
    all_bad = sorted(set(ocular_bad_idx) | set(wrong_ans))
    print(f"\nOdrzucanie: {len(ocular_bad_idx)} artefakty oczne + {len(wrong_ans)} błędne odpowiedzi = {len(all_bad)} epok łącznie")
    epochs = epochs_temp.copy()
    if len(all_bad) > 0:
        epochs = epochs.drop(all_bad, reason="REJECT")
    if verbose:
        n_total = len(epochs_temp.events)
        n_keep = len(epochs.events)
        print("\n" + "="*60)
        print("KOŃCOWA STATYSTYKA EPOK")
        print("="*60)
        print(epochs)
        print(f"\nWszystkich epok: {n_total}")
        print(f"Zaakceptowane:   {n_keep} ({n_keep/n_total*100:.1f}%)")
        print(f"Odrzucone:       {n_total - n_keep}")
        for k in ["left_valid", "left_invalid", "right_valid", "right_invalid"]:
            if k in epochs.event_id:
                print(f"  {k}: {len(epochs[k])} prób")
        print("="*60)
    return epochs


def run_artifact_rejection(raw, events, event_dict, wrong_ans=None, plot_rejected=True, show_drop_log=True):
    """
    Pełny pipeline: epochs_temp -> ptp stats -> ocular reject -> drop (ocular + wrong_ans) -> epochs_clean.
    Zwraca (epochs_clean, epochs_temp).
    """
    if wrong_ans is None:
        wrong_ans = WRONG_ANS
    epochs_temp = mne.Epochs(
        raw, events, event_id=event_dict,
        tmin=-0.2, tmax=0.8, baseline=(-0.1, 0),
        preload=True, reject=None, picks="eeg", verbose=False,
    )
    ptp_stats(epochs_temp, show_hist=True)
    bad_idx, threshold, data_full, times, mask_t, max_ptp_ocular, idx_ocular = get_ocular_bad_epochs(epochs_temp, verbose=True)
    if plot_rejected and len(bad_idx) > 0:
        plot_rejected_epochs(epochs_temp, bad_idx, threshold, data_full, times, max_ptp_ocular, idx_ocular)
    epochs_clean = drop_bad_epochs(epochs_temp, bad_idx, wrong_ans=wrong_ans, verbose=True)
    if show_drop_log:
        epochs_clean.plot_drop_log()
        plt.show()
    drop_log_stats(epochs_clean)
    return epochs_clean, epochs_temp


def drop_log_stats(epochs):
    """Wypisuje przyczyny odrzucenia z drop_log."""
    log = epochs.drop_log
    ch_list = []
    for entry in log:
        if entry:
            ch_list.extend(entry)
    from collections import Counter
    cnt = Counter(ch_list)
    print("\n=== Przyczyny odrzucenia epok ===")
    for cause, n in cnt.most_common(10):
        print(f"  {str(cause):12s}: {n:3d} razy ({n/len(log)*100:.1f}% epok)")
    print(f"\nKanały w finalnych epokach ({len(epochs.ch_names)}): {epochs.ch_names}")
