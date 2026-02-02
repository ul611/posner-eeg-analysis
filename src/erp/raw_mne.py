# -*- coding: utf-8 -*-
"""Konwersja bloku Neo na MNE Raw."""

import numpy as np
import mne

from .constants import CH_NAMES_10_20


def block_to_raw(block, ch_names=None, verbose=True):
    """
    Z segmentu 0 bloku Neo wyciąga sygnał EEG (kanały × czas),
    tworzy MNE Raw. Zwraca (raw, eeg_data, sfreq).
    """
    if ch_names is None:
        ch_names = CH_NAMES_10_20
    seg = block.segments[0]
    signal = seg.analogsignals[0]
    eeg_data = signal.magnitude.T  # (n_channels, n_times)
    sfreq = float(signal.sampling_rate)
    if verbose:
        print(f"Kształt danych: {signal.shape}")
        print(f"Dane: {eeg_data.shape}")
        print(f"Częstotliwość: {sfreq} Hz")
        print(f"Kanały: {ch_names}")
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=["eeg"] * len(ch_names),
    )
    raw = mne.io.RawArray(eeg_data, info)
    if verbose:
        print(raw.info)
    return raw, eeg_data, sfreq
