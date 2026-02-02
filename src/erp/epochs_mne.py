# -*- coding: utf-8 -*-
"""Tworzenie epok MNE i konwersja µV -> V."""

import numpy as np
import mne


def uv_to_v_if_needed(raw, verbose=True):
    """Jeśli dane w Raw są w µV (max > 1e-3 V), konwertuje na V i zwraca nowy Raw."""
    data = raw.get_data()
    if data.max() > 1e-3:
        if verbose:
            print("Dane w µV, konwertuję na V...")
        data = data * 1e-6
        raw = mne.io.RawArray(data, raw.info)
    return raw


def create_epochs(raw, events, event_dict, tmin=-0.2, tmax=0.8, baseline=(-0.1, 0), verbose=True):
    """Tworzy obiekt mne.Epochs bez automatycznego odrzucania."""
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_dict,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        reject=None,
        picks="eeg",
    )
    if verbose:
        print(epochs)
        for k in event_dict:
            print(f"{k}: {len(epochs[k])} trials")
    return epochs


def drop_channel(raw, ch_names_to_drop=("F8",), verbose=True):
    """Usuwa kanały (np. F8) z Raw."""
    raw = raw.copy()
    raw.drop_channels(list(ch_names_to_drop))
    if verbose:
        print(f"Kanały po usunięciu {ch_names_to_drop}: {raw.ch_names}")
    return raw
