# -*- coding: utf-8 -*-
"""Wykrywanie pików ERP (N70, P1, N1, P3) i tabele amplitud/latencji."""

import numpy as np
import pandas as pd
import mne

from .constants import PEAK_WINDOWS

CONDITION_LABELS = [
    ("left_valid", "Lewo Poprawne"),
    ("left_invalid", "Lewo Niepoprawne"),
    ("right_valid", "Prawo Poprawne"),
    ("right_invalid", "Prawo Niepoprawne"),
]
CHANNELS = ["O1", "O2", "P3", "P4", "C3", "C4"]


def find_peaks_simple(evoked_dict, channels=None, peak_windows=None, verbose=True):
    """Proste wyszukiwanie pików w oknach. Zwraca DataFrame z kolumnami Warunek, Kanał, *_Amp_uV, *_Lat_ms."""
    if channels is None:
        channels = CHANNELS
    if peak_windows is None:
        peak_windows = PEAK_WINDOWS
    results = []
    for cond_name, cond_label in CONDITION_LABELS:
        evoked = evoked_dict[cond_name]
        for ch in channels:
            try:
                ch_idx = evoked.ch_names.index(ch)
                data_uV = evoked.data[ch_idx, :] * 1e6
            except Exception:
                data_uV = evoked.copy().pick(ch).get_data(units="uV")[0, 0]
            times = evoked.times
            times_ms = times * 1000
            row = {"Warunek": cond_label, "Kanał": ch}
            for comp_name, (tmin, tmax) in peak_windows.items():
                mask = (times_ms >= tmin) & (times_ms <= tmax)
                if not np.any(mask):
                    row[f"{comp_name}_Amp_uV"] = np.nan
                    row[f"{comp_name}_Lat_ms"] = np.nan
                    continue
                w_data = data_uV[mask]
                w_times = times_ms[mask]
                if comp_name.startswith("P"):
                    peak_idx = np.argmax(w_data)
                else:
                    peak_idx = np.argmin(w_data)
                row[f"{comp_name}_Amp_uV"] = np.round(w_data[peak_idx], 2)
                row[f"{comp_name}_Lat_ms"] = np.round(w_times[peak_idx], 1)
            results.append(row)
    df = pd.DataFrame(results)
    if verbose:
        col_order = ["Warunek", "Kanał"] + [c for comp in peak_windows for c in [f"{comp}_Amp_uV", f"{comp}_Lat_ms"]]
        print("\n" + "="*100)
        print("TABELA: Amplitudy i latencje składowych ERP")
        print("="*100)
        print(df[col_order].to_string(index=False))
        print("="*100)
    return df


def find_peaks_validated(evoked_dict, channels=None, verbose=True):
    """Wyszukiwanie pików z weryfikacją sekwencji P1->N1->P3. Zwraca DataFrame."""
    if channels is None:
        channels = CHANNELS

    def _find_peaks_validated(data_uV, times_ms):
        out = {}
        mask_p1 = (times_ms >= 90) & (times_ms <= 130)
        if np.any(mask_p1):
            p1_data, p1_times = data_uV[mask_p1], times_ms[mask_p1]
            p1_idx = np.argmax(p1_data)
            p1_amp, p1_lat = p1_data[p1_idx], p1_times[p1_idx]
            out["P1"] = (p1_amp, p1_lat)
            mask_n1 = (times_ms >= p1_lat + 20) & (times_ms <= 200)
            if np.any(mask_n1):
                n1_data, n1_times = data_uV[mask_n1], times_ms[mask_n1]
                n1_idx = np.argmin(n1_data)
                n1_amp, n1_lat = n1_data[n1_idx], n1_times[n1_idx]
                out["N1"] = (n1_amp, n1_lat)
                mask_p3 = (times_ms >= n1_lat + 50) & (times_ms <= 600)
                if np.any(mask_p3):
                    p3_data, p3_times = data_uV[mask_p3], times_ms[mask_p3]
                    p3_idx = np.argmax(p3_data)
                    p3_amp, p3_lat = p3_data[p3_idx], p3_times[p3_idx]
                    if p1_lat < n1_lat < p3_lat and abs(p3_amp) >= abs(n1_amp) * 0.7:
                        out["P3"] = (p3_amp, p3_lat)
        mask_n70 = (times_ms >= 50) & (times_ms <= 90)
        if np.any(mask_n70):
            n70_data, n70_times = data_uV[mask_n70], times_ms[mask_n70]
            n70_idx = np.argmin(n70_data)
            out["N70"] = (n70_data[n70_idx], n70_times[n70_idx])
        return out

    results = []
    for cond_name, cond_label in CONDITION_LABELS:
        evoked = evoked_dict[cond_name]
        for ch in channels:
            try:
                ch_idx = evoked.ch_names.index(ch)
                data_uV = evoked.data[ch_idx, :] * 1e6
            except Exception:
                data_uV = evoked.copy().pick(ch).get_data(units="uV")[0, 0]
            times_ms = evoked.times * 1000
            peaks = _find_peaks_validated(data_uV, times_ms)
            row = {"Warunek": cond_label, "Kanał": ch}
            for comp in ["N70", "P1", "N1", "P3"]:
                if comp in peaks:
                    amp, lat = peaks[comp]
                    row[f"{comp}_Amp_uV"] = np.round(amp, 2)
                    row[f"{comp}_Lat_ms"] = np.round(lat, 1)
                else:
                    row[f"{comp}_Amp_uV"] = np.nan
                    row[f"{comp}_Lat_ms"] = np.nan
            results.append(row)
    df = pd.DataFrame(results)
    if verbose:
        col_order = ["Warunek", "Kanał", "N70_Amp_uV", "N70_Lat_ms", "P1_Amp_uV", "P1_Lat_ms", "N1_Amp_uV", "N1_Lat_ms", "P3_Amp_uV", "P3_Lat_ms"]
        print("\n" + "="*100)
        print("TABELA: Amplitudy i latencje (z weryfikacją sekwencji)")
        print("="*100)
        print(df[col_order].to_string(index=False))
        print("="*100)
    return df


def save_peak_tables(df_simple, df_validated, output_dir="results", path_simple="ERP_peak_analysis_single_subject.csv", path_validated="ERP_peaks_validated.csv"):
    """Zapisuje DataFrame pików do CSV w output_dir."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    path_simple = os.path.join(output_dir, path_simple)
    path_validated = os.path.join(output_dir, path_validated)
    df_simple.to_csv(path_simple, index=False, encoding="utf-8-sig")
    df_validated.to_csv(path_validated, index=False, encoding="utf-8-sig")
    print(f"✓ Zapisano: {path_simple}, {path_validated}")
