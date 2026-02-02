# -*- coding: utf-8 -*-
"""Analiza asymetrii i statystyki amplitud ERP."""

import numpy as np
import pandas as pd

CHANNELS = ["O1", "O2", "P3", "P4", "C3", "C4"]


def asymmetry_analysis(df_results, verbose=True):
    """Porównanie lewej vs prawej strony i ipsi/contra dla P1."""
    if verbose:
        print("\n" + "="*100)
        print("ANALIZA ASYMETRII: LEWA vs PRAWA STRONA")
        print("="*100)
    df_left = df_results[df_results["Warunek"].str.contains("Lewo")]
    df_right = df_results[df_results["Warunek"].str.contains("Prawo")]
    if verbose:
        print("\nŚREDNIE AMPLITUDY LEWA:")
        for comp in ["P1", "N1", "P3"]:
            v = df_left[df_left["Warunek"] == "Lewo Poprawne"][f"{comp}_Amp_uV"].mean()
            i = df_left[df_left["Warunek"] == "Lewo Niepoprawne"][f"{comp}_Amp_uV"].mean()
            print(f"  {comp}: Poprawne={v:.2f}, Niepoprawne={i:.2f}, Różnica={i-v:+.2f} µV")
        print("\nŚREDNIE AMPLITUDY PRAWA:")
        for comp in ["P1", "N1", "P3"]:
            v = df_right[df_right["Warunek"] == "Prawo Poprawne"][f"{comp}_Amp_uV"].mean()
            i = df_right[df_right["Warunek"] == "Prawo Niepoprawne"][f"{comp}_Amp_uV"].mean()
            print(f"  {comp}: Poprawne={v:.2f}, Niepoprawne={i:.2f}, Różnica={i-v:+.2f} µV")
        print("="*100)
    return df_left, df_right


def full_amplitude_stats(df_results, verbose=True):
    """Pełna analiza statystyczna: podsumowanie P1/N1/P3, N70 latencje, P1 według kanałów."""
    summary = []
    for comp in ["P1", "N1", "P3"]:
        for cond in ["Lewo Poprawne", "Lewo Niepoprawne", "Prawo Poprawne", "Prawo Niepoprawne"]:
            mask = df_results["Warunek"] == cond
            d = df_results.loc[mask, f"{comp}_Amp_uV"]
            if len(d) > 0:
                summary.append({
                    "Komponent": comp, "Warunek": cond,
                    "Średnia": d.mean(), "SD": d.std(),
                    "Min": d.min(), "Max": d.max(), "n": len(d),
                })
    df_sum = pd.DataFrame(summary)
    if verbose:
        print("\n" + "="*100)
        print("PEŁNA ANALIZA STATYSTYCZNA AMPLITUD ERP")
        print("="*100)
        print(df_sum.round(2).to_string(index=False))
        print("="*100)
    return df_sum
