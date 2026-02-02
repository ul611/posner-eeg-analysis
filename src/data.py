# -*- coding: utf-8 -*-
"""Wczytywanie i przygotowanie danych Posnera."""

import pandas as pd


def get_cue_validity(port_value):
    """Określa trafność wskazówki na podstawie portu."""
    if port_value in [1, 4]:
        return "valid"
    if port_value in [2, 8]:
        return "invalid"
    return "unknown"


def load_posner_csv(csv_path):
    """Wczytuje surowe dane z pliku CSV eksperymentu Posnera."""
    return pd.read_csv(csv_path)


def prepare_posner_data(
    df,
    epocs_bad_eye,
    monitor_delay_sec,
    verbose=True,
):
    """
    Filtruje poprawne próby i dodaje zmienne: block, position_in_block,
    trial_order, rt_clean, response_type, cue_validity.

    Zwraca DataFrame (kopia) z tylko poprawnymi próbami, bez epok z ruchami oczu.
    """
    df_correct = df[
        (df["button_resp.corr"] == 1) & (df["trials_3.thisRepN"].isna())
    ].copy()

    col_rep = "trials_5.thisRepN"
    df_correct[col_rep] = df_correct[col_rep].fillna(6)
    df_correct["block"] = df_correct[col_rep]
    df_correct["position_in_block"] = df_correct["thisN"]
    df_correct["trial_order"] = df_correct["block"] * 60 + df_correct["position_in_block"]
    df_correct = df_correct[~df_correct["trial_order"].isin(epocs_bad_eye)]

    df_correct["rt_clean"] = df_correct["button_resp.rt"] - monitor_delay_sec
    df_correct["response_type"] = df_correct["correct"].map({4: "left", 5: "right"})
    df_correct["cue_validity"] = df_correct["port"].apply(get_cue_validity)

    if verbose:
        cue_counts = df_correct["cue_validity"].value_counts()
        print("=== Rozkład typów wskazówek ===")
        print(cue_counts)
        print(f"\nUdział trafnych wskazówek: {cue_counts.get('valid', 0) / len(df_correct):.2%}")
        block_cue_stats = df_correct.groupby(
            ["block", "cue_validity", "response_type"]
        ).agg(
            n_trials=("rt_clean", "count"),
            mean_rt=("rt_clean", "mean"),
            std_rt=("rt_clean", "std"),
            median_rt=("rt_clean", "median"),
        ).round(3)
        print("\n=== Statystyki według bloków i typu wskazówki ===")
        print(block_cue_stats)

    return df_correct


def load_and_prepare_posner(
    csv_path,
    wrong_ans=None,
    epocs_bad_eye=None,
    monitor_delay_sec=None,
    verbose=True,
):
    """
    Wczytuje CSV i zwraca przygotowany DataFrame (poprawne próby, zmienne).
    Używa stałych z constants, jeśli nie podano.
    """
    from .constants import EPOCS_BAD_EYE, MONITOR_DELAY_SEC

    if epocs_bad_eye is None:
        epocs_bad_eye = EPOCS_BAD_EYE
    if monitor_delay_sec is None:
        monitor_delay_sec = MONITOR_DELAY_SEC

    df = load_posner_csv(csv_path)
    return prepare_posner_data(df, epocs_bad_eye, monitor_delay_sec, verbose=verbose)
