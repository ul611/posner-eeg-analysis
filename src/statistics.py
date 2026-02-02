# -*- coding: utf-8 -*-
"""Funkcje statystyczne dla analizy Posnera."""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t as t_dist


def p_to_stars(p):
    """Zwraca string gwiazdek istotności dla p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def cohens_d_interpretation(d):
    """Interpretacja wielkości efektu Cohen's d."""
    if abs(d) < 0.2:
        return "mały"
    if abs(d) < 0.5:
        return "średni"
    if abs(d) < 0.8:
        return "duży"
    return "bardzo duży"


def posner_effect_stats(df_correct, save_path=None, verbose=True):
    """
    Welch t-test valid vs invalid; zwraca dict z t, p, cohens_d, df_welch,
    ci_lower, ci_upper, effect_ms oraz DataFrame tabeli statystyk.
    """
    valid_rt = df_correct[df_correct["cue_validity"] == "valid"]["rt_clean"]
    invalid_rt = df_correct[df_correct["cue_validity"] == "invalid"]["rt_clean"]

    t_stat, p_val = stats.ttest_ind(valid_rt, invalid_rt, equal_var=False)
    n1, n2 = len(valid_rt), len(invalid_rt)
    s1, s2 = valid_rt.var(), invalid_rt.var()
    df_welch = (s1 / n1 + s2 / n2) ** 2 / (
        (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)
    )
    pooled_std = np.sqrt(
        ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    )
    cohens_d = (invalid_rt.mean() - valid_rt.mean()) / pooled_std
    se_diff = np.sqrt(s1 / n1 + s2 / n2)
    diff = invalid_rt.mean() - valid_rt.mean()
    ci_lower = diff - t_dist.ppf(0.975, df_welch) * se_diff
    ci_upper = diff + t_dist.ppf(0.975, df_welch) * se_diff

    results = {
        "Warunek": ["Trafna (valid)", "Nietrafna (invalid)", "Różnica"],
        "n": [n1, n2, ""],
        "M (ms)": [
            valid_rt.mean() * 1000,
            invalid_rt.mean() * 1000,
            diff * 1000,
        ],
        "SD (ms)": [valid_rt.std() * 1000, invalid_rt.std() * 1000, ""],
        "Median (ms)": [
            valid_rt.median() * 1000,
            invalid_rt.median() * 1000,
            "",
        ],
        "Min (ms)": [valid_rt.min() * 1000, invalid_rt.min() * 1000, ""],
        "Max (ms)": [valid_rt.max() * 1000, invalid_rt.max() * 1000, ""],
    }
    df_stats = pd.DataFrame(results)
    for col in ["M (ms)", "SD (ms)", "Median (ms)", "Min (ms)", "Max (ms)"]:
        df_stats[col] = df_stats[col].apply(
            lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x
        )

    out = {
        "t": t_stat,
        "p": p_val,
        "df_welch": df_welch,
        "cohens_d": cohens_d,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "effect_ms": diff * 1000,
        "df_stats": df_stats,
    }

    if verbose:
        print("\n" + "=" * 90)
        print("TABELA: Efekt Posnera - analiza statystyczna (Valid vs Invalid)")
        print("=" * 90)
        print(df_stats.to_string(index=False))
        print("\n" + "-" * 90)
        print("TEST STATYSTYCZNY (Welch's t-test):")
        print("-" * 90)
        print(f"t({df_welch:.1f}) = {t_stat:.3f}")
        print(f"p = {p_val:.4f} {p_to_stars(p_val)}")
        print(f"Cohen's d = {cohens_d:.3f}")
        print(f"95% CI dla różnicy: [{ci_lower*1000:.1f}, {ci_upper*1000:.1f}] ms")
        print("-" * 90)
        print(f"\nWielkość efektu (Cohen's d = {cohens_d:.3f}): {cohens_d_interpretation(cohens_d)}")
        print(f"Efekt Posnera: {diff*1000:.1f} ms")
        print("=" * 90)

    if save_path:
        df_stats.to_csv(save_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"\n✓ Tabela zapisana do pliku: {save_path}")

    return out


def anova_hand_cue(df_correct, verbose=True):
    """ANOVA 2×2: response_type × cue_validity. Zwraca model i tabelę ANOVA."""
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    model = ols(
        "rt_clean ~ C(response_type) * C(cue_validity)", data=df_correct
    ).fit()
    anova_table = anova_lm(model, typ=2)
    interaction_p = anova_table.loc["C(response_type):C(cue_validity)", "PR(>F)"]
    if verbose:
        print("=" * 80)
        print(anova_table.round(4))
        print(f"\nInterakcja ręka × wskazówka: p = {interaction_p:.4f}")
    return {"model": model, "anova_table": anova_table, "interaction_p": interaction_p}


def block_effects(df_correct, verbose=True, save_path=None):
    """
    Dla każdego bloku: efekt Posnera (ms), t-test, Cohen's d.
    Zwraca DataFrame z kolumnami Blok, n_valid, n_invalid, M_valid, M_invalid,
    Efekt (ms), t, df, p, d, significant.
    """
    block_effects_list = []
    for block in sorted(df_correct["block"].unique()):
        valid = df_correct[
            (df_correct["block"] == block) & (df_correct["cue_validity"] == "valid")
        ]["rt_clean"]
        invalid = df_correct[
            (df_correct["block"] == block) & (df_correct["cue_validity"] == "invalid")
        ]["rt_clean"]
        effect = (invalid.mean() - valid.mean()) * 1000
        t_stat, p_val = stats.ttest_ind(valid, invalid)
        n1, n2 = len(valid), len(invalid)
        s1, s2 = valid.var(), invalid.var()
        df_welch = (s1 / n1 + s2 / n2) ** 2 / (
            (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)
        )
        pooled_std = np.sqrt(
            ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
        )
        cohens_d = effect / 1000 / pooled_std
        block_effects_list.append({
            "Blok": int(block),
            "n_valid": n1,
            "n_invalid": n2,
            "M_valid": valid.mean() * 1000,
            "M_invalid": invalid.mean() * 1000,
            "Efekt (ms)": effect,
            "t": t_stat,
            "df": df_welch,
            "p": p_val,
            "d": cohens_d,
            "significant": p_val < 0.05,
        })

    block_df = pd.DataFrame(block_effects_list)

    if verbose:
        display_df = block_df.copy()
        display_df["M_valid"] = display_df["M_valid"].round(1)
        display_df["M_invalid"] = display_df["M_invalid"].round(1)
        display_df["Efekt (ms)"] = display_df["Efekt (ms)"].round(1)
        display_df["t"] = display_df["t"].round(3)
        display_df["df"] = display_df["df"].round(1)
        display_df["d"] = display_df["d"].round(3)
        display_df["Istotność"] = display_df["p"].apply(p_to_stars)
        display_df["p"] = display_df["p"].apply(
            lambda x: f"{x:.4f}" if x >= 0.001 else "< 0.001"
        )
        output_df = display_df[
            [
                "Blok", "n_valid", "n_invalid", "M_valid", "M_invalid",
                "Efekt (ms)", "t", "df", "p", "d", "Istotność",
            ]
        ]
        output_df.columns = [
            "Blok", "n trafne", "n nietrafne", "M trafne", "M nietrafne",
            "Efekt (ms)", "t", "df", "p", "Cohen's d", "Istotność",
        ]
        print("\n" + "=" * 100)
        print("TABELA: Efekt Posnera według bloków - szczegółowa analiza statystyczna")
        print("=" * 100)
        print(output_df.to_string(index=False))
        print("=" * 100)
        sig_blocks = block_df[block_df["significant"]]["Blok"].tolist()
        nonsig_blocks = block_df[~block_df["significant"]]["Blok"].tolist()
        print(f"\nBloki z istotnym efektem (p < 0.05): {sig_blocks}")
        print(f"Bloki bez istotnego efektu (p ≥ 0.05): {nonsig_blocks}")
        if save_path:
            output_df.to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"\n✓ Tabela zapisana do pliku: {save_path}")

    return block_df


def hand_cue_stats(df_correct, verbose=True):
    """Statystyki RT według ręki i typu wskazówki oraz efekt Posnera per ręka."""
    hand_cue = df_correct.groupby(["response_type", "cue_validity"]).agg(
        n=("rt_clean", "count"),
        M=("rt_clean", lambda x: x.mean() * 1000),
        SD=("rt_clean", lambda x: x.std() * 1000),
        Median=("rt_clean", lambda x: x.median() * 1000),
    ).round(1)
    hand_cue.index = hand_cue.index.set_levels(
        ["Lewa", "Prawa"], level=0
    ).set_levels(
        ["Trafna", "Nietrafna"], level=1
    )
    hand_cue.index.names = ["Ręka", "Wskazówka"]

    if verbose:
        print("\n" + "=" * 80)
        print("TABELA: Statystyki RT według ręki i typu wskazówki")
        print("=" * 80)
        print(hand_cue)
        print("\n=== Efekt Posnera dla każdej ręki ===")
        for hand, hand_pl in [("left", "Lewa"), ("right", "Prawa")]:
            valid = df_correct[
                (df_correct["response_type"] == hand)
                & (df_correct["cue_validity"] == "valid")
            ]["rt_clean"]
            invalid = df_correct[
                (df_correct["response_type"] == hand)
                & (df_correct["cue_validity"] == "invalid")
            ]["rt_clean"]
            effect = (invalid.mean() - valid.mean()) * 1000
            t_stat, p_val = stats.ttest_ind(valid, invalid, equal_var=False)
            print(f"{hand_pl:5s}: Efekt = {effect:5.1f} ms, t = {t_stat:6.3f}, p = {p_val:.4f} {p_to_stars(p_val)}")
        print("=" * 80)

    return hand_cue
