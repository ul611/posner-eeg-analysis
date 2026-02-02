# -*- coding: utf-8 -*-
"""Moduły analizy eksperymentu Posnera."""

from .constants import WRONG_ANS, EPOCS_BAD_EYE, MONITOR_DELAY_SEC
from .data import get_cue_validity, load_posner_csv, prepare_posner_data, load_and_prepare_posner
from .statistics import (
    p_to_stars,
    cohens_d_interpretation,
    posner_effect_stats,
    anova_hand_cue,
    block_effects,
    hand_cue_stats,
)
from .plots import (
    plot_posner_effect,
    plot_block_dynamics,
    plot_blocks_violin,
    plot_hand_cue_interaction,
)

__all__ = [
    "WRONG_ANS",
    "EPOCS_BAD_EYE",
    "MONITOR_DELAY_SEC",
    "get_cue_validity",
    "load_posner_csv",
    "prepare_posner_data",
    "load_and_prepare_posner",
    "p_to_stars",
    "cohens_d_interpretation",
    "posner_effect_stats",
    "anova_hand_cue",
    "block_effects",
    "hand_cue_stats",
    "plot_posner_effect",
    "plot_block_dynamics",
    "plot_blocks_violin",
    "plot_hand_cue_interaction",
]
