# -*- coding: utf-8 -*-
"""Moduły analizy ERP (Posner, Spike2, MNE)."""

from .constants import (
    WRONG_ANS,
    MONITOR_DELAY_SEC,
    CH_NAMES_10_20,
    EVENT_MAPPING,
    EVENT_DICT,
    KANALY_OCZNE,
    PEAK_WINDOWS,
)
from .io_spike2 import load_smr_block, shift_events_42ms
from .raw_mne import block_to_raw
from .events import build_events
from .epochs_mne import uv_to_v_if_needed, create_epochs, drop_channel
from .artifacts import (
    ptp_stats,
    get_ocular_bad_epochs,
    drop_bad_epochs,
    run_artifact_rejection,
    drop_log_stats,
)
from .erp import compute_evokeds, plot_all_erp, get_global_ylim
from .peaks import find_peaks_simple, find_peaks_validated, save_peak_tables
from .stats import asymmetry_analysis, full_amplitude_stats

__all__ = [
    "WRONG_ANS",
    "MONITOR_DELAY_SEC",
    "CH_NAMES_10_20",
    "EVENT_MAPPING",
    "EVENT_DICT",
    "KANALY_OCZNE",
    "PEAK_WINDOWS",
    "load_smr_block",
    "shift_events_42ms",
    "block_to_raw",
    "build_events",
    "uv_to_v_if_needed",
    "create_epochs",
    "drop_channel",
    "ptp_stats",
    "get_ocular_bad_epochs",
    "drop_bad_epochs",
    "run_artifact_rejection",
    "drop_log_stats",
    "compute_evokeds",
    "plot_all_erp",
    "get_global_ylim",
    "find_peaks_simple",
    "find_peaks_validated",
    "save_peak_tables",
    "asymmetry_analysis",
    "full_amplitude_stats",
]
