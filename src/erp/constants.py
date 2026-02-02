# -*- coding: utf-8 -*-
"""Stałe dla analizy ERP (Posner)."""

# Epoki z błędną odpowiedzią (indeksy od 0)
WRONG_ANS = [95, 153, 172, 173, 174, 186, 339, 405]

# Przesunięcie zdarzeń (monitor delay), sekundy
MONITOR_DELAY_SEC = 0.042

# Nazwy kanałów 10-20 (19 kanałów)
CH_NAMES_10_20 = [
    "Fp1", "F3", "F7", "C3", "T3", "P3", "T5",
    "O1", "Fz", "Cz", "Pz", "Fp2", "F4", "F8", "C4", "T4", "P4", "T6", "O2",
]

# Mapowanie nazw zdarzeń Spike2 -> event_id (left_valid, right_invalid, right_valid, left_invalid)
EVENT_MAPPING = {
    "LewoLewo": 0,   # left valid
    "LewoPraw": 1,   # right invalid
    "PrawPraw": 2,   # right valid
    "PrawLew": 3,    # left invalid
}

EVENT_DICT = {
    "left_valid": 0,
    "right_invalid": 1,
    "right_valid": 2,
    "left_invalid": 3,
}

# Kanały do detekcji artefaktów ocznych
KANALY_OCZNE = ["Fp1", "Fp2", "F7"]

# Okno czasowe dla artefaktów (s)
TMIN_ARTEFAKT, TMAX_ARTEFAKT = 0.0, 0.6

# Okna czasowe pików ERP (ms): (tmin, tmax)
PEAK_WINDOWS = {
    "N70": (50, 85),
    "P1": (90, 130),
    "N1": (130, 200),
    "P3": (200, 600),
}
