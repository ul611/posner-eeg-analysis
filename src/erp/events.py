# -*- coding: utf-8 -*-
"""Budowanie tablicy zdarzeń MNE z segmentu Neo."""

import numpy as np

from .constants import EVENT_MAPPING, EVENT_DICT


def build_events(seg, sfreq, event_mapping=None, verbose=True):
    """
    Z segmentu Neo (zdarzenia) buduje events (n_events, 3) dla MNE.
    Zwraca (events, event_dict).
    """
    if event_mapping is None:
        event_mapping = EVENT_MAPPING
    events_list = []
    for event in seg.events:
        event_name = event.name
        event_id = event_mapping.get(event_name, 0)
        times = event.times.magnitude
        samples = (times * sfreq).astype(int)
        for sample in samples:
            events_list.append([sample, 0, event_id])
        if verbose:
            print(f"{event_name}: {len(times)} zdarzeń, kod {event_id}")
    events = np.array(events_list)
    events = events[events[:, 0].argsort()]
    if verbose:
        print(f"\nŁącznie zdarzeń: {len(events)}")
    return events, EVENT_DICT.copy()
