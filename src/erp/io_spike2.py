# -*- coding: utf-8 -*-
"""Wczytywanie danych Spike2 (Neo) i przesunięcie zdarzeń o opóźnienie monitora."""

import quantities as pq
from neo.io import Spike2IO
from neo.core import Event

from .constants import MONITOR_DELAY_SEC


def load_smr_block(filename, verbose=True):
    """Wczytuje plik .smr i zwraca block (Neo)."""
    reader = Spike2IO(filename=filename)
    block = reader.read_block()
    if verbose:
        print("Dostępne sygnały:")
        for i, seg in enumerate(block.segments):
            print(f"\nSegment {i}:")
            for j, sig in enumerate(seg.analogsignals):
                print(f"  Kanał {j}: {sig.name}, częstotliwość: {sig.sampling_rate}, długość: {sig.shape}")
            for k, event in enumerate(seg.events):
                print(f"  Zdarzenia {k}: {event.name}, liczba: {len(event.times)}")
    return block


def shift_events_42ms(block):
    """Przesuwa czasy zdarzeń o MONITOR_DELAY_SEC w każdym segmencie."""
    delay = MONITOR_DELAY_SEC * pq.s
    for seg in block.segments:
        new_events = []
        for event in seg.events:
            shifted_times = event.times + delay
            new_events.append(Event(times=shifted_times, name=event.name))
        seg.events[:] = new_events
    return block
