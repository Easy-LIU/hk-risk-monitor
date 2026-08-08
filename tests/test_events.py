"""Tests for the static reference-event list: structural sanity checks
only (this is hand-curated data, not logic to verify against expected
computed output)."""

from datetime import date

import pytest

from src.events import REFERENCE_EVENTS

VALID_CATEGORIES = {"shock", "structural"}


def test_every_event_has_a_valid_iso_date():
    for event in REFERENCE_EVENTS:
        date.fromisoformat(event.date)  # raises ValueError if malformed


def test_every_event_has_a_valid_category():
    for event in REFERENCE_EVENTS:
        assert event.category in VALID_CATEGORIES


def test_every_event_has_a_non_empty_label_and_source():
    for event in REFERENCE_EVENTS:
        assert event.label.strip()
        assert event.source.strip()


def test_no_duplicate_dates():
    dates = [event.date for event in REFERENCE_EVENTS]
    assert len(dates) == len(set(dates))


def test_events_are_within_the_tool_sample_period():
    # The rolling engine's sample starts 2015-01-01 (see src/precompute.py).
    # An event before that date would never be visible on the chart.
    for event in REFERENCE_EVENTS:
        assert date.fromisoformat(event.date) >= date(2015, 1, 1)
