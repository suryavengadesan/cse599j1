"""
Tests for simulator/tracking.py — UsageTracker.

Includes:
  - Unit tests for basic record/summary/reset/cost/export_csv behaviour
  - Property test (task 2.2): totals equal sum of recorded calls
    Feature: conversation-simulator-refactor, Property 5: UsageTracker totals equal sum of recorded calls
"""

import csv
import os
import tempfile

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from simulator.tracking import CallRecord, UsageTracker


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_empty_summary():
    t = UsageTracker()
    s = t.summary()
    assert s["total_tokens"] == 0
    assert s["total_input_tokens"] == 0
    assert s["total_output_tokens"] == 0
    assert s["total_api_calls"] == 0
    assert s["by_type"] == {}


def test_single_record():
    t = UsageTracker()
    t.record("survey", "Alice", 100, 50, "q1")
    s = t.summary()
    assert s["total_input_tokens"] == 100
    assert s["total_output_tokens"] == 50
    assert s["total_tokens"] == 150
    assert s["total_api_calls"] == 1
    assert "survey" in s["by_type"]
    assert s["by_type"]["survey"]["calls"] == 1


def test_reset_clears_state():
    t = UsageTracker()
    t.record("conversation", "Bob", 200, 80)
    t.reset()
    s = t.summary()
    assert s["total_tokens"] == 0
    assert s["total_api_calls"] == 0
    assert s["by_type"] == {}


def test_cost_calculation():
    t = UsageTracker()
    t.record("conversation", "Alice", 1_000_000, 500_000)
    c = t.cost(3.0, 15.0)
    assert abs(c["input_cost_usd"] - 3.0) < 1e-9
    assert abs(c["output_cost_usd"] - 7.5) < 1e-9
    assert abs(c["total_cost_usd"] - 10.5) < 1e-9


def test_export_csv_structure():
    t = UsageTracker()
    t.record("survey", "Alice", 100, 50, "q1")
    t.record("conversation", "Bob", 200, 80)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name

    try:
        t.export_csv(path)
        with open(path, newline="") as fh:
            reader = csv.reader(fh)
            rows = list(reader)
        assert rows[0] == ["call_number", "type", "persona", "question_id",
                           "input_tokens", "output_tokens", "total_tokens"]
        assert len(rows) == 3  # header + 2 data rows
    finally:
        os.unlink(path)


def test_by_type_aggregation():
    t = UsageTracker()
    t.record("survey", "Alice", 100, 50, "q1")
    t.record("survey", "Alice", 80, 40, "q2")
    t.record("conversation", "Bob", 200, 100)
    s = t.summary()
    assert s["by_type"]["survey"]["calls"] == 2
    assert s["by_type"]["survey"]["input_tokens"] == 180
    assert s["by_type"]["conversation"]["calls"] == 1


# ---------------------------------------------------------------------------
# Property test — task 2.2
# Feature: conversation-simulator-refactor, Property 5: UsageTracker totals equal sum of recorded calls
# ---------------------------------------------------------------------------

_call_type = st.sampled_from(["survey", "conversation"])
_token_count = st.integers(min_value=0, max_value=100_000)
_name = st.text(min_size=1, max_size=20)
_q_id = st.one_of(st.none(), st.text(min_size=1, max_size=10))

_record_args = st.tuples(_call_type, _name, _token_count, _token_count, _q_id)


@given(records=st.lists(_record_args, min_size=0, max_size=50))
@settings(max_examples=100)
def test_property_totals_equal_sum_of_records(records):
    """
    Property 5: UsageTracker totals equal sum of recorded calls
    Validates: Requirements 4.1, 4.2
    """
    t = UsageTracker()
    for call_type, persona, inp, out, q_id in records:
        t.record(call_type, persona, inp, out, q_id)

    s = t.summary()

    expected_input = sum(r[2] for r in records)
    expected_output = sum(r[3] for r in records)
    expected_total = expected_input + expected_output

    assert s["total_input_tokens"] == expected_input
    assert s["total_output_tokens"] == expected_output
    assert s["total_tokens"] == expected_total
    assert s["total_api_calls"] == len(records)

    # Per-type aggregates must also be consistent
    for call_type, bucket in s["by_type"].items():
        type_records = [r for r in records if r[0] == call_type]
        assert bucket["input_tokens"] == sum(r[2] for r in type_records)
        assert bucket["output_tokens"] == sum(r[3] for r in type_records)
        assert bucket["calls"] == len(type_records)
