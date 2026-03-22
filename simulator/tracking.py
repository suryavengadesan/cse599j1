"""
simulator/tracking.py — Unified token usage tracker.

Replaces the duplicated _track_token_usage / _track_token_usage_hf methods
and the raw dict stored on the legacy ConversationSimulator.
"""

import csv
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


@dataclass
class CallRecord:
    call_number: int
    call_type: str          # "survey" or "conversation"
    persona_name: str
    input_tokens: int
    output_tokens: int
    question_id: Optional[str]
    timestamp: str


class UsageTracker:
    """Records token consumption across API calls and provides aggregated statistics."""

    def __init__(self) -> None:
        self._records: List[CallRecord] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def record(
        self,
        call_type: str,
        persona_name: str,
        input_tokens: int,
        output_tokens: int,
        question_id: Optional[str] = None,
    ) -> None:
        """Append a new call record."""
        self._records.append(
            CallRecord(
                call_number=len(self._records) + 1,
                call_type=call_type,
                persona_name=persona_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                question_id=question_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        )

    def reset(self) -> None:
        """Clear all recorded calls."""
        self._records = []

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        """Return totals and per-type breakdowns.

        Keys:
            total_input_tokens, total_output_tokens, total_tokens,
            total_api_calls, by_type
        """
        total_input = sum(r.input_tokens for r in self._records)
        total_output = sum(r.output_tokens for r in self._records)

        by_type: Dict[str, Dict] = {}
        for r in self._records:
            bucket = by_type.setdefault(
                r.call_type,
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "calls": 0},
            )
            bucket["input_tokens"] += r.input_tokens
            bucket["output_tokens"] += r.output_tokens
            bucket["total_tokens"] += r.input_tokens + r.output_tokens
            bucket["calls"] += 1

        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_api_calls": len(self._records),
            "by_type": by_type,
        }

    def cost(
        self,
        input_price_per_million: float,
        output_price_per_million: float,
    ) -> Dict:
        """Estimate cost in USD given per-million-token prices."""
        s = self.summary()
        input_cost = (s["total_input_tokens"] / 1_000_000) * input_price_per_million
        output_cost = (s["total_output_tokens"] / 1_000_000) * output_price_per_million
        return {
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": input_cost + output_cost,
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, filepath: Optional[str] = None) -> str:
        """Write records to CSV.

        If *filepath* is None, writes to
        ``results/debug/token_counts/token_counts_<timestamp>.csv``.

        Returns the path written.
        """
        if filepath is None:
            os.makedirs("results/debug/token_counts", exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filepath = f"results/debug/token_counts/token_counts_{ts}.csv"

        with open(filepath, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "call_number",
                    "type",
                    "persona",
                    "question_id",
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                ]
            )
            for r in self._records:
                writer.writerow(
                    [
                        r.call_number,
                        r.call_type,
                        r.persona_name,
                        r.question_id or "",
                        r.input_tokens,
                        r.output_tokens,
                        r.input_tokens + r.output_tokens,
                    ]
                )

        return filepath
