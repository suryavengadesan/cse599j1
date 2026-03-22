"""
simulator/utils.py — API log formatting utilities.

Can be imported as a library or run as a CLI:
    python -m simulator.utils <json_file> [--mode full|concise] [--output <file>]
"""

import json
import sys
from typing import IO, List, Dict, Optional


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _default_output_path(json_file: str, mode: str) -> str:
    base = json_file.removesuffix(".json")
    suffix = "readable" if mode == "full" else "concise"
    return f"{base}_{suffix}.txt"


def _write_header(f: IO, title: str, total: int) -> None:
    f.write("=" * 100 + "\n")
    f.write(f"{title}\n")
    f.write(f"Total API Calls: {total}\n")
    f.write("=" * 100 + "\n\n")


def _format_full(logs: List[Dict], f: IO) -> None:
    """Write every message in the history for each call."""
    _write_header(f, "API CALL LOGS - HUMAN READABLE FORMAT", len(logs))

    for log in logs:
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"API CALL #{log['call_number']}\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Type:      {log['type'].upper()}\n")
        f.write(f"Persona:   {log['persona']}\n")
        f.write(f"Timestamp: {log['timestamp']}\n")
        if log.get("metadata"):
            f.write(f"Metadata:  {log['metadata']}\n")
        f.write("\n")

        f.write("-" * 100 + "\nSYSTEM PROMPT:\n" + "-" * 100 + "\n")
        f.write(log["system_prompt"])
        f.write("\n\n")

        f.write("-" * 100 + "\nMESSAGES SENT TO API:\n" + "-" * 100 + "\n\n")
        for i, msg in enumerate(log["messages"], 1):
            f.write(f"[Message {i} - Role: {msg['role'].upper()}]\n")
            f.write(msg["content"])
            f.write("\n\n")

        f.write("-" * 100 + "\nRESPONSE FROM API:\n" + "-" * 100 + "\n")
        f.write(log["response"])
        f.write("\n\n")


def _format_concise(logs: List[Dict], f: IO) -> None:
    """Write only the new message per turn; show full survey calls."""
    _write_header(f, "API CALL LOGS - CONCISE FORMAT", len(logs))

    for log in logs:
        call_type = log["type"]

        f.write("\n" + "=" * 100 + "\n")
        f.write(f"API CALL #{log['call_number']} - {call_type.upper()} - {log['persona']}\n")
        f.write("=" * 100 + "\n")

        if call_type == "survey":
            meta = log.get("metadata", {})
            f.write(f"Stage: {meta.get('stage', 'unknown').upper()}\n")
            f.write(f"Question ID: {meta.get('question_id', 'unknown')}\n\n")

            f.write("SYSTEM PROMPT:\n" + "-" * 80 + "\n")
            f.write(log["system_prompt"])
            f.write("\n\n")

            f.write("QUESTION:\n" + "-" * 80 + "\n")
            f.write(log["messages"][0]["content"])
            f.write("\n\n")

            f.write("RESPONSE:\n" + "-" * 80 + "\n")
            f.write(log["response"])
            f.write("\n\n")

        else:  # conversation
            messages = log["messages"]
            f.write(f"Conversation history length: {len(messages)} messages\n\n")

            # Show system prompt only on the first turn (single message in history)
            if len(messages) <= 1:
                f.write("SYSTEM PROMPT:\n" + "-" * 80 + "\n")
                f.write(log["system_prompt"])
                f.write("\n\n")

            new_message = messages[-1]
            f.write("NEW INPUT TO MODEL:\n" + "-" * 80 + "\n")
            f.write(f"[{new_message['role'].upper()}]\n")
            f.write(new_message["content"])
            f.write("\n\n")

            f.write("RESPONSE:\n" + "-" * 80 + "\n")
            f.write(log["response"])
            f.write("\n\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def format_logs(
    json_file: str,
    mode: str = "full",
    output_file: Optional[str] = None,
) -> str:
    """Format API call logs from *json_file* and write to a text file.

    Args:
        json_file:   Path to the JSON log file produced by debug mode.
        mode:        ``"full"`` (all messages) or ``"concise"`` (new message only).
        output_file: Destination path. Auto-generated from *json_file* if None.

    Returns:
        The path of the written output file.

    Raises:
        ValueError: If *mode* is not ``"full"`` or ``"concise"``.
        FileNotFoundError: If *json_file* does not exist.
        json.JSONDecodeError: If *json_file* is not valid JSON.
    """
    if mode not in ("full", "concise"):
        raise ValueError(f"Invalid mode {mode!r}. Choose 'full' or 'concise'.")

    with open(json_file, "r") as fh:
        logs = json.load(fh)

    if output_file is None:
        output_file = _default_output_path(json_file, mode)

    formatter = _format_full if mode == "full" else _format_concise

    with open(output_file, "w") as fh:
        formatter(logs, fh)

    print(f"✓ Logs written to: {output_file}")
    return output_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m simulator.utils",
        description="Format API call log JSON files into human-readable text.",
    )
    parser.add_argument("json_file", help="Path to the JSON log file")
    parser.add_argument(
        "--mode",
        choices=["full", "concise"],
        default="full",
        help="'full' shows all messages; 'concise' shows only the new message per turn (default: full)",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Output file path (default: auto-generated from input filename)",
    )
    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        format_logs(args.json_file, mode=args.mode, output_file=args.output)
        return 0
    except FileNotFoundError:
        print(f"Error: file '{args.json_file}' not found", file=sys.stderr)
        return 1
    except json.JSONDecodeError:
        print(f"Error: '{args.json_file}' is not valid JSON", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
