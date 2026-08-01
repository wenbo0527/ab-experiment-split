"""Backwards-compatible wrapper for the old `python bias_vs_traffic.py` command.

This thin shim delegates to the canonical module under `abexp.`. New code
should import directly from `abexp`; this file exists only so existing
commands and CI pipelines keep working.

Usage:
    python scripts/bias_vs_traffic.py
"""
from _run_module import main

if __name__ == "__main__":
    main()
