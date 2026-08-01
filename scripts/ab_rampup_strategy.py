"""Backwards-compatible wrapper for the old `python ab_rampup_strategy.py` command.

This thin shim delegates to the canonical module under `abexp.`. New code
should import directly from `abexp`; this file exists only so existing
commands and CI pipelines keep working.

Usage:
    python scripts/ab_rampup_strategy.py
"""
from _run_module import main

if __name__ == "__main__":
    main()
