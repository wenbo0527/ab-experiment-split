"""Backwards-compatible wrapper for the old `python bucket_count_analysis.py` command.

This thin shim delegates to the canonical module under `abexp.`. New code
should import directly from `abexp`; this file exists only so existing
commands and CI pipelines keep working.

Usage:
    python scripts/bucket_count_analysis.py
"""
from _run_module import main

if __name__ == "__main__":
    main()
