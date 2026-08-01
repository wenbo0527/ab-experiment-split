"""
Backwards-compatible wrapper.

Re-executes the module's `__main__` block by importing it under the
given module path. This preserves the original script semantics
(whatever was inside `if __name__ == "__main__":`) without the
caller having to know which entry function to invoke.

Usage:
    python scripts/ab_split_validator.py
        -> runs abexp.routing.ab_split_validator.__main__
"""
from __future__ import annotations

import os
import runpy
import sys

# Map: legacy script name -> new module path under the abexp package.
_LEGACY_TO_MODULE = {
    "ab_split_validator":      "abexp.routing.ab_split_validator",
    "realtime_prebucket":      "abexp.routing.realtime_prebucket",
    "realtime_remedy":         "abexp.routing.realtime_remedy",
    "realtime_breakthrough":   "abexp.routing.realtime_breakthrough",
    "realtime_adaptive":       "abexp.routing.realtime_adaptive",
    "calibration":             "abexp.routing.calibration",
    "orthogonal_layers":       "abexp.routing.orthogonal_layers",
    "bucket_count_analysis":   "abexp.routing.bucket_count_analysis",
    "streaming_vs_batch":      "abexp.routing.streaming_vs_batch",
    "bias_vs_traffic":         "abexp.routing.bias_vs_traffic",
    "did_cuped_analysis":      "abexp.analysis.did_cuped_analysis",
    "did_cuped_kaggle":        "abexp.analysis.did_cuped_kaggle",
    "did_cuped_consumption":   "abexp.analysis.did_cuped_consumption",
    "beta_binomial":           "abexp.analysis.beta_binomial",
    "outlier_handling":        "abexp.analysis.outlier_handling",
    "experiment_validation_report": "abexp.validation.experiment_validation_report",
    "aa_test":                 "abexp.validation.aa_test",
    "stratified_bucketing":    "abexp.validation.stratified_bucketing",
    "seasonal_early_stop":     "abexp.validation.seasonal_early_stop",
    "full_scale_validation":   "abexp.validation.full_scale_validation",
    "mab_vs_ab":               "abexp.advanced.mab_vs_ab",
    "mab_vs_ab_when":          "abexp.advanced.mab_vs_ab_when",
    "ab_rampup_strategy":      "abexp.advanced.ab_rampup_strategy",
    "finance_repayment_experiment": "abexp.advanced.finance_repayment_experiment",
    "generate_test_data":      "abexp.tools.generate_test_data",
    "monte_carlo_100":         "abexp.tools.monte_carlo_100",
    "sample_size_table":       "abexp.tools.sample_size_table",
}


def _resolve_legacy_name() -> str:
    name = os.path.basename(sys.argv[0])
    if name.endswith(".py"):
        name = name[:-3]
    return name


def main() -> None:
    legacy_name = _resolve_legacy_name()
    module_path = _LEGACY_TO_MODULE.get(legacy_name)
    if module_path is None:
        sys.stderr.write(
            f"Unknown legacy script: {legacy_name}. "
            f"Known: {sorted(_LEGACY_TO_MODULE)}\n"
        )
        sys.exit(1)

    # 1) Remove the scripts/ dir itself from sys.path, otherwise Python
    #    may resolve imports like `from realtime_prebucket import ...`
    #    to scripts/realtime_prebucket.py (the wrapper) instead of the
    #    real abexp.routing.realtime_prebucket module.
    _SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _SCRIPTS_DIR]

    # 2) Make sure the project root (where abexp/ lives) is on sys.path
    #    so that `from abexp.routing.x import ...` resolves correctly.
    _PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    else:
        # Move to the front so the package is found before any other entry.
        sys.path.remove(_PROJECT_ROOT)
        sys.path.insert(0, _PROJECT_ROOT)

    runpy.run_module(module_path, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
