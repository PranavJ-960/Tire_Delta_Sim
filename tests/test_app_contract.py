"""Regression checks for the Streamlit module's reusable backend contract.

``app.py`` currently defines its UI at import time and exposes no standalone
helper functions.  The UI delegates these calculations to ``explore`` and
``optimizer``; their behavior is covered in the dedicated test modules.
"""

import ast
from pathlib import Path


def test_app_delegates_to_shared_analysis_helpers():
    tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
    imported_names = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module in {"explore", "optimizer"}
        for alias in node.names
    }

    assert {
        "load_race",
        "clean_laps",
        "attach_tire_age",
        "compute_pit_loss",
        "fit_degradation_models",
        "optimize_strategy",
        "compare_driver_to_optimal",
        "format_strategy",
    }.issubset(imported_names)
