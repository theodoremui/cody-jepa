#!/usr/bin/env python3
"""Compatibility entry point for Grounded Factorial Completion."""

from cody_jepa.evaluation.gfc.runner import RecordingRow, main, run_gfc

__all__ = ["RecordingRow", "main", "run_gfc"]


if __name__ == "__main__":
    main()
