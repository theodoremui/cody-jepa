"""Grounded Factorial Completion evaluation pipeline."""


def run_gfc(*args, **kwargs):
    """Run GFC without eagerly importing the command and I/O layer."""
    from .runner import run_gfc as _run_gfc

    return _run_gfc(*args, **kwargs)


__all__ = ["run_gfc"]
