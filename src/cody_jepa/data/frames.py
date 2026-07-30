"""Shared frame discovery, ordering, and contiguous-window utilities."""

from pathlib import Path

from .schema import IMAGE_SUFFIXES


def frame_sort_key(path):
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.name.casefold())


def list_frame_paths(frame_dir):
    frame_dir = Path(frame_dir)
    if not frame_dir.is_dir():
        return []
    frames = [
        path
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    ]
    return sorted(frames, key=frame_sort_key)


def contiguous_window_starts(frame_paths, clip_length, *, strict=True):
    if len(frame_paths) < clip_length:
        return []
    if not strict:
        return list(range(len(frame_paths) - clip_length + 1))
    if any(not path.stem.isdigit() for path in frame_paths):
        return []
    indices = [int(path.stem) for path in frame_paths]
    if len(indices) != len(set(indices)):
        return []
    return [
        start
        for start in range(len(indices) - clip_length + 1)
        if indices[start : start + clip_length]
        == list(range(indices[start], indices[start] + clip_length))
    ]


def resolve_frame_dir(frame_dir, repo_root, *, resolve=True):
    path = Path(frame_dir).expanduser()
    value = path if path.is_absolute() else Path(repo_root) / path
    return value.resolve() if resolve else value


def is_within(path, root):
    try:
        Path(path).relative_to(Path(root))
    except ValueError:
        return False
    return True


__all__ = [
    "contiguous_window_starts",
    "frame_sort_key",
    "is_within",
    "list_frame_paths",
    "resolve_frame_dir",
]
