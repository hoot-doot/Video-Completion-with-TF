"""Video input/output and conversion to a third-order tensor."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_video_tensor(
    path: str | Path,
    size: tuple[int, int] = (128, 128),
    max_frames: int = 80,
    frame_stride: int = 1,
    grayscale: bool = True,
) -> np.ndarray:
    """Read a real video into H x W x T (or H x W x T x 3), scaled to [0, 1].

    The primary pipeline is grayscale because it explicitly studies a
    third-order tensor. RGB is retained as an optional fourth-order extension.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    frames: list[np.ndarray] = []
    index = 0
    while len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if index % frame_stride == 0:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float64) / 255.0)
        index += 1
    capture.release()
    if not frames:
        raise ValueError("No frames were extracted. Check video and parameters.")
    if grayscale:
        return np.stack(frames, axis=2)
    return np.stack(frames, axis=2)


def write_grayscale_video(tensor: np.ndarray, path: str | Path, fps: float = 15.0) -> None:
    """Write an H x W x T normalized tensor as a browser-playable MP4 preview.

    H.264/AVC is attempted first because HTML video players generally cannot
    decode OpenCV's common ``mp4v`` output. ``mp4v`` is retained only as a
    fallback for OpenCV builds without an H.264 encoder.
    """
    if tensor.ndim != 3:
        raise ValueError("write_grayscale_video expects an H x W x T tensor")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = tensor.shape
    writer = None
    for codec in ("avc1", "H264", "mp4v"):
        candidate = cv2.VideoWriter(
            str(destination), cv2.VideoWriter_fourcc(*codec), fps, (w, h), True
        )
        if candidate.isOpened():
            writer = candidate
            break
        candidate.release()
    if writer is None:
        raise RuntimeError(f"Could not create video: {destination}")
    for frame in np.moveaxis(np.clip(tensor, 0, 1), 2, 0):
        uint8_frame = (frame * 255).astype(np.uint8)
        writer.write(cv2.cvtColor(uint8_frame, cv2.COLOR_GRAY2BGR))
    writer.release()
