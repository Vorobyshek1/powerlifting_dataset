from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd


@dataclass
class PoseExtractionConfig:
    frame_step: int = 3
    max_duration_sec: float = 30.0
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1


LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


def _empty_row(frame_index: int, time_sec: float) -> Dict[str, float]:
    row: Dict[str, float] = {"frame": frame_index, "time": time_sec, "pose_found": 0.0}
    for name in LANDMARK_NAMES:
        row[f"{name}_x"] = np.nan
        row[f"{name}_y"] = np.nan
        row[f"{name}_z"] = np.nan
        row[f"{name}_visibility"] = 0.0
    return row




def _find_precomputed_pose(video_path: str | Path) -> Path | None:
    """Find saved pose CSV for a local demo video.

    This is used only for bundled educational clips. For normal user videos the
    function returns None and MediaPipe is used as usual.
    """
    path = Path(video_path)
    candidates = [
        path.with_suffix(path.suffix + ".pose.csv"),
        path.with_suffix(".pose.csv"),
        path.parent / "sample_pose" / f"{path.stem}.csv",
        path.parent.parent / "sample_pose" / f"{path.stem}.csv",
        Path(__file__).resolve().parent.parent / "sample_pose" / f"{path.stem}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _video_meta(video_path: str | Path) -> Dict[str, float | str]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {
            "fps": 0.0,
            "total_frames": 0.0,
            "width": 0.0,
            "height": 0.0,
            "duration_sec": 0.0,
            "analyzed_duration_sec": 0.0,
            "sampled_frames": 0.0,
        }
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return {
        "fps": float(fps),
        "total_frames": float(total_frames),
        "width": float(width),
        "height": float(height),
        "duration_sec": float(total_frames / fps) if fps and total_frames else 0.0,
        "analyzed_duration_sec": 0.0,
        "sampled_frames": 0.0,
    }


def _load_precomputed_pose(video_path: str | Path, csv_path: Path, config: PoseExtractionConfig) -> Tuple[pd.DataFrame, Dict[str, float | str]]:
    df = pd.read_csv(csv_path)
    if "frame" not in df or "time" not in df:
        raise RuntimeError(f"Файл разметки позы имеет неправильный формат: {csv_path}")

    df = df[df["time"] <= float(config.max_duration_sec)].copy()
    step = max(1, int(config.frame_step))
    df = df[df["frame"].astype(int) % step == 0].reset_index(drop=True)

    meta = _video_meta(video_path)
    meta["analyzed_duration_sec"] = float(df["time"].max()) if not df.empty else 0.0
    meta["sampled_frames"] = float(len(df))
    meta["pose_source"] = "precomputed"
    meta["pose_csv"] = str(csv_path)
    return df, meta


def extract_pose_series(video_path: str | Path, config: PoseExtractionConfig) -> Tuple[pd.DataFrame, Dict[str, float | str]]:
    """Extract pose landmarks for sampled video frames.

    If a matching CSV file exists in sample_pose, it is used. This makes bundled
    demo videos deterministic. Otherwise the function runs MediaPipe Pose on the
    video frames.
    """
    precomputed = _find_precomputed_pose(video_path)
    if precomputed is not None:
        return _load_precomputed_pose(video_path, precomputed, config)

    try:
        import mediapipe as mp
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("Не установлен mediapipe. Выполните pip install -r requirements.txt") from exc

    path = str(video_path)
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError("Не удалось открыть видеофайл.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    max_frames = int(min(total_frames if total_frames else fps * config.max_duration_sec, fps * config.max_duration_sec))

    rows: List[Dict[str, float]] = []
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=config.model_complexity,
        enable_segmentation=False,
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    ) as pose:
        frame_index = 0
        while frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % max(1, config.frame_step) != 0:
                frame_index += 1
                continue
            time_sec = frame_index / fps if fps else 0.0
            row = _empty_row(frame_index, time_sec)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks:
                row["pose_found"] = 1.0
                for idx, landmark in enumerate(result.pose_landmarks.landmark):
                    if idx >= len(LANDMARK_NAMES):
                        break
                    name = LANDMARK_NAMES[idx]
                    row[f"{name}_x"] = float(landmark.x)
                    row[f"{name}_y"] = float(landmark.y)
                    row[f"{name}_z"] = float(landmark.z)
                    row[f"{name}_visibility"] = float(landmark.visibility)
            rows.append(row)
            frame_index += 1

    capture.release()
    df = pd.DataFrame(rows)
    meta = {
        "fps": float(fps),
        "total_frames": float(total_frames),
        "width": float(width),
        "height": float(height),
        "duration_sec": float(total_frames / fps) if fps and total_frames else 0.0,
        "analyzed_duration_sec": float(max_frames / fps) if fps else 0.0,
        "sampled_frames": float(len(df)),
        "pose_source": "mediapipe",
    }
    return df, meta
