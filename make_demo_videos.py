from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from powerlift_analyzer.pose import LANDMARK_NAMES

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "sample_videos"
POSE_DIR = BASE_DIR / "sample_pose"
VIDEO_DIR.mkdir(exist_ok=True)
POSE_DIR.mkdir(exist_ok=True)

W, H = 960, 540
FPS = 30
DURATION = 8.0
FRAMES = int(FPS * DURATION)
REPS = 2


def interp(a, b, q):
    return (a[0] * (1 - q) + b[0] * q, a[1] * (1 - q) + b[1] * q)


def n2p(pt):
    return int(pt[0] * W), int(pt[1] * H)


def progress(i: int) -> float:
    # 0 -> 1 -> 0, repeated. Starts and ends at the top position.
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * REPS * i / (FRAMES - 1)))


def base_row(frame: int, t: float):
    row = {"frame": frame, "time": t, "pose_found": 1.0}
    for name in LANDMARK_NAMES:
        row[f"{name}_x"] = np.nan
        row[f"{name}_y"] = np.nan
        row[f"{name}_z"] = 0.0
        row[f"{name}_visibility"] = 0.95
    return row


def add_pose_points(row, points):
    # Fill both sides so the side selector works. Right side is shifted a little.
    for side in ["left", "right"]:
        dx = 0.0 if side == "left" else 0.025
        for joint in ["shoulder", "elbow", "wrist", "hip", "knee", "ankle"]:
            x, y = points[joint]
            row[f"{side}_{joint}_x"] = x + dx
            row[f"{side}_{joint}_y"] = y
            row[f"{side}_{joint}_z"] = 0.0
            row[f"{side}_{joint}_visibility"] = 0.95
        # simplified foot/heel landmarks
        ax, ay = points["ankle"]
        row[f"{side}_heel_x"] = ax - 0.03 + dx
        row[f"{side}_heel_y"] = ay + 0.02
        row[f"{side}_foot_index_x"] = ax + 0.05 + dx
        row[f"{side}_foot_index_y"] = ay + 0.02
        row[f"{side}_heel_visibility"] = 0.9
        row[f"{side}_foot_index_visibility"] = 0.9
        # hand landmarks near wrist
        wx, wy = points["wrist"]
        for hand in ["pinky", "index", "thumb"]:
            row[f"{side}_{hand}_x"] = wx + dx
            row[f"{side}_{hand}_y"] = wy
            row[f"{side}_{hand}_visibility"] = 0.9
    # head landmarks
    sx, sy = points["shoulder"]
    row["nose_x"] = sx + 0.01
    row["nose_y"] = sy - 0.12
    row["nose_visibility"] = 0.9
    for name, ox, oy in [
        ("left_eye_inner", -0.01, -0.135), ("left_eye", -0.015, -0.135), ("left_eye_outer", -0.02, -0.135),
        ("right_eye_inner", 0.01, -0.135), ("right_eye", 0.015, -0.135), ("right_eye_outer", 0.02, -0.135),
        ("left_ear", -0.035, -0.125), ("right_ear", 0.035, -0.125),
        ("mouth_left", -0.01, -0.105), ("mouth_right", 0.01, -0.105),
    ]:
        row[f"{name}_x"] = sx + ox
        row[f"{name}_y"] = sy + oy
        row[f"{name}_visibility"] = 0.85
    return row


def draw_skeleton(frame, points, exercise: str):
    # floor / bench
    cv2.line(frame, (40, int(0.88 * H)), (W - 40, int(0.88 * H)), (190, 190, 190), 2)
    if exercise == "bench":
        cv2.rectangle(frame, (220, int(0.60 * H)), (780, int(0.65 * H)), (200, 200, 200), -1)

    # body segments
    segments = [("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle"), ("shoulder", "elbow"), ("elbow", "wrist")]
    for a, b in segments:
        cv2.line(frame, n2p(points[a]), n2p(points[b]), (25, 25, 25), 8, cv2.LINE_AA)
    for joint in ["shoulder", "elbow", "wrist", "hip", "knee", "ankle"]:
        cv2.circle(frame, n2p(points[joint]), 10, (50, 50, 50), -1, cv2.LINE_AA)
    # head
    sx, sy = n2p(points["shoulder"])
    cv2.circle(frame, (sx, sy - 65), 26, (50, 50, 50), 3, cv2.LINE_AA)

    # barbell
    if exercise == "squat":
        bx, by = n2p(points["shoulder"])
        cv2.line(frame, (bx - 150, by - 8), (bx + 170, by - 8), (30, 30, 30), 6, cv2.LINE_AA)
        cv2.circle(frame, (bx - 170, by - 8), 18, (30, 30, 30), 4, cv2.LINE_AA)
        cv2.circle(frame, (bx + 190, by - 8), 18, (30, 30, 30), 4, cv2.LINE_AA)
    else:
        wx, wy = n2p(points["wrist"])
        cv2.line(frame, (wx - 150, wy), (wx + 150, wy), (30, 30, 30), 6, cv2.LINE_AA)
        cv2.circle(frame, (wx - 170, wy), 18, (30, 30, 30), 4, cv2.LINE_AA)
        cv2.circle(frame, (wx + 170, wy), 18, (30, 30, 30), 4, cv2.LINE_AA)


def get_points(exercise: str, q: float):
    if exercise == "squat":
        top = {
            "shoulder": (0.50, 0.28), "hip": (0.50, 0.48), "knee": (0.55, 0.67), "ankle": (0.56, 0.84),
            "elbow": (0.45, 0.34), "wrist": (0.40, 0.30),
        }
        bottom = {
            "shoulder": (0.50, 0.45), "hip": (0.38, 0.75), "knee": (0.54, 0.72), "ankle": (0.62, 0.86),
            "elbow": (0.45, 0.49), "wrist": (0.40, 0.45),
        }
    elif exercise == "bench":
        top = {
            "shoulder": (0.42, 0.58), "hip": (0.65, 0.58), "knee": (0.77, 0.73), "ankle": (0.88, 0.76),
            "elbow": (0.42, 0.37), "wrist": (0.42, 0.18),
        }
        bottom = {
            "shoulder": (0.42, 0.58), "hip": (0.65, 0.58), "knee": (0.77, 0.73), "ankle": (0.88, 0.76),
            "elbow": (0.35, 0.45), "wrist": (0.45, 0.35),
        }
    elif exercise == "deadlift":
        top = {
            "shoulder": (0.52, 0.28), "hip": (0.52, 0.50), "knee": (0.55, 0.68), "ankle": (0.56, 0.86),
            "elbow": (0.55, 0.55), "wrist": (0.56, 0.72),
        }
        bottom = {
            "shoulder": (0.64, 0.45), "hip": (0.43, 0.58), "knee": (0.52, 0.68), "ankle": (0.56, 0.86),
            "elbow": (0.60, 0.62), "wrist": (0.56, 0.80),
        }
    else:
        raise ValueError(exercise)
    return {k: interp(top[k], bottom[k], q) for k in top}


def make_clip(stem: str, title: str, exercise: str):
    out_path = VIDEO_DIR / f"{stem}.mp4"
    pose_path = POSE_DIR / f"{stem}.csv"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed for {out_path}")

    rows = []
    for i in range(FRAMES):
        t = i / FPS
        q = progress(i)
        points = get_points(exercise, q)
        frame = np.full((H, W, 3), 246, dtype=np.uint8)
        draw_skeleton(frame, points, exercise)
        cv2.putText(frame, title, (32, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 30), 3, cv2.LINE_AA)
        cv2.putText(frame, "demo video from local sample_videos", (32, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (90, 90, 90), 2, cv2.LINE_AA)
        writer.write(frame)
        rows.append(add_pose_points(base_row(i, t), points))
    writer.release()
    pd.DataFrame(rows).to_csv(pose_path, index=False)
    print(f"created {out_path} and {pose_path}")


if __name__ == "__main__":
    make_clip("squat_demo_local", "Barbell squat demo", "squat")
    make_clip("bench_demo_local", "Bench press demo", "bench")
    make_clip("deadlift_demo_local", "Deadlift demo", "deadlift")
