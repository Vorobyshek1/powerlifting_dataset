from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .geometry import angle_deg, angle_to_vertical_deg, rolling_median, safe_mean

SIDE_POINTS = ["shoulder", "elbow", "wrist", "hip", "knee", "ankle"]


def _point(row: pd.Series, name: str) -> Tuple[float, float]:
    return float(row.get(f"{name}_x", np.nan)), float(row.get(f"{name}_y", np.nan))


def _visibility_mean(df: pd.DataFrame, side: str) -> float:
    cols = [f"{side}_{p}_visibility" for p in SIDE_POINTS]
    values = []
    for col in cols:
        if col in df:
            values.extend(df[col].dropna().tolist())
    return safe_mean(values, default=0.0)


def choose_body_side(pose_df: pd.DataFrame) -> str:
    left_score = _visibility_mean(pose_df, "left")
    right_score = _visibility_mean(pose_df, "right")
    return "left" if left_score >= right_score else "right"


def build_feature_table(pose_df: pd.DataFrame, side: str | None = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Convert pose landmarks to angles and movement features."""
    if pose_df.empty:
        return pd.DataFrame(), {"side": "left", "visibility": 0.0}
    chosen_side = side or choose_body_side(pose_df)
    rows = []
    for _, row in pose_df.iterrows():
        shoulder = _point(row, f"{chosen_side}_shoulder")
        elbow = _point(row, f"{chosen_side}_elbow")
        wrist = _point(row, f"{chosen_side}_wrist")
        hip = _point(row, f"{chosen_side}_hip")
        knee = _point(row, f"{chosen_side}_knee")
        ankle = _point(row, f"{chosen_side}_ankle")
        opposite_hip = _point(row, "right_hip" if chosen_side == "left" else "left_hip")

        knee_angle = angle_deg(hip, knee, ankle)
        hip_angle = angle_deg(shoulder, hip, knee)
        elbow_angle = angle_deg(shoulder, elbow, wrist)
        torso_vertical_angle = angle_to_vertical_deg(shoulder, hip)
        hip_mid_y = np.nanmean([hip[1], opposite_hip[1]])

        rows.append(
            {
                "frame": row["frame"],
                "time": row["time"],
                "pose_found": row["pose_found"],
                "side": chosen_side,
                "shoulder_x": shoulder[0],
                "shoulder_y": shoulder[1],
                "elbow_x": elbow[0],
                "elbow_y": elbow[1],
                "wrist_x": wrist[0],
                "wrist_y": wrist[1],
                "hip_x": hip[0],
                "hip_y": hip[1],
                "hip_mid_y": hip_mid_y,
                "knee_x": knee[0],
                "knee_y": knee[1],
                "ankle_x": ankle[0],
                "ankle_y": ankle[1],
                "knee_angle": knee_angle,
                "hip_angle": hip_angle,
                "elbow_angle": elbow_angle,
                "torso_vertical_angle": torso_vertical_angle,
                "visibility": np.nanmean(
                    [
                        row.get(f"{chosen_side}_{p}_visibility", np.nan)
                        for p in SIDE_POINTS
                    ]
                ),
            }
        )
    features = pd.DataFrame(rows)
    for col in ["knee_angle", "hip_angle", "elbow_angle", "torso_vertical_angle", "wrist_y", "hip_y", "hip_mid_y"]:
        if col in features:
            features[f"{col}_smooth"] = rolling_median(features[col].to_numpy(), window=5)
    meta = {
        "side": chosen_side,
        "visibility": float(features["visibility"].dropna().mean()) if not features.empty else 0.0,
        "pose_found_ratio": float(features["pose_found"].mean()) if not features.empty else 0.0,
    }
    return features, meta


def detect_exercise(features: pd.DataFrame) -> str:
    """Heuristic exercise type detector.

    Returns one of: squat, bench, deadlift. The rules are simple on purpose:
    this is not a classifier, only automatic selection for the interface.
    """
    if features.empty:
        return "squat"
    knee_range = float(np.nanpercentile(features["knee_angle_smooth"], 90) - np.nanpercentile(features["knee_angle_smooth"], 10))
    hip_range = float(np.nanpercentile(features["hip_angle_smooth"], 90) - np.nanpercentile(features["hip_angle_smooth"], 10))
    elbow_range = float(np.nanpercentile(features["elbow_angle_smooth"], 90) - np.nanpercentile(features["elbow_angle_smooth"], 10))
    torso_mean = float(np.nanmean(features["torso_vertical_angle_smooth"]))
    knee_min = float(np.nanmin(features["knee_angle_smooth"]))
    depth_margin = float(np.nanpercentile(features["hip_y"] - features["knee_y"], 90))

    # Bench press is easiest to separate: torso is close to horizontal and elbows move a lot.
    if torso_mean > 60 and elbow_range > 25:
        return "bench"

    # In squat the knee flexes strongly and the hip often goes below knee level.
    if knee_range >= 30 and (knee_min < 125 or depth_margin > 0.01):
        return "squat"

    # In deadlift the main motion is hip extension.
    if hip_range >= 25:
        return "deadlift"

    if elbow_range >= 20:
        return "bench"
    if knee_range >= 20:
        return "squat"
    return "deadlift"
