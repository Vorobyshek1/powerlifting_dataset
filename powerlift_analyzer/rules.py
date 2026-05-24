from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .geometry import safe_max, safe_mean, safe_min


@dataclass
class Rep:
    start_idx: int
    bottom_idx: int
    end_idx: int


@dataclass
class RepResult:
    rep_no: int
    start_time: float
    bottom_time: float
    end_time: float
    score: int
    status: str
    issues: List[str]
    metrics: Dict[str, float]


def _detect_flexion_reps(values: pd.Series, top_threshold: float, bottom_threshold: float) -> List[Rep]:
    arr = values.to_numpy(dtype=float)
    reps: List[Rep] = []
    state = "search_top"
    start_idx = 0
    bottom_idx = 0
    best_bottom = 999.0

    for i, value in enumerate(arr):
        if np.isnan(value):
            continue
        if state == "search_top":
            if value >= top_threshold:
                start_idx = i
                state = "ready"
        elif state == "ready":
            if value >= top_threshold:
                start_idx = i
            if value <= bottom_threshold:
                bottom_idx = i
                best_bottom = value
                state = "bottom"
        elif state == "bottom":
            if value < best_bottom:
                best_bottom = value
                bottom_idx = i
            if value >= top_threshold and i - start_idx >= 3:
                reps.append(Rep(start_idx=start_idx, bottom_idx=bottom_idx, end_idx=i))
                start_idx = i
                state = "ready"
    return reps


def _score_label(score: int) -> str:
    if score >= 85:
        return "хорошо"
    if score >= 70:
        return "есть небольшие замечания"
    if score >= 50:
        return "нужно исправить"
    return "плохое выполнение"


def _movement_reversal(series: pd.Series, start_idx: int, bottom_idx: int, end_idx: int, expected: str) -> float:
    """Return largest movement against the expected direction after the bottom point."""
    segment = series.iloc[bottom_idx : end_idx + 1].to_numpy(dtype=float)
    segment = segment[~np.isnan(segment)]
    if segment.size < 4:
        return 0.0
    diffs = np.diff(segment)
    if expected == "increase":
        bad = -diffs[diffs < 0]
    else:
        bad = diffs[diffs > 0]
    return float(np.sum(bad)) if bad.size else 0.0


def _segment(features: pd.DataFrame, rep: Rep) -> pd.DataFrame:
    return features.iloc[rep.start_idx : rep.end_idx + 1]


def analyze_squat(features: pd.DataFrame) -> List[RepResult]:
    reps = _detect_flexion_reps(features["knee_angle_smooth"], top_threshold=150, bottom_threshold=120)
    results: List[RepResult] = []
    if not reps and not features.empty:
        min_idx = int(np.nanargmin(features["knee_angle_smooth"].to_numpy(dtype=float)))
        reps = [Rep(start_idx=0, bottom_idx=min_idx, end_idx=len(features) - 1)]

    for n, rep in enumerate(reps, start=1):
        seg = _segment(features, rep)
        bottom = features.iloc[rep.bottom_idx]
        end = features.iloc[rep.end_idx]
        score = 100
        issues: List[str] = []

        knee_min = safe_min(seg["knee_angle_smooth"])
        top_knee = safe_max(seg["knee_angle_smooth"])
        depth_margin = float(bottom["hip_y"] - bottom["knee_y"])
        torso_bottom = float(bottom["torso_vertical_angle_smooth"])
        reversal = _movement_reversal(features["knee_angle_smooth"], rep.start_idx, rep.bottom_idx, rep.end_idx, "increase")
        visibility = safe_mean(seg["visibility"])

        if not (depth_margin > 0.01 or knee_min < 100):
            score -= 25
            issues.append("глубина приседа недостаточная или плохо видна")
        if top_knee < 150:
            score -= 15
            issues.append("в верхней точке колени не полностью выпрямлены")
        if torso_bottom > 55:
            score -= 10
            issues.append("корпус сильно наклонен вперед")
        if reversal > 12:
            score -= 10
            issues.append("есть обратное движение во время подъема")
        if visibility < 0.55:
            score -= 15
            issues.append("ключевые точки тела видны плохо")
        if not issues:
            issues.append("основные критерии выполнены")

        score = max(0, min(100, int(score)))
        results.append(
            RepResult(
                rep_no=n,
                start_time=float(features.iloc[rep.start_idx]["time"]),
                bottom_time=float(bottom["time"]),
                end_time=float(end["time"]),
                score=score,
                status=_score_label(score),
                issues=issues,
                metrics={
                    "knee_angle_min": knee_min,
                    "depth_margin_y": depth_margin,
                    "torso_angle_bottom": torso_bottom,
                    "top_knee_angle": top_knee,
                    "reversal_sum": reversal,
                    "visibility": visibility,
                },
            )
        )
    return results


def analyze_bench(features: pd.DataFrame) -> List[RepResult]:
    reps = _detect_flexion_reps(features["elbow_angle_smooth"], top_threshold=150, bottom_threshold=125)
    results: List[RepResult] = []
    if not reps and not features.empty:
        min_idx = int(np.nanargmin(features["elbow_angle_smooth"].to_numpy(dtype=float)))
        reps = [Rep(start_idx=0, bottom_idx=min_idx, end_idx=len(features) - 1)]

    for n, rep in enumerate(reps, start=1):
        seg = _segment(features, rep)
        bottom = features.iloc[rep.bottom_idx]
        end = features.iloc[rep.end_idx]
        score = 100
        issues: List[str] = []

        elbow_min = safe_min(seg["elbow_angle_smooth"])
        top_elbow = safe_max(seg["elbow_angle_smooth"])
        hip_motion = float(np.nanmax(seg["hip_mid_y"]) - np.nanmin(seg["hip_mid_y"])) if len(seg) else 0.0
        wrist_x_motion = float(np.nanmax(seg["wrist_x"]) - np.nanmin(seg["wrist_x"])) if len(seg) else 0.0
        bottom_frames = int(np.sum(np.abs(seg["elbow_angle_smooth"] - elbow_min) < 4))
        pause_sec = bottom_frames * safe_mean(np.diff(seg["time"]).tolist(), default=0.0)
        reversal = _movement_reversal(features["elbow_angle_smooth"], rep.start_idx, rep.bottom_idx, rep.end_idx, "increase")
        visibility = safe_mean(seg["visibility"])

        if elbow_min > 120:
            score -= 20
            issues.append("амплитуда жима недостаточная или касание груди не видно")
        if top_elbow < 150:
            score -= 15
            issues.append("в верхней точке руки не полностью выпрямлены")
        if hip_motion > 0.05:
            score -= 15
            issues.append("таз заметно смещается во время жима")
        if wrist_x_motion > 0.18:
            score -= 10
            issues.append("траектория кисти сильно смещается по горизонтали")
        if pause_sec < 0.10:
            score -= 5
            issues.append("пауза в нижней точке почти не выражена")
        if reversal > 10:
            score -= 10
            issues.append("есть обратное движение во время подъема")
        if visibility < 0.55:
            score -= 15
            issues.append("ключевые точки тела видны плохо")
        if not issues:
            issues.append("основные критерии выполнены")

        score = max(0, min(100, int(score)))
        results.append(
            RepResult(
                rep_no=n,
                start_time=float(features.iloc[rep.start_idx]["time"]),
                bottom_time=float(bottom["time"]),
                end_time=float(end["time"]),
                score=score,
                status=_score_label(score),
                issues=issues,
                metrics={
                    "elbow_angle_min": elbow_min,
                    "top_elbow_angle": top_elbow,
                    "hip_motion_y": hip_motion,
                    "wrist_x_motion": wrist_x_motion,
                    "pause_sec_proxy": pause_sec,
                    "reversal_sum": reversal,
                    "visibility": visibility,
                },
            )
        )
    return results


def analyze_deadlift(features: pd.DataFrame) -> List[RepResult]:
    reps = _detect_flexion_reps(features["hip_angle_smooth"], top_threshold=150, bottom_threshold=125)
    results: List[RepResult] = []
    if not reps and not features.empty:
        min_idx = int(np.nanargmin(features["hip_angle_smooth"].to_numpy(dtype=float)))
        reps = [Rep(start_idx=0, bottom_idx=min_idx, end_idx=len(features) - 1)]

    for n, rep in enumerate(reps, start=1):
        seg = _segment(features, rep)
        bottom = features.iloc[rep.bottom_idx]
        end = features.iloc[rep.end_idx]
        score = 100
        issues: List[str] = []

        hip_top = safe_max(seg["hip_angle_smooth"])
        knee_top = safe_max(seg["knee_angle_smooth"])
        torso_start = float(bottom["torso_vertical_angle_smooth"])
        wrist_ankle_dx = abs(float(bottom["wrist_x"] - bottom["ankle_x"]))
        wrist_reversal = _movement_reversal(features["wrist_y_smooth"], rep.start_idx, rep.bottom_idx, rep.end_idx, "decrease")
        visibility = safe_mean(seg["visibility"])

        if hip_top < 150:
            score -= 20
            issues.append("нет полного разгибания таза в конце тяги")
        if knee_top < 150:
            score -= 15
            issues.append("колени не полностью выпрямлены в конце тяги")
        if torso_start > 65:
            score -= 10
            issues.append("в начале движения корпус слишком наклонен")
        if wrist_ankle_dx > 0.18:
            score -= 10
            issues.append("штанга находится далеко от линии стопы")
        if wrist_reversal > 0.04:
            score -= 15
            issues.append("видно заметное обратное движение снаряда")
        if visibility < 0.55:
            score -= 15
            issues.append("ключевые точки тела видны плохо")
        if not issues:
            issues.append("основные критерии выполнены")

        score = max(0, min(100, int(score)))
        results.append(
            RepResult(
                rep_no=n,
                start_time=float(features.iloc[rep.start_idx]["time"]),
                bottom_time=float(bottom["time"]),
                end_time=float(end["time"]),
                score=score,
                status=_score_label(score),
                issues=issues,
                metrics={
                    "hip_top_angle": hip_top,
                    "knee_top_angle": knee_top,
                    "torso_start_angle": torso_start,
                    "wrist_ankle_dx": wrist_ankle_dx,
                    "wrist_reversal_y": wrist_reversal,
                    "visibility": visibility,
                },
            )
        )
    return results


def analyze_exercise(features: pd.DataFrame, exercise: str) -> Tuple[List[RepResult], Dict[str, object]]:
    if features.empty:
        return [], {"score": 0, "status": "поза не найдена", "issues": ["нет данных для анализа"]}
    if exercise == "squat":
        reps = analyze_squat(features)
    elif exercise == "bench":
        reps = analyze_bench(features)
    elif exercise == "deadlift":
        reps = analyze_deadlift(features)
    else:
        raise ValueError(f"Unknown exercise: {exercise}")

    if not reps:
        return [], {"score": 0, "status": "повторения не найдены", "issues": ["система не смогла выделить повторение"]}

    score = int(round(float(np.mean([r.score for r in reps]))))
    all_issues: List[str] = []
    for result in reps:
        for issue in result.issues:
            if issue not in all_issues and issue != "основные критерии выполнены":
                all_issues.append(issue)
    if not all_issues:
        all_issues = ["основные критерии выполнены"]
    return reps, {"score": score, "status": _score_label(score), "issues": all_issues}
