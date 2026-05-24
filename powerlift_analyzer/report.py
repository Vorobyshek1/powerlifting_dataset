from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd

from .rules import RepResult

EXERCISE_NAMES = {
    "squat": "присед",
    "bench": "жим лежа",
    "deadlift": "становая тяга",
}


def reps_to_dataframe(reps: Iterable[RepResult]) -> pd.DataFrame:
    rows = []
    for rep in reps:
        row = {
            "Повторение": rep.rep_no,
            "Начало, с": round(rep.start_time, 2),
            "Нижняя точка, с": round(rep.bottom_time, 2),
            "Конец, с": round(rep.end_time, 2),
            "Оценка": rep.score,
            "Статус": rep.status,
            "Замечания": "; ".join(rep.issues),
        }
        for key, value in rep.metrics.items():
            row[key] = round(float(value), 3) if value == value else value
        rows.append(row)
    return pd.DataFrame(rows)


def make_text_report(
    exercise: str,
    summary: Dict[str, object],
    reps: List[RepResult],
    pose_meta: Dict[str, float],
    video_meta: Dict[str, float],
) -> str:
    lines: List[str] = []
    lines.append("ОТЧЕТ ПО АНАЛИЗУ УПРАЖНЕНИЯ")
    lines.append("")
    lines.append(f"Упражнение: {EXERCISE_NAMES.get(exercise, exercise)}")
    lines.append(f"Итоговая оценка: {summary.get('score', 0)} из 100")
    lines.append(f"Итоговый статус: {summary.get('status', '-')}")
    lines.append(f"Найдено повторений: {len(reps)}")
    lines.append(f"Сторона тела для анализа: {pose_meta.get('side', '-')}")
    lines.append(f"Доля кадров с найденной позой: {pose_meta.get('pose_found_ratio', 0):.2f}")
    lines.append(f"Средняя видимость ключевых точек: {pose_meta.get('visibility', 0):.2f}")
    lines.append(f"Проанализированная длительность: {video_meta.get('analyzed_duration_sec', 0):.1f} с")
    lines.append("")
    lines.append("Основные замечания:")
    for issue in summary.get("issues", []):
        lines.append(f"- {issue}")
    lines.append("")
    lines.append("Повторения:")
    for rep in reps:
        lines.append(
            f"{rep.rep_no}. {rep.start_time:.2f}-{rep.end_time:.2f} с, "
            f"оценка {rep.score}, статус: {rep.status}."
        )
        for issue in rep.issues:
            lines.append(f"   - {issue}")
    lines.append("")
    lines.append("Примечание: это автоматический учебный анализ по видео. Он зависит от ракурса, освещения и качества определения позы.")
    return "\n".join(lines)
