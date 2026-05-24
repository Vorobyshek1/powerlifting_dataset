from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from powerlift_analyzer.features import build_feature_table, detect_exercise
from powerlift_analyzer.pose import PoseExtractionConfig, extract_pose_series
from powerlift_analyzer.report import EXERCISE_NAMES, make_text_report, reps_to_dataframe
from powerlift_analyzer.rules import analyze_exercise
from powerlift_analyzer.video_loader import (
    VideoLoadError,
    copy_to_workdir,
    list_local_videos,
    save_uploaded_video,
)

st.set_page_config(page_title="Powerlift Quality Analyzer", page_icon="🏋️", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_VIDEO_DIR = BASE_DIR / "sample_videos"

EXERCISE_OPTIONS = {
    "Авто": "auto",
    "Присед": "squat",
    "Жим лежа": "bench",
    "Становая тяга": "deadlift",
}


def _exercise_label(code: str) -> str:
    return EXERCISE_NAMES.get(code, code)


def _run_analysis(video_path: Path, exercise_label: str, frame_step: int, max_duration: int, model_complexity: int):
    st.video(str(video_path))

    config = PoseExtractionConfig(
        frame_step=int(frame_step),
        max_duration_sec=float(max_duration),
        model_complexity=int(model_complexity),
    )

    try:
        with st.status("Извлечение позы и расчет признаков...", expanded=True) as status:
            pose_df, video_meta = extract_pose_series(video_path, config)
            features_df, pose_meta = build_feature_table(pose_df)
            selected = EXERCISE_OPTIONS[exercise_label]
            exercise = detect_exercise(features_df) if selected == "auto" else selected
            reps, summary = analyze_exercise(features_df, exercise)
            status.update(label="Анализ завершен", state="complete")
    except Exception as exc:
        st.error(f"Ошибка анализа: {exc}")
        st.stop()

    st.subheader("Итог")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Упражнение", _exercise_label(exercise))
    col2.metric("Оценка", f"{summary.get('score', 0)} / 100")
    col3.metric("Повторений", len(reps))
    col4.metric("Кадров с позой", f"{pose_meta.get('pose_found_ratio', 0):.2f}")

    source = video_meta.get("pose_source", "mediapipe")
    if source == "precomputed":
        st.caption("Для этого учебного ролика используется заранее сохраненная разметка позы из папки sample_pose.")
    else:
        st.caption("Поза извлечена из видео через MediaPipe Pose.")

    st.write(f"**Статус:** {summary.get('status', '-')}")
    st.write("**Замечания:**")
    for issue in summary.get("issues", []):
        st.write(f"- {issue}")

    reps_table = reps_to_dataframe(reps)
    if not reps_table.empty:
        st.subheader("Повторения")
        st.dataframe(reps_table, use_container_width=True)

    st.subheader("Графики")
    chart_cols = ["time", "knee_angle_smooth", "hip_angle_smooth", "elbow_angle_smooth", "torso_vertical_angle_smooth"]
    available = [c for c in chart_cols if c in features_df]
    if available:
        chart_df = features_df[available].rename(
            columns={
                "time": "Время, с",
                "knee_angle_smooth": "Угол колена",
                "hip_angle_smooth": "Угол таза",
                "elbow_angle_smooth": "Угол локтя",
                "torso_vertical_angle_smooth": "Наклон корпуса",
            }
        )
        st.line_chart(chart_df.set_index("Время, с"))

    text_report = make_text_report(exercise, summary, reps, pose_meta, video_meta)
    st.subheader("Текстовый отчет")
    st.text_area("Отчет", text_report, height=320)

    csv_bytes = reps_table.to_csv(index=False).encode("utf-8-sig") if not reps_table.empty else b""
    col_a, col_b = st.columns(2)
    col_a.download_button("Скачать отчет TXT", data=text_report.encode("utf-8"), file_name="powerlift_report.txt")
    col_b.download_button("Скачать таблицу CSV", data=csv_bytes, file_name="powerlift_reps.csv", disabled=not bool(csv_bytes))

    with st.expander("Сырые признаки"):
        st.dataframe(features_df.head(300), use_container_width=True)


st.title("Анализ качества упражнений по пауэрлифтингу")
st.write(
    "Программа работает с локальными видеофайлами. Выберите ролик из папки sample_videos "
    "или загрузите свой файл через интерфейс. Ссылки больше не нужны."
)

with st.sidebar:
    st.header("Настройки")
    exercise_label = st.selectbox("Упражнение", list(EXERCISE_OPTIONS.keys()), index=0)
    frame_step = st.slider("Шаг кадров", min_value=1, max_value=10, value=3, help="1 - точнее, но медленнее")
    max_duration = st.slider("Максимальная длительность анализа, секунд", min_value=5, max_value=120, value=30, step=5)
    model_complexity = st.selectbox("Сложность модели позы", [0, 1, 2], index=1)
    st.caption("Лучший ракурс: сбоку, один спортсмен в кадре, тело видно полностью.")

source_mode = st.radio(
    "Источник видео",
    ["Из папки sample_videos", "Загрузить свой файл"],
    horizontal=True,
)

video_path: Path | None = None
work_dir = Path(tempfile.mkdtemp(prefix="powerlift_app_"))

if source_mode == "Из папки sample_videos":
    videos = list_local_videos(SAMPLE_VIDEO_DIR)
    if not videos:
        st.warning("В папке sample_videos пока нет видео. Добавьте туда .mp4, .webm, .mov, .avi или .mkv файл.")
        st.stop()

    selected_video = st.selectbox("Видео из локальной папки", videos, format_func=lambda p: p.name)
    st.caption(f"Путь: {selected_video}")
    run_button = st.button("Проанализировать выбранное видео", type="primary")

    if run_button:
        try:
            video_path = copy_to_workdir(selected_video, work_dir)
        except VideoLoadError as exc:
            st.error(str(exc))
            st.stop()
        _run_analysis(video_path, exercise_label, frame_step, max_duration, model_complexity)
    else:
        st.info("Выберите локальное видео и нажмите кнопку анализа.")
else:
    uploaded = st.file_uploader("Выберите видеофайл", type=["mp4", "mov", "avi", "mkv", "webm"])
    run_button = st.button("Проанализировать загруженное видео", type="primary")

    if run_button:
        try:
            video_path = save_uploaded_video(uploaded, work_dir)
        except VideoLoadError as exc:
            st.error(str(exc))
            st.stop()
        _run_analysis(video_path, exercise_label, frame_step, max_duration, model_complexity)
    else:
        st.info("Загрузите файл и нажмите кнопку анализа.")
