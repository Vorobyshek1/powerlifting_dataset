from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterable, List

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class VideoLoadError(RuntimeError):
    pass


# Оставлено для совместимости со старой версией проекта.
VideoDownloadError = VideoLoadError


def is_video_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


def list_local_videos(video_dir: str | Path = "sample_videos") -> List[Path]:
    """Return videos from a local folder.

    The app no longer downloads videos from links. It works with files that are
    already stored in sample_videos, or with a file uploaded through the UI.
    """
    folder = Path(video_dir)
    if not folder.exists():
        return []
    videos = [p for p in folder.iterdir() if p.is_file() and is_video_file(p)]
    return sorted(videos, key=lambda p: p.name.lower())


def save_uploaded_video(uploaded_file, output_dir: str | Path | None = None) -> Path:
    """Save Streamlit UploadedFile to a temporary local path."""
    if uploaded_file is None:
        raise VideoLoadError("Файл не выбран.")
    name = Path(uploaded_file.name).name
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise VideoLoadError(f"Неподдерживаемый формат файла. Можно: {allowed}")

    target_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="powerlift_upload_"))
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / name
    with target_path.open("wb") as f:
        f.write(uploaded_file.getbuffer())
    return target_path


def copy_to_workdir(video_path: str | Path, output_dir: str | Path | None = None) -> Path:
    """Copy an existing local video into a working folder and return the copy path."""
    src = Path(video_path)
    if not src.exists():
        raise VideoLoadError(f"Видео не найдено: {src}")
    if not is_video_file(src):
        raise VideoLoadError(f"Файл не похож на видео: {src.name}")
    target_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="powerlift_local_"))
    target_dir.mkdir(parents=True, exist_ok=True)
    dst = target_dir / src.name
    shutil.copy2(src, dst)
    return dst
