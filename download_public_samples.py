from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "sample_videos"
VIDEO_DIR.mkdir(exist_ok=True)

SAMPLES = {
    "squat_wikimedia.webm": "https://upload.wikimedia.org/wikipedia/commons/5/5c/Squat_-_exercise_demonstration_video.webm",
    "bench_press_wikimedia.webm": "https://upload.wikimedia.org/wikipedia/commons/d/df/Bench_press_-_exercise_demonstration_video.webm",
    "deadlift_wikimedia.webm": "https://upload.wikimedia.org/wikipedia/commons/6/62/Deadlift_-_exercise_demonstration_video.webm",
}


def download(url: str, target: Path) -> None:
    request = Request(url, headers={"User-Agent": "powerlift-quality-app/1.0"})
    with urlopen(request, timeout=60) as response:
        data = response.read()
    if len(data) < 1000:
        raise RuntimeError(f"Too small response for {url}")
    target.write_bytes(data)


if __name__ == "__main__":
    for filename, url in SAMPLES.items():
        target = VIDEO_DIR / filename
        if target.exists():
            print(f"skip: {target.name} already exists")
            continue
        print(f"download: {target.name}")
        download(url, target)
    print("done")
