# Источники примерных видео

В архив включены локальные учебные MP4-ролики, созданные для демонстрации работы алгоритма. Они лежат в `sample_videos`, а разметка позы лежит в `sample_pose`.

Если нужны реальные открытые ролики, можно скачать их командой:

```bash
python download_public_samples.py
```

Скрипт скачивает короткие `.webm` файлы из Wikimedia Commons:

1. `Squat - exercise demonstration video.webm` — автор FitnessScape, лицензия CC BY 3.0.
2. `Bench press - exercise demonstration video.webm` — автор FitnessScape, лицензия CC BY 3.0.
3. `Deadlift - exercise demonstration video.webm` — автор FitnessScape, лицензия CC BY 3.0.

После скачивания они появятся в `sample_videos` и будут доступны в выпадающем списке приложения.
