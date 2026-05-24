# Быстрый запуск БД

В архиве уже есть готовая SQLite-БД `powerlifting_dataset.sqlite`. Самый быстрый вариант:

```bash
sqlite3 db/powerlifting_dataset.sqlite
.tables
SELECT video_id, filename, exercise, quality_label FROM videos;
SELECT video_id, program_score, program_status FROM analysis_results;
```

Если нужен PostgreSQL:

```bash
cd db
docker compose up -d
cd ..
psql postgresql://powerlift_user:powerlift_pass@localhost:5432/powerlift_db -f db/init.sql
psql postgresql://powerlift_user:powerlift_pass@localhost:5432/powerlift_db -f db/seed.sql
```

Что вставляется в БД:

- `videos` — список 10 видеороликов, упражнение, ручная метка good/bad, тип ошибки, путь к CSV-разметке;
- `analysis_results` — результат работы программы: автоопределение упражнения, оценка, статус, замечания;
- `rep_results` — результаты по отдельным повторениям и числовые метрики.

Координаты ключевых точек хранятся не в SQL, а в CSV-файлах `sample_pose/*.csv`, потому что это удобнее для Git и для повторной обработки.
