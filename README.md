# Powerlift Quality Analyzer

Учебный репозиторий для дипломной работы по теме автоматического анализа выполнения упражнений по пауэрлифтингу методами компьютерного зрения.

## Что внутри

- `app.py` — графический интерфейс Streamlit;
- `powerlift_analyzer/` — модули анализа видео, позы, признаков и правил;
- `sample_videos/` — 10 учебных видеороликов: хорошие и плохие примеры;
- `sample_pose/` — CSV-разметка позы для этих роликов;
- `dataset/annotations/` — ручная разметка и результаты работы программы;
- `db/` — быстрый запуск SQLite/PostgreSQL и SQL-скрипты;
- `scripts/` — вспомогательные скрипты;
- `tests/` — базовые тесты геометрии;
- `notebooks/` — место для ноутбуков экспериментов.

## Быстрый запуск приложения

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
streamlit run app.py
```

После запуска выбрать видео из папки `sample_videos` и нажать кнопку анализа.

## Быстрый запуск БД

SQLite уже лежит в `db/powerlifting_dataset.sqlite`. Для PostgreSQL:

```bash
cd db
docker compose up -d
cd ..
psql postgresql://powerlift_user:powerlift_pass@localhost:5432/powerlift_db -f db/init.sql
psql postgresql://powerlift_user:powerlift_pass@localhost:5432/powerlift_db -f db/seed.sql
```

## Что заливать в Git

Заливать весь этот каталог как репозиторий. После загрузки на GitHub/GitVerse в дипломе нужно заменить строку `<ссылка на репозиторий>` на реальную ссылку.
