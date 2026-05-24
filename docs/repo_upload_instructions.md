# Как быстро загрузить в репозиторий

```bash
git init
git add .
git commit -m "add powerlifting dataset and analyzer"
git branch -M main
git remote add origin <ссылка на пустой репозиторий>
git push -u origin main
```

После загрузки можно дать в дипломе такие ссылки:

- код программы: `<repo>/tree/main/powerlift_analyzer`;
- видео: `<repo>/tree/main/sample_videos`;
- разметка: `<repo>/tree/main/sample_pose`;
- таблицы датасета: `<repo>/tree/main/dataset/annotations`;
- БД: `<repo>/tree/main/db`.
