# Практическая работа № 5

Линейная регрессия на данных Boston Housing: загрузка и проверка данных, выбор признака по корреляции, расчёт коэффициентов методом наименьших квадратов, предсказание и графики.

## Самостоятельная проверка преподавателем

Скрипт `check_work.py` выполняет всю практическую работу без Jupyter и VS Code. Нужен Python 3.11 или 3.12. После клонирования репозитория откройте терминал в его папке.

На macOS/Linux:

```sh
git clone https://github.com/Monopoly450/-05-ML--.git
cd ./-05-ML--
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-check.txt
python check_work.py
```

На Windows (PowerShell):

```powershell
git clone https://github.com/Monopoly450/-05-ML--.git
cd ./-05-ML--
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-check.txt
.\.venv\Scripts\python.exe check_work.py
```

Если репозиторий уже скачан, вместо повторного клонирования выполните `git pull` в его папке. На этом Mac можно использовать готовое окружение: `conda activate ml05`, затем `python check_work.py`.

Скрипт проверяет размер и корректность данных, выбор признака, коэффициенты и прогноз. Ручной расчёт сравнивается со scikit-learn. Ожидаются `w0 = 34.55384088`, `w1 = -0.95004935`, прогноз `29.63 тыс. долларов` и сообщение **«ПРОВЕРКА ПРОЙДЕНА»**. При несовпадении скрипт завершится с ошибкой и ненулевым кодом выхода. Два графика сохраняются в `results/`. Проверяется выполнение задания и правильность расчётов, а не качество прогноза на новых данных.

## Запуск на этом Mac

1. Откройте эту папку в VS Code и файл `05 ML.ipynb`.
2. Выберите ядро **Python 3.11 (ml05)**, если оно не выбрано автоматически.
3. Нажмите **Run All**. Ноутбук уже содержит результаты проверенного запуска.

Окружение установлено в `/opt/anaconda3/envs/ml05`. В терминале его можно активировать командой `conda activate ml05`.

## Воспроизведение окружения

```sh
conda create --override-channels -c conda-forge -n ml05 python=3.11 pip -y
conda activate ml05
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ml05 --display-name "Python 3.11 (ml05)"
```

Установленные версии зависимостей сохранены в `requirements-lock.txt`.

## Файлы и результат

- `05 ML.ipynb` - выполненный ноутбук с пояснениями и выводом.
- `check_work.py` - вся работа и автоматическая проверка одним скриптом.
- `requirements-check.txt` - проверенные версии библиотек для скрипта.
- `housing.csv` - исходные данные.
- `results/correlations.png`, `results/regression.png` - графики.
- `.vscode` - локальные настройки Python и рекомендуемые расширения.

Модель: `MEDV ≈ 34.55 − 0.95 × LSTAT`. При `LSTAT = 5.18` прогноз составляет **29.63 тыс. долларов**. Это обучение с учителем, поскольку известны целевые значения `MEDV`.

## Git

Репозиторий: `https://github.com/Monopoly450/-05-ML--.git`.

Команда клонирования: `git clone https://github.com/Monopoly450/-05-ML--.git`. Слово `bash` перед `git` не добавляется: в PDF оно обозначает язык блока кода.
