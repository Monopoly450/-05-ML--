"""Выполнение и проверка практической работы № 5 без Jupyter.

Установка библиотек: python -m pip install -r requirements-check.txt
Запуск: python check_work.py
"""

from pathlib import Path

import matplotlib

# Сохраняем графики и на компьютерах без графического интерфейса.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


COLUMNS = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV",
]
PROJECT_DIR = Path(__file__).resolve().parent


def check(condition, description):
    """Останавливаем проверку с ненулевым кодом выхода при ошибке."""
    if not condition:
        raise ValueError(f"Проверка не пройдена: {description}")
    print(f"[OK] {description}")


def linear_regression(X, y):
    """Рассчитываем w = (X.T @ X)^(-1) @ X.T @ y вручную."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    Q = np.linalg.inv(X_with_bias.T @ X_with_bias)
    return Q @ X_with_bias.T @ y


def main():
    # 1. Загрузка и проверка данных. Путь не зависит от папки запуска.
    data_path = PROJECT_DIR / "housing.csv"
    data = pd.read_csv(data_path, header=None, sep=r"\s+")
    check(data.shape == (506, 14), "506 объектов и 14 столбцов")
    data.columns = COLUMNS
    check(not data.isna().any().any(), "Пропущенных значений нет")
    check(
        all(pd.api.types.is_numeric_dtype(dtype) for dtype in data.dtypes),
        "Все признаки числовые",
    )
    check(np.isfinite(data.to_numpy()).all(), "Все значения конечные")
    print("\nПервые пять строк:")
    print(data.head().to_string(index=False))

    # 2. Корреляции с целевой ценой MEDV.
    correlations = data.corr()["MEDV"].drop("MEDV")
    abs_correlations = correlations.abs().sort_values()
    selected_feature = abs_correlations.idxmax()
    print("\nКорреляции с MEDV (по убыванию абсолютного значения):")
    print(correlations.loc[abs_correlations.index[::-1]].round(4).to_string())
    check(selected_feature == "LSTAT", "Самая сильная связь по модулю у LSTAT")

    # 3. Наблюдения X и правильные ответы y.
    X = data[[selected_feature]]
    y = data["MEDV"]

    # 4. Ручное вычисление параметров методом наименьших квадратов.
    w = linear_regression(X, y)
    print(f"\nw0 = {w[0]:.8f}")
    print(f"w1 = {w[1]:.8f}")
    print(f"Модель: MEDV = {w[0]:.2f} {w[1]:+.2f} × LSTAT")

    # 5. Независимая проверка через scikit-learn.
    model = LinearRegression().fit(X, y)
    w_sklearn = np.r_[model.intercept_, model.coef_]
    check(
        np.allclose(w, w_sklearn, rtol=1e-10, atol=1e-10),
        "Ручные коэффициенты совпадают с LinearRegression из scikit-learn",
    )
    check(
        np.allclose(w, [34.55384088, -0.95004935], rtol=0, atol=1e-7),
        "Коэффициенты соответствуют ожидаемому результату задания",
    )

    # 6. Предсказание для заданного значения LSTAT.
    x_example = 5.18
    prediction = float(w[0] + w[1] * x_example)
    reference_prediction = float(
        model.predict(pd.DataFrame({selected_feature: [x_example]}))[0]
    )
    print(f"\nПри LSTAT = {x_example}% цена = {prediction:.2f} тыс. долларов")
    check(
        np.isclose(prediction, reference_prediction, rtol=1e-10, atol=1e-10),
        "Предсказание совпадает с результатом scikit-learn",
    )
    check(
        abs(prediction - 29.63) < 0.005,
        "Прогноз после округления составляет 29.63 тыс. долларов",
    )

    # 7. Графики: корреляции и наблюдения с прямой регрессии.
    output_dir = PROJECT_DIR / "results"
    output_dir.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(abs_correlations.index, abs_correlations.values)
    ax.set_title("Модуль корреляции признаков с MEDV")
    ax.set_xlabel("Признак")
    ax.set_ylabel("|Коэффициент корреляции|")
    fig.tight_layout()
    fig.savefig(output_dir / "correlations.png", dpi=160)
    plt.close(fig)

    x_values = X[selected_feature].to_numpy()
    y_pred = w[0] + w[1] * x_values
    order = np.argsort(x_values)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(x_values, y, alpha=0.65, label="Исходные данные")
    ax.plot(
        x_values[order], y_pred[order], color="black", linewidth=2,
        label="Линейная регрессия",
    )
    ax.set_title("Зависимость стоимости домов от LSTAT")
    ax.set_xlabel("LSTAT (%)")
    ax.set_ylabel("MEDV (тыс. долларов)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "regression.png", dpi=160)
    plt.close(fig)

    print(f"\nГрафики сохранены в: {output_dir}")
    print("\nВывод: получена модель MEDV ≈ 34.55 − 0.95 × LSTAT.")
    print("Рост LSTAT на один процентный пункт связан со снижением")
    print("предсказанной стоимости примерно на 0.95 тыс. долларов.")
    print("Это обучение с учителем: в данных известны целевые цены.")
    print("Качество прогноза на новых данных отдельно не оценивалось.")
    print("\nПРОВЕРКА ПРОЙДЕНА: все расчёты выполнены успешно.")


if __name__ == "__main__":
    main()
