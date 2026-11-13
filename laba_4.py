import numpy as np
import pandas as pd
from scipy.optimize import minimize

# 1. Вихідні дані: котирування акцій за 11 періодів
data = {
    "Жидачівський ЦПК": [0.16, 0.16, 0.20, 0.25, 0.10, 0.25, 0.10, 0.08, 0.07, 0.15, 0.15],
    "Запоріж-транспорт": [0.15, 0.12, 0.15, 0.10, 0.11, 0.11, 0.11, 0.12, 0.11, 0.11, 0.18],
    "Пивзавод «Рогань»": [40.0, 37.0, 35.0, 40.0, 80.0, 110.0, 40.0, 30.0, 33.0, 35.0, 60.0]
}

prices = pd.DataFrame(data, index=range(1, 12))
prices.index.name = "Період"

# Обчислення норм прибутку (10 інтервалів): R_t = (P_t - P_{t-1}) / P_{t-1}
returns = prices.pct_change().dropna()
returns.index = [f"{i}-{i+1}" for i in range(1, 11)]

print("=== Таблиця норм прибутку за періодами (%) ===")
print((returns * 100).round(2))
print()

# а) Сподівана норма прибутку (m_i) та ризик (СКВ, sigma_i)
# ddof=0 відповідає формулі дисперсії з посібника: D(X) = sum(p_j * (x_j - M(X))^2)
m = returns.mean()
sigma = returns.std(ddof=0)

print("=== а) Сподівана норма прибутку та ризик кожної акції ===")
stats_df = pd.DataFrame({
    "Сподівана норма прибутку m (%)": (m * 100).round(2),
    "Ризик (СКВ) sigma (%)": (sigma * 100).round(2)
})
print(stats_df)
print()

# б) Коваріаційна та кореляційна матриці
cov_matrix = np.cov(returns.T, ddof=0)
corr_matrix = returns.corr()

print("=== б) Коваріаційна матриця (в частках одиниці) ===")
print(pd.DataFrame(cov_matrix, index=prices.columns, columns=prices.columns).round(6))
print("\n=== Кореляційна матриця ===")
print(corr_matrix.round(4))
print()

# Допоміжні функції оптимізації портфеля Марковіца
m_vec = m.values
n = len(m_vec)
bounds = tuple((0.0, 1.0) for _ in range(n))  # x_i >= 0
init_w = np.array([1.0 / n] * n)

def port_variance(w):
    return float(w.T @ cov_matrix @ w)

def port_return(w):
    return float(w @ m_vec)

# в) Портфель збереження капіталу: min sigma_П^2 при sum(x_i) = 1, x_i >= 0
res_min_risk = minimize(
    port_variance,
    init_w,
    method="SLSQP",
    bounds=bounds,
    constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
)

w_min = res_min_risk.x
m_min = port_return(w_min)
s_min = np.sqrt(port_variance(w_min))

print("=== в) ПЦП щодо збереження капіталу (мінімальний ризик) ===")
for name, weight in zip(prices.columns, w_min):
    print(f"  {name}: {weight:.4f} ({weight*100:.2f}%)")
print(f"Сподівана норма прибутку m_П = {m_min*100:.2f}%")
print(f"Ризик портфеля sigma_П = {s_min*100:.2f}%")
print()

# г) Портфель бажаного прибутку: min sigma_П^2 при m_П >= 5% (0.05), sum(x_i) = 1, x_i >= 0
target_m = 0.05
res_target = minimize(
    port_variance,
    init_w,
    method="SLSQP",
    bounds=bounds,
    constraints=[
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: port_return(w) - target_m}
    ]
)

w_target = res_target.x
m_target = port_return(w_target)
s_target = np.sqrt(port_variance(w_target))

print(f"=== г) ПЦП щодо одержання бажаного прибутку (m_П = {target_m*100:.0f}%) ===")
for name, weight in zip(prices.columns, w_target):
    print(f"  {name}: {weight:.4f} ({weight*100:.2f}%)")
print(f"Сподівана норма прибутку m_П = {m_target*100:.2f}%")
print(f"Ризик портфеля sigma_П = {s_target*100:.2f}%")