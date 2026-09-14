"""Точное решение задачи Римана для уравнений мелкой воды (dam break, влажное дно).

Даёт эталон ("ground truth") для оценки погрешности и PINN, и HLL.
Структура решения при h_L > h_R > 0, u_L = u_R = 0:
    волна разрежения влево  |  звезда (h*, u*)  |  ударная волна вправо
Ссылка: E.F. Toro, "Shock-Capturing Methods for Free-Surface Shallow Flows", гл. 5.
"""

import numpy as np
from scipy.optimize import brentq

from swe_constants import g, H_LEFT, H_RIGHT, X_DAM


def _shock_branch(h_star, h_k):
    """Скорость за ударной волной (соотношение Рэнкина-Гюгонио)."""
    return (h_star - h_k) * np.sqrt(0.5 * g * (h_star + h_k) / (h_star * h_k))


def _rarefaction_branch(h_star, h_k):
    """Скорость за волной разрежения (инвариант Римана)."""
    return 2.0 * (np.sqrt(g * h_star) - np.sqrt(g * h_k))


def solve_star_region(h_l=H_LEFT, u_l=0.0, h_r=H_RIGHT, u_r=0.0):
    """Находит (h*, u*) в промежуточной области между волнами."""

    def f(h_star, h_k):
        # вклад левой/правой волны: ударная при h* > h_k, разрежение при h* < h_k
        return _shock_branch(h_star, h_k) if h_star > h_k else _rarefaction_branch(h_star, h_k)

    # f_L(h*) + f_R(h*) + (u_r - u_l) = 0
    def total(h_star):
        return f(h_star, h_l) + f(h_star, h_r) + (u_r - u_l)

    h_star = brentq(total, 1e-12, 1e6 * max(h_l, h_r), xtol=1e-14, rtol=1e-14)
    u_star = 0.5 * (u_l + u_r) + 0.5 * (f(h_star, h_r) - f(h_star, h_l))
    return h_star, u_star


def exact_dam_break(x, t, h_l=H_LEFT, u_l=0.0, h_r=H_RIGHT, u_r=0.0, x_dam=X_DAM):
    """Точное (h, u) в точках x в момент t. Векторизовано, x - np.ndarray."""
    x = np.asarray(x, dtype=float)
    h = np.empty_like(x)
    u = np.empty_like(x)

    if t <= 0:
        h[:] = np.where(x <= x_dam, h_l, h_r)
        u[:] = np.where(x <= x_dam, u_l, u_r)
        return h, u

    h_star, u_star = solve_star_region(h_l, u_l, h_r, u_r)
    c_l, c_r, c_star = np.sqrt(g * h_l), np.sqrt(g * h_r), np.sqrt(g * h_star)

    xi = (x - x_dam) / t  # автомодельная переменная

    # --- левая волна: разрежение (h* < h_l) ---
    s_head = u_l - c_l          # голова веера
    s_tail = u_star - c_star    # хвост веера

    # --- правая волна: ударная (h* > h_r), скорость из сохранения массы ---
    s_shock = (h_star * u_star - h_r * u_r) / (h_star - h_r)

    left = xi <= s_head
    fan = (xi > s_head) & (xi < s_tail)
    star = (xi >= s_tail) & (xi < s_shock)
    right = xi >= s_shock

    h[left], u[left] = h_l, u_l
    h[star], u[star] = h_star, u_star
    h[right], u[right] = h_r, u_r

    # внутри веера разрежения — инварианты Римана
    c_fan = (u_l + 2.0 * c_l - xi[fan]) / 3.0
    u[fan] = (u_l + 2.0 * c_l + 2.0 * xi[fan]) / 3.0
    h[fan] = c_fan ** 2 / g

    return h, u


def wave_speeds(h_l=H_LEFT, u_l=0.0, h_r=H_RIGHT, u_r=0.0):
    """Скорости характерных волн — удобно для выбора размера области."""
    h_star, u_star = solve_star_region(h_l, u_l, h_r, u_r)
    c_l, c_star = np.sqrt(g * h_l), np.sqrt(g * h_star)
    return {
        "h_star": h_star,
        "u_star": u_star,
        "rarefaction_head": u_l - c_l,
        "rarefaction_tail": u_star - c_star,
        "shock": (h_star * u_star - h_r * u_r) / (h_star - h_r),
    }


if __name__ == "__main__":
    info = wave_speeds()
    for k, v in info.items():
        print(f"{k:>18}: {v:.6f}")
