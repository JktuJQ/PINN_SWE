"""Количественное сравнение PINN и HLL с точным решением задачи Римана.

Модель — параметризованная: model(x, y, t, h_L, h_R).
Все функции принимают h_L, h_R с дефолтами H_LEFT, H_RIGHT.
"""

import time

import numpy as np
import torch

import swe_constants as C
from swe_constants import (X_MIN, X_MAX, T_MAX, DEVICE,
                           H_LEFT, H_RIGHT)
from swe_exact import exact_dam_break, solve_star_region
from hll_solver_vectorized import solve_2d_dam_break


# ============================================================
#  Вспомогательное
# ============================================================

def _make_params(x, h_L, h_R):
    """Создать тензоры h_L, h_R формы (N, 1) по x."""
    n = x.shape[0]
    hL = torch.full((n, 1), float(h_L), device=x.device, dtype=x.dtype)
    hR = torch.full((n, 1), float(h_R), device=x.device, dtype=x.dtype)
    return hL, hR


# ============================================================
#  Профиль PINN
# ============================================================

def pinn_profile(model, t, n_x=800, y_val=0.0,
                 h_L=H_LEFT, h_R=H_RIGHT):
    """Профиль (x, h, u, v) вдоль y = y_val в момент t.

    Аргументы:
        model  — PINN_for_all_initial_conditions
        t      — момент времени
        n_x    — число точек по x
        y_val  — координата y (по умолчанию 0)
        h_L    — глубина слева
        h_R    — глубина справа

    Возвращает: x, h, u, v — массивы numpy.
    """
    x = torch.linspace(X_MIN, X_MAX, n_x, device=DEVICE).view(-1, 1)
    y = torch.full_like(x, y_val)
    tt = torch.full_like(x, float(t))

    hL, hR = _make_params(x, h_L, h_R)

    with torch.no_grad():
        h, u, v = model(x, y, tt, hL, hR)

    return (x.cpu().numpy().ravel(),
            h.cpu().numpy().ravel(),
            u.cpu().numpy().ravel(),
            v.cpu().numpy().ravel())


# ============================================================
#  Ошибки
# ============================================================

def _errors(x, h, u, t, h_L, h_R):
    """L2 и Linf ошибки относительно точного решения."""
    he, ue = exact_dam_break(x, t, h_l=h_L, h_r=h_R)
    return {
        "L2_h": float(np.linalg.norm(h - he) / np.linalg.norm(he)),
        "Linf_h": float(np.max(np.abs(h - he))),
        "L2_u": float(np.linalg.norm(u - ue) / max(np.linalg.norm(ue), 1e-12)),
        "Linf_u": float(np.max(np.abs(u - ue))),
    }


def pinn_errors(model, times, n_x=800,
                h_L=H_LEFT, h_R=H_RIGHT):
    """Ошибки PINN в заданные моменты.

    Аргументы:
        model  — PINN_for_all_initial_conditions
        times  — список моментов времени
        h_L    — глубина слева
        h_R    — глубина справа
    """
    out = {}
    for t in times:
        x, h, u, _ = pinn_profile(model, t, n_x=n_x, h_L=h_L, h_R=h_R)
        out[float(t)] = _errors(x, h, u, float(t), h_L, h_R)
    return out


def hll_errors(times, Nx=C.HLL_NX, Ny=C.HLL_NY,
               h_L=H_LEFT, h_R=H_RIGHT):
    """Ошибки HLL в заданные моменты.

    HLL-солвер должен принимать h_L, h_R.
    """
    snaps, X, Y, ts = solve_2d_dam_break(
        Nx=Nx, Ny=Ny, t_end=max(times),
        save_times=list(times),
        h_left=h_L, h_right=h_R
    )

    j = X.shape[1] // 2
    out = {}
    for (H, HU, HV), t in zip(snaps, ts):
        x = X[:, j]
        h = H[:, j]
        u = np.where(h > 1e-8, HU[:, j] / np.maximum(h, 1e-8), 0.0)
        out[float(t)] = _errors(x, h, u, float(t), h_L, h_R)
    return out


# ============================================================
#  Резкость фронта
# ============================================================

def shock_sharpness(x, h, t, h_L=H_LEFT, h_R=H_RIGHT):
    """Ширина фронта ударной волны на уровне 10–90% от скачка.

    Ключевая метрика для задачи: насколько метод «размазывает» разрыв.
    Для точного решения ширина равна 0.
    """
    h_star, u_star = solve_star_region(h_l=h_L, h_r=h_R)

    lo, hi = h_R, h_star
    if t <= 0:
        return 0.0

    s_shock = (h_star * u_star) / (h_star - h_R)
    x_sh = C.X_DAM + s_shock * t

    # окно вокруг истинного положения фронта
    win = (x > x_sh - 2.0) & (x < x_sh + 2.0)
    if win.sum() < 5:
        return float("nan")

    order = np.argsort(x[win])
    xs, hs = x[win][order], h[win][order]

    def crossing(level):
        """Положение пересечения профилем уровня level (линейная интерполяция)."""
        d = hs - level
        sign_change = np.where(np.diff(np.sign(d)) != 0)[0]
        if len(sign_change) == 0:
            return None
        i = sign_change[0]
        y0, y1 = d[i], d[i + 1]
        if y1 == y0:
            return float(xs[i])
        return float(xs[i] - y0 * (xs[i + 1] - xs[i]) / (y1 - y0))

    x90 = crossing(lo + 0.90 * (hi - lo))
    x10 = crossing(lo + 0.10 * (hi - lo))
    if x90 is None or x10 is None:
        return float("nan")
    return abs(x10 - x90)


# ============================================================
#  Время счёта
# ============================================================

def timing_report(model, Nx_list=(100, 200, 400), n_eval=800,
                  h_L=H_LEFT, h_R=H_RIGHT):
    """Время: инференс PINN против счёта HLL."""
    rep = {}

    t0 = time.time()
    pinn_profile(model, T_MAX, n_x=n_eval, h_L=h_L, h_R=h_R)
    rep["pinn_inference_s"] = time.time() - t0

    for Nx in Nx_list:
        t0 = time.time()
        solve_2d_dam_break(Nx=Nx, Ny=Nx, t_end=T_MAX,
                           save_times=[T_MAX],
                           h_left=h_L, h_right=h_R)
        rep[f"hll_{Nx}x{Nx}_s"] = time.time() - t0

    return rep


# ============================================================
#  Таблица
# ============================================================

def print_table(pinn_err, hll_err, title="PINN vs HLL vs точное решение"):
    print(f"\n{title}")
    print(f"{'t':>6} | {'PINN L2(h)':>11} {'PINN Linf(h)':>13} | "
          f"{'HLL L2(h)':>10} {'HLL Linf(h)':>12}")
    print("-" * 62)
    for t in sorted(pinn_err):
        p = pinn_err[t]
        hk = min(hll_err, key=lambda s: abs(s - t)) if hll_err else None
        hq = hll_err[hk] if hk is not None else None
        if hq:
            print(f"{t:>6.2f} | {100*p['L2_h']:>10.2f}% {p['Linf_h']:>13.4f} | "
                  f"{100*hq['L2_h']:>9.2f}% {hq['Linf_h']:>12.4f}")
        else:
            print(f"{t:>6.2f} | {100*p['L2_h']:>10.2f}% {p['Linf_h']:>13.4f} |")