"""Выборка коллокационных точек и постановка начально-краевых условий.

Параметризованная версия: h_L, h_R — случайные параметры задачи,
передаются как входы во все сэмплеры и в начальные условия.
"""

import torch

import swe_constants as C
from swe_constants import (
    X_MIN, X_MAX, Y_MIN, Y_MAX, T_MIN, T_MAX,
    N_F, N_IC, N_BC, X_DAM, DEVICE,
    H_L_MIN, H_L_MAX, H_R_MIN, H_R_MAX,
)


# ============================================================
#  Вспомогательные
# ============================================================

def _u(n, lo, hi):
    """Равномерная выборка n чисел из [lo, hi] формы (n, 1)."""
    return (hi - lo) * torch.rand(n, 1, device=DEVICE) + lo


def _sample_h(n, h_range):
    """Равномерная выборка n чисел из h_range = (lo, hi) формы (n, 1)."""
    lo, hi = h_range
    return (hi - lo) * torch.rand(n, 1, device=DEVICE) + lo


# ============================================================
#  Точки внутри области (PDE)
# ============================================================

def sample_interior(n=N_F, t_max=T_MAX, cone_frac=C.CONE_FRAC,
                    h_L_range=(H_L_MIN, H_L_MAX),
                    h_R_range=(H_R_MIN, H_R_MAX)):
    """Точки внутри области + случайные параметры задачи.

    Часть точек — равномерно, часть — в конусе влияния разрыва
    |x - X_DAM| <= c*t + margin, где вся нетривиальная динамика.

    Возвращает: x, y, t, h_L, h_R — формы (n, 1).
    """
    n_cone = int(n * cone_frac)
    n_uni = n - n_cone

    # --- равномерные точки ---
    x_u = _u(n_uni, X_MIN, X_MAX)
    y_u = _u(n_uni, Y_MIN, Y_MAX)
    t_u = _u(n_uni, T_MIN, t_max)

    # --- точки в конусе влияния ---
    t_c = _u(n_cone, T_MIN, t_max)
    half = C.MAX_WAVE_SPEED * t_c + C.CONE_MARGIN
    x_c = X_DAM + (2.0 * torch.rand(n_cone, 1, device=DEVICE) - 1.0) * half
    x_c = x_c.clamp(X_MIN, X_MAX)
    y_c = _u(n_cone, Y_MIN, Y_MAX)

    # --- склейка ---
    x = torch.cat([x_u, x_c])
    y = torch.cat([y_u, y_c])
    t = torch.cat([t_u, t_c])

    # --- случайные параметры задачи ---
    h_L = _sample_h(n, h_L_range)
    h_R = _sample_h(n, h_R_range)

    return x, y, t, h_L, h_R


# ============================================================
#  Начальные условия (t = 0)
# ============================================================

def sample_initial(n=N_IC,
                   h_L_range=(H_L_MIN, H_L_MAX),
                   h_R_range=(H_R_MIN, H_R_MAX)):
    """Точки на t = 0 + случайные параметры задачи.

    Возвращает: x, y, t, h_L, h_R — формы (n, 1).
    """
    x = _u(n, X_MIN, X_MAX)
    y = _u(n, Y_MIN, Y_MAX)
    t = torch.zeros(n, 1, device=DEVICE)

    h_L = _sample_h(n, h_L_range)
    h_R = _sample_h(n, h_R_range)

    return x, y, t, h_L, h_R


# ============================================================
#  Граница по x (дальнее поле — точное условие Дирихле)
# ============================================================

def sample_boundary_x(n=N_BC, t_max=T_MAX,
                      h_L_range=(H_L_MIN, H_L_MAX),
                      h_R_range=(H_R_MIN, H_R_MAX)):
    """Точки на стенках x = X_MIN и x = X_MAX.

    За t <= T_MAX волны туда не доходят, поэтому там ТОЧНО
    h = h_L (слева) или h = h_R (справа), u = v = 0.

    Возвращает: x, y, t, h_L, h_R — формы (n, 1).
    """
    n2 = n // 2
    t = _u(n2, T_MIN, t_max)

    xl = torch.full((n2, 1), X_MIN, device=DEVICE)
    yl = _u(n2, Y_MIN, Y_MAX)
    xr = torch.full((n2, 1), X_MAX, device=DEVICE)
    yr = _u(n2, Y_MIN, Y_MAX)

    x = torch.cat([xl, xr])      # (n, 1)
    y = torch.cat([yl, yr])
    t = torch.cat([t, t])

    # параметры задачи — случайные (по одной на точку)
    h_L = _sample_h(n, h_L_range)
    h_R = _sample_h(n, h_R_range)

    return x, y, t, h_L, h_R


# ============================================================
#  Граница по y (только непротекание v = 0)
# ============================================================

def sample_boundary_y(n=N_BC, t_max=T_MAX,
                      h_L_range=(H_L_MIN, H_L_MAX),
                      h_R_range=(H_R_MIN, H_R_MAX)):
    """Точки на стенках y = Y_MIN и y = Y_MAX.

    Здесь ставится ТОЛЬКО условие непротекания v = 0
    (свободное скольжение, симметрия по y).

    Возвращает: x, y, t, h_L, h_R — формы (n, 1).
    """
    n2 = n // 2
    t = _u(n2, T_MIN, t_max)

    xb = _u(n2, X_MIN, X_MAX)
    yb = torch.full((n2, 1), Y_MIN, device=DEVICE)
    xt = _u(n2, X_MIN, X_MAX)
    yt = torch.full((n2, 1), Y_MAX, device=DEVICE)

    x = torch.cat([xb, xt])
    y = torch.cat([yb, yt])
    t = torch.cat([t, t])

    h_L = _sample_h(n, h_L_range)
    h_R = _sample_h(n, h_R_range)

    return x, y, t, h_L, h_R


# ============================================================
#  Начальное условие — параметризованное
# ============================================================

def initial_condition(x, y, h_L, h_R):
    """Начальные h, u, v — разрыв в x = X_DAM.

    h_L, h_R — тензоры формы (N, 1) — глубины слева / справа.
    """
    h0 = torch.where(x <= X_DAM, h_L, h_R).float()
    u0 = torch.zeros_like(x)
    v0 = torch.zeros_like(x)
    return h0, u0, v0


# ============================================================
#  RAR — адаптивное уточнение
# ============================================================

@torch.no_grad()
def _pool(n, t_max, h_L_range, h_R_range):
    x = _u(n, X_MIN, X_MAX)
    y = _u(n, Y_MIN, Y_MAX)
    t = _u(n, T_MIN, t_max)
    h_L = _sample_h(n, h_L_range)
    h_R = _sample_h(n, h_R_range)
    return x, y, t, h_L, h_R


def refine_points(model, residual_fn, nu, t_max=T_MAX,
                  pool=C.RAR_POOL, keep=C.RAR_KEEP,
                  h_L_range=(H_L_MIN, H_L_MAX),
                  h_R_range=(H_R_MIN, H_R_MAX)):
    """RAR: из большого пула оставить точки с наибольшей невязкой.

    Сеть сама показывает, где ей тяжело, вместо того чтобы
    угадывать заранее.
    """
    x, y, t, h_L, h_R = _pool(pool, t_max, h_L_range, h_R_range)
    R = residual_fn(model, x, y, t, h_L, h_R, nu=nu)
    score = (R ** 2).mean(dim=1)
    idx = torch.topk(score, keep).indices
    return (x[idx].detach(), y[idx].detach(), t[idx].detach(),
            h_L[idx].detach(), h_R[idx].detach())