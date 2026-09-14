"""Выборка коллокационных точек и постановка начально-краевых условий."""

import torch

import swe_constants as C
from swe_constants import (X_MIN, X_MAX, Y_MIN, Y_MAX, T_MIN, T_MAX,
                           N_F, N_IC, N_BC, H_LEFT, H_RIGHT, X_DAM, DEVICE)


def _u(n, lo, hi):
    return (hi - lo) * torch.rand(n, 1, device=DEVICE) + lo


def sample_interior(n=N_F, t_max=T_MAX, cone_frac=C.CONE_FRAC):
    """Точки внутри области.

    Часть точек размещается равномерно, часть — в конусе влияния разрыва
    |x - X_DAM| <= c*t + margin, где и находится вся нетривиальная динамика
    (волна разрежения и ударная волна). Вне конуса решение постоянно и
    выучивается тривиально, поэтому равномерная выборка тратит точки впустую.
    """
    n_cone = int(n * cone_frac)
    n_uni = n - n_cone

    x_u = _u(n_uni, X_MIN, X_MAX)
    y_u = _u(n_uni, Y_MIN, Y_MAX)
    t_u = _u(n_uni, T_MIN, t_max)

    t_c = _u(n_cone, T_MIN, t_max)
    half = C.MAX_WAVE_SPEED * t_c + C.CONE_MARGIN
    x_c = (X_DAM + (2.0 * torch.rand(n_cone, 1, device=DEVICE) - 1.0) * half)
    x_c = x_c.clamp(X_MIN, X_MAX)
    y_c = _u(n_cone, Y_MIN, Y_MAX)

    x = torch.cat([x_u, x_c])
    y = torch.cat([y_u, y_c])
    t = torch.cat([t_u, t_c])
    return x, y, t


def sample_initial(n=N_IC):
    """Точки на t = 0."""
    x = _u(n, X_MIN, X_MAX)
    y = _u(n, Y_MIN, Y_MAX)
    t = torch.zeros(n, 1, device=DEVICE)
    return x, y, t


def sample_boundary_x(n=N_BC, t_max=T_MAX):
    """Точки на стенках x = X_MIN и x = X_MAX.

    За t <= T_MAX волны туда не доходят, поэтому там ТОЧНО выполняется
    h = h_left / h_right, u = v = 0. Это условие Дирихле не приближённое.
    """
    n2 = n // 2
    t = _u(n2, T_MIN, t_max)

    xl = torch.full((n2, 1), X_MIN, device=DEVICE)
    yl = _u(n2, Y_MIN, Y_MAX)
    xr = torch.full((n2, 1), X_MAX, device=DEVICE)
    yr = _u(n2, Y_MIN, Y_MAX)

    x = torch.cat([xl, xr])
    y = torch.cat([yl, yr])
    t = torch.cat([t, t])
    h_far = torch.where(x <= X_DAM, torch.tensor(H_LEFT, device=DEVICE),
                        torch.tensor(H_RIGHT, device=DEVICE)).float()
    return x, y, t, h_far


def sample_boundary_y(n=N_BC, t_max=T_MAX):
    """Точки на стенках y = Y_MIN и y = Y_MAX.

    Здесь ставится ТОЛЬКО условие непротекания v = 0 (свободное скольжение).

    В старой версии на всех четырёх стенках требовалось u = v = 0. Но истинное
    решение имеет u != 0 при y = +-Y_MAX (оно не зависит от y), поэтому это
    условие противоречило самому уравнению: невязка BC падала до 1e-6, а затем
    РОСЛА до 5e-2 и там застревала — сеть «отказывалась» его выполнять.
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
    return x, y, t


def initial_condition(x, y):
    """Начальные h, u, v — разрыв в x = X_DAM."""
    h0 = torch.where(x <= X_DAM, torch.tensor(H_LEFT, device=x.device),
                     torch.tensor(H_RIGHT, device=x.device)).float()
    u0 = torch.zeros_like(x)
    v0 = torch.zeros_like(x)
    return h0, u0, v0


@torch.no_grad()
def _pool(n, t_max):
    x = _u(n, X_MIN, X_MAX)
    y = _u(n, Y_MIN, Y_MAX)
    t = _u(n, T_MIN, t_max)
    return x, y, t


def refine_points(model, residual_fn, nu, t_max=T_MAX,
                  pool=C.RAR_POOL, keep=C.RAR_KEEP):
    """Адаптивное уточнение (RAR): из большого пула оставить точки с наибольшей
    невязкой. Это принципиальная версия «сгустить точки у разрыва» — сеть сама
    показывает, где ей тяжело, вместо того чтобы мы угадывали заранее."""
    x, y, t = _pool(pool, t_max)
    R = residual_fn(model, x, y, t, nu=nu)
    score = (R ** 2).mean(dim=1)
    idx = torch.topk(score, keep).indices
    return x[idx].detach(), y[idx].detach(), t[idx].detach()
