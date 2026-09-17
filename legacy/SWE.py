"""Невязка двумерных уравнений мелкой воды в консервативной форме.

    h_t  + (hu)_x            + (hv)_y            = nu*lap(h)
    (hu)_t + (hu^2 + gh^2/2)_x + (huv)_y         = -g*h*z_x + nu*lap(hu)
    (hv)_t + (huv)_x + (hv^2 + gh^2/2)_y         = -g*h*z_y + nu*lap(hv)

nu — искусственная вязкость (метод исчезающей вязкости), см. swe_constants.
"""

import torch

import swe_constants as C
from swe_constants import g


def topography(x, y):
    return torch.zeros_like(x)


def topography_grad(x, y):
    """Частные производные z_x, z_y."""
    return torch.zeros_like(x), torch.zeros_like(y)


def _d(out, wrt):
    """Производная out по wrt с сохранением графа."""
    return torch.autograd.grad(
        out, wrt, grad_outputs=torch.ones_like(out), create_graph=True
    )[0]


# Характерные масштабы невязок. Без нормировки уравнения импульса (~g*h^2 ~ 20)
# на порядок перевешивают уравнение неразрывности (~1), и MSE фактически
# оптимизирует только импульс.
_U_REF = C.MAX_WAVE_SPEED
_SCALE_MASS = C.H_LEFT / C.T_MAX
_SCALE_MOM = C.H_LEFT * _U_REF / C.T_MAX


def residual(model, x, y, t, nu=0.0):
    """Возвращает (N, 3) — безразмерные невязки трёх уравнений."""
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)

    h, u, v = model(x, y, t)

    hu = h * u
    hv = h * v
    huv = hu * v

    # --- производные по времени ---
    h_t = _d(h, t)
    hu_t = _d(hu, t)
    hv_t = _d(hv, t)

    # --- потоки ---
    F1 = hu
    F2 = hu * u + 0.5 * g * h * h
    F3 = huv

    G1 = hv
    G2 = huv
    G3 = hv * v + 0.5 * g * h * h

    F1_x = _d(F1, x)
    F2_x = _d(F2, x)
    F3_x = _d(F3, x)

    G1_y = _d(G1, y)
    G2_y = _d(G2, y)
    G3_y = _d(G3, y)

    z_x, z_y = topography_grad(x, y)

    R1 = h_t + F1_x + G1_y
    R2 = hu_t + F2_x + G2_y + g * h * z_x
    R3 = hv_t + F3_x + G3_y + g * h * z_y

    # --- искусственная вязкость ---
    if nu > 0.0:
        for q, name in ((h, "R1"), (hu, "R2"), (hv, "R3")):
            lap = _d(_d(q, x), x) + _d(_d(q, y), y)
            if name == "R1":
                R1 = R1 - nu * lap
            elif name == "R2":
                R2 = R2 - nu * lap
            else:
                R3 = R3 - nu * lap

    return torch.cat([R1 / _SCALE_MASS, R2 / _SCALE_MOM, R3 / _SCALE_MOM], dim=1)

def residual_for_all_initial_conditions(model, x, y, t, h_L, h_R, nu=0):
    """Возвращает (N, 3) — безразмерные невязки трёх уравнений."""
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)

    h, u, v = model(x, y, t, h_L, h_R)

    hu = h * u
    hv = h * v
    huv = hu * v

    # --- производные по времени ---
    h_t = _d(h, t)
    hu_t = _d(hu, t)
    hv_t = _d(hv, t)

    # --- потоки ---
    F1 = hu
    F2 = hu * u + 0.5 * g * h * h
    F3 = huv

    G1 = hv
    G2 = huv
    G3 = hv * v + 0.5 * g * h * h

    F1_x = _d(F1, x)
    F2_x = _d(F2, x)
    F3_x = _d(F3, x)

    G1_y = _d(G1, y)
    G2_y = _d(G2, y)
    G3_y = _d(G3, y)

    z_x, z_y = topography_grad(x, y)

    R1 = h_t + F1_x + G1_y
    R2 = hu_t + F2_x + G2_y + g * h * z_x
    R3 = hv_t + F3_x + G3_y + g * h * z_y

    # --- искусственная вязкость ---
    if nu > 0.0:
        for q, name in ((h, "R1"), (hu, "R2"), (hv, "R3")):
            lap = _d(_d(q, x), x) + _d(_d(q, y), y)
            if name == "R1":
                R1 = R1 - nu * lap
            elif name == "R2":
                R2 = R2 - nu * lap
            else:
                R3 = R3 - nu * lap

    return torch.cat([R1 / _SCALE_MASS, R2 / _SCALE_MOM, R3 / _SCALE_MOM], dim=1)