"""Векторизованный HLL-решатель для двумерных уравнений мелкой воды.

Полностью на numpy-срезах, без питоновских циклов по ячейкам
(исходная версия в hll_solver_for_swe.py делает Nx*Ny итераций на шаг).
Ускорение ~500-1000x при Nx=Ny=200.
"""

import numpy as np

from swe_constants import g, CFL, X_MIN, X_MAX, Y_MIN, Y_MAX, H_LEFT, H_RIGHT, X_DAM, T_MAX

H_MIN = 1e-8  # порог "сухого дна"


def _velocity(h, hu):
    """u = hu/h с защитой от деления на ноль на сухом дне."""
    return np.where(h > H_MIN, hu / np.maximum(h, H_MIN), 0.0)


def _hll_flux(hL, huL, hvL, hR, huR, hvR, normal):
    """HLL-поток через набор граней. normal='x' или 'y'. Все входы - массивы.

    Возвращает (F_h, F_hu, F_hv) той же формы.
    """
    uL, vL = _velocity(hL, huL), _velocity(hL, hvL)
    uR, vR = _velocity(hR, huR), _velocity(hR, hvR)

    cL, cR = np.sqrt(g * np.maximum(hL, 0.0)), np.sqrt(g * np.maximum(hR, 0.0))

    # нормальная компонента скорости
    if normal == "x":
        wL, wR = uL, uR
    else:
        wL, wR = vL, vR

    sL = np.minimum(wL - cL, wR - cR)
    sR = np.maximum(wL + cL, wR + cR)

    if normal == "x":
        FL = (huL, huL * uL + 0.5 * g * hL ** 2, huL * vL)
        FR = (huR, huR * uR + 0.5 * g * hR ** 2, huR * vR)
    else:
        FL = (hvL, huL * vL, hvL * vL + 0.5 * g * hL ** 2)
        FR = (hvR, huR * vR, hvR * vR + 0.5 * g * hR ** 2)

    UL = (hL, huL, hvL)
    UR = (hR, huR, hvR)

    denom = np.where(np.abs(sR - sL) > 1e-14, sR - sL, 1.0)

    out = []
    for fL, fR, ul, ur in zip(FL, FR, UL, UR):
        f_hll = (sR * fL - sL * fR + sL * sR * (ur - ul)) / denom
        f = np.where(sL >= 0.0, fL, np.where(sR <= 0.0, fR, f_hll))
        out.append(f)
    return out


def _faces_x(H, HU, HV):
    """Состояния слева/справа от каждой вертикальной грани, с прозрачными границами."""
    # Nx+1 граней: дублируем крайние ячейки (zero-gradient / transmissive)
    hL = np.concatenate([H[:1, :], H], axis=0)
    hR = np.concatenate([H, H[-1:, :]], axis=0)
    huL = np.concatenate([HU[:1, :], HU], axis=0)
    huR = np.concatenate([HU, HU[-1:, :]], axis=0)
    hvL = np.concatenate([HV[:1, :], HV], axis=0)
    hvR = np.concatenate([HV, HV[-1:, :]], axis=0)
    return hL, huL, hvL, hR, huR, hvR


def _faces_y(H, HU, HV):
    hL = np.concatenate([H[:, :1], H], axis=1)
    hR = np.concatenate([H, H[:, -1:]], axis=1)
    huL = np.concatenate([HU[:, :1], HU], axis=1)
    huR = np.concatenate([HU, HU[:, -1:]], axis=1)
    hvL = np.concatenate([HV[:, :1], HV], axis=1)
    hvR = np.concatenate([HV, HV[:, -1:]], axis=1)
    return hL, huL, hvL, hR, huR, hvR


def hll_step(H, HU, HV, dx, dy, dt):
    """Один шаг схемы. H, HU, HV - массивы (Nx, Ny)."""
    Fh, Fhu, Fhv = _hll_flux(*_faces_x(H, HU, HV), normal="x")
    Gh, Ghu, Ghv = _hll_flux(*_faces_y(H, HU, HV), normal="y")

    H_new = H - dt / dx * (Fh[1:, :] - Fh[:-1, :]) - dt / dy * (Gh[:, 1:] - Gh[:, :-1])
    HU_new = HU - dt / dx * (Fhu[1:, :] - Fhu[:-1, :]) - dt / dy * (Ghu[:, 1:] - Ghu[:, :-1])
    HV_new = HV - dt / dx * (Fhv[1:, :] - Fhv[:-1, :]) - dt / dy * (Ghv[:, 1:] - Ghv[:, :-1])

    H_new = np.maximum(H_new, H_MIN)
    return H_new, HU_new, HV_new


def solve_2d_dam_break(x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX,
                       Nx=200, Ny=200, h_left=H_LEFT, h_right=H_RIGHT,
                       x_dam=X_DAM, t_end=T_MAX, save_times=None, verbose=False):
    """Решает задачу о прорыве плотины.

    save_times: список моментов, в которые сохранять состояние (по умолчанию - только t_end).
    Возвращает (snapshots, X, Y, saved_times), где snapshots[k] = (H, HU, HV).
    """
    x = np.linspace(x_min, x_max, Nx + 1)
    y = np.linspace(y_min, y_max, Ny + 1)
    dx, dy = x[1] - x[0], y[1] - y[0]
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])
    X, Y = np.meshgrid(xc, yc, indexing="ij")

    H = np.where(X <= x_dam, h_left, h_right).astype(float)
    HU = np.zeros_like(H)
    HV = np.zeros_like(H)

    if save_times is None:
        save_times = [t_end]
    save_times = sorted(float(s) for s in save_times)

    snapshots, saved_times = [], []
    next_save = 0

    t = 0.0
    # снимок при t=0, если запрошен
    while next_save < len(save_times) and save_times[next_save] <= 0.0:
        snapshots.append((H.copy(), HU.copy(), HV.copy()))
        saved_times.append(0.0)
        next_save += 1

    steps = 0
    while t < t_end - 1e-15:
        U = _velocity(H, HU)
        V = _velocity(H, HV)
        c = np.sqrt(g * np.maximum(H, 0.0))
        max_sx = np.max(np.abs(U) + c)
        max_sy = np.max(np.abs(V) + c)
        dt = CFL / (max_sx / dx + max_sy / dy)  # 2D-условие устойчивости

        # не перескочить ближайший момент сохранения и конец счёта
        target = save_times[next_save] if next_save < len(save_times) else t_end
        dt = min(dt, target - t, t_end - t)

        H, HU, HV = hll_step(H, HU, HV, dx, dy, dt)
        t += dt
        steps += 1

        while next_save < len(save_times) and t >= save_times[next_save] - 1e-12:
            snapshots.append((H.copy(), HU.copy(), HV.copy()))
            saved_times.append(t)
            next_save += 1

        if verbose and steps % 50 == 0:
            print(f"  HLL t={t:.4f}/{t_end}  ({steps} шагов)")

    if verbose:
        print(f"  HLL завершён: {steps} шагов, сетка {Nx}x{Ny}")

    return snapshots, X, Y, saved_times


if __name__ == "__main__":
    import time

    t0 = time.time()
    snaps, X, Y, ts = solve_2d_dam_break(Nx=200, Ny=200, verbose=True)
    print(f"время: {time.time() - t0:.2f} c")
