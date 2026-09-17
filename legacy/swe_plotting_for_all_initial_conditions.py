import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

import swe_constants as C
from swe_constants import T_MIN, T_MAX, X_MIN, X_MAX, DEVICE
from swe_exact import exact_dam_break
from hll_solver_vectorized import solve_2d_dam_break
from swe_metrics_for_all_initial_conditions import pinn_profile


def plot_history(history, save_path="data/loss.png", show=False):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].semilogy(history["loss"], label="Итоговый Loss", linewidth=2)
    ax[0].semilogy(history["pde"], label="Невязка ФДУ", alpha=0.75)
    ax[0].semilogy(history["ic"], label="Начальные условия", alpha=0.75)
    ax[0].semilogy(history["bc"], label="Граничные условия", alpha=0.75)
    ax[0].set_xlabel("Эпоха")
    ax[0].set_ylabel("Loss (лог. шкала)")
    ax[0].set_title("История обучения PINN")
    ax[0].legend()
    ax[0].grid(True, which="both", alpha=0.3)

    ax2 = ax[1]
    ax2.plot(history["nu"], color="tab:red", label=r"искусств. вязкость $\nu$")
    ax2.set_xlabel("Эпоха")
    ax2.set_ylabel(r"$\nu$", color="tab:red")
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax3 = ax2.twinx()
    ax3.plot(history["t_max"], color="tab:blue", label="горизонт $t_{max}$")
    ax3.set_ylabel(r"горизонт $t_{max}$", color="tab:blue")
    ax3.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_title("Расписание: вязкость и горизонт по времени")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_comparison(model, times=(0.0, 0.25, 0.5, 1.0),
                    save_path="data/comparison.png", Nx=C.HLL_NX, show=False):
    """Главный рисунок: PINN vs HLL vs точное решение."""
    snaps, X, Y, ts = solve_2d_dam_break(Nx=Nx, Ny=Nx, t_end=max(times),
                                         save_times=list(times), h_left=C.H_LEFT, h_right=C.H_RIGHT)
    j = X.shape[1] // 2

    n = len(times)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8), squeeze=False)

    for k, t in enumerate(times):
        xp, hp, up, _ = pinn_profile(model, t)
        xe = np.linspace(X_MIN, X_MAX, 2000)
        he, ue = exact_dam_break(xe, float(t))

        H, HU, HV = snaps[k]
        xh = X[:, j]
        hh = H[:, j]
        uh = np.where(hh > 1e-8, HU[:, j] / np.maximum(hh, 1e-8), 0.0)

        a = axes[0][k]
        a.plot(xe, he, "k-", lw=2.2, label="точное")
        a.plot(xh, hh, "--", color="tab:green", lw=1.6, label=f"HLL {Nx}×{Nx}")
        a.plot(xp, hp, "-", color="tab:red", lw=1.6, label="PINN")
        a.set_title(f"h(x, y=0),  t = {t:.2f}")
        a.set_xlabel("x")
        a.grid(alpha=0.3)
        if k == 0:
            a.set_ylabel("h")
            a.legend(fontsize=9)

        b = axes[1][k]
        b.plot(xe, ue, "k-", lw=2.2, label="точное")
        b.plot(xh, uh, "--", color="tab:green", lw=1.6, label=f"HLL {Nx}×{Nx}")
        b.plot(xp, up, "-", color="tab:red", lw=1.6, label="PINN")
        b.set_title(f"u(x, y=0),  t = {t:.2f}")
        b.set_xlabel("x")
        b.grid(alpha=0.3)
        if k == 0:
            b.set_ylabel("u")
            b.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_error_curves(pinn_err, hll_err, save_path="data/errors.png", show=False):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    tp = sorted(pinn_err)
    th = sorted(hll_err)

    ax[0].plot(tp, [100 * pinn_err[t]["L2_h"] for t in tp], "o-",
               color="tab:red", label="PINN")
    ax[0].plot(th, [100 * hll_err[t]["L2_h"] for t in th], "s--",
               color="tab:green", label="HLL")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("отн. ошибка $L_2$ по h, %")
    ax[0].set_title("Относительная ошибка $L_2$")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(tp, [pinn_err[t]["Linf_h"] for t in tp], "o-",
               color="tab:red", label="PINN")
    ax[1].plot(th, [hll_err[t]["Linf_h"] for t in th], "s--",
               color="tab:green", label="HLL")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel(r"$L_\infty$ по h")
    ax[1].set_title(r"Максимальная ошибка ($L_\infty$, достигается на фронте)")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_field_2d(model, t=T_MAX, h_L=C.H_LEFT, h_R=C.H_RIGHT, save_path="data/field_2d.png", n=200, show=False):
    """Двумерное поле h и проверка y-инвариантности решения."""
    xs = torch.linspace(X_MIN, X_MAX, n, device=DEVICE)
    ys = torch.linspace(C.Y_MIN, C.Y_MAX, n, device=DEVICE)
    XX, YY = torch.meshgrid(xs, ys, indexing="ij")
    flat_x = XX.reshape(-1, 1)
    flat_y = YY.reshape(-1, 1)
    flat_t = torch.full_like(flat_x, float(t))
    flat_hL = torch.full_like(flat_x, float(h_L))
    flat_hR = torch.full_like(flat_x, float(h_R))
    with torch.no_grad():
        h, u, v = model(flat_x, flat_y, flat_t, flat_hL, flat_hR)
    H = h.reshape(n, n).cpu().numpy()
    V = v.reshape(n, n).cpu().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    im0 = ax[0].pcolormesh(XX.cpu().numpy(), YY.cpu().numpy(), H,
                           shading="auto", cmap="viridis")
    ax[0].set_title(f"PINN: поле h(x, y) при t = {t:.2f}")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    plt.colorbar(im0, ax=ax[0])

    # решение обязано не зависеть от y — отклонение показывает качество
    dev = H - H.mean(axis=1, keepdims=True)
    im1 = ax[1].pcolormesh(XX.cpu().numpy(), YY.cpu().numpy(), dev,
                           shading="auto", cmap="coolwarm")
    ax[1].set_title(f"Отклонение от y-инвариантности\nmax |h - <h>_y| = {np.abs(dev).max():.2e}")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    plt.colorbar(im1, ax=ax[1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# --- обратная совместимость со старым интерфейсом ---
def plot_hx_with_constant_y(model, times=None, save_path="data/h_profile.png", show=False):
    if times is None:
        times = torch.linspace(T_MIN, T_MAX, 6)
    fig = plt.figure(figsize=(10, 6))
    for t_val in times:
        x, h, u, v = pinn_profile(model, float(t_val))
        plt.plot(x, h, label=f"t = {float(t_val):.2f}")
    xe = np.linspace(X_MIN, X_MAX, 2000)
    he, _ = exact_dam_break(xe, float(times[-1]))
    plt.plot(xe, he, "k--", lw=1.5, label=f"точное, t={float(times[-1]):.2f}")
    plt.xlabel("x")
    plt.ylabel("h (глубина)")
    plt.title("Профиль h(x, y=0, t)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
