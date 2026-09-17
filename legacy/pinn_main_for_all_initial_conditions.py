"""Обучение PINN для двумерной задачи Римана (мелкая вода) и сравнение с HLL.

Запуск:
    python pinn_main.py              # полное обучение
    python pinn_main.py --quick      # быстрая проверка работоспособности
    python pinn_main.py --eval-only  # только оценка уже обученной модели
"""

import argparse
import time

import numpy as np
import torch

import swe_constants as C
from pinn_model_for_all_initial_conditions import PINN_for_all_initial_conditions, load_trained
from swe_metrics_for_all_initial_conditions import (pinn_errors, hll_errors, print_table, pinn_profile,
                         shock_sharpness, timing_report)
from swe_plotting_for_all_initial_conditions import (plot_history, plot_comparison, plot_error_curves,
                          plot_field_2d, plot_hx_with_constant_y)

EVAL_TIMES = (0.25, 0.5, 0.75, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="короткий прогон")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)

    if args.eval_only:
        model = load_trained()
        history = None
    else:
        epochs = args.epochs or (400 if args.quick else C.ADAM_EPOCHS)
        lbfgs = 20 if args.quick else C.LBFGS_STEPS
        model = PINN_for_all_initial_conditions()
        t0 = time.time()
        model, history = model.pinn_train(adam_epochs=epochs, lbfgs_steps=lbfgs)
        train_time = time.time() - t0
        print(f"\nОбучение заняло {train_time/60:.1f} мин")

    model.eval()

    # ---------------- метрики ----------------
    print("\nСчитаем ошибки относительно точного решения задачи Римана...")
    p_err = pinn_errors(model, EVAL_TIMES)
    h_err = hll_errors(EVAL_TIMES)
    print_table(p_err, h_err)

    # резкость фронта — главный показатель для разрывных решений
    print(f"\n{'t':>6} | {'ширина фронта PINN':>20} | {'ширина фронта HLL':>19}"
          f" | точное")
    print("-" * 62)
    from hll_solver_vectorized import solve_2d_dam_break
    snaps, X, Y, ts = solve_2d_dam_break(Nx=C.HLL_NX, Ny=C.HLL_NY,
                                         t_end=max(EVAL_TIMES),
                                         save_times=list(EVAL_TIMES))
    j = X.shape[1] // 2
    for t, (H, HU, HV) in zip(EVAL_TIMES, snaps):
        xp, hp, up, _ = pinn_profile(model, t)
        wp = shock_sharpness(xp, hp, t)
        wh = shock_sharpness(X[:, j], H[:, j], t)
        print(f"{t:>6.2f} | {wp:>20.4f} | {wh:>19.4f} |    0.0")

    # ---------------- время счёта ----------------
    print("\nВремя счёта:")
    rep = timing_report(model)
    for k, v in rep.items():
        print(f"  {k:>22}: {v:.3f} c")

    # ---------------- графики ----------------
    print("\nСтроим графики...")
    if history is not None:
        plot_history(history)
    plot_comparison(model, times=(0.0, 0.25, 0.5, 1.0))
    plot_error_curves(p_err, h_err)
    plot_field_2d(model)
    plot_hx_with_constant_y(model)
    print("Готово: data/loss.png, data/comparison.png, data/errors.png, "
          "data/field_2d.png, data/h_profile.png")


if __name__ == "__main__":
    main()
