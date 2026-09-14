"""
Error metrics: L2, Linf, relative errors.
"""

import numpy as np


def relative_l2(predicted: np.ndarray, exact: np.ndarray) -> float:
    """Relative L2 error: ||predicted - exact|| / ||exact||.

    Args:
        predicted: Array of predicted values.
        exact: Array of exact (reference) values.

    Returns:
        Relative L2 error as a float. Returns 0.0 if exact is zero.
    """
    norm_exact = np.linalg.norm(exact)
    if norm_exact < 1e-14:
        return 0.0
    return float(np.linalg.norm(predicted - exact) / norm_exact)


def absolute_l2(predicted: np.ndarray, exact: np.ndarray) -> float:
    """Absolute L2 error: ||predicted - exact||."""
    return float(np.linalg.norm(predicted - exact))


def linf(predicted: np.ndarray, exact: np.ndarray) -> float:
    """L-infinity (max) error: max|predicted - exact|."""
    return float(np.max(np.abs(predicted - exact)))


def compute_all_errors(
    predicted: dict[str, np.ndarray],
    exact: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute L2 and Linf errors for all state variables.

    Args:
        predicted: Dictionary {'h': array, 'u': array, 'v': array}.
        exact: Dictionary with the same keys.

    Returns:
        Dictionary with keys like 'L2_h', 'Linf_h', 'L2_u', 'Linf_u', etc.

    Example:
        >>> pred = {'h': h_pinn, 'u': u_pinn, 'v': v_pinn}
        >>> exact = problem.exact_solution(x, t)
        >>> errors = compute_all_errors(pred, exact)
        >>> print(f"L2(h) = {errors['L2_h']:.2%}")
    """
    errors = {}
    for var in predicted:
        if var in exact:
            p = predicted[var]
            e = exact[var]
            errors[f"L2_{var}"] = relative_l2(p, e)
            errors[f"Linf_{var}"] = linf(p, e)
    return errors


def print_error_table(
    pinn_errors: dict[float, dict[str, float]],
    fvm_errors: dict[float, dict[str, float]],
    title: str = "PINN vs FVM errors",
) -> None:
    """Print a formatted table comparing errors at different times.

    Args:
        pinn_errors: {t: {'L2_h': ..., 'Linf_h': ..., ...}}.
        fvm_errors: Same format.
        title: Table title.
    """
    print(f"\n{title}")
    print(
        f"{'t':>6} | {'PINN L2(h)':>11} {'PINN Linf(h)':>13} | "
        f"{'FVM L2(h)':>10} {'FVM Linf(h)':>12}"
    )
    print("-" * 62)

    for t in sorted(pinn_errors):
        p = pinn_errors[t]
        fvm_key = min(fvm_errors, key=lambda s: abs(s - t)) if fvm_errors else None
        f = fvm_errors[fvm_key] if fvm_key is not None else None

        if f:
            print(
                f"{t:>6.2f} | {100 * p['L2_h']:>10.2f}% {p['Linf_h']:>13.4f} | "
                f"{100 * f['L2_h']:>9.2f}% {f['Linf_h']:>12.4f}"
            )
        else:
            print(f"{t:>6.2f} | {100 * p['L2_h']:>10.2f}% {p['Linf_h']:>13.4f} |")
