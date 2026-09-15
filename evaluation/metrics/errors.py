"""Error metrics: L2, Linf, relative errors."""

import numpy as np


def relative_l2(predicted: np.ndarray, exact: np.ndarray) -> float:
    """Relative L2 error: ||predicted - exact|| / ||exact||.

    Returns 0.0 if the exact norm is (numerically) zero.
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
    """Compute L2 and Linf errors for every variable present in both dicts.

    Returns a flat dictionary with keys ``'L2_<var>'`` and ``'Linf_<var>'``.
    """
    errors: dict[str, float] = {}
    for var in predicted:
        if var in exact:
            p = predicted[var]
            e = exact[var]
            errors[f"L2_{var}"] = relative_l2(p, e)
            errors[f"Linf_{var}"] = linf(p, e)
    return errors


def print_error_table(
    pinn_errors: dict[float, dict[str, float]],
    fvm_errors: dict[float, dict[str, float]] | None = None,
    title: str = "PINN vs FVM errors",
    variables: tuple[str, ...] = ("h",),
) -> None:
    """Print a formatted table of L2 and Linf errors at each time.

    Args:
        pinn_errors: ``{t: {'L2_h': ..., 'Linf_h': ..., ...}}``.
        fvm_errors: Same format, or None to print only PINN columns.
        title: Table header.
        variables: Which state variables to show. Must match keys present
            in the error dicts (as ``'L2_<var>'`` / ``'Linf_<var>'``).
    """
    print(f"\n{title}")

    header = f"{'t':>6}"
    for var in variables:
        header += f" | {'PINN L2(' + var + ')':>14} {'PINN Linf(' + var + ')':>16}"
    if fvm_errors:
        for var in variables:
            header += f" | {'FVM L2(' + var + ')':>13} {'FVM Linf(' + var + ')':>15}"
    print(header)
    print("-" * len(header))

    for t in sorted(pinn_errors):
        row = f"{t:>6.2f}"
        for var in variables:
            row += (
                f" | {100 * pinn_errors[t].get(f'L2_{var}', float('nan')):>13.2f}%"
                f" {pinn_errors[t].get(f'Linf_{var}', float('nan')):>16.4f}"
            )
        if fvm_errors:
            key = min(fvm_errors, key=lambda s: abs(s - t))
            f = fvm_errors[key]
            for var in variables:
                row += (
                    f" | {100 * f.get(f'L2_{var}', float('nan')):>12.2f}%"
                    f" {f.get(f'Linf_{var}', float('nan')):>15.4f}"
                )
        print(row)
