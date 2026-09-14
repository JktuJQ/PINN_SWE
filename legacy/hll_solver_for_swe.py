import numpy as np

from swe_constants import g, CFL, X_MIN, X_MAX, Y_MIN, Y_MAX, H_LEFT, H_RIGHT, X_DAM, T_MAX


def find_flux_x(U_L, U_R):
    """
    U_L - состояние в клетке слева от границы
    U_R - состояние в клетке справа от границы
    Нам нужно найти поток через границу между ними
    """

    h_L, hu_L, hv_L = U_L
    h_R, hu_R, hv_R = U_R

    u_L = hu_L / h_L if h_L > 1e-8 else 0.0
    v_L = hv_L / h_L if h_L > 1e-8 else 0.0
    u_R = hu_R / h_R if h_R > 1e-8 else 0.0
    v_R = hv_R / h_R if h_R > 1e-8 else 0.0

    # Скорости гравитационных волн (я не разобрался зачем они)
    c_L = np.sqrt(g * h_L)
    c_R = np.sqrt(g * h_R)

    s_L = min(u_L - c_L, u_R - c_R) # Самая сильная волна, идущая влево
    s_R = max(u_L + c_L, u_R + c_R) # ... вправо

    # Считаем потоки по тем же формулам, что и в PINN
    F_L = np.array([
        hu_L,
        hu_L * u_L + 0.5 * g * h_L ** 2,
        hu_L * v_L
    ])
    F_R = np.array([
        hu_R,
        hu_R * u_R + 0.5 * g * h_R ** 2,
        hu_R * v_R,
    ])

    # Если волны с обеих сторон границы направлены вправо, то берем левую
    if s_L >= 0:
        return F_L
    elif s_R <= 0:
        return F_R
    else:
        return (s_R * F_L - s_L * F_R + s_L * s_R * (U_R - U_L)) / (s_R - s_L)
        # Как-то выводится из заакона сохранения потока


def find_flux_y(U_L, U_R):
    """
    U_L - состояние в клетке слева(снизу) от границы
    U_R - состояние в клетке справа(сверху) от границы
    Нам нужно найти поток через границу между ними
    """

    h_L, hu_L, hv_L = U_L
    h_R, hu_R, hv_R = U_R

    u_L = hu_L / h_L if h_L > 1e-8 else 0.0
    v_L = hv_L / h_L if h_L > 1e-8 else 0.0
    u_R = hu_R / h_R if h_R > 1e-8 else 0.0
    v_R = hv_R / h_R if h_R > 1e-8 else 0.0

    # Скорости гравитационных волн (я не разобрался зачем они)
    c_L = np.sqrt(g * h_L)
    c_R = np.sqrt(g * h_R)

    s_L = min(v_L - c_L, v_R - c_R) # Самая сильная волна, идущая влево
    s_R = max(v_L + c_L, v_R + c_R) # ... вправо

    # Считаем потоки по тем же формулам, что и в PINN
    G_L = np.array([
        hv_L,
        hu_L * v_L,
        hv_L * v_L + 0.5 * g * h_L ** 2,
    ])
    G_R = np.array([
        hv_R,
        hu_R * v_R,
        hv_R * v_R + 0.5 * g * h_R ** 2,
    ])

    if s_L >= 0:
        return G_L
    elif s_R <= 0:
        return G_R
    else:
        return (s_R * G_L - s_L * G_R + s_L * s_R * (U_R - U_L)) / (s_R - s_L)


def hll_step(H, HU, HV, dx, dy, dt, Nx, Ny):
    """H, HU, HV - массивы Nx * Ny c состояниями сетки на текущий момент"""

    # Находим потоки на всех границах по x
    F = np.zeros((3, Nx + 1, Ny))
    for i in range(Nx + 1):
        for j in range(Ny):
            if i == 0:
                # Если самая левая граница, у которой нет соседа слева
                U_L = np.array([H[0, j], HU[0, j], HV[0, j]])
                U_R = np.array([H[0, j], HU[0, j], HV[0, j]])
            elif i == Nx:
                U_L = np.array([H[-1, j], HU[-1, j], HV[-1, j]])
                U_R = np.array([H[-1, j], HU[-1, j], HV[-1, j]])
            else:
                U_L = np.array([H[i-1, j], HU[i-1, j], HV[i-1, j]])
                U_R = np.array([H[i, j], HU[i, j], HV[i, j]])
            F[:, i, j] = find_flux_x(U_L, U_R)

    # Находим потоки на всех границах по y
    G = np.zeros((3, Nx, Ny + 1))
    for i in range(Nx):
        for j in range(Ny + 1):
            if j == 0:
                # Если самая левая граница, у которой нет соседа слева
                U_L = np.array([H[i, 0], HU[i, 0], HV[i, 0]])
                U_R = np.array([H[i, 0], HU[i, 0], HV[i, 0]])
            elif j == Ny:
                U_L = np.array([H[i, -1], HU[i, -1], HV[i, -1]])
                U_R = np.array([H[i, -1], HU[i, -1], HV[i, -1]])
            else:
                U_L = np.array([H[i, j-1], HU[i, j-1], HV[i, j-1]])
                U_R = np.array([H[i, j], HU[i, j], HV[i, j]])
            G[:, i, j] = find_flux_y(U_L, U_R)

    H_new = H.copy()
    HU_new = HU.copy()
    HV_new = HV.copy()

    for i in range(Nx):
        for j in range(Ny):
            H_new[i, j] = H[i, j] - dt/dx * (F[0, i+1, j] - F[0, i, j]) - dt/dy * (G[0, i, j+1] - G[0, i, j])
            HU_new[i, j] = HU[i, j] - dt / dx * (F[1, i + 1, j] - F[1, i, j]) - dt / dy * (G[1, i, j + 1] - G[1, i, j])
            HV_new[i, j] = HV[i, j] - dt / dx * (F[2, i + 1, j] - F[2, i, j]) - dt / dy * (G[2, i, j + 1] - G[2, i, j])

    H_new = np.maximum(H_new, 1e-8)

    return H_new, HU_new, HV_new


def solve_2d_dam_break(x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX, Nx=100, Ny=100,
                       h_left=H_LEFT, h_right=H_RIGHT, x_dam=X_DAM, t_end=T_MAX):
    # Создаем сетку
    x = np.linspace(x_min, x_max, Nx+1)
    y = np.linspace(y_min, y_max, Ny+1)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    x_center = 0.5 * (x[:-1] + x[1:])
    y_center = 0.5 * (y[:-1] + y[1:])

    X, Y = np.meshgrid(x_center, y_center, indexing="ij")

    # Начальные условия
    H = np.where(X <= x_dam, h_left, h_right).astype(float)
    U = np.zeros_like(H)
    V = np.zeros_like(H)
    HU = H * U
    HV = H * V

    history = []
    times = []

    t = 0.0
    step_count = 0
    while t < t_end:
        # Адаптивное вычисление dt, чтобы схема оставалась устойчивой при быстром течении.
        #
        # ИСПРАВЛЕНО (было две ошибки):
        #  1) использовался массив U, который создаётся при инициализации и больше
        #     НИКОГДА не обновляется (шаг схемы меняет только HU/HV). То есть скорость
        #     всегда считалась нулевой, и dt не реагировал на разгон потока.
        #  2) бралось одномерное условие CFL*min(dx,dy)/s_max; для двумерной схемы
        #     вклады обоих направлений складываются, иначе шаг завышен примерно вдвое.
        c = np.sqrt(g * H)
        U = np.where(H > 1e-8, HU / np.maximum(H, 1e-8), 0.0)
        V = np.where(H > 1e-8, HV / np.maximum(H, 1e-8), 0.0)
        dt = CFL / (np.max(np.abs(U) + c) / dx + np.max(np.abs(V) + c) / dy)
        dt = min(dt, t_end - t)

        H, HU, HV = hll_step(H, HU, HV, dx, dy, dt, Nx, Ny)

        t += dt

        # состояние и момент времени должны соответствовать друг другу:
        # раньше сохранялось состояние ПОСЛЕ шага, но время ДО него
        history.append(H)
        times.append(t)
        step_count += 1
        if step_count % 10 == 0:
            print(f"Текущее время: {round(t, 2)}   Всего время: {T_MAX}")

    return history, X, times