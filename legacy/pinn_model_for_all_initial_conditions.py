import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import swe_constants as C
from swe_constants import (X_MIN, X_MAX, Y_MIN, Y_MAX, T_MIN, T_MAX, H_MIN, H_MAX, DEVICE)
from swe_sampler_for_all_initial_conditions import (sample_interior, sample_initial, sample_boundary_x,
                         sample_boundary_y, initial_condition, refine_points)
from SWE import residual_for_all_initial_conditions


def _inv_softplus(z):
    return math.log(math.expm1(z))


class PINN_for_all_initial_conditions(nn.Module):
    """Полносвязная сеть (x, y, t, h_L, h_R) -> (h, u, v).

    Отличия от первой версии:
      * только гладкие активации (tanh) — ReLU даёт нулевую вторую производную,
        из-за чего невязка ФДУ теряет смысл, а профиль скорости идёт изломами;
      * вход нормируется в [-1, 1] — иначе x ~ 10 насыщает tanh с первого слоя;
      * h = H_FLOOR + softplus(...) — положительность глубины по построению,
        поэтому штраф W_POS и лишний прямой проход больше не нужны.
    """

    def __init__(self, hidden=C.HIDDEN, depth=C.DEPTH, fourier=C.USE_FOURIER):
        super().__init__()
        self.fourier = fourier

        in_dim = 5
        if fourier:
            B = torch.randn(in_dim, C.FOURIER_FEATURES) * C.FOURIER_SCALE
            self.register_buffer("B", B)
            in_dim = 2 * C.FOURIER_FEATURES

        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 3)]
        self.net = nn.Sequential(*layers)

        self.register_buffer("in_lo", torch.tensor([[X_MIN, Y_MIN, T_MIN, H_MIN, H_MIN]]))
        self.register_buffer("in_hi", torch.tensor([[X_MAX, Y_MAX, T_MAX, H_MAX, H_MAX]]))

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        # стартовать с разумного уровня воды, а не с softplus(0) = 0.69
        with torch.no_grad():
            self.net[-1].bias[0] = _inv_softplus(C.H_INIT - C.H_FLOOR)

    def forward(self, x, y, t, h_L, h_R):
        inp = torch.cat([x, y, t, h_L, h_R], dim=1)
        z = 2.0 * (inp - self.in_lo) / (self.in_hi - self.in_lo) - 1.0

        if self.fourier:
            proj = 2.0 * math.pi * (z @ self.B)
            z = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)

        out = self.net(z)
        h = C.H_FLOOR + F.softplus(out[:, 0:1])
        u = out[:, 1:2]
        v = out[:, 2:3]
        return h, u, v

    # ------------------------------------------------------------------
    # Функция потерь
    # ------------------------------------------------------------------
    def _pde_loss(self, x, y, t, h_L, h_R, nu, t_max, causal=True):
        R = residual_for_all_initial_conditions(self, x, y, t, h_L, h_R, nu=nu)
        r2 = (R ** 2).mean(dim=1)

        if not causal:
            return r2.mean()

        # --- причинностное взвешивание ---
        n_bins = C.N_TIME_BINS
        span = max(t_max - T_MIN, 1e-8)
        idx = ((t.squeeze(1) - T_MIN) / span * n_bins).long().clamp(0, n_bins - 1)

        ones = torch.ones_like(r2)
        bin_sum = torch.zeros(n_bins, device=r2.device).index_add_(0, idx, r2)
        bin_cnt = torch.zeros(n_bins, device=r2.device).index_add_(0, idx, ones)
        occupied = bin_cnt > 0
        bin_mean = bin_sum / bin_cnt.clamp(min=1.0)

        # вес бина гасится накопленной невязкой всех предыдущих моментов
        cum_prev = torch.cumsum(bin_mean.detach(), 0) - bin_mean.detach()
        w = torch.exp(-C.CAUSAL_EPS * cum_prev) * occupied.float()

        return (w * bin_mean).sum() / w.sum().clamp(min=1e-12)

    def compute_losses(self, batch, nu, t_max):
        x_f, y_f, t_f, h_L_f, h_R_f = batch["interior"]
        L_pde = self._pde_loss(x_f, y_f, t_f, h_L_f, h_R_f, nu, t_max)

        x_i, y_i, t_i, h_L_i, h_R_i = batch["ic"]
        h_p, u_p, v_p = self(x_i, y_i, t_i, h_L_i, h_R_i)
        h_t_, u_t_, v_t_ = initial_condition(x_i, y_i, h_L_i, h_R_i)
        L_ic = (((h_p - h_t_) ** 2).mean()
                + ((u_p - u_t_) ** 2).mean()
                + ((v_p - v_t_) ** 2).mean())

        x_b, y_b, t_b, h_L_b, h_R_b = batch["bc_x"]
        h_bx, u_bx, v_bx = self(x_b, y_b, t_b, h_L_b, h_R_b)
        # дальнее поле: h_L слева от X_DAM, h_R справа
        h_far = torch.where(x_b <= C.X_DAM, h_L_b, h_R_b)
        L_bc = (((h_bx - h_far) ** 2).mean()
                + (u_bx ** 2).mean()
                + (v_bx ** 2).mean())

        x_y, y_y, t_y, h_L_y, h_R_y = batch["bc_y"]
        _, _, v_by = self(x_y, y_y, t_y, h_L_y, h_R_y)
        L_bc = L_bc + (v_by ** 2).mean()

        return {"pde": L_pde, "ic": L_ic, "bc": L_bc}

    # ------------------------------------------------------------------
    def _grad_norm(self, loss):
        gs = torch.autograd.grad(loss, list(self.parameters()),
                                 retain_graph=True, allow_unused=True)
        total = 0.0
        for gi in gs:
            if gi is not None:
                total += (gi ** 2).sum()
        return torch.sqrt(total + 1e-30)

    def _rebalance(self, losses, weights):
        """Автобалансировка весов по нормам градиентов (Wang et al. 2021).

        Снимает ручной подбор W_IC/W_BC: каждый член получает вес, выравнивающий
        его вклад в градиент с вкладом невязки ФДУ.
        """
        ref = self._grad_norm(losses["pde"]).item()
        for k in ("ic", "bc"):
            gk = self._grad_norm(losses[k]).item()
            if gk < 1e-20:
                continue
            target = ref / gk
            new = C.ADAPT_ALPHA * weights[k] + (1.0 - C.ADAPT_ALPHA) * target
            weights[k] = float(min(max(new, C.W_MIN), C.W_MAX))
        return weights

    # ------------------------------------------------------------------
    def pinn_train(self, adam_epochs=C.ADAM_EPOCHS, lbfgs_steps=C.LBFGS_STEPS,
                   verbose=True):
        self.to(DEVICE)

        ok, reach = C.check_domain_is_large_enough()
        if verbose:
            print(f"Устройство: {DEVICE}")
            print(f"Волны за t={T_MAX} доходят до |x|={reach:.2f}, граница |x|={X_MAX}"
                  f" -> условие Дирихле {'точное' if ok else 'НЕТОЧНОЕ (!)'}")

        optim = torch.optim.Adam(self.parameters(), lr=C.LR)
        gamma = (C.LR_FINAL / C.LR) ** (1.0 / max(adam_epochs, 1))
        sched = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)

        weights = {"pde": C.W_PDE, "ic": C.W_IC, "bc": C.W_BC}
        history = {k: [] for k in
                   ("loss", "pde", "ic", "bc", "nu", "t_max", "w_ic", "w_bc")}

        rar = None
        pbar = tqdm(range(adam_epochs), desc="Adam", disable=not verbose)
        for epoch in pbar:
            frac = epoch / max(adam_epochs - 1, 1)

            # --- отжиг искусственной вязкости ---
            a = min(frac / C.VISC_ANNEAL_FRAC, 1.0)
            nu = C.VISC_START * (C.VISC_END / C.VISC_START) ** a

            # --- наращивание горизонта по времени ---
            if frac < C.CURRICULUM_FRAC:
                stage = int(frac / C.CURRICULUM_FRAC * C.CURRICULUM_STAGES)
                t_max = T_MIN + (T_MAX - T_MIN) * (stage + 1) / C.CURRICULUM_STAGES
            else:
                t_max = T_MAX

            # --- адаптивное добавление сложных точек ---
            if C.RAR_EVERY and epoch > 0 and epoch % C.RAR_EVERY == 0:
                rar = refine_points(self, residual_for_all_initial_conditions,
                                    nu, t_max=t_max)

            x_f, y_f, t_f, h_L_f, h_R_f = sample_interior(t_max=t_max)
            if rar is not None:
                x_f = torch.cat([x_f, rar[0]])
                y_f = torch.cat([y_f, rar[1]])
                t_f = torch.cat([t_f, rar[2]])
                h_L_f = torch.cat([h_L_f, rar[3]])
                h_R_f = torch.cat([h_R_f, rar[4]])

            batch = {
                "interior": (x_f, y_f, t_f, h_L_f, h_R_f),
                "ic": sample_initial(),
                "bc_x": sample_boundary_x(t_max=t_max),
                "bc_y": sample_boundary_y(t_max=t_max),
            }

            optim.zero_grad(set_to_none=True)
            losses = self.compute_losses(batch, nu, t_max)

            if C.ADAPTIVE_WEIGHTS and epoch % C.ADAPT_EVERY == 0 and epoch > 0:
                weights = self._rebalance(losses, weights)

            loss = sum(weights[k] * losses[k] for k in losses)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optim.step()
            sched.step()

            history["loss"].append(loss.item())
            for k in ("pde", "ic", "bc"):
                history[k].append(losses[k].item())
            history["nu"].append(nu)
            history["t_max"].append(t_max)
            history["w_ic"].append(weights["ic"])
            history["w_bc"].append(weights["bc"])

            if verbose and epoch % 100 == 0:
                pbar.set_description(
                    f"Adam L={loss.item():.2e} PDE={losses['pde'].item():.2e} "
                    f"IC={losses['ic'].item():.2e} BC={losses['bc'].item():.2e} "
                    f"nu={nu:.3f} T={t_max:.2f}"
                )

        # ------------------------------------------------------------------
        # Дожим L-BFGS на фиксированном наборе точек с финальной вязкостью
        # ------------------------------------------------------------------
        if lbfgs_steps > 0:
            if verbose:
                print("L-BFGS дожим...")
            nu = C.VISC_END
            x_f, y_f, t_f, h_L_f, h_R_f = sample_interior(n=C.N_F * 2, t_max=T_MAX)
            batch = {
                "interior": (x_f, y_f, t_f, h_L_f, h_R_f),
                "ic": sample_initial(n=C.N_IC * 2),
                "bc_x": sample_boundary_x(n=C.N_BC * 2),
                "bc_y": sample_boundary_y(n=C.N_BC * 2),
            }
            lb = torch.optim.LBFGS(self.parameters(), lr=1.0,
                                   max_iter=lbfgs_steps,
                                   history_size=50,
                                   tolerance_grad=1e-12,
                                   tolerance_change=1e-14,
                                   line_search_fn="strong_wolfe")
            state = {"n": 0}

            def closure():
                lb.zero_grad(set_to_none=True)
                ls = self.compute_losses(batch, nu, T_MAX)
                l = sum(weights[k] * ls[k] for k in ls)
                l.backward()
                state["n"] += 1
                if state["n"] % 25 == 0:
                    history["loss"].append(l.item())
                    for k in ("pde", "ic", "bc"):
                        history[k].append(ls[k].item())
                    history["nu"].append(nu)
                    history["t_max"].append(T_MAX)
                    history["w_ic"].append(weights["ic"])
                    history["w_bc"].append(weights["bc"])
                    if verbose:
                        print(f"  L-BFGS {state['n']:4d}  L={l.item():.3e}  "
                              f"PDE={ls['pde'].item():.3e}")
                return l

            try:
                lb.step(closure)
            except RuntimeError as e:
                print(f"  L-BFGS прерван: {e}")

        torch.save(self.state_dict(), C.MODEL_PATH_FOR_ALL_INITIAL_CONDITIONS)
        return self, history


def load_trained(path=C.MODEL_PATH_FOR_ALL_INITIAL_CONDITIONS):
    model = PINN_for_all_initial_conditions().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model
