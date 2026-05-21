"""
Занятие 21. Кинематическая модель наведения (9.19), (9.8)–(9.9).

Запуск: python 21.py  или  F5
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np

import _bootstrap  # noqa: F401

from angles import wrap_pi

_t6 = importlib.import_module("6")
ground_velocity_from_heading = _t6.ground_velocity_from_heading
drift_angle = _t6.drift_angle


@dataclass
class AircraftState:
    p_n: float
    p_e: float
    psi: float
    chi: float


def step_guidance_model_919_heading(
    state: AircraftState,
    psi_c: float,
    V_a: float,
    w_n: float,
    w_e: float,
    b_psi: float,
    dt: float,
) -> AircraftState:
    vgn, vge, _, chi = ground_velocity_from_heading(state.psi, V_a, w_n, w_e)
    psi_dot = b_psi * wrap_pi(psi_c - state.psi)
    psi = state.psi + psi_dot * dt
    p_n = state.p_n + vgn * dt
    p_e = state.p_e + vge * dt
    _, _, _, chi = ground_velocity_from_heading(psi, V_a, w_n, w_e)
    return AircraftState(p_n=p_n, p_e=p_e, psi=psi, chi=chi)


def step_guidance_model_919_course(
    state: AircraftState,
    chi_c: float,
    V_a: float,
    w_n: float,
    w_e: float,
    b_chi: float,
    dt: float,
) -> AircraftState:
    chi = state.chi
    chi_dot = b_chi * wrap_pi(chi_c - chi)
    chi = chi + chi_dot * dt
    p_n = state.p_n + (V_a * np.cos(chi) + w_n) * dt
    p_e = state.p_e + (V_a * np.sin(chi) + w_e) * dt
    return AircraftState(p_n=p_n, p_e=p_e, psi=chi, chi=chi)


def trajectory_from_velocity(
    ts: np.ndarray,
    V_g: np.ndarray,
    chi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    chi = np.asarray(chi)
    V_g = np.asarray(V_g)
    vx = V_g * np.cos(chi)
    vy = V_g * np.sin(chi)
    p_n = np.zeros_like(ts)
    p_e = np.zeros_like(ts)
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i - 1]
        p_n[i] = p_n[i - 1] + 0.5 * dt * (vx[i] + vx[i - 1])
        p_e[i] = p_e[i - 1] + 0.5 * dt * (vy[i] + vy[i - 1])
    return p_n, p_e


def psi_c_value(psi: float, chi: float) -> float:
    return drift_angle(psi, chi)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from _bootstrap import save_or_show

    V_a, b_chi, dt, T = 22.0, 0.4, 0.05, 80.0
    w_n, w_e = 3.0, -3.0
    n = int(T / dt)
    ts = np.linspace(0, T, n)
    chi_c = np.deg2rad(30.0)

    state = AircraftState(p_n=0.0, p_e=0.0, psi=0.0, chi=0.0)
    pn = np.zeros(n)
    pe = np.zeros(n)
    chis = np.zeros(n)

    for k in range(n):
        pn[k], pe[k], chis[k] = state.p_n, state.p_e, state.chi
        state = step_guidance_model_919_course(state, chi_c, V_a, w_n, w_e, b_chi, dt)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(pn, pe, "k-")
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True)
    ax1.set_xlabel("p_n, м")
    ax1.set_ylabel("p_e, м")
    ax1.set_title("Траектория (9.19)")

    ax2.plot(ts, np.degrees(chis), label="χ(t)")
    ax2.axhline(np.degrees(chi_c), color="r", ls="--", label="χ_c")
    ax2.grid(True)
    ax2.set_xlabel("t, с")
    ax2.set_ylabel("χ, °")
    ax2.legend()

    plt.tight_layout()
    save_or_show(fig, "21.png")
    print("Готово: 21.py -> 21.png")
