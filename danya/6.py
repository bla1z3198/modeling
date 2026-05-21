"""
Тема 6. Нелинейные модели и ветровые воздействия (гл. 2.3–2.4).

Ветровой треугольник (2.6)–(2.12), модели ветра, угол сноса ψ_c = χ − ψ.

Запуск: python 6.py  или  F5
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

import _bootstrap  # noqa: F401

from angles import wrap_pi

WindField = Callable[[float, float, float], tuple[float, float]]


# --- ветровой треугольник ---


def ground_velocity_from_heading(
    psi: float,
    V_a: float,
    w_n: float,
    w_e: float,
) -> tuple[float, float, float, float]:
    van = V_a * np.cos(psi)
    vae = V_a * np.sin(psi)
    vgn = van + w_n
    vge = vae + w_e
    V_g = float(np.hypot(vgn, vge))
    chi = float(np.arctan2(vge, vgn))
    return vgn, vge, V_g, chi


def drift_angle(psi: float, chi: float) -> float:
    return float(wrap_pi(chi - psi))


def airspeed_from_ground(
    chi: float,
    gamma: float,
    V_g: float,
    w_n: float,
    w_e: float,
    w_d: float = 0.0,
) -> float:
    vgn = V_g * np.cos(gamma) * np.cos(chi)
    vge = V_g * np.cos(gamma) * np.sin(chi)
    vgd = V_g * np.sin(gamma)
    diff_n = vgn - w_n
    diff_e = vge - w_e
    diff_d = vgd - w_d
    return float(np.sqrt(diff_n * diff_n + diff_e * diff_e + diff_d * diff_d))


def heading_from_wind(
    chi: float,
    gamma: float,
    V_a: float,
    w_n: float,
    w_e: float,
) -> float:
    num = w_e * np.cos(chi) - w_n * np.sin(chi)
    den = V_a + w_n * np.cos(chi) + w_e * np.sin(chi)
    if abs(den) < 1e-9:
        return float(chi)
    return float(wrap_pi(chi - np.arctan2(num, den)))


def heading_for_desired_course(
    chi_des: float,
    V_a: float,
    w_n: float,
    w_e: float,
    V_g_des: float | None = None,
) -> tuple[float, float]:
    if V_g_des is None:
        V_g_des = V_a
    gn = V_g_des * np.cos(chi_des) - w_n
    ge = V_g_des * np.sin(chi_des) - w_e
    psi_cmd = float(np.arctan2(ge, gn))
    mag = float(np.hypot(gn, ge))
    return psi_cmd, mag


def ground_velocity(psi: float, V_a: float, W_x: float, W_y: float) -> tuple[float, float, float, float]:
    return ground_velocity_from_heading(psi, V_a, W_x, W_y)


def drift_angle_chi_c(psi: float, chi: float) -> float:
    return drift_angle(psi, chi)


# --- модели ветра ---


def wind_constant(w_n: float, w_e: float) -> WindField:
    def w(_t: float, _p_n: float, _p_e: float) -> tuple[float, float]:
        return float(w_n), float(w_e)

    return w


def wind_sinusoidal(
    w_mean_n: float,
    w_mean_e: float,
    amp_n: float,
    amp_e: float,
    omega: float,
    phi_n: float = 0.0,
    phi_e: float = 0.0,
) -> WindField:
    def w(t: float, _p_n: float, _p_e: float) -> tuple[float, float]:
        wn = w_mean_n + amp_n * np.sin(omega * t + phi_n)
        we = w_mean_e + amp_e * np.cos(omega * t + phi_e)
        return float(wn), float(we)

    return w


def wind_shear_linear(w_ref_n: float, w_ref_e: float, k_e: float) -> WindField:
    def w(_t: float, _p_n: float, p_e: float) -> tuple[float, float]:
        return float(w_ref_n), float(w_ref_e + k_e * p_e)

    return w


def combine_winds(*fields: WindField) -> WindField:
    def w(t: float, p_n: float, p_e: float) -> tuple[float, float]:
        sn, se = 0.0, 0.0
        for f in fields:
            fn, fe = f(t, p_n, p_e)
            sn += fn
            se += fe
        return sn, se

    return w


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from _bootstrap import save_or_show

    V_a = 22.0
    w_n, w_e = 3.0, -3.0
    psi_deg = np.linspace(-180, 180, 361)
    chi_deg = np.zeros_like(psi_deg)
    psi_c_deg = np.zeros_like(psi_deg)
    for i, psi_d in enumerate(psi_deg):
        psi = np.deg2rad(psi_d)
        _, _, _, chi = ground_velocity_from_heading(psi, V_a, w_n, w_e)
        chi_deg[i] = np.degrees(chi)
        psi_c_deg[i] = np.degrees(drift_angle(psi, chi))

    t = np.linspace(0, 60, 600)
    base = wind_constant(w_n, w_e)
    gust = wind_sinusoidal(0.0, 0.0, 1.2, 0.9, 0.35, phi_e=0.7)
    wn_t = np.array([combine_winds(base, gust)(ti, 0, 0)[0] for ti in t])
    we_t = np.array([combine_winds(base, gust)(ti, 0, 0)[1] for ti in t])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(psi_deg, chi_deg, "b-")
    axes[0].plot([-180, 180], [-180, 180], "k--", alpha=0.4)
    axes[0].set_xlabel("ψ, °")
    axes[0].set_ylabel("χ, °")
    axes[0].set_title("Ветровой треугольник")
    axes[0].grid(True)

    axes[1].plot(psi_deg, psi_c_deg, "r-")
    axes[1].set_xlabel("ψ, °")
    axes[1].set_ylabel("ψ_c, °")
    axes[1].set_title("Угол сноса ψ_c = χ − ψ")
    axes[1].grid(True)

    axes[2].plot(t, wn_t, label="w_n")
    axes[2].plot(t, we_t, label="w_e")
    axes[2].set_xlabel("t, с")
    axes[2].set_ylabel("м/с")
    axes[2].set_title("Модели ветра")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    save_or_show(fig, "6.png")
    print("Готово: 6.py -> 6.png")
