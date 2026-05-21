"""
Тема 7. Комплексное наведение (гл. 10–11).

Алгоритмы 3–4: прямая и орбита; менеджер маршрута PathManager.

Запуск: python 7.py  или  F5
"""
from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

import _bootstrap  # noqa: F401

from angles import wrap_to_reference

_t6 = importlib.import_module("6")
heading_for_desired_course = _t6.heading_for_desired_course
ground_velocity_from_heading = _t6.ground_velocity_from_heading


# --- наведение (гл. 10) ---


def cross_track_error(
    p_n: float,
    p_e: float,
    r_n: float,
    r_e: float,
    chi_q: float,
) -> float:
    return float(-np.sin(chi_q) * (p_n - r_n) + np.cos(chi_q) * (p_e - r_e))


def course_from_line_direction(q_n: float, q_e: float) -> float:
    return float(np.arctan2(q_e, q_n))


def follow_straight_line(
    p_n: float,
    p_e: float,
    chi: float,
    r_n: float,
    r_e: float,
    q_n: float,
    q_e: float,
    *,
    chi_inf: float = np.pi / 2.0,
    k_path: float = 0.05,
) -> float:
    chi_q = wrap_to_reference(course_from_line_direction(q_n, q_e), chi)
    e_py = cross_track_error(p_n, p_e, r_n, r_e, chi_q)
    chi_d = -chi_inf * (2.0 / np.pi) * np.arctan(k_path * e_py)
    return float(chi_q + chi_d)


def follow_orbit(
    p_n: float,
    p_e: float,
    chi: float,
    c_n: float,
    c_e: float,
    rho: float,
    lam: int,
    *,
    k_orbit: float = 0.1,
) -> float:
    d_n = p_n - c_n
    d_e = p_e - c_e
    d = float(np.hypot(d_n, d_e))
    eta = wrap_to_reference(float(np.arctan2(d_e, d_n)), chi)
    chi_o = eta + lam * np.pi / 2.0
    rho_safe = max(rho, 1e-3)
    chi_d = (2.0 / np.pi) * np.arctan(k_orbit * (d - rho) / rho_safe)
    return float(chi_o + chi_d)


def course_to_heading(
    chi_c: float,
    V_a: float,
    w_n: float,
    w_e: float,
    *,
    use_wind_compensation: bool,
    psi_geom: float | None = None,
) -> float:
    if use_wind_compensation:
        psi_cmd, _ = heading_for_desired_course(chi_c, V_a, w_n, w_e, V_g_des=V_a)
        return psi_cmd
    if psi_geom is not None:
        return psi_geom
    return chi_c


# --- менеджер маршрута (гл. 11) ---


class PathFlag(IntEnum):
    LINE = 1
    ORBIT = 2


@dataclass
class LineSegment:
    r_n: float
    r_e: float
    q_n: float
    q_e: float
    length: float

    @property
    def chi_q(self) -> float:
        return math.atan2(self.q_e, self.q_n)


@dataclass
class OrbitSegment:
    c_n: float
    c_e: float
    rho: float
    lam: int
    turns: float


@dataclass
class Route:
    segments: list[LineSegment | OrbitSegment]


@dataclass
class PathCommand:
    flag: PathFlag
    r_n: float = 0.0
    r_e: float = 0.0
    q_n: float = 1.0
    q_e: float = 0.0
    c_n: float = 0.0
    c_e: float = 0.0
    rho: float = 0.0
    lam: int = -1


class RouteManager:
    def __init__(self, route: Route) -> None:
        self.route = route
        self.idx = 0
        self._orbit_angle = 0.0

    @property
    def finished(self) -> bool:
        return self.idx >= len(self.route.segments)

    def current(self) -> LineSegment | OrbitSegment | None:
        if self.finished:
            return None
        return self.route.segments[self.idx]

    def command(self) -> PathCommand | None:
        seg = self.current()
        if seg is None:
            return None
        if isinstance(seg, LineSegment):
            return PathCommand(
                flag=PathFlag.LINE,
                r_n=seg.r_n,
                r_e=seg.r_e,
                q_n=seg.q_n,
                q_e=seg.q_e,
            )
        return PathCommand(
            flag=PathFlag.ORBIT,
            c_n=seg.c_n,
            c_e=seg.c_e,
            rho=seg.rho,
            lam=seg.lam,
        )

    def update(self, p_n: float, p_e: float, vgn: float, vge: float, dt: float) -> None:
        seg = self.current()
        if seg is None or dt <= 0.0:
            return
        if isinstance(seg, LineSegment):
            along = (p_n - seg.r_n) * seg.q_n + (p_e - seg.r_e) * seg.q_e
            if along >= seg.length:
                self._advance()
        else:
            rx = p_n - seg.c_n
            ry = p_e - seg.c_e
            dist2 = max(rx * rx + ry * ry, 1e-6)
            omega = (rx * vge - ry * vgn) / dist2
            self._orbit_angle += abs(omega) * dt
            if self._orbit_angle >= seg.turns * (2.0 * math.pi):
                self._advance()

    def _advance(self) -> None:
        self.idx += 1
        self._orbit_angle = 0.0


def line_segment(r_n: float, r_e: float, chi_q: float, length: float) -> LineSegment:
    return LineSegment(
        r_n=r_n,
        r_e=r_e,
        q_n=math.cos(chi_q),
        q_e=math.sin(chi_q),
        length=length,
    )


def orbit_segment(
    c_n: float,
    c_e: float,
    rho: float,
    clockwise: bool,
    turns: float,
) -> OrbitSegment:
    return OrbitSegment(
        c_n=c_n,
        c_e=c_e,
        rho=rho,
        lam=1 if clockwise else -1,
        turns=turns,
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from _bootstrap import save_or_show

    _t21 = importlib.import_module("21")
    AircraftState = _t21.AircraftState
    step_guidance_model_919_heading = _t21.step_guidance_model_919_heading

    print("7.py — менеджер маршрута (консоль):")
    route = Route(
        segments=[
            line_segment(0.0, 0.0, np.deg2rad(45.0), 300.0),
            orbit_segment(350.0, 350.0, 100.0, clockwise=False, turns=0.5),
            line_segment(450.0, 450.0, np.deg2rad(-135.0), 200.0),
        ]
    )
    mgr = RouteManager(route)
    p_n, p_e, vgn, vge, dt = 0.0, 0.0, 15.0, 15.0, 1.0
    for step in range(40):
        cmd = mgr.command()
        if cmd is None:
            print(f"  t={step * dt:5.1f} с  маршрут завершён")
            break
        name = "LINE" if cmd.flag == PathFlag.LINE else "ORBIT"
        print(f"  t={step * dt:5.1f} с  сегмент {mgr.idx + 1}: {name}")
        mgr.update(p_n, p_e, vgn, vge, dt)
        p_n += vgn * dt
        p_e += vge * dt

    V_a, b_psi, dt, T = 22.0, 0.35, 0.05, 120.0
    w_n, w_e = 3.0, -3.0
    n = int(T / dt)
    pn = np.zeros(n)
    pe = np.zeros(n)
    state = AircraftState(p_n=0.0, p_e=-80.0, psi=0.0, chi=0.0)

    for k in range(n):
        _, _, _, chi = ground_velocity_from_heading(state.psi, V_a, w_n, w_e)
        pn[k], pe[k] = state.p_n, state.p_e
        if k <= n // 2:
            chi_c = follow_straight_line(state.p_n, state.p_e, chi, 0.0, 0.0, 1.0, 0.0)
        else:
            chi_c = follow_orbit(state.p_n, state.p_e, chi, 400.0, 200.0, 120.0, -1)
        psi_c = course_to_heading(chi_c, V_a, w_n, w_e, use_wind_compensation=True)
        state = step_guidance_model_919_heading(state, psi_c, V_a, w_n, w_e, b_psi, dt)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(pn[: n // 2 + 1], pe[: n // 2 + 1], "b-", label="прямая (алг. 3)")
    ax.plot(pn[n // 2 :], pe[n // 2 :], "r-", label="орбита (алг. 4)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_xlabel("p_n, м")
    ax.set_ylabel("p_e, м")
    ax.set_title("Наведение по прямой и окружности")
    ax.legend()
    plt.tight_layout()
    save_or_show(fig, "7.png")
    print("Готово: 7.py -> 7.png")
