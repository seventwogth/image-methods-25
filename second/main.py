from __future__ import annotations

import csv
from dataclasses import dataclass
from math import pi, sqrt
from typing import Iterable

Vector = tuple[float, float, float]
Color = tuple[float, float, float]


@dataclass(frozen=True)
class LightSource:
    """Точечный источник света с цветной интенсивностью и осью излучения."""

    name: str
    intensity_rgb: Color  # I0(RGB)
    axis: Vector  # O
    position: Vector  # P_L


@dataclass(frozen=True)
class TrianglePlane:
    p0: Vector
    p1: Vector
    p2: Vector


@dataclass(frozen=True)
class Material:
    color_rgb: Color  # K(RGB)
    kd: float
    ks: float
    ke: float


def dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def add(a: Vector, b: Vector) -> Vector:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def sub(a: Vector, b: Vector) -> Vector:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def mul(v: Vector, k: float) -> Vector:
    return v[0] * k, v[1] * k, v[2] * k


def cross(a: Vector, b: Vector) -> Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(v: Vector) -> float:
    return sqrt(dot(v, v))


def normalize(v: Vector) -> Vector:
    length = norm(v)
    if length == 0:
        raise ValueError("Нулевой вектор нельзя нормализовать")
    return v[0] / length, v[1] / length, v[2] / length


def clamp_nonnegative(x: float) -> float:
    return max(0.0, x)


def to_global_point(plane: TrianglePlane, x: float, y: float) -> Vector:
    """Перевод локальных координат в глобальные: P_T = P0 + e1*x + e2*y."""
    e1 = normalize(sub(plane.p1, plane.p0))
    e2 = normalize(sub(plane.p2, plane.p0))
    return add(plane.p0, add(mul(e1, x), mul(e2, y)))


def plane_normal(plane: TrianglePlane) -> Vector:
    e1 = sub(plane.p1, plane.p0)
    e2 = sub(plane.p2, plane.p0)
    return normalize(cross(e1, e2))


def radiation_intensity(light: LightSource, direction_from_light: Vector) -> Color:
    """I(RGB, s) = I0(RGB) * max(0, cos(theta))."""
    light_axis = normalize(light.axis)
    d = normalize(direction_from_light)
    cos_theta = clamp_nonnegative(dot(d, light_axis))
    return tuple(channel * cos_theta for channel in light.intensity_rgb)  # type: ignore[return-value]


def illuminance_from_light(light: LightSource, plane: TrianglePlane, point: Vector) -> Color:
    """E_i(RGB, P_T) = I_i(RGB,s_i) * max(0, cos(alpha_i)) / R_i^2."""
    n = plane_normal(plane)

    # Вектор от источника к точке (для диаграммы излучения)
    s_from_light = sub(point, light.position)
    distance_sq = dot(s_from_light, s_from_light)
    if distance_sq == 0:
        raise ValueError("Точка совпадает с положением источника света")

    # Вектор от точки к источнику (для угла с нормалью поверхности)
    s_to_light = mul(s_from_light, -1.0)
    cos_alpha = clamp_nonnegative(dot(normalize(s_to_light), n))

    i_rgb = radiation_intensity(light, s_from_light)
    return tuple(channel * cos_alpha / distance_sq for channel in i_rgb)  # type: ignore[return-value]


def brdf(material: Material, n: Vector, v: Vector, s_to_light: Vector) -> Color:
    """f(RGB, P_T, v, s) = K(RGB) * [kd + ks * (h·N)^ke]."""
    h = normalize(add(normalize(v), normalize(s_to_light)))
    spec = clamp_nonnegative(dot(h, n)) ** material.ke
    factor = material.kd + material.ks * spec
    return tuple(channel * factor for channel in material.color_rgb)  # type: ignore[return-value]


def brightness_at_point(
    lights: list[LightSource],
    material: Material,
    plane: TrianglePlane,
    point: Vector,
    view_direction: Vector,
) -> tuple[list[Color], Color]:
    """Возвращает [E1, E2, ...] и итоговую яркость L(RGB, P_T, v)."""
    n = plane_normal(plane)
    v = normalize(view_direction)

    e_values: list[Color] = []
    l_rgb = [0.0, 0.0, 0.0]

    for light in lights:
        e_i = illuminance_from_light(light, plane, point)
        e_values.append(e_i)

        s_to_light = sub(light.position, point)
        f_rgb = brdf(material, n, v, s_to_light)

        for idx in range(3):
            l_rgb[idx] += e_i[idx] * f_rgb[idx] / pi

    return e_values, (l_rgb[0], l_rgb[1], l_rgb[2])


def format_color(c: Color) -> str:
    return f"({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})"


def print_table(title: str, x_values: list[float], y_values: list[float], table: list[list[Color]]) -> None:
    print(f"\n{title}")
    header = ["y\\x"] + [f"{x:.2f}" for x in x_values]
    print(" | ".join(f"{h:>20}" for h in header))
    print("-" * (24 * len(header)))
    for y, row in zip(y_values, table):
        values = [f"{y:.2f}"] + [format_color(c) for c in row]
        print(" | ".join(f"{v:>20}" for v in values))


def save_table_to_csv(
    output_path: str,
    title: str,
    x_values: list[float],
    y_values: list[float],
    table: list[list[Color]],
) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow([title])
        writer.writerow(["y\\x", *[f"{x:.2f}" for x in x_values]])

        for y, row in zip(y_values, table):
            writer.writerow([f"{y:.2f}", *[format_color(color) for color in row]])


def main() -> None:
    # Два источника света из задания ЛР2
    lights = [
        LightSource(
            name="L1",
            intensity_rgb=(1400.0, 1200.0, 1000.0),
            axis=(0.4, -0.8, -0.4),
            position=(1.0, 6.0, 5.0),
        ),
        LightSource(
            name="L2",
            intensity_rgb=(900.0, 1100.0, 1300.0),
            axis=(0.2, -1.0, -0.2),
            position=(-1.0, 4.5, 3.8),
        ),
    ]

    plane = TrianglePlane(
        p0=(0.0, 0.0, 0.0),
        p1=(4.0, 1.2, 0.0),
        p2=(1.1, 3.8, 0.0),
    )

    material = Material(
        color_rgb=(0.9, 0.75, 0.7),
        kd=0.65,
        ks=0.35,
        ke=18.0,
    )

    view_direction = (2.0, 2.0, 3.0)

    x_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    y_values = [0.0, 0.4, 0.8, 1.2, 1.6]

    e1_table: list[list[Color]] = []
    e2_table: list[list[Color]] = []
    l_table: list[list[Color]] = []

    for y in y_values:
        e1_row: list[Color] = []
        e2_row: list[Color] = []
        l_row: list[Color] = []
        for x in x_values:
            point = to_global_point(plane, x, y)
            e_values, l_rgb = brightness_at_point(lights, material, plane, point, view_direction)
            e1_row.append(e_values[0])
            e2_row.append(e_values[1])
            l_row.append(l_rgb)
        e1_table.append(e1_row)
        e2_table.append(e2_row)
        l_table.append(l_row)

    print_table("E1(RGB, P_T) для первого источника:", x_values, y_values, e1_table)
    print_table("E2(RGB, P_T) для второго источника:", x_values, y_values, e2_table)
    print_table("L(RGB, P_T, v) с учетом BRDF:", x_values, y_values, l_table)

    save_table_to_csv(
        output_path="second/e1_table.csv",
        title="E1(RGB, P_T) для первого источника",
        x_values=x_values,
        y_values=y_values,
        table=e1_table,
    )
    save_table_to_csv(
        output_path="second/e2_table.csv",
        title="E2(RGB, P_T) для второго источника",
        x_values=x_values,
        y_values=y_values,
        table=e2_table,
    )
    save_table_to_csv(
        output_path="second/l_table.csv",
        title="L(RGB, P_T, v) с учетом BRDF",
        x_values=x_values,
        y_values=y_values,
        table=l_table,
    )


if __name__ == "__main__":
    main()
