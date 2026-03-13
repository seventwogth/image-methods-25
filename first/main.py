from __future__ import annotations

import csv
from dataclasses import dataclass
from math import sqrt
from typing import Iterable


Vector = tuple[float, float, float]
Color = tuple[float, float, float]


@dataclass(frozen=True)
class LightSource:
    intensity_rgb: Color
    axis: Vector
    position: Vector


@dataclass(frozen=True)
class TrianglePlane:
    p0: Vector
    p1: Vector
    p2: Vector


def dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def sub(a: Vector, b: Vector) -> Vector:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def add(a: Vector, b: Vector) -> Vector:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def mul(a: Vector, k: float) -> Vector:
    return a[0] * k, a[1] * k, a[2] * k


def norm(a: Vector) -> float:
    return sqrt(dot(a, a))


def normalize(a: Vector) -> Vector:
    length = norm(a)
    if length == 0:
        raise ValueError("Нулевой вектор нельзя нормализовать")
    return a[0] / length, a[1] / length, a[2] / length


def cross(a: Vector, b: Vector) -> Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def clamp_nonnegative(value: float) -> float:
    return max(value, 0.0)


def to_global_point(plane: TrianglePlane, x: float, y: float) -> Vector:
    e1 = sub(plane.p1, plane.p0)
    e2 = sub(plane.p2, plane.p0)
    u1 = normalize(e1)
    u2 = normalize(e2)
    return add(plane.p0, add(mul(u1, x), mul(u2, y)))


def plane_normal(plane: TrianglePlane) -> Vector:
    e1 = sub(plane.p1, plane.p0)
    e2 = sub(plane.p2, plane.p0)
    return normalize(cross(e2, e1))


def radiation_intensity(light: LightSource, s: Vector) -> Color:
    s_unit = normalize(s)
    axis_unit = normalize(light.axis)
    cos_theta = clamp_nonnegative(dot(s_unit, axis_unit))
    return tuple(channel * cos_theta for channel in light.intensity_rgb)  # type: ignore[return-value]


def illuminance_at_point(light: LightSource, plane: TrianglePlane, point: Vector) -> Color:
    normal = plane_normal(plane)
    s = sub(point, light.position)
    distance_sq = dot(s, s)
    if distance_sq == 0:
        raise ValueError("Точка совпадает с источником света, деление на ноль")

    s_unit = normalize(s)
    cos_alpha = clamp_nonnegative(dot(s_unit, normal))
    i_rgb = radiation_intensity(light, s)
    return tuple(channel * cos_alpha / distance_sq for channel in i_rgb)  # type: ignore[return-value]


def build_illuminance_table(
    light: LightSource,
    plane: TrianglePlane,
    x_values: Iterable[float],
    y_values: Iterable[float],
) -> list[list[Color]]:
    table: list[list[Color]] = []
    x_list = list(x_values)
    for y in y_values:
        row: list[Color] = []
        for x in x_list:
            point = to_global_point(plane, x, y)
            row.append(illuminance_at_point(light, plane, point))
        table.append(row)
    return table


def format_color(color: Color) -> str:
    return f"({color[0]:.4f}, {color[1]:.4f}, {color[2]:.4f})"


def print_table(x_values: list[float], y_values: list[float], table: list[list[Color]]) -> None:
    header = ["y\\x"] + [f"{x:.2f}" for x in x_values]
    print(" | ".join(f"{item:>20}" for item in header))
    print("-" * (24 * len(header)))

    for y, row in zip(y_values, table):
        values = [f"{y:.2f}"] + [format_color(c) for c in row]
        print(" | ".join(f"{item:>20}" for item in values))


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
    light = LightSource(
        intensity_rgb=(1200.0, 900.0, 700.0),
        axis=(0.3, -0.9, -0.3),
        position=(2.0, 5.0, 4.0),
    )

    plane = TrianglePlane(
        p0=(0.0, 0.0, 0.0),
        p1=(4.0, 1.0, 0.0),
        p2=(1.0, 3.5, 0.0),
    )

    x_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    y_values = [0.0, 0.4, 0.8, 1.2, 1.6]

    table = build_illuminance_table(light, plane, x_values, y_values)

    print("Вычисленные значения цветной освещенности E(RGB, P_T):")
    print_table(x_values, y_values, table)
    save_table_to_csv(
        output_path="first/illuminance_table.csv",
        title="E(RGB, P_T) для точек, заданных локальными координатами",
        x_values=x_values,
        y_values=y_values,
        table=table,
    )


if __name__ == "__main__":
    main()
