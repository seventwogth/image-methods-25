from __future__ import annotations

import csv
from dataclasses import dataclass
from math import pi, sqrt
from pathlib import Path

Vector = tuple[float, float, float]
Color = tuple[float, float, float]

OUTPUT_DIR = Path(__file__).resolve().parent


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


@dataclass(frozen=True)
class PointSample:
    point_id: str
    local_x: float
    local_y: float
    global_point: Vector
    illuminance_by_light: tuple[Color, ...]
    total_illuminance: Color
    brightness: Color


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
    return abs(x);


def add_colors(a: Color, b: Color) -> Color:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def sum_colors(colors: tuple[Color, ...]) -> Color:
    total = (0.0, 0.0, 0.0)
    for color in colors:
        total = add_colors(total, color)
    return total


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
    """E_i(RGB, P_T) = I_i(RGB, s_i) * max(0, cos(alpha_i)) / R_i^2."""
    n = plane_normal(plane)

    # Вектор от источника к точке нужен для диаграммы излучения.
    s_from_light = sub(point, light.position)
    distance_sq = dot(s_from_light, s_from_light)
    if distance_sq == 0:
        raise ValueError("Точка совпадает с положением источника света")

    # Вектор от точки к источнику нужен для угла между нормалью и светом.
    s_to_light = mul(s_from_light, -1.0)
    cos_alpha = clamp_nonnegative(dot(normalize(s_to_light), n))

    i_rgb = radiation_intensity(light, s_from_light)
    return tuple(channel * cos_alpha / distance_sq for channel in i_rgb)  # type: ignore[return-value]


def brdf(material: Material, n: Vector, v: Vector, s_to_light: Vector) -> Color:
    """f(RGB, P_T, v, l) = K(RGB) * [kd + ks * (h·N)^ke]."""
    half_vector_base = add(normalize(v), normalize(s_to_light))
    spec = 0.0
    if norm(half_vector_base) != 0:
        h = normalize(half_vector_base)
        spec = clamp_nonnegative(dot(h, n)) ** material.ke

    factor = material.kd + material.ks * spec
    return tuple(channel * factor for channel in material.color_rgb)  # type: ignore[return-value]


def brightness_at_point(
    lights: list[LightSource],
    material: Material,
    plane: TrianglePlane,
    point: Vector,
    observer_position: Vector,
) -> tuple[tuple[Color, ...], Color]:
    """Возвращает освещённости от всех источников и итоговую яркость L(RGB, P_T, v)."""
    n = plane_normal(plane)
    v = normalize(sub(observer_position, point))

    e_values: list[Color] = []
    l_rgb = [0.0, 0.0, 0.0]

    for light in lights:
        e_i = illuminance_from_light(light, plane, point)
        e_values.append(e_i)

        s_to_light = sub(light.position, point)
        f_rgb = brdf(material, n, v, s_to_light)

        for idx in range(3):
            l_rgb[idx] += e_i[idx] * f_rgb[idx] / pi

    return tuple(e_values), (l_rgb[0], l_rgb[1], l_rgb[2])


def format_color(c: Color) -> str:
    return f"({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})"


def format_number(value: float) -> str:
    return f"{value:.6f}"


def build_point_samples(
    lights: list[LightSource],
    material: Material,
    plane: TrianglePlane,
    observer_position: Vector,
    x_values: list[float],
    y_values: list[float],
) -> list[PointSample]:
    samples: list[PointSample] = []
    point_index = 1

    for y in y_values:
        for x in x_values:
            point = to_global_point(plane, x, y)
            illuminance_by_light, brightness = brightness_at_point(lights, material, plane, point, observer_position)
            samples.append(
                PointSample(
                    point_id=f"P{point_index:02d}",
                    local_x=x,
                    local_y=y,
                    global_point=point,
                    illuminance_by_light=illuminance_by_light,
                    total_illuminance=sum_colors(illuminance_by_light),
                    brightness=brightness,
                )
            )
            point_index += 1

    return samples


def print_rows_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    print(f"\n{title}")
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def format_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


def save_rows_to_csv(output_path: Path, headers: list[str], rows: list[list[str]]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(headers)
        writer.writerows(rows)


def global_coordinate_rows(samples: list[PointSample]) -> list[list[str]]:
    rows: list[list[str]] = []
    for sample in samples:
        rows.append(
            [
                sample.point_id,
                format_number(sample.local_x),
                format_number(sample.local_y),
                format_number(sample.global_point[0]),
                format_number(sample.global_point[1]),
                format_number(sample.global_point[2]),
            ]
        )
    return rows


def illuminance_rows(samples: list[PointSample], lights: list[LightSource], coordinate_mode: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for sample in samples:
        if coordinate_mode == "local":
            row = [sample.point_id, format_number(sample.local_x), format_number(sample.local_y)]
        elif coordinate_mode == "global":
            row = [
                sample.point_id,
                format_number(sample.global_point[0]),
                format_number(sample.global_point[1]),
                format_number(sample.global_point[2]),
            ]
        else:
            raise ValueError(f"Неизвестный режим координат: {coordinate_mode}")

        for light, illuminance in zip(lights, sample.illuminance_by_light):
            row.extend(
                [
                    format_number(illuminance[0]),
                    format_number(illuminance[1]),
                    format_number(illuminance[2]),
                ]
            )

        row.extend(
            [
                format_number(sample.total_illuminance[0]),
                format_number(sample.total_illuminance[1]),
                format_number(sample.total_illuminance[2]),
            ]
        )
        rows.append(row)
    return rows


def brightness_rows(samples: list[PointSample]) -> list[list[str]]:
    rows: list[list[str]] = []
    for sample in samples:
        rows.append(
            [
                sample.point_id,
                format_number(sample.local_x),
                format_number(sample.local_y),
                format_number(sample.global_point[0]),
                format_number(sample.global_point[1]),
                format_number(sample.global_point[2]),
                format_number(sample.brightness[0]),
                format_number(sample.brightness[1]),
                format_number(sample.brightness[2]),
            ]
        )
    return rows


def compact_local_illuminance_rows(samples: list[PointSample], lights: list[LightSource]) -> list[list[str]]:
    rows: list[list[str]] = []
    for sample in samples:
        row = [sample.point_id, format_number(sample.local_x), format_number(sample.local_y)]
        for light, illuminance in zip(lights, sample.illuminance_by_light):
            row.append(f"{light.name}: {format_color(illuminance)}")
        row.append(f"E_total: {format_color(sample.total_illuminance)}")
        rows.append(row)
    return rows


def compact_global_illuminance_rows(samples: list[PointSample], lights: list[LightSource]) -> list[list[str]]:
    rows: list[list[str]] = []
    for sample in samples:
        row = [
            sample.point_id,
            format_number(sample.global_point[0]),
            format_number(sample.global_point[1]),
            format_number(sample.global_point[2]),
        ]
        for light, illuminance in zip(lights, sample.illuminance_by_light):
            row.append(f"{light.name}: {format_color(illuminance)}")
        row.append(f"E_total: {format_color(sample.total_illuminance)}")
        rows.append(row)
    return rows


def compact_brightness_rows(samples: list[PointSample]) -> list[list[str]]:
    rows: list[list[str]] = []
    for sample in samples:
        rows.append(
            [
                sample.point_id,
                format_number(sample.local_x),
                format_number(sample.local_y),
                format_number(sample.global_point[0]),
                format_number(sample.global_point[1]),
                format_number(sample.global_point[2]),
                format_color(sample.brightness),
            ]
        )
    return rows


def illuminance_headers(lights: list[LightSource], coordinate_mode: str) -> list[str]:
    if coordinate_mode == "local":
        headers = ["point_id", "local_x", "local_y"]
    elif coordinate_mode == "global":
        headers = ["point_id", "global_x", "global_y", "global_z"]
    else:
        raise ValueError(f"Неизвестный режим координат: {coordinate_mode}")

    for light in lights:
        headers.extend(
            [
                f"E_{light.name}_R",
                f"E_{light.name}_G",
                f"E_{light.name}_B",
            ]
        )

    headers.extend(["E_total_R", "E_total_G", "E_total_B"])
    return headers


def main() -> None:
    lights = [
        LightSource(
            name="L1",
            intensity_rgb=(1000.0, 1000.0, 1000.0),
            axis=(0.0, 0.0, -1.0),
            position=(0.0, 0.0, 4.0),
        ),
        LightSource(
            name="L2",
            intensity_rgb=(900.0, 1100.0, 1300.0),
            axis=(0.0, 0.0, 1.0),
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

    observer_position = (2.0, 2.0, 3.0)

    x_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    y_values = [0.0, 0.4, 0.8, 1.2, 1.6]

    samples = build_point_samples(lights, material, plane, observer_position, x_values, y_values)

    print_rows_table(
        "Точки в глобальных координатах:",
        ["point_id", "local_x", "local_y", "global_x", "global_y", "global_z"],
        global_coordinate_rows(samples),
    )
    print_rows_table(
        "Освещенность для точек, заданных локальными координатами:",
        ["point_id", "local_x", "local_y", *[f"{light.name}(RGB)" for light in lights], "E_total(RGB)"],
        compact_local_illuminance_rows(samples, lights),
    )
    print_rows_table(
        "Освещенность для тех же точек, заданных глобальными координатами:",
        ["point_id", "global_x", "global_y", "global_z", *[f"{light.name}(RGB)" for light in lights], "E_total(RGB)"],
        compact_global_illuminance_rows(samples, lights),
    )
    print_rows_table(
        "Яркость точек с учетом BRDF:",
        ["point_id", "local_x", "local_y", "global_x", "global_y", "global_z", "L(RGB)"],
        compact_brightness_rows(samples),
    )

    save_rows_to_csv(
        OUTPUT_DIR / "global_point_coordinates.csv",
        ["point_id", "local_x", "local_y", "global_x", "global_y", "global_z"],
        global_coordinate_rows(samples),
    )
    save_rows_to_csv(
        OUTPUT_DIR / "local_illuminance_results.csv",
        illuminance_headers(lights, "local"),
        illuminance_rows(samples, lights, "local"),
    )
    save_rows_to_csv(
        OUTPUT_DIR / "global_illuminance_results.csv",
        illuminance_headers(lights, "global"),
        illuminance_rows(samples, lights, "global"),
    )
    save_rows_to_csv(
        OUTPUT_DIR / "brightness_results.csv",
        [
            "point_id",
            "local_x",
            "local_y",
            "global_x",
            "global_y",
            "global_z",
            "L_R",
            "L_G",
            "L_B",
        ],
        brightness_rows(samples),
    )

    print(
        "\nCSV-файлы сохранены:\n"
        f"- {OUTPUT_DIR / 'global_point_coordinates.csv'}\n"
        f"- {OUTPUT_DIR / 'local_illuminance_results.csv'}\n"
        f"- {OUTPUT_DIR / 'global_illuminance_results.csv'}\n"
        f"- {OUTPUT_DIR / 'brightness_results.csv'}"
    )


if __name__ == "__main__":
    main()
