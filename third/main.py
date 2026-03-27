from __future__ import annotations

import csv
from dataclasses import dataclass
from math import sqrt


Vector = tuple[float, float, float]
Color = tuple[float, float, float]


@dataclass(frozen=True)
class LightSource:
    """Точечный источник света с цветной интенсивностью и осью излучения."""

    intensity_rgb: Color
    axis: Vector
    position: Vector


@dataclass(frozen=True)
class MirrorSurface:
    """Зеркальная поверхность, заданная точкой, нормалью и коэффициентом отражения."""

    point: Vector
    normal: Vector
    color_rgb: Color
    kd: float


@dataclass(frozen=True)
class Observer:
    """Наблюдатель, для которого вычисляется яркость в заданном направлении."""

    name: str
    position: Vector


@dataclass(frozen=True)
class ObservationResult:
    """Результат расчета для одного положения наблюдателя."""

    observer_name: str
    observer_position: Vector
    virtual_observer_position: Vector
    reflected_direction: Vector
    observer_direction: Vector
    brightness_rgb: Color


def dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def add(a: Vector, b: Vector) -> Vector:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def sub(a: Vector, b: Vector) -> Vector:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def mul(v: Vector, k: float) -> Vector:
    return v[0] * k, v[1] * k, v[2] * k


def norm(v: Vector) -> float:
    return sqrt(dot(v, v))


def normalize(v: Vector) -> Vector:
    length = norm(v)
    if length == 0:
        raise ValueError("Нулевой вектор нельзя нормализовать")
    return v[0] / length, v[1] / length, v[2] / length


def clamp_nonnegative(x: float) -> float:
    return max(0.0, x)


def reflect(vector: Vector, normal: Vector) -> Vector:
    """Зеркальное отражение вектора относительно нормали: O' = O - 2(O·N)N."""

    n = normalize(normal)
    projection = 2.0 * dot(vector, n)
    return sub(vector, mul(n, projection))


def reflect_point_across_plane(point: Vector, plane_point: Vector, plane_normal: Vector) -> Vector:
    """Положение виртуального наблюдателя как отражение точки относительно плоскости."""

    n = normalize(plane_normal)
    offset = sub(point, plane_point)
    distance = dot(offset, n)
    return sub(point, mul(n, 2.0 * distance))


def radiation_intensity(light: LightSource, direction_from_light: Vector) -> Color:
    """Цветная сила излучения I(RGB, s) = I0(RGB) * max(0, cos(theta))."""

    axis_unit = normalize(light.axis)
    direction_unit = normalize(direction_from_light)
    cos_theta = clamp_nonnegative(dot(direction_unit, axis_unit))
    return tuple(channel * cos_theta for channel in light.intensity_rgb)  # type: ignore[return-value]


def illuminance_at_point(light: LightSource, surface: MirrorSurface) -> Color:
    """Цветная освещенность точки зеркальной поверхности от точечного источника света."""

    surface_normal = normalize(surface.normal)
    direction_from_light = sub(surface.point, light.position)
    distance_sq = dot(direction_from_light, direction_from_light)
    if distance_sq == 0:
        raise ValueError("Точка наблюдения на поверхности совпадает с источником света")

    cos_alpha = clamp_nonnegative(dot(normalize(direction_from_light), surface_normal))
    intensity_rgb = radiation_intensity(light, direction_from_light)
    return tuple(channel * cos_alpha / distance_sq for channel in intensity_rgb)  # type: ignore[return-value]


def directions_match(a: Vector, b: Vector, tolerance: float = 1e-9) -> bool:
    """Проверка, что два единичных направления совпадают с заданной точностью."""

    a_unit = normalize(a)
    b_unit = normalize(b)
    return dot(a_unit, b_unit) >= 1.0 - tolerance


def reflected_light_direction(light: LightSource, surface: MirrorSurface) -> Vector:
    """Направление отраженного луча от точки зеркальной поверхности."""

    incoming = normalize(sub(surface.point, light.position))
    return normalize(reflect(incoming, surface.normal))


def observer_sees_reflection(
    light: LightSource,
    surface: MirrorSurface,
    observer_position: Vector,
    virtual_observer_position: Vector,
) -> bool:
    """Проверка закона зеркального отражения через отраженный луч и виртуального наблюдателя."""

    incoming_direction = normalize(sub(surface.point, light.position))
    reflected_direction = reflected_light_direction(light, surface)
    observer_direction = normalize(sub(observer_position, surface.point))
    virtual_observer_direction = normalize(sub(virtual_observer_position, surface.point))
    return directions_match(reflected_direction, observer_direction) and directions_match(
        incoming_direction,
        virtual_observer_direction,
    )


def brightness_for_observer(light: LightSource, surface: MirrorSurface, observer: Observer) -> ObservationResult:
    """Расчет яркости L(RGB, P_T, v) для заданного положения наблюдателя.

    В текущей реализации яркость вычисляется как E(RGB, P_T) * K(RGB) * kd
    при выполнении условия зеркального отражения для выбранного направления
    наблюдения. Если наблюдатель не лежит на отраженном луче, яркость
    считается равной нулю.
    """

    virtual_observer = reflect_point_across_plane(
        point=observer.position,
        plane_point=surface.point,
        plane_normal=surface.normal,
    )

    reflected_direction = reflected_light_direction(light, surface)
    observer_direction = normalize(sub(observer.position, surface.point))
    illuminance_rgb = illuminance_at_point(light, surface)
    if observer_sees_reflection(light, surface, observer.position, virtual_observer):
        brightness_rgb = tuple(
            illuminance_rgb[idx] * surface.color_rgb[idx] * surface.kd for idx in range(3)
        )
    else:
        brightness_rgb = (0.0, 0.0, 0.0)

    return ObservationResult(
        observer_name=observer.name,
        observer_position=observer.position,
        virtual_observer_position=virtual_observer,
        reflected_direction=reflected_direction,
        observer_direction=observer_direction,
        brightness_rgb=brightness_rgb,
    )


def format_vector(v: Vector) -> str:
    return f"({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})"


def format_color(color: Color) -> str:
    return f"({color[0]:.6f}, {color[1]:.6f}, {color[2]:.6f})"


def save_light_input_to_csv(output_path: str, light: LightSource) -> None:
    """Сохранение входных данных источника света в отдельную таблицу."""

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(["parameter", "x_or_r", "y_or_g", "z_or_b"])
        writer.writerow(["I01(RGB)", f"{light.intensity_rgb[0]:.6f}", f"{light.intensity_rgb[1]:.6f}", f"{light.intensity_rgb[2]:.6f}"])
        writer.writerow(["O1", f"{light.axis[0]:.6f}", f"{light.axis[1]:.6f}", f"{light.axis[2]:.6f}"])
        writer.writerow(["P_L", f"{light.position[0]:.6f}", f"{light.position[1]:.6f}", f"{light.position[2]:.6f}"])


def save_observers_to_csv(output_path: str, observers: list[Observer]) -> None:
    """Сохранение положений наблюдателей из блока входных данных."""

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(["observer", "x", "y", "z"])
        for observer in observers:
            writer.writerow(
                [
                    observer.name,
                    f"{observer.position[0]:.6f}",
                    f"{observer.position[1]:.6f}",
                    f"{observer.position[2]:.6f}",
                ]
            )


def save_surface_input_to_csv(output_path: str, surface: MirrorSurface) -> None:
    """Сохранение входных параметров отражающей поверхности, указанных в docs."""

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(["parameter", "x_or_r", "y_or_g", "z_or_b"])
        writer.writerow(["K(RGB)", f"{surface.color_rgb[0]:.6f}", f"{surface.color_rgb[1]:.6f}", f"{surface.color_rgb[2]:.6f}"])
        writer.writerow(["kd", f"{surface.kd:.6f}", "", ""])


def save_brightness_results_to_csv(output_path: str, results: list[ObservationResult]) -> None:
    """Сохранение требуемых итоговых значений яркости для направлений наблюдения."""

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(
            [
                "observer",
                "observer_x",
                "observer_y",
                "observer_z",
                "virtual_observer_x",
                "virtual_observer_y",
                "virtual_observer_z",
                "view_dir_x",
                "view_dir_y",
                "view_dir_z",
                "reflected_dir_x",
                "reflected_dir_y",
                "reflected_dir_z",
                "L_r",
                "L_g",
                "L_b",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.observer_name,
                    f"{result.observer_position[0]:.6f}",
                    f"{result.observer_position[1]:.6f}",
                    f"{result.observer_position[2]:.6f}",
                    f"{result.virtual_observer_position[0]:.6f}",
                    f"{result.virtual_observer_position[1]:.6f}",
                    f"{result.virtual_observer_position[2]:.6f}",
                    f"{result.observer_direction[0]:.6f}",
                    f"{result.observer_direction[1]:.6f}",
                    f"{result.observer_direction[2]:.6f}",
                    f"{result.reflected_direction[0]:.6f}",
                    f"{result.reflected_direction[1]:.6f}",
                    f"{result.reflected_direction[2]:.6f}",
                    f"{result.brightness_rgb[0]:.6f}",
                    f"{result.brightness_rgb[1]:.6f}",
                    f"{result.brightness_rgb[2]:.6f}",
                ]
            )


def print_input_data(light: LightSource, surface: MirrorSurface, observers: list[Observer]) -> None:
    """Печать входных данных в форме, удобной для отчета."""

    print("Входные данные:")
    print(f"1. I01(RGB) = {format_color(light.intensity_rgb)}")
    print(f"2. O1 = {format_vector(light.axis)}")
    print(f"3. P_L = {format_vector(light.position)}")
    for index, observer in enumerate(observers, start=1):
        print(f"4.{index} {observer.name} = {format_vector(observer.position)}")
    print(f"5. K(RGB) = {format_color(surface.color_rgb)}")
    print(f"6. kd = {surface.kd:.4f}")
    print()


def print_results(results: list[ObservationResult]) -> None:
    """Печать яркостей для заданных направлений наблюдения."""

    print("Результаты вычисления яркости L(RGB, P_T, v):")
    print(f"{'Наблюдатель':>12} | {'P0':>28} | {'P0`':>28} | {'L(RGB)':>28}")
    print("-" * 110)
    for result in results:
        print(
            f"{result.observer_name:>12} | "
            f"{format_vector(result.observer_position):>28} | "
            f"{format_vector(result.virtual_observer_position):>28} | "
            f"{format_color(result.brightness_rgb):>28}"
        )


def main() -> None:
    """Пример расчета яркости в заданных направлениях для ЛР_3-1."""

    # Входные данные из постановки ЛР_3-1:
    # 1. I01(RGB)
    # 2. O1
    # 3. P_L
    # 4. P01..P05
    # 5. K(RGB)
    # 6. kd
    light = LightSource(
        intensity_rgb=(1400.0, 1100.0, 900.0),
        axis=(-2.0, 1.0, -5.0),
        position=(2.0, -1.0, 5.0),
    )

    # Поверхность задана одной отражающей точкой P_T и нормалью к зеркалу.
    # Нормаль ориентирована в сторону источника, чтобы формула освещенности из docs
    # давала положительное значение cos(alpha).
    surface = MirrorSurface(
        point=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, -1.0),
        color_rgb=(0.92, 0.82, 0.74),
        kd=0.85,
    )

    observers = [
        Observer(name="P01", position=(-2.0, 1.0, 5.0)),
        Observer(name="P02", position=(-1.5, 1.0, 5.0)),
        Observer(name="P03", position=(-2.0, 2.0, 5.0)),
        Observer(name="P04", position=(0.0, 0.0, 5.0)),
        Observer(name="P05", position=(-4.0, 2.0, 10.0)),
    ]

    results = [brightness_for_observer(light, surface, observer) for observer in observers]

    print_input_data(light, surface, observers)
    print_results(results)
    save_light_input_to_csv("third/input_light_source.csv", light)
    save_observers_to_csv("third/input_observers.csv", observers)
    save_surface_input_to_csv("third/input_surface.csv", surface)
    save_brightness_results_to_csv("third/brightness_results.csv", results)


if __name__ == "__main__":
    main()
