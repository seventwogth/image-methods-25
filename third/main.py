from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from lighting import compute_brightness
from scene import LightSource, Material, SceneInput, SurfacePoint, create_default_scene
from vector import Vec3


OUTPUT_DIR = Path(__file__).resolve().parent
EPS = 1e-9


@dataclass(frozen=True)
class ObserverBrightness:
    observer_id: str
    observer_position: Vec3
    raw_brightness: Vec3
    normalized_brightness: Vec3


@dataclass(frozen=True)
class ControlScenario:
    scenario_id: str
    description: str
    surface_point: SurfacePoint
    material: Material
    light: LightSource
    observer_position: Vec3
    material_type: str
    expected_behavior: str
    expect_nonzero: bool


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    description: str
    light_position: Vec3
    observer_position: Vec3
    surface_normal: Vec3
    material_type: str
    expected_behavior: str
    raw_brightness: Vec3
    passed: bool


def format_number(value: float) -> str:
    return f"{value:.6f}"


def format_vec3(vector: Vec3) -> str:
    return f"({format_number(vector.x)}, {format_number(vector.y)}, {format_number(vector.z)})"


def vec3_cells(vector: Vec3) -> tuple[str, str, str]:
    return (
        format_number(vector.x),
        format_number(vector.y),
        format_number(vector.z),
    )


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


def normalize_brightness_values(values: list[Vec3]) -> list[Vec3]:
    max_component = 0.0
    for value in values:
        max_component = max(max_component, value.x, value.y, value.z)

    if max_component == 0.0:
        return [Vec3(0.0, 0.0, 0.0) for _ in values]

    return [value / max_component for value in values]


def compute_results(scene: SceneInput) -> list[ObserverBrightness]:
    raw_values = [
        compute_brightness(
            surface_point=scene.surface_point,
            material=scene.material,
            light=scene.light,
            observer_position=observer_position,
        )
        for observer_position in scene.observer_positions
    ]
    normalized_values = normalize_brightness_values(raw_values)

    results: list[ObserverBrightness] = []
    for index, (observer_position, raw_brightness, normalized_brightness) in enumerate(
        zip(scene.observer_positions, raw_values, normalized_values),
        start=1,
    ):
        results.append(
            ObserverBrightness(
                observer_id=f"P{index:02d}",
                observer_position=observer_position,
                raw_brightness=raw_brightness,
                normalized_brightness=normalized_brightness,
            )
        )
    return results


def build_control_scenarios() -> list[ControlScenario]:
    origin_point = SurfacePoint(position=Vec3(0.0, 0.0, 0.0), normal=Vec3(0.0, 0.0, 1.0))
    mirror_material = Material(
        diffuse_color=Vec3(0.0, 0.0, 0.0),
        specular_color=Vec3(1.0, 1.0, 1.0),
        kd=0.0,
        ks=1.0,
        shininess=16.0,
    )
    diffuse_material = Material(
        diffuse_color=Vec3(0.7, 0.6, 0.5),
        specular_color=Vec3(0.0, 0.0, 0.0),
        kd=1.0,
        ks=0.0,
        shininess=8.0,
    )

    return [
        ControlScenario(
            scenario_id="mirror_valid_reflection",
            description="Mirror surface, observer is aligned with the reflected ray.",
            surface_point=origin_point,
            material=mirror_material,
            light=LightSource(position=Vec3(1.0, 0.0, 1.0), intensity=Vec3(1.0, 1.0, 1.0)),
            observer_position=Vec3(-2.0, 0.0, 2.0),
            material_type="mirror",
            expected_behavior="Non-zero brightness is expected.",
            expect_nonzero=True,
        ),
        ControlScenario(
            scenario_id="observer_below_mirror",
            description="Observer is below the mirror surface.",
            surface_point=origin_point,
            material=mirror_material,
            light=LightSource(position=Vec3(1.0, 0.0, 1.0), intensity=Vec3(1.0, 1.0, 1.0)),
            observer_position=Vec3(-2.0, 0.0, -2.0),
            material_type="mirror",
            expected_behavior="Brightness must be zero for a back-side observer.",
            expect_nonzero=False,
        ),
        ControlScenario(
            scenario_id="light_below_surface",
            description="Light source is below the one-sided surface.",
            surface_point=origin_point,
            material=mirror_material,
            light=LightSource(position=Vec3(0.0, 0.0, -2.0), intensity=Vec3(1.0, 1.0, 1.0)),
            observer_position=Vec3(0.0, 0.0, 3.0),
            material_type="mirror",
            expected_behavior="Brightness must be zero because light comes from the back side.",
            expect_nonzero=False,
        ),
        ControlScenario(
            scenario_id="diffuse_front_lit",
            description="Diffuse surface lit from the front side.",
            surface_point=origin_point,
            material=diffuse_material,
            light=LightSource(position=Vec3(0.0, 0.0, 2.0), intensity=Vec3(1.0, 1.0, 1.0)),
            observer_position=Vec3(0.0, 0.0, 3.0),
            material_type="diffuse",
            expected_behavior="Non-zero diffuse brightness is expected.",
            expect_nonzero=True,
        ),
        ControlScenario(
            scenario_id="diffuse_back_lit",
            description="Diffuse surface lit from the back side.",
            surface_point=origin_point,
            material=diffuse_material,
            light=LightSource(position=Vec3(0.0, 0.0, -2.0), intensity=Vec3(1.0, 1.0, 1.0)),
            observer_position=Vec3(0.0, 0.0, 3.0),
            material_type="diffuse",
            expected_behavior="Brightness must be zero because the surface is back-lit.",
            expect_nonzero=False,
        ),
        ControlScenario(
            scenario_id="observer_back_side",
            description="Observer is behind a front-lit one-sided surface.",
            surface_point=origin_point,
            material=diffuse_material,
            light=LightSource(position=Vec3(0.0, 0.0, 2.0), intensity=Vec3(1.0, 1.0, 1.0)),
            observer_position=Vec3(0.0, 0.0, -3.0),
            material_type="diffuse",
            expected_behavior="Brightness must be zero for a back-side observer.",
            expect_nonzero=False,
        ),
    ]


def compute_scenario_result(scenario: ControlScenario) -> ScenarioResult:
    raw_brightness = compute_brightness(
        surface_point=scenario.surface_point,
        material=scenario.material,
        light=scenario.light,
        observer_position=scenario.observer_position,
    )

    if scenario.expect_nonzero:
        passed = (
            raw_brightness.x > EPS
            or raw_brightness.y > EPS
            or raw_brightness.z > EPS
        )
    else:
        passed = (
            abs(raw_brightness.x) < EPS
            and abs(raw_brightness.y) < EPS
            and abs(raw_brightness.z) < EPS
        )

    return ScenarioResult(
        scenario_id=scenario.scenario_id,
        description=scenario.description,
        light_position=scenario.light.position,
        observer_position=scenario.observer_position,
        surface_normal=scenario.surface_point.normal,
        material_type=scenario.material_type,
        expected_behavior=scenario.expected_behavior,
        raw_brightness=raw_brightness,
        passed=passed,
    )


def light_input_rows(light: LightSource) -> list[list[str]]:
    px, py, pz = vec3_cells(light.position)
    ir, ig, ib = vec3_cells(light.intensity)
    return [[px, py, pz, ir, ig, ib]]


def surface_input_rows(surface_point: SurfacePoint, material: Material) -> list[list[str]]:
    point_x, point_y, point_z = vec3_cells(surface_point.position)
    normal_x, normal_y, normal_z = vec3_cells(surface_point.normal)
    diffuse_r, diffuse_g, diffuse_b = vec3_cells(material.diffuse_color)
    specular_r, specular_g, specular_b = vec3_cells(material.specular_color)
    return [
        [
            point_x,
            point_y,
            point_z,
            normal_x,
            normal_y,
            normal_z,
            diffuse_r,
            diffuse_g,
            diffuse_b,
            specular_r,
            specular_g,
            specular_b,
            format_number(material.kd),
            format_number(material.ks),
            format_number(material.shininess),
        ]
    ]


def observer_input_rows(observer_positions: list[Vec3]) -> list[list[str]]:
    rows: list[list[str]] = []
    for index, observer_position in enumerate(observer_positions, start=1):
        observer_x, observer_y, observer_z = vec3_cells(observer_position)
        rows.append([f"P{index:02d}", observer_x, observer_y, observer_z])
    return rows


def brightness_rows(results: list[ObserverBrightness]) -> list[list[str]]:
    rows: list[list[str]] = []
    for result in results:
        observer_x, observer_y, observer_z = vec3_cells(result.observer_position)
        raw_r, raw_g, raw_b = vec3_cells(result.raw_brightness)
        norm_r, norm_g, norm_b = vec3_cells(result.normalized_brightness)
        rows.append(
            [
                result.observer_id,
                observer_x,
                observer_y,
                observer_z,
                raw_r,
                raw_g,
                raw_b,
                norm_r,
                norm_g,
                norm_b,
            ]
        )
    return rows


def control_scenario_rows(results: list[ScenarioResult]) -> list[list[str]]:
    rows: list[list[str]] = []
    for result in results:
        raw_r, raw_g, raw_b = vec3_cells(result.raw_brightness)
        rows.append(
            [
                result.scenario_id,
                result.description,
                format_vec3(result.light_position),
                format_vec3(result.observer_position),
                format_vec3(result.surface_normal),
                result.material_type,
                result.expected_behavior,
                raw_r,
                raw_g,
                raw_b,
                "yes" if result.passed else "no",
            ]
        )
    return rows


def main() -> None:
    scene = create_default_scene()
    results = compute_results(scene)
    control_results = [compute_scenario_result(scenario) for scenario in build_control_scenarios()]

    light_headers = [
        "position_x",
        "position_y",
        "position_z",
        "intensity_R",
        "intensity_G",
        "intensity_B",
    ]
    surface_headers = [
        "point_x",
        "point_y",
        "point_z",
        "normal_x",
        "normal_y",
        "normal_z",
        "diffuse_R",
        "diffuse_G",
        "diffuse_B",
        "specular_R",
        "specular_G",
        "specular_B",
        "kd",
        "ks",
        "shininess",
    ]
    observer_headers = [
        "observer_id",
        "observer_x",
        "observer_y",
        "observer_z",
    ]
    brightness_headers = [
        "observer_id",
        "observer_x",
        "observer_y",
        "observer_z",
        "raw_L_R",
        "raw_L_G",
        "raw_L_B",
        "normalized_L_R",
        "normalized_L_G",
        "normalized_L_B",
    ]
    control_headers = [
        "scenario_id",
        "description",
        "light_position",
        "observer_position",
        "surface_normal",
        "material_type",
        "expected_behavior",
        "raw_L_R",
        "raw_L_G",
        "raw_L_B",
        "passed",
    ]

    light_rows = light_input_rows(scene.light)
    surface_rows = surface_input_rows(scene.surface_point, scene.material)
    observer_rows = observer_input_rows(scene.observer_positions)
    result_rows = brightness_rows(results)
    scenario_rows = control_scenario_rows(control_results)

    print_rows_table("Входные данные источника света:", light_headers, light_rows)
    print_rows_table("Входные данные поверхности:", surface_headers, surface_rows)
    print_rows_table("Положения наблюдателей:", observer_headers, observer_rows)
    print_rows_table(
        "Яркость для заданных направлений наблюдения:",
        brightness_headers,
        result_rows,
    )
    print_rows_table(
        "Контрольные и пограничные сценарии:",
        control_headers,
        scenario_rows,
    )

    save_rows_to_csv(OUTPUT_DIR / "input_light_source.csv", light_headers, light_rows)
    save_rows_to_csv(OUTPUT_DIR / "input_surface.csv", surface_headers, surface_rows)
    save_rows_to_csv(OUTPUT_DIR / "input_observers.csv", observer_headers, observer_rows)
    save_rows_to_csv(OUTPUT_DIR / "brightness_results.csv", brightness_headers, result_rows)
    save_rows_to_csv(OUTPUT_DIR / "control_scenarios.csv", control_headers, scenario_rows)

    print(
        "\nCSV-файлы сохранены:\n"
        f"- {OUTPUT_DIR / 'input_light_source.csv'}\n"
        f"- {OUTPUT_DIR / 'input_surface.csv'}\n"
        f"- {OUTPUT_DIR / 'input_observers.csv'}\n"
        f"- {OUTPUT_DIR / 'brightness_results.csv'}\n"
        f"- {OUTPUT_DIR / 'control_scenarios.csv'}"
    )


if __name__ == "__main__":
    main()
