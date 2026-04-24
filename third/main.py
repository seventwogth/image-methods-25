from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lighting import compute_brightness
from scene import SceneInput, create_default_scene
from vector import Vec3


RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULT_FILE = RESULTS_DIR / "brightness_results.txt"


@dataclass(frozen=True)
class ObserverBrightness:
    observer_position: Vec3
    raw_brightness: Vec3
    normalized_brightness: Vec3


def format_vec3(vector: Vec3) -> str:
    return f"({vector.x:.4f}, {vector.y:.4f}, {vector.z:.4f})"


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

    return [
        ObserverBrightness(
            observer_position=observer_position,
            raw_brightness=raw_brightness,
            normalized_brightness=normalized_brightness,
        )
        for observer_position, raw_brightness, normalized_brightness in zip(
            scene.observer_positions,
            raw_values,
            normalized_values,
        )
    ]


def build_report(scene: SceneInput) -> str:
    results = compute_results(scene)
    lines = [
        "Input scene:",
        f"Surface point: {format_vec3(scene.surface_point.position)}",
        f"Normal: {format_vec3(scene.surface_point.normal)}",
        f"Light position: {format_vec3(scene.light.position)}",
        f"Light intensity RGB: {format_vec3(scene.light.intensity)}",
        f"Material diffuse color: {format_vec3(scene.material.diffuse_color)}",
        f"Material specular color: {format_vec3(scene.material.specular_color)}",
        (
            "Material coefficients: "
            f"kd={scene.material.kd}, ks={scene.material.ks}, "
            f"shininess={scene.material.shininess}"
        ),
        "",
        "Results:",
    ]

    for index, result in enumerate(results, start=1):
        lines.append(
            f"Observer P{index:02d} = {format_vec3(result.observer_position)} "
            f"-> raw L(RGB) = {format_vec3(result.raw_brightness)} "
            f"-> normalized L(RGB) = {format_vec3(result.normalized_brightness)}"
        )

    return "\n".join(lines)


def main() -> None:
    scene = create_default_scene()
    report = build_report(scene)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(report, encoding="utf-8")

    print(report)
    print()
    print(f"Saved to: {RESULT_FILE}")


if __name__ == "__main__":
    main()
