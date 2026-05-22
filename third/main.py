from __future__ import annotations

import csv
from pathlib import Path

from lighting import compute_two_stage_brightness
from scene import SceneInput, create_default_scene
from vector import Vec3


OUTPUT_DIR = Path(__file__).resolve().parent


def format_number(value: float) -> str:
    return f"{value:.6f}"


def format_vec3(vector: Vec3) -> str:
    return f"({format_number(vector.x)}, {format_number(vector.y)}, {format_number(vector.z)})"


def save_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(headers)
        writer.writerows(rows)


def print_report(scene: SceneInput, brightness_rows: list[tuple[str, Vec3]]) -> None:
    # Report input block (format close to methodical guide).
    print("Входные данные:")
    print(f"- I01(RGB) = {format_vec3(scene.light.intensity)}")
    print(f"- O1 = {format_vec3(scene.light.position - scene.mirror.reflection_point)}")
    print(f"- PL = {format_vec3(scene.light.position)}")
    print(f"- PT = {format_vec3(scene.mirror.reflection_point)}")
    print(f"- N_mirror = {format_vec3(scene.mirror.normal)}")

    for idx, observer in enumerate(scene.observer_positions, start=1):
        print(f"- P{idx:02d} = {format_vec3(observer)}")

    print(f"- K(RGB) = {format_vec3(scene.diffuse.color)}")
    print(f"- kd = {format_number(scene.diffuse.kd)}")
    print(f"- ks = {format_number(scene.mirror.ks)}")

    print("\nРезультаты:")
    for observer_id, brightness in brightness_rows:
        print(f"- {observer_id} L(RGB, PT, {observer_id}) = {format_vec3(brightness)}")


def main() -> None:
    scene = create_default_scene()

    report_rows: list[tuple[str, Vec3]] = []
    csv_rows: list[list[str]] = []

    for idx, observer in enumerate(scene.observer_positions, start=1):
        observer_id = f"P{idx:02d}"
        brightness, state = compute_two_stage_brightness(scene, observer)
        report_rows.append((observer_id, brightness))
        csv_rows.append(
            [
                observer_id,
                format_vec3(observer),
                format_vec3(state.incident_to_mirror),
                format_vec3(state.reflected_from_mirror),
                format_vec3(state.diffuse_hit_point) if state.diffuse_hit_point else "NONE",
                format_vec3(brightness),
            ]
        )

    print_report(scene, report_rows)

    save_csv(
        OUTPUT_DIR / "brightness_results.csv",
        [
            "observer_id",
            "observer_position",
            "incident_to_mirror",
            "reflected_from_mirror",
            "diffuse_hit_point",
            "L_rgb",
        ],
        csv_rows,
    )


if __name__ == "__main__":
    main()
