from __future__ import annotations

import csv
import random
from pathlib import Path

from lighting import ObserverLightingResult, compute_observer_lighting
from scene import SceneInput, create_default_scene
from vector import Vec3


OUTPUT_DIR = Path(__file__).resolve().parent
SAMPLES_PER_OBSERVER = 5
RANDOM_SEED = 42


def format_number(value: float) -> str:
    return f"{value:.6f}"


def format_vec3(vector: Vec3) -> str:
    return f"({format_number(vector.x)}, {format_number(vector.y)}, {format_number(vector.z)})"


def save_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(headers)
        writer.writerows(rows)


def print_report(scene: SceneInput, results: list[ObserverLightingResult]) -> None:
    print("Входные данные:")
    print(f"- I01(RGB) = {format_vec3(scene.light.intensity)}")
    print(f"- PL = {format_vec3(scene.light.position)}")
    print("- Mirror:")
    print(f"  A = {format_vec3(scene.mirror.p1)}")
    print(f"  B = {format_vec3(scene.mirror.p2)}")
    print(f"  C = {format_vec3(scene.mirror.p3)}")
    print(f"  N_mirror = {format_vec3(scene.mirror.normal)}")
    print(f"  ks = {format_number(scene.mirror.ks)}")
    print("- Diffuse:")
    print(f"  A = {format_vec3(scene.diffuse.p1)}")
    print(f"  B = {format_vec3(scene.diffuse.p2)}")
    print(f"  C = {format_vec3(scene.diffuse.p3)}")
    print(f"  N_diffuse = {format_vec3(scene.diffuse.normal)}")
    print(f"  K(RGB) = {format_vec3(scene.diffuse.color)}")
    print(f"  kd = {format_number(scene.diffuse.kd)}")
    print(f"- samples_per_observer = {SAMPLES_PER_OBSERVER}")
    print(f"- random_seed = {RANDOM_SEED}")
    for idx, observer in enumerate(scene.observer_positions, start=1):
        print(f"- P{idx:02d} = {format_vec3(observer)}")

    print("\nРезультаты:")
    for observer_result in results:
        print(f"{observer_result.observer_id}:")
        for sample in observer_result.samples:
            hit = format_vec3(sample.diffuse_hit_point) if sample.diffuse_hit_point else "NONE"
            print(f"  {sample.sample_id}:")
            print(f"    PT = {format_vec3(sample.mirror_point)}")
            print(f"    O = {format_vec3(sample.incident_to_mirror)}")
            print(f"    O' = {format_vec3(sample.reflected_from_mirror)}")
            print(f"    P_hit = {hit}")
            print(f"    L(RGB) = {format_vec3(sample.brightness)}")
        print(
            f"  average L(RGB, {observer_result.observer_id}) = "
            f"{format_vec3(observer_result.average_brightness)}"
        )


def build_csv_rows(results: list[ObserverLightingResult]) -> list[list[str]]:
    rows: list[list[str]] = []
    for observer_result in results:
        avg = observer_result.average_brightness
        for sample in observer_result.samples:
            rows.append(
                [
                    observer_result.observer_id,
                    sample.sample_id,
                    format_vec3(observer_result.observer_position),
                    format_vec3(sample.mirror_point),
                    format_vec3(sample.incident_to_mirror),
                    format_vec3(sample.reflected_from_mirror),
                    format_vec3(sample.diffuse_hit_point) if sample.diffuse_hit_point else "NONE",
                    format_number(sample.brightness.x),
                    format_number(sample.brightness.y),
                    format_number(sample.brightness.z),
                    format_number(avg.x),
                    format_number(avg.y),
                    format_number(avg.z),
                ]
            )
    return rows


def main() -> None:
    scene = create_default_scene()
    rng = random.Random(RANDOM_SEED)

    results: list[ObserverLightingResult] = []
    for idx, observer_position in enumerate(scene.observer_positions, start=1):
        observer_id = f"P{idx:02d}"
        results.append(
            compute_observer_lighting(
                scene=scene,
                observer_id=observer_id,
                observer_position=observer_position,
                samples_per_observer=SAMPLES_PER_OBSERVER,
                rng=rng,
            )
        )

    print_report(scene, results)

    csv_headers = [
        "observer_id",
        "sample_id",
        "observer_position",
        "mirror_point",
        "incident_to_mirror",
        "reflected_from_mirror",
        "diffuse_hit_point",
        "L_R",
        "L_G",
        "L_B",
        "average_L_R",
        "average_L_G",
        "average_L_B",
    ]
    csv_rows = build_csv_rows(results)
    save_csv(OUTPUT_DIR / "brightness_results.csv", csv_headers, csv_rows)


if __name__ == "__main__":
    main()
