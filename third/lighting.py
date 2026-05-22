from __future__ import annotations

import random
from dataclasses import dataclass

from scene import SceneInput
from vector import Vec3, clamp_nonnegative

EPS = 1e-9


@dataclass(frozen=True)
class SampleLightingResult:
    sample_id: str
    mirror_point: Vec3
    view_incident_to_mirror: Vec3
    reflected_view_ray: Vec3
    diffuse_hit_point: Vec3 | None
    light_direction_to_source: Vec3 | None
    brightness: Vec3


@dataclass(frozen=True)
class ObserverLightingResult:
    observer_id: str
    observer_position: Vec3
    samples: list[SampleLightingResult]
    average_brightness: Vec3


def reflect_from_mirror(incident: Vec3, normal: Vec3) -> Vec3:
    """Mirror reflection: O' = O - 2 * (O · N) * N."""
    incident_unit = incident.normalize()
    normal_unit = normal.normalize()
    return (incident_unit - 2.0 * incident_unit.dot(normal_unit) * normal_unit).normalize()


def is_point_inside_triangle(point: Vec3, a: Vec3, b: Vec3, c: Vec3) -> bool:
    v0 = b - a
    v1 = c - a
    v2 = point - a

    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)

    denominator = d00 * d11 - d01 * d01
    if abs(denominator) < EPS:
        return False

    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    u = 1.0 - v - w

    return (
        -EPS <= u <= 1.0 + EPS
        and -EPS <= v <= 1.0 + EPS
        and -EPS <= w <= 1.0 + EPS
    )


def intersect_ray_with_surface_triangle(
    ray_origin: Vec3,
    ray_direction: Vec3,
    p1: Vec3,
    p2: Vec3,
    p3: Vec3,
    normal: Vec3,
) -> Vec3 | None:
    direction = ray_direction.normalize()
    unit_normal = normal.normalize()

    denominator = direction.dot(unit_normal)
    if abs(denominator) < EPS:
        return None

    t = (p1 - ray_origin).dot(unit_normal) / denominator
    if t <= EPS:
        return None

    hit = ray_origin + direction * t
    if not is_point_inside_triangle(hit, p1, p2, p3):
        return None

    return hit


def random_point_on_triangle(a: Vec3, b: Vec3, c: Vec3, rng: random.Random) -> Vec3:
    r1 = rng.random()
    r2 = rng.random()

    if r1 + r2 > 1.0:
        r1 = 1.0 - r1
        r2 = 1.0 - r2

    return a + (b - a) * r1 + (c - a) * r2


def compute_reflected_view_brightness(
    scene: SceneInput,
    observer_position: Vec3,
    mirror_point: Vec3,
    sample_id: str,
) -> SampleLightingResult:
    # 1) Observer view ray arrives to mirror point.
    view_incident = (mirror_point - observer_position).normalize()

    # 2) Mirror reflects observer view direction.
    reflected_view = reflect_from_mirror(view_incident, scene.mirror.normal)

    # 3) Reflected view ray intersects finite diffuse triangle.
    hit_point = intersect_ray_with_surface_triangle(
        ray_origin=mirror_point,
        ray_direction=reflected_view,
        p1=scene.diffuse.p1,
        p2=scene.diffuse.p2,
        p3=scene.diffuse.p3,
        normal=scene.diffuse.normal,
    )

    if hit_point is None:
        return SampleLightingResult(
            sample_id=sample_id,
            mirror_point=mirror_point,
            view_incident_to_mirror=view_incident,
            reflected_view_ray=reflected_view,
            diffuse_hit_point=None,
            light_direction_to_source=None,
            brightness=Vec3(0.0, 0.0, 0.0),
        )

    # 4) Diffuse point is lit directly by source (Lambert), then scaled by mirror ks.
    light_direction = (scene.light.position - hit_point).normalize()
    lambert = clamp_nonnegative(scene.diffuse.normal.dot(light_direction))
    diffuse = scene.light.intensity.component_mul(scene.diffuse.color) * (scene.diffuse.kd * lambert)
    visible = diffuse * scene.mirror.ks

    return SampleLightingResult(
        sample_id=sample_id,
        mirror_point=mirror_point,
        view_incident_to_mirror=view_incident,
        reflected_view_ray=reflected_view,
        diffuse_hit_point=hit_point,
        light_direction_to_source=light_direction,
        brightness=visible,
    )


def compute_observer_lighting(
    scene: SceneInput,
    observer_id: str,
    observer_position: Vec3,
    samples_per_observer: int,
    rng: random.Random,
) -> ObserverLightingResult:
    sample_results: list[SampleLightingResult] = []
    total = Vec3(0.0, 0.0, 0.0)

    for sample_index in range(1, samples_per_observer + 1):
        mirror_point = random_point_on_triangle(scene.mirror.p1, scene.mirror.p2, scene.mirror.p3, rng)
        sample = compute_reflected_view_brightness(
            scene=scene,
            observer_position=observer_position,
            mirror_point=mirror_point,
            sample_id=f"sample_{sample_index:02d}",
        )
        sample_results.append(sample)
        total = total + sample.brightness

    return ObserverLightingResult(
        observer_id=observer_id,
        observer_position=observer_position,
        samples=sample_results,
        average_brightness=total / float(samples_per_observer),
    )
