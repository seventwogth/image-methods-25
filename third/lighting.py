from __future__ import annotations

import random
from dataclasses import dataclass

from scene import DiffuseSurface, MirrorSurface, SceneInput
from vector import Vec3, clamp_nonnegative

EPS = 1e-9


@dataclass(frozen=True)
class SampleLightingResult:
    sample_id: str
    mirror_point: Vec3
    incident_to_mirror: Vec3
    reflected_from_mirror: Vec3
    diffuse_hit_point: Vec3 | None
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
    """Check point inclusion in triangle by barycentric coordinates."""
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
    """Intersect ray with surface plane and keep only hits inside the triangle."""
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
    """Generate uniformly distributed random point inside triangle."""
    r1 = rng.random()
    r2 = rng.random()

    if r1 + r2 > 1.0:
        r1 = 1.0 - r1
        r2 = 1.0 - r2

    return a + (b - a) * r1 + (c - a) * r2


def compute_sample_brightness(
    scene: SceneInput,
    observer_position: Vec3,
    mirror_point: Vec3,
    sample_id: str,
) -> SampleLightingResult:
    # 1) Build incident ray from source to random mirror point PT.
    incident = (mirror_point - scene.light.position).normalize()

    # 2) Reflect from mirror using mirror normal computed from triangle points.
    reflected = reflect_from_mirror(incident, scene.mirror.normal)

    # 3) Intersect reflected ray with finite diffuse triangle.
    hit_point = intersect_ray_with_surface_triangle(
        ray_origin=mirror_point,
        ray_direction=reflected,
        p1=scene.diffuse.p1,
        p2=scene.diffuse.p2,
        p3=scene.diffuse.p3,
        normal=scene.diffuse.normal,
    )

    if hit_point is None:
        return SampleLightingResult(
            sample_id=sample_id,
            mirror_point=mirror_point,
            incident_to_mirror=incident,
            reflected_from_mirror=reflected,
            diffuse_hit_point=None,
            brightness=Vec3(0.0, 0.0, 0.0),
        )

    # 4) Compute Lambert brightness toward observer from diffuse hit point.
    mirrored_intensity = scene.light.intensity * scene.mirror.ks
    view_direction = (observer_position - hit_point).normalize()
    lambert_factor = clamp_nonnegative(scene.diffuse.normal.dot(view_direction))
    brightness = mirrored_intensity.component_mul(scene.diffuse.color) * (scene.diffuse.kd * lambert_factor)

    return SampleLightingResult(
        sample_id=sample_id,
        mirror_point=mirror_point,
        incident_to_mirror=incident,
        reflected_from_mirror=reflected,
        diffuse_hit_point=hit_point,
        brightness=brightness,
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
        mirror_point = random_point_on_triangle(
            scene.mirror.p1,
            scene.mirror.p2,
            scene.mirror.p3,
            rng,
        )
        sample = compute_sample_brightness(
            scene=scene,
            observer_position=observer_position,
            mirror_point=mirror_point,
            sample_id=f"sample_{sample_index:02d}",
        )
        sample_results.append(sample)
        total = total + sample.brightness

    average = total / float(samples_per_observer)
    return ObserverLightingResult(
        observer_id=observer_id,
        observer_position=observer_position,
        samples=sample_results,
        average_brightness=average,
    )
