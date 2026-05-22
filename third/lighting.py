from __future__ import annotations

from dataclasses import dataclass

from scene import DiffuseSurface, SceneInput
from vector import Vec3, clamp_nonnegative


@dataclass(frozen=True)
class TwoStageLightingState:
    incident_to_mirror: Vec3
    reflected_from_mirror: Vec3
    diffuse_hit_point: Vec3 | None


def reflect_from_mirror(incident: Vec3, normal: Vec3) -> Vec3:
    """Mirror reflection: O' = O - 2 * (O · N) * N."""
    incident_unit = incident.normalize()
    normal_unit = normal.normalize()
    return (incident_unit - 2.0 * incident_unit.dot(normal_unit) * normal_unit).normalize()


def intersect_ray_with_plane(ray_origin: Vec3, ray_direction: Vec3, plane: DiffuseSurface) -> Vec3 | None:
    """Return intersection point of ray and diffuse plane, or None if absent."""
    direction = ray_direction.normalize()
    plane_normal = plane.normal.normalize()

    denominator = direction.dot(plane_normal)
    if abs(denominator) < 1e-9:
        return None

    t = (plane.point - ray_origin).dot(plane_normal) / denominator
    if t <= 1e-9:
        return None

    return ray_origin + direction * t


def compute_two_stage_brightness(scene: SceneInput, observer_position: Vec3) -> tuple[Vec3, TwoStageLightingState]:
    # 1) Build incident ray from source to mirror reflection point PT.
    pt = scene.mirror.reflection_point
    incident = (pt - scene.light.position).normalize()

    # 2) Reflect from mirror at PT.
    reflected = reflect_from_mirror(incident, scene.mirror.normal)

    # Mirror scales incoming RGB intensity.
    mirrored_intensity = scene.light.intensity * scene.mirror.ks

    # 3) Intersect reflected ray with diffuse plane.
    hit_point = intersect_ray_with_plane(pt, reflected, scene.diffuse)
    if hit_point is None:
        return Vec3(0.0, 0.0, 0.0), TwoStageLightingState(incident, reflected, None)

    # 4) Compute Lambert brightness toward observer from diffuse hit point.
    view_dir = (observer_position - hit_point).normalize()
    diffuse_cos = clamp_nonnegative(scene.diffuse.normal.normalize().dot(view_dir))

    brightness = mirrored_intensity.component_mul(scene.diffuse.color) * (scene.diffuse.kd * diffuse_cos)
    return brightness, TwoStageLightingState(incident, reflected, hit_point)
