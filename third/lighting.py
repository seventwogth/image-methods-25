from __future__ import annotations

from dataclasses import dataclass

from scene import LightSource, Material, SceneInput, SurfacePoint
from vector import Vec3, Vector3, clamp_nonnegative


@dataclass(frozen=True)
class BrightnessResult:
    light_direction: Vector3
    view_direction: Vector3
    reflection_direction: Vector3
    alignment: float
    estimated_brightness: Vec3
    note: str


def reflect(incident: Vec3, normal: Vec3) -> Vec3:
    """
    Reflect an incident ray relative to the surface normal.

    Formula:
        O' = O - 2 * (O dot N) * N
    """
    incident_unit = incident.normalize()
    normal_unit = normal.normalize()
    reflected = incident_unit - normal_unit * (2.0 * incident_unit.dot(normal_unit))
    return reflected.normalize()


def compute_brightness(
    surface_point: SurfacePoint,
    material: Material,
    light: LightSource,
    observer_position: Vec3,
) -> Vec3:
    """
    Compute RGB brightness in the surface point for one observer direction.

    Local lighting model:
        L = L_diffuse + L_specular
    """
    point = surface_point.position
    normal = surface_point.normal.normalize()

    light_direction = (light.position - point).normalize()
    view_direction = (observer_position - point).normalize()
    incident_direction = (point - light.position).normalize()
    reflection_direction = reflect(incident_direction, normal)

    diffuse_factor = clamp_nonnegative(normal.dot(light_direction))
    diffuse = light.intensity.component_mul(material.diffuse_color) * (
        material.kd * diffuse_factor
    )

    specular_factor = clamp_nonnegative(reflection_direction.dot(view_direction))
    specular = light.intensity.component_mul(material.specular_color) * (
        material.ks * (specular_factor ** material.shininess)
    )

    return diffuse + specular


def estimate_specular_brightness(scene: SceneInput) -> BrightnessResult:
    """
    Prepare intermediate values and compute brightness for the first observer.
    """
    if not scene.observer_positions:
        raise ValueError("Scene must contain at least one observer position.")

    point = scene.surface_point.position
    normal = scene.surface_point.normal
    observer_position = scene.observer_positions[0]

    light_direction = (scene.light.position - point).normalize()
    view_direction = (observer_position - point).normalize()
    incident_direction = (point - scene.light.position).normalize()
    reflection_direction = reflect(incident_direction, normal)

    alignment = clamp_nonnegative(reflection_direction.dot(view_direction))
    brightness = compute_brightness(
        surface_point=scene.surface_point,
        material=scene.material,
        light=scene.light,
        observer_position=observer_position,
    )

    return BrightnessResult(
        light_direction=light_direction,
        view_direction=view_direction,
        reflection_direction=reflection_direction,
        alignment=alignment,
        estimated_brightness=brightness,
        note=(
            "Brightness is computed with the local model "
            "for the first observer from observer_positions."
        ),
    )
