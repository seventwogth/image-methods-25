from __future__ import annotations

from dataclasses import dataclass

from vector import Vec3


@dataclass(frozen=True)
class LightSource:
    position: Vec3
    intensity: Vec3  # RGB intensity


@dataclass(frozen=True)
class SurfacePoint:
    position: Vec3
    normal: Vec3


@dataclass(frozen=True)
class Material:
    diffuse_color: Vec3
    specular_color: Vec3
    kd: float
    ks: float
    shininess: float


@dataclass(frozen=True)
class SceneInput:
    light: LightSource
    surface_point: SurfacePoint
    material: Material
    observer_positions: list[Vec3]


def create_default_scene() -> SceneInput:
    """
    Return a simple scene for the first part of lab work 3.

    The values are intentionally small and easy to explain:
    - the surface point is at the origin;
    - the normal is directed upward along the Z axis;
    - the light source is placed to the side and above the point;
    - five observers are placed around the point.
    """
    return SceneInput(
        light=LightSource(
            position=Vec3(2.0, 2.0, 3.0),
            intensity=Vec3(1.0, 0.9, 0.8),
        ),
        surface_point=SurfacePoint(
            position=Vec3(0.0, 0.0, 0.0),
            normal=Vec3(0.0, 0.0, 1.0),
        ),
        material=Material(
            diffuse_color=Vec3(0.7, 0.7, 0.7),
            specular_color=Vec3(1.0, 1.0, 1.0),
            kd=0.6,
            ks=0.8,
            shininess=16.0,
        ),
        observer_positions=[
            Vec3(-3.0, 0.0, 2.0),
            Vec3(-1.5, 2.0, 2.5),
            Vec3(0.0, 3.0, 3.0),
            Vec3(1.5, 2.0, 2.5),
            Vec3(3.0, 0.0, 2.0),
        ],
    )
