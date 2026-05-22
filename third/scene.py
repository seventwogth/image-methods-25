from __future__ import annotations

from dataclasses import dataclass

from vector import Vec3


@dataclass(frozen=True)
class LightSource:
    position: Vec3
    intensity: Vec3  # I01(RGB)


@dataclass(frozen=True)
class MirrorSurface:
    reflection_point: Vec3  # PT
    normal: Vec3  # N_mirror
    ks: float


@dataclass(frozen=True)
class DiffuseSurface:
    point: Vec3  # a point on diffuse plane
    normal: Vec3
    color: Vec3  # K(RGB)
    kd: float


@dataclass(frozen=True)
class SceneInput:
    light: LightSource
    mirror: MirrorSurface
    diffuse: DiffuseSurface
    observer_positions: list[Vec3]


def create_default_scene() -> SceneInput:
    # Mirror on z=0, diffuse plane above it on z=2.
    # Light arrives to mirror and reflects upward to diffuse plane.
    return SceneInput(
        light=LightSource(
            position=Vec3(2.0, 0.0, 2.0),
            intensity=Vec3(1.0, 0.9, 0.8),
        ),
        mirror=MirrorSurface(
            reflection_point=Vec3(0.0, 0.0, 0.0),
            normal=Vec3(0.0, 0.0, 1.0),
            ks=1.0,
        ),
        diffuse=DiffuseSurface(
            point=Vec3(0.0, 0.0, 2.0),
            normal=Vec3(0.0, 0.0, -1.0),
            color=Vec3(0.7, 0.7, 0.7),
            kd=1.0,
        ),
        observer_positions=[
            Vec3(-3.0, 0.0, 1.0),
            Vec3(-1.5, 2.0, 1.0),
            Vec3(0.0, 3.0, 1.0),
            Vec3(1.5, 2.0, 1.0),
            Vec3(3.0, 0.0, 1.0),
        ],
    )
