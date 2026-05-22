from __future__ import annotations

from dataclasses import dataclass

from vector import Vec3, cross


@dataclass(frozen=True)
class LightSource:
    position: Vec3
    intensity: Vec3  # I01(RGB)


@dataclass(frozen=True)
class MirrorSurface:
    p1: Vec3
    p2: Vec3
    p3: Vec3
    ks: float

    @property
    def normal(self) -> Vec3:
        return cross(self.p2 - self.p1, self.p3 - self.p1).normalize()


@dataclass(frozen=True)
class DiffuseSurface:
    p1: Vec3
    p2: Vec3
    p3: Vec3
    color: Vec3  # K(RGB)
    kd: float

    @property
    def normal(self) -> Vec3:
        return cross(self.p2 - self.p1, self.p3 - self.p1).normalize()


@dataclass(frozen=True)
class SceneInput:
    light: LightSource
    mirror: MirrorSurface
    diffuse: DiffuseSurface
    observer_positions: list[Vec3]


def create_default_scene() -> SceneInput:
    mirror = MirrorSurface(
        p1=Vec3(-2.0, -1.5, 0.0),
        p2=Vec3(2.0, -1.5, 0.0),
        p3=Vec3(0.0, 1.5, 0.0),
        ks=1.0,
    )

    # p2/p3 order chosen so normal points down (negative Z).
    diffuse = DiffuseSurface(
        p1=Vec3(-2.5, -1.5, 2.0),
        p2=Vec3(0.0, 2.0, 2.0),
        p3=Vec3(2.5, -1.5, 2.0),
        color=Vec3(0.7, 0.7, 0.7),
        kd=1.0,
    )

    return SceneInput(
        light=LightSource(
            position=Vec3(2.0, 0.0, 2.0),
            intensity=Vec3(1.0, 0.9, 0.8),
        ),
        mirror=mirror,
        diffuse=diffuse,
        observer_positions=[
            Vec3(-3.0, 0.0, 1.0),
            Vec3(-1.5, 2.0, 1.0),
            Vec3(0.0, 3.0, 1.0),
            Vec3(1.5, 2.0, 1.0),
            Vec3(3.0, 0.0, 1.0),
        ],
    )
