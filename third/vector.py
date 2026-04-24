from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import TypeAlias


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: VecLike) -> Vec3:
        other_vec = to_vec3(other)
        return Vec3(self.x + other_vec.x, self.y + other_vec.y, self.z + other_vec.z)

    def __sub__(self, other: VecLike) -> Vec3:
        other_vec = to_vec3(other)
        return Vec3(self.x - other_vec.x, self.y - other_vec.y, self.z - other_vec.z)

    def __mul__(self, scalar: float) -> Vec3:
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> Vec3:
        return self * scalar

    def __truediv__(self, scalar: float) -> Vec3:
        if scalar == 0:
            raise ValueError("Cannot divide a vector by zero.")
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other: VecLike) -> float:
        other_vec = to_vec3(other)
        return self.x * other_vec.x + self.y * other_vec.y + self.z * other_vec.z

    def length(self) -> float:
        return sqrt(self.dot(self))

    def normalize(self) -> Vec3:
        length = self.length()
        if length == 0:
            raise ValueError("Cannot normalize a zero vector.")
        return self / length

    def distance(self, other: VecLike) -> float:
        return (self - other).length()

    def component_mul(self, other: VecLike) -> Vec3:
        other_vec = to_vec3(other)
        return Vec3(self.x * other_vec.x, self.y * other_vec.y, self.z * other_vec.z)

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


Vector3: TypeAlias = Vec3
VecLike: TypeAlias = Vec3 | tuple[float, float, float]


def to_vec3(value: VecLike) -> Vec3:
    if isinstance(value, Vec3):
        return value
    return Vec3(*value)


def add(a: VecLike, b: VecLike) -> Vec3:
    return to_vec3(a) + b


def subtract(a: VecLike, b: VecLike) -> Vec3:
    return to_vec3(a) - b


def scale(vector: VecLike, factor: float) -> Vec3:
    return to_vec3(vector) * factor


def divide(vector: VecLike, factor: float) -> Vec3:
    return to_vec3(vector) / factor


def dot(a: VecLike, b: VecLike) -> float:
    return to_vec3(a).dot(b)


def cross(a: VecLike, b: VecLike) -> Vec3:
    left = to_vec3(a)
    right = to_vec3(b)
    return Vec3(
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x,
    )


def length(vector: VecLike) -> float:
    return to_vec3(vector).length()


def norm(vector: VecLike) -> float:
    return length(vector)


def normalize(vector: VecLike) -> Vec3:
    return to_vec3(vector).normalize()


def distance(a: VecLike, b: VecLike) -> float:
    return to_vec3(a).distance(b)


def component_multiply(a: VecLike, b: VecLike) -> Vec3:
    return to_vec3(a).component_mul(b)


def reflect(incident: VecLike, normal: VecLike) -> Vec3:
    """Reflect incident direction relative to a unit surface normal."""
    incident_unit = normalize(incident)
    normal_unit = normalize(normal)
    projection = 2.0 * dot(incident_unit, normal_unit)
    return incident_unit - normal_unit * projection


def clamp_nonnegative(value: float) -> float:
    return max(0.0, value)
