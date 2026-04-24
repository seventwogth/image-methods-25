from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from lighting import compute_brightness, reflect
from scene import LightSource, Material, SurfacePoint
from vector import Vec3


class LightingReflectionTests(unittest.TestCase):
    def test_reflects_vertical_ray_from_horizontal_plane(self) -> None:
        incident = Vec3(0.0, 0.0, -1.0)
        normal = Vec3(0.0, 0.0, 1.0)

        reflected = reflect(incident, normal)

        self.assertAlmostEqual(reflected.x, 0.0)
        self.assertAlmostEqual(reflected.y, 0.0)
        self.assertAlmostEqual(reflected.z, 1.0)

    def test_reflects_angled_ray(self) -> None:
        incident = Vec3(1.0, 0.0, -1.0)
        normal = Vec3(0.0, 0.0, 1.0)

        reflected = reflect(incident, normal)

        self.assertAlmostEqual(reflected.x, 2 ** -0.5)
        self.assertAlmostEqual(reflected.y, 0.0)
        self.assertAlmostEqual(reflected.z, 2 ** -0.5)

    def test_reflection_result_is_normalized(self) -> None:
        incident = Vec3(2.0, -1.0, -3.0)
        normal = Vec3(0.0, 0.0, 5.0)

        reflected = reflect(incident, normal)

        self.assertAlmostEqual(reflected.length(), 1.0)

    def test_diffuse_brightness_is_positive_for_front_lighting(self) -> None:
        brightness = compute_brightness(
            surface_point=SurfacePoint(position=Vec3(0.0, 0.0, 0.0), normal=Vec3(0.0, 0.0, 1.0)),
            material=Material(
                diffuse_color=Vec3(0.7, 0.6, 0.5),
                specular_color=Vec3(0.0, 0.0, 0.0),
                kd=1.0,
                ks=0.0,
                shininess=8.0,
            ),
            light=LightSource(position=Vec3(0.0, 0.0, 2.0), intensity=Vec3(1.0, 1.0, 1.0)),
            observer_position=Vec3(0.0, 0.0, 3.0),
        )

        self.assertGreater(brightness.x, 0.0)
        self.assertGreater(brightness.y, 0.0)
        self.assertGreater(brightness.z, 0.0)

    def test_diffuse_brightness_is_zero_when_light_is_behind_surface(self) -> None:
        brightness = compute_brightness(
            surface_point=SurfacePoint(position=Vec3(0.0, 0.0, 0.0), normal=Vec3(0.0, 0.0, 1.0)),
            material=Material(
                diffuse_color=Vec3(0.7, 0.6, 0.5),
                specular_color=Vec3(0.0, 0.0, 0.0),
                kd=1.0,
                ks=0.0,
                shininess=8.0,
            ),
            light=LightSource(position=Vec3(0.0, 0.0, -2.0), intensity=Vec3(1.0, 1.0, 1.0)),
            observer_position=Vec3(0.0, 0.0, 3.0),
        )

        self.assertAlmostEqual(brightness.x, 0.0)
        self.assertAlmostEqual(brightness.y, 0.0)
        self.assertAlmostEqual(brightness.z, 0.0)

    def test_specular_component_increases_near_reflected_direction(self) -> None:
        surface_point = SurfacePoint(position=Vec3(0.0, 0.0, 0.0), normal=Vec3(0.0, 0.0, 1.0))
        material = Material(
            diffuse_color=Vec3(0.0, 0.0, 0.0),
            specular_color=Vec3(1.0, 1.0, 1.0),
            kd=0.0,
            ks=1.0,
            shininess=12.0,
        )
        light = LightSource(position=Vec3(0.0, 0.0, 2.0), intensity=Vec3(1.0, 1.0, 1.0))

        near_highlight = compute_brightness(
            surface_point=surface_point,
            material=material,
            light=light,
            observer_position=Vec3(0.0, 0.0, 3.0),
        )
        far_from_highlight = compute_brightness(
            surface_point=surface_point,
            material=material,
            light=light,
            observer_position=Vec3(2.0, 0.0, 1.0),
        )

        self.assertGreater(near_highlight.x, far_from_highlight.x)
        self.assertGreater(near_highlight.y, far_from_highlight.y)
        self.assertGreater(near_highlight.z, far_from_highlight.z)


if __name__ == "__main__":
    unittest.main()
