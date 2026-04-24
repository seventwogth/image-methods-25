from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from lighting import estimate_specular_brightness
from scene import create_default_scene
from vector import dot, normalize


class ProjectSmokeTests(unittest.TestCase):
    def test_default_scene_can_be_processed(self) -> None:
        scene = create_default_scene()
        result = estimate_specular_brightness(scene)

        self.assertGreaterEqual(result.estimated_brightness.x, 0.0)
        self.assertGreaterEqual(result.estimated_brightness.y, 0.0)
        self.assertGreaterEqual(result.estimated_brightness.z, 0.0)
        self.assertGreaterEqual(result.alignment, 0.0)

    def test_default_scene_contains_five_observers(self) -> None:
        scene = create_default_scene()
        self.assertEqual(len(scene.observer_positions), 5)

    def test_normalize_returns_unit_vector(self) -> None:
        vector = normalize((3.0, 0.0, 4.0))
        self.assertAlmostEqual(dot(vector, vector), 1.0)


if __name__ == "__main__":
    unittest.main()
