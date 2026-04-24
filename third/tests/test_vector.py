from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from vector import Vec3


class Vec3Tests(unittest.TestCase):
    def test_addition_and_subtraction(self) -> None:
        left = Vec3(1.0, 2.0, 3.0)
        right = Vec3(0.5, -1.0, 4.0)

        self.assertEqual(left + right, Vec3(1.5, 1.0, 7.0))
        self.assertEqual(left - right, Vec3(0.5, 3.0, -1.0))

    def test_dot_product(self) -> None:
        left = Vec3(1.0, 3.0, -5.0)
        right = Vec3(4.0, -2.0, -1.0)

        self.assertEqual(left.dot(right), 3.0)

    def test_normalize(self) -> None:
        vector = Vec3(3.0, 0.0, 4.0).normalize()

        self.assertAlmostEqual(vector.x, 0.6)
        self.assertAlmostEqual(vector.y, 0.0)
        self.assertAlmostEqual(vector.z, 0.8)
        self.assertAlmostEqual(vector.length(), 1.0)

    def test_component_multiplication(self) -> None:
        left = Vec3(0.5, 0.25, 0.75)
        right = Vec3(0.2, 0.8, 0.4)
        result = left.component_mul(right)

        self.assertAlmostEqual(result.x, 0.1)
        self.assertAlmostEqual(result.y, 0.2)
        self.assertAlmostEqual(result.z, 0.3)

    def test_normalize_zero_vector_raises_error(self) -> None:
        with self.assertRaises(ValueError):
            Vec3(0.0, 0.0, 0.0).normalize()


if __name__ == "__main__":
    unittest.main()
