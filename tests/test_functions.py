import unittest
import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from codes.functions import nearest_neighbor_derivative, compute_f_hat_with_nearest_neighbor, f_obs


class TestFunctions(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Create test data
        self.n_points = 50
        self.x_samples = tf.random.uniform((1, self.n_points), 0, 3)
        self.y_samples = tf.random.uniform((1, self.n_points), 0, 3)
        self.g_values = tf.random.normal((1, self.n_points))

    def test_nearest_neighbor_derivative_dimensions(self):
        """Test dimension handling of nearest_neighbor_derivative"""

        # Test correct input dimensions
        d_g_dx, d_g_dy = nearest_neighbor_derivative(
            self.x_samples,
            self.y_samples,
            self.g_values
        )

        # Check output dimensions
        self.assertEqual(d_g_dx.shape, (1, self.n_points))
        self.assertEqual(d_g_dy.shape, (1, self.n_points))

        # Test invalid input dimensions - missing batch dimension
        invalid_x = tf.random.uniform((self.n_points,))
        invalid_y = tf.random.uniform((self.n_points,))
        invalid_g = tf.random.normal((self.n_points,))

        with self.assertRaises(tf.errors.InvalidArgumentError):
            nearest_neighbor_derivative(invalid_x, invalid_y, invalid_g)

        # Test mismatched number of points
        invalid_x = tf.random.uniform((1, self.n_points + 1))
        with self.assertRaises(tf.errors.InvalidArgumentError):
            nearest_neighbor_derivative(invalid_x, self.y_samples, self.g_values)

    def test_nearest_neighbor_derivative_values(self):
        """Test computation results of nearest_neighbor_derivative"""
        d_g_dx, d_g_dy = nearest_neighbor_derivative(
            self.x_samples,
            self.y_samples,
            self.g_values
        )

        # Check output finiteness
        self.assertTrue(tf.reduce_all(tf.math.is_finite(d_g_dx)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(d_g_dy)))

    def test_compute_f_hat_dimensions(self):
        """Test dimension handling of compute_f_hat_with_nearest_neighbor"""
        # Get sample data
        f_obs_values, x_samples, y_samples = f_obs()

        # Calculate u_x and u_y
        u_x = tf.cos(2 * x_samples) * 2
        u_y = tf.cos(2 * y_samples) * 2

        # Create test q values
        q = tf.random.normal((1, self.n_points))

        # Test correct input dimensions
        f_hat = compute_f_hat_with_nearest_neighbor(
            x_samples, y_samples, q, u_x, u_y
        )

        # Check output dimensions
        self.assertEqual(f_hat.shape, (1, self.n_points))

        # Test invalid input dimensions - missing batch dimension
        invalid_x = tf.random.uniform((self.n_points,))
        invalid_y = tf.random.uniform((self.n_points,))
        invalid_q = tf.random.normal((self.n_points,))
        invalid_u_x = tf.cos(2 * invalid_x) * 2
        invalid_u_y = tf.cos(2 * invalid_y) * 2

        with self.assertRaises(tf.errors.InvalidArgumentError):
            compute_f_hat_with_nearest_neighbor(
                invalid_x, invalid_y, invalid_q, invalid_u_x, invalid_u_y
            )

        # Test mismatched number of points
        invalid_x = tf.random.uniform((1, self.n_points + 1))
        with self.assertRaises(tf.errors.InvalidArgumentError):
            compute_f_hat_with_nearest_neighbor(
                invalid_x, y_samples, q, u_x, u_y
            )

    def test_compute_f_hat_values(self):
        """Test computation results of compute_f_hat_with_nearest_neighbor"""
        # Get sample data
        f_obs_values, x_samples, y_samples = f_obs()

        # Calculate u_x and u_y
        u_x = tf.cos(2 * x_samples) * 2
        u_y = tf.cos(2 * y_samples) * 2

        # Create test q values
        q = tf.random.normal((1, self.n_points))

        # Compute f_hat
        f_hat = compute_f_hat_with_nearest_neighbor(
            x_samples, y_samples, q, u_x, u_y
        )

        # Check output finiteness
        self.assertTrue(tf.reduce_all(tf.math.is_finite(f_hat)))

        # Check if output is within clipping range
        self.assertTrue(tf.reduce_all(tf.abs(f_hat) <= 200.0))

    def tearDown(self):
        """Clean up test environment"""
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    unittest.main()