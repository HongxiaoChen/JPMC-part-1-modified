import unittest
import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from codes.functions import nearest_neighbor_derivative, compute_f_hat_with_nearest_neighbor, f_obs, get_target_log_prob


class TestGetTargetLogProb(unittest.TestCase):
    """Test the get_target_log_prob function"""
    
    def setUp(self):
        """Set up the test environment"""
        # Define test configurations
        self.test_configs = {
            '1D_Gauss_mix': {
                'dims': [1],  # Position coordinate dimensions
                'input_dim': 2  # Total dimensions (position + momentum)
            },
            '2D_Neal_funnel': {
                'dims': [2],
                'input_dim': 4
            },
            '5D_illconditioned_Gaussian': {
                'dims': [5],
                'input_dim': 10
            },
            'nD_Rosenbrock': {
                'dims': [3, 5],  # Test different dimensions
                'input_dim': [6, 10]  # Corresponding total dimensions (position + momentum)
            }
        }
    
    def test_single_chain_dimensions(self):
        """Test single chain dimension handling"""
        for dist_name, config in self.test_configs.items():
            for dim_idx, dim in enumerate(config['dims']):
                with self.subTest(distribution=dist_name, dim=dim):
                    # Create single chain input (shape [dim])
                    state_parts = tf.random.normal([dim])
                    input_dim = config['input_dim']
                    if isinstance(input_dim, list):
                        input_dim = input_dim[dim_idx]
                    
                    # Calculate log probability
                    log_prob = get_target_log_prob(state_parts, dist_name, input_dim)
                    
                    # Check if output is a scalar
                    self.assertEqual(log_prob.shape, ())
                    
                    # Check if output is finite
                    self.assertTrue(tf.math.is_finite(log_prob))

    def test_multiple_chains_dimensions(self):
        """Test multiple chains dimension handling"""
        batch_sizes = [1, 5, 10]  # Test different batch sizes
        
        for dist_name, config in self.test_configs.items():
            for dim_idx, dim in enumerate(config['dims']):
                for batch_size in batch_sizes:
                    with self.subTest(distribution=dist_name, dim=dim, batch_size=batch_size):
                        # Create multiple chains input (shape [batch_size, dim])
                        state_parts = tf.random.normal([batch_size, dim])
                        input_dim = config['input_dim']
                        if isinstance(input_dim, list):
                            input_dim = input_dim[dim_idx]
                        
                        # Calculate log probability
                        log_prob = get_target_log_prob(state_parts, dist_name, input_dim)
                        
                        # Check if output shape is [batch_size]
                        self.assertEqual(log_prob.shape, (batch_size,))
                        
                        # Check if output is finite
                        self.assertTrue(tf.reduce_all(tf.math.is_finite(log_prob)))
    
    def test_invalid_distribution(self):
        """Test invalid distribution name"""
        invalid_dist = "invalid_distribution"
        state_parts = tf.random.normal([2])
        
        # Verify if an invalid distribution name raises ValueError
        with self.assertRaises(ValueError):
            get_target_log_prob(state_parts, invalid_dist, 4)
    
    def test_input_type_conversion(self):
        """Test input type conversion"""
        # Test numpy array input
        for dist_name in ['1D_Gauss_mix', '2D_Neal_funnel']:
            with self.subTest(distribution=dist_name):
                # Use numpy array as input
                if dist_name == '1D_Gauss_mix':
                    state_parts = np.random.normal(size=1).astype(np.float32)
                    input_dim = 2
                else:
                    state_parts = np.random.normal(size=2).astype(np.float32)
                    input_dim = 4
                
                # Verify numpy array can be converted and processed
                log_prob = get_target_log_prob(state_parts, dist_name, input_dim)
                
                # Check if output is a scalar
                self.assertEqual(log_prob.shape, ())
                
                # Check if output is finite
                self.assertTrue(tf.math.is_finite(log_prob))
                
    def tearDown(self):
        """Clean up test environment"""
        tf.keras.backend.clear_session()


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