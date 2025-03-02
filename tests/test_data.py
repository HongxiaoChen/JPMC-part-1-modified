import unittest
import tensorflow as tf
import numpy as np
import os
import shutil
from pathlib import Path
import sys

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class TestDataGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up environment before all tests"""
        # Change to codes directory
        os.chdir(os.path.join(PROJECT_ROOT, 'codes'))
        # Add codes directory to Python path
        if str(PROJECT_ROOT / 'codes') not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT / 'codes'))

        # Import required modules after changing working directory
        global get_trajectory, get_dataset, get_args, from_pickle
        from codes.data import get_trajectory, get_dataset
        from codes.get_args import get_args
        from codes.utils import from_pickle

    def setUp(self):
        """Set up test environment"""
        # Create temporary save directory
        self.temp_save_dir = PROJECT_ROOT / 'temp_test_data'
        self.temp_save_dir.mkdir(exist_ok=True)

        # Configure test cases
        self.test_configs = [
            {
                'name': 'nD_Rosenbrock',
                'input_dim': 6,
                'samples': 5,
                'len_sample': 2,
                'test_fraction': 0.2
            },
            {
                'name': '2D_Neal_funnel',
                'input_dim': 4,
                'samples': 5,
                'len_sample': 2,
                'test_fraction': 0.2
            },
            {
                'name': '5D_illconditioned_Gaussian',
                'input_dim': 10,
                'samples': 5,
                'len_sample': 2,
                'test_fraction': 0.2
            },
            {
                'name': 'Allen_Cahn',
                'input_dim': 50,
                'samples': 5,
                'len_sample': 2,
                'test_fraction': 0.2
            }
        ]

    def get_modified_args(self, config):
        """Get modified arguments

        Args:
            config: Dictionary containing configuration parameters

        Returns:
            Modified args object
        """
        args = get_args()
        args.dist_name = config['name']
        args.input_dim = config['input_dim']
        args.num_samples = config['samples']
        args.len_sample = config['len_sample']
        args.test_fraction = config['test_fraction']
        args.save_dir = str(self.temp_save_dir)
        return args

    def test_get_trajectory(self):
        """Test trajectory generation function"""
        for config in self.test_configs:
            with self.subTest(distribution=config['name']):
                # Get modified parameters
                args = self.get_modified_args(config)

                # Test default parameters
                traj_split, deriv_split, t_eval = get_trajectory(args=args)

                # Check output dimensions
                self.assertEqual(len(traj_split), config['input_dim'])
                self.assertEqual(len(deriv_split), config['input_dim'])

                # Check dimensions of each component
                for traj, deriv in zip(traj_split, deriv_split):
                    self.assertEqual(traj.shape[0], 1)  # batch size
                    self.assertEqual(deriv.shape[0], 1)  # batch size
                    self.assertEqual(traj.shape[2], 1)  # each component is scalar
                    self.assertEqual(deriv.shape[2], 1)  # each component is scalar

                # Test custom parameters
                custom_t_span = [0, 2]
                custom_dt = 0.1
                y0 = tf.zeros([1, config['input_dim']])

                traj_split, deriv_split, t_eval = get_trajectory(
                    t_span=custom_t_span,
                    dt=custom_dt,
                    y0=y0,
                    args=args
                )

                # Check number of time points
                expected_steps = int((custom_t_span[1] - custom_t_span[0]) / custom_dt)
                self.assertEqual(t_eval.shape[0], expected_steps + 1)

                # Check trajectory continuity
                for traj in traj_split:
                    traj_squeezed = tf.squeeze(traj, axis=2)  # remove last dimension of 1
                    diff = tf.reduce_max(tf.abs(traj_squeezed[:, 1:] - traj_squeezed[:, :-1]))
                    self.assertLess(float(diff), 10.0)

    def test_get_dataset(self):
        """Test dataset generation function"""
        for config in self.test_configs:
            with self.subTest(distribution=config['name']):
                # Get modified parameters
                args = self.get_modified_args(config)

                # Set custom time range
                custom_t_span = [0, 2]

                # Generate dataset
                dataset = get_dataset(
                    seed=42,
                    samples=config['samples'],
                    test_split=0.8,  # 80% training set
                    args=args,
                    t_span=custom_t_span
                )

                # Check dataset structure
                expected_keys = ['coords', 'dcoords', 'test_coords', 'test_dcoords']
                self.assertTrue(all(key in dataset for key in expected_keys))

                # Calculate expected number of data points
                dt = 0.025  # default step size
                n_steps = int((custom_t_span[1] - custom_t_span[0]) / dt)
                points_per_sample = n_steps + 1
                total_points = config['samples'] * points_per_sample

                train_size = int(total_points * 0.8)  # 80% training set
                test_size = total_points - train_size

                # Check dimensions
                self.assertEqual(dataset['coords'].shape[0], train_size)
                self.assertEqual(dataset['coords'].shape[1], config['input_dim'])
                self.assertEqual(dataset['test_coords'].shape[0], test_size)
                self.assertEqual(dataset['test_coords'].shape[1], config['input_dim'])

                # Verify saved files
                save_path = self.temp_save_dir / f"{config['name']}{args.len_sample}.pkl"
                self.assertTrue(save_path.exists())

                # Load and verify saved data
                loaded_data = from_pickle(save_path)
                for key in expected_keys:
                    self.assertTrue(tf.reduce_all(tf.equal(dataset[key], loaded_data[key])))

                # Check data finiteness
                for key in expected_keys:
                    self.assertTrue(tf.reduce_all(tf.math.is_finite(dataset[key])))

    def tearDown(self):
        """Clean up test environment"""
        # Delete temporary directory and its contents
        if self.temp_save_dir.exists():
            shutil.rmtree(self.temp_save_dir)
        # Clear TensorFlow session
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    unittest.main()