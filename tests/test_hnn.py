import unittest
import tensorflow as tf
import numpy as np
import os
import sys
from pathlib import Path

# Get the project root path
PROJECT_ROOT = Path(__file__).parent.parent

# Add codes directory to Python path
sys.path.append(str(PROJECT_ROOT))

from codes.hnn import HNN
from codes.nn_models import MLP
from codes.get_args import get_args


class TestHNNUtils(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.args = get_args()
        self.model_configs = [
            {
                'name': 'nD_Rosenbrock100',
                'input_dim': 6,
                'latent_dim': 100,
                'dist_name': 'nD_Rosenbrock'
            },
            {
                'name': '2D_Neal_funnel250',
                'input_dim': 4,
                'latent_dim': 2,
                'dist_name': '2D_Neal_funnel'
            },
            {
                'name': '5D_illconditioned_Gaussian250',
                'input_dim': 10,
                'latent_dim': 5,
                'dist_name': '5D_illconditioned_Gaussian'
            },
            {
                'name': '10D_Rosenbrock250',
                'input_dim': 20,
                'latent_dim': 10,
                'dist_name': 'nD_Rosenbrock'
            }
        ]

    def load_model(self, config):
        """Load model with specified configuration

        Args:
            config: Dictionary containing model configuration parameters

        Returns:
            Model instance with loaded weights
        """
        self.args.input_dim = config['input_dim']
        self.args.latent_dim = config['latent_dim']
        self.args.dist_name = config['dist_name']

        # Create MLP model
        differentiable_model = MLP(
            input_dim=self.args.input_dim,
            hidden_dim=self.args.hidden_dim,
            latent_dim=self.args.latent_dim,
            nonlinearity=self.args.nonlinearity
        )

        # Create HNN model
        model = HNN(
            input_dim=self.args.input_dim,
            differentiable_model=differentiable_model
        )

        # Compile model
        model.compile(optimizer=tf.keras.optimizers.Adam(self.args.learn_rate))

        # Build complete weight file path
        weight_path = os.path.join(PROJECT_ROOT, 'codes', 'files', config['name'])
        try:
            # Check if weight file exists
            if os.path.exists(weight_path + '.index'):
                model.load_weights(weight_path)
                print(f"Successfully loaded weights for {config['name']}")
            else:
                print(f"Warning: Weight file not found for {config['name']}")
        except Exception as e:
            print(f"Error loading weights for {config['name']}: {str(e)}")

        return model

    def test_mlp(self):
        """Test MLP model"""
        for config in self.model_configs:
            with self.subTest(model_name=config['name']):
                # Test different activation functions
                for nonlinearity in ['sine', 'tanh', 'relu']:
                    mlp = MLP(
                        input_dim=config['input_dim'],
                        hidden_dim=self.args.hidden_dim,
                        latent_dim=config['latent_dim'],
                        nonlinearity=nonlinearity
                    )

                    # Test forward pass
                    x = tf.random.normal([4, config['input_dim']])
                    output = mlp(x)

                    # Check output shape
                    self.assertEqual(output.shape, (4, config['latent_dim']))

                    # Check output finiteness
                    self.assertTrue(tf.reduce_all(tf.math.is_finite(output)))

                # Test invalid activation function
                with self.assertRaises(ValueError):
                    MLP(
                        input_dim=config['input_dim'],
                        hidden_dim=self.args.hidden_dim,
                        latent_dim=config['latent_dim'],
                        nonlinearity='invalid_activation'
                    )

    def test_hnn(self):
        """Test HNN model"""
        for config in self.model_configs:
            with self.subTest(model_name=config['name']):
                model = self.load_model(config)

                # Test kinetic energy computation
                p = tf.random.normal([4, config['input_dim'] // 2])
                kinetic = model.kinetic_energy(p)
                self.assertEqual(kinetic.shape, (4,))
                self.assertTrue(tf.reduce_all(kinetic >= 0))  # Kinetic energy should be non-negative

                # Test Hamiltonian computation
                x = tf.random.normal([4, config['input_dim']])
                H = model.compute_hamiltonian(x)
                self.assertEqual(H.shape, (4, 1))
                self.assertTrue(tf.reduce_all(tf.math.is_finite(H)))

                # Test mass matrix
                # Default mass matrix
                self.assertEqual(model.M.shape, (config['input_dim'] // 2,))
                self.assertTrue(tf.reduce_all(model.M > 0))  # Mass should be positive

                # Custom mass matrix
                custom_mass = tf.ones(config['input_dim'] // 2) * 2.0
                model_custom_mass = HNN(
                    input_dim=config['input_dim'],
                    differentiable_model=model.differentiable_model,
                    mass_matrix=custom_mass
                )
                self.assertTrue(tf.reduce_all(tf.equal(model_custom_mass.M, custom_mass)))


    def tearDown(self):
        """Clean up test environment"""
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    unittest.main()
