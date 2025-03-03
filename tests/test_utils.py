import unittest
import os
import tensorflow as tf
import tempfile
import sys


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'codes'))


from codes.utils import (dynamics_fn, traditional_leapfrog, L2_loss,
                        to_pickle, from_pickle, compute_ess, hamiltonian_wrapper,
                        hnn_wrapper)
from codes.functions import functions
from codes.get_args import get_args





class TestUtils(unittest.TestCase):
    def setUp(self):
        self.args = get_args()
        self.args.dist_name = '1D_Gauss_mix'
        self.args.input_dim = 2

        
        self.model_configs = {
            'nD_Rosenbrock100': {'input_dim': 200},
            '2D_Neal_funnel250': {'input_dim': 4},
            '5D_illconditioned_Gaussian250': {'input_dim': 10},
            '10D_Rosenbrock250': {'input_dim': 20}
        }

        self.differentiable_models = {}
        for config in self.model_configs.values():
            class SimpleModel(tf.keras.Model):
                def __init__(self, input_dim):
                    super(SimpleModel, self).__init__()
                    self.input_dim = input_dim
                    self.dense1 = tf.keras.layers.Dense(32, activation='relu', input_shape=(input_dim,))
                    self.dense2 = tf.keras.layers.Dense(1)

                def call(self, inputs):
                    x = self.dense1(inputs)
                    return self.dense2(x)

            self.differentiable_models[config['input_dim']] = SimpleModel(config['input_dim'])

        self.test_batch_size = 32

    def test_dynamics_fn(self):
        test_cases = [
            ('1D_Gauss_mix', 2),
            ('2D_Neal_funnel', 4),
            ('5D_illconditioned_Gaussian', 10),
            ('nD_Rosenbrock', 6),
            ('Allen_Cahn', 50),
            ('Elliptic', 100)
        ]

        for dist_name, input_dim in test_cases:
            with self.subTest(dist_name=dist_name):
                self.args.dist_name = dist_name
                self.args.input_dim = input_dim

                z = tf.constant([[1.0] * input_dim], dtype=tf.float32)
                derivatives = dynamics_fn(lambda x, args: functions(x, args), z, self.args)
                self.assertEqual(derivatives.shape, z.shape)
                self.assertTrue(tf.reduce_all(tf.math.is_finite(derivatives)))

    def test_traditional_leapfrog(self):
        z0 = tf.constant([[1.0, 0.5]], dtype=tf.float32)
        t_span = [0.0, 1.0]
        n_steps = 10

        traj, derivatives = traditional_leapfrog(
            lambda x, args: functions(x, args),
            z0, t_span, n_steps, self.args
        )

        self.assertEqual(traj.shape, (n_steps + 1, z0.shape[0], z0.shape[1]))
        self.assertEqual(derivatives.shape, (n_steps + 1, z0.shape[0], z0.shape[1]))

        diff = tf.reduce_max(tf.abs(traj[1:] - traj[:-1]))
        self.assertLess(diff, 1.0)

        diff_thresholds = {
            '1D_Gauss_mix': 1.0,
            '2D_Neal_funnel': 2.0,
            '5D_illconditioned_Gaussian': 10.0,
            'nD_Rosenbrock': 5.0
        }

        test_cases = [
            ('1D_Gauss_mix', 2),
            ('2D_Neal_funnel', 4),
            ('5D_illconditioned_Gaussian', 10),
            ('nD_Rosenbrock', 6)
        ]

        for dist_name, input_dim in test_cases:
            with self.subTest(dist_name=dist_name):
                self.args.dist_name = dist_name
                self.args.input_dim = input_dim

                input_shapes = [
                    (tf.constant([[1.0] * input_dim], dtype=tf.float32), "batch_shape"),
                    (tf.constant([1.0] * input_dim, dtype=tf.float32), "single_shape")
                ]

                for z0_test, shape_type in input_shapes:
                    traj, derivatives = traditional_leapfrog(
                        lambda x, args: functions(x, args),
                        z0_test, t_span, n_steps, self.args
                    )

                    expected_shape = (n_steps + 1, 1, input_dim)
                    self.assertEqual(traj.shape, expected_shape)
                    diff = tf.reduce_max(tf.abs(traj[1:] - traj[:-1]))
                    self.assertLess(diff, diff_thresholds[dist_name])
                    self.assertTrue(tf.reduce_all(tf.math.is_finite(derivatives)))

                    if dist_name in ['1D_Gauss_mix', '2D_Neal_funnel']:
                        initial_energy = functions(traj[0], self.args)
                        final_energy = functions(traj[-1], self.args)
                        energy_diff = tf.abs(final_energy - initial_energy)
                        self.assertLess(float(energy_diff), 0.1)

    def test_traditional_leapfrog_invalid_inputs(self):
        z0 = tf.constant([[1.0, 0.5]], dtype=tf.float32)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            traditional_leapfrog(
                lambda x, args: functions(x, args),
                z0, [0.0, 1.0], -1, self.args
            )

        with self.assertRaises(tf.errors.InvalidArgumentError):
            traditional_leapfrog(
                lambda x, args: functions(x, args),
                z0, [0.0, 1.0], 0, self.args
            )

    def test_L2_loss(self):
        u = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        v = tf.constant([[1.1, 2.1], [3.1, 4.1]])

        loss = L2_loss(u, v)

        self.assertEqual(loss.shape, ())
        self.assertGreater(loss, 0)
        expected_loss = ((1 - 1.1) ** 2 + (2 - 2.1) ** 2 + (3 - 3.1) ** 2 + (4 - 4.1) ** 2) / 4
        self.assertAlmostEqual(float(loss), expected_loss, places=6)

    def test_pickle_operations(self):
        test_data = {"test": "data"}
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            to_pickle(test_data, tmp.name)
            loaded_data = from_pickle(tmp.name)
        self.assertEqual(test_data, loaded_data)
        os.unlink(tmp.name)

    def test_ess_computations(self):
        test_dims = [2, 4, 6, 8]
        for dim in test_dims:
            samples = tf.random.normal([1, 1000, dim])
            burn_in = 100
            ess_values = compute_ess(samples, burn_in)
            self.assertEqual(len(ess_values), dim)

    def test_hamiltonian_wrapper(self):
        # Test with 1D input
        coords_1d = tf.constant([1.0, 2.0], dtype=tf.float32)
        output_1d = hamiltonian_wrapper(coords_1d, self.args, functions)
        self.assertEqual(output_1d.shape, (1,))

        # Test with batch input
        coords_batch = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
        output_batch = hamiltonian_wrapper(coords_batch, self.args, functions)
        self.assertEqual(output_batch.shape, (2,))

        # Test output is scalar per sample
        self.assertEqual(len(output_batch.shape), 1)

    def test_hnn_wrapper(self):
        # Create a simple HNN model for testing
        input_dim = 4
        model = self.differentiable_models[input_dim]

        # Test with 1D input
        coords_1d = tf.constant([1.0] * input_dim, dtype=tf.float32)
        output_1d = hnn_wrapper(coords_1d, model)
        self.assertEqual(output_1d.shape, (1,))

        # Test with batch input
        batch_size = 3
        coords_batch = tf.constant([[1.0] * input_dim] * batch_size, dtype=tf.float32)
        output_batch = hnn_wrapper(coords_batch, model)
        self.assertEqual(output_batch.shape, (batch_size,))

        # Test output is scalar per sample
        self.assertEqual(len(output_batch.shape), 1)

    def tearDown(self):
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    unittest.main()
