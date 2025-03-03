import unittest
import tensorflow as tf
import numpy as np
import sys
import os
import tensorflow_probability as tfp

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'codes'))  

from codes.tfp_modified_kernels.hnn_leapfrog import HNNLeapfrogIntegrator
from codes.tfp_modified_kernels.tfp_hnn_hmc import HNNHMC, UncalibratedHNNHMC
from codes.tfp_modified_kernels.tfp_hnn_nuts_online import NoUTurnSampler
from codes.utils import run_sampling, hnn_wrapper, hamiltonian_wrapper, process_samples, compute_metrics
from codes.functions import functions, FunctionModel, get_target_log_prob
from codes.hnn import HNN
from codes.nn_models import MLP


class TestHNNLeapfrogIntegrator(unittest.TestCase):
    """Test HNNLeapfrogIntegrator class"""

    def setUp(self):
        """Set up test environment"""
        # Create parameters
        class Args:
            def __init__(self):
                self.input_dim = 2  # 1D_Gauss_mix requires 2D input (1D position + 1D momentum)
                self.hidden_dim = 16
                self.output_dim = 1
                self.latent_dim = 1 
                self.learning_rate = 0.001
                self.nonlinearity = 'tanh'  
                self.batch_size = 2
                self.test_batch_size = 2
                self.num_samples = 100
                self.seed = 42
                self.dist_name = '1D_Gauss_mix'  
        
        self.args = Args()
        tf.random.set_seed(self.args.seed)
        
        # Create model
        self.function_model = FunctionModel(self.args)

        nn_model = MLP(
            input_dim=self.args.input_dim,
            hidden_dim=self.args.hidden_dim,
            latent_dim=self.args.latent_dim,
            nonlinearity=self.args.nonlinearity
        )
        self.hnn_model = HNN(
            input_dim=self.args.input_dim,
            differentiable_model=nn_model
        )
        
        # Initialize model
        dummy_input = tf.zeros([1, self.args.input_dim])
        _ = self.hnn_model(dummy_input)
        
        # Set integrator parameters
        self.step_size = 0.1
        self.num_steps = 3
        
        # Create integrator
        self.integrator = HNNLeapfrogIntegrator(
            hnn_model=self.hnn_model,
            target_fn=lambda z: get_target_log_prob(z, self.args.dist_name, self.args.input_dim),
            step_sizes=[self.step_size],
            num_steps=self.num_steps
        )

    def test_integrator_properties(self):
        """Test integrator properties"""
        self.assertEqual(self.integrator.hnn_model, self.hnn_model)
        self.assertEqual(self.integrator.step_sizes, [self.step_size])
        self.assertEqual(self.integrator.num_steps, self.num_steps)

    def test_integrator_call(self):
        """Test integrator call method"""
        # Create initial momentum and state
        batch_size = 2
        initial_momentum = [tf.random.normal([batch_size, 1])]
        initial_state = [tf.zeros_like(initial_momentum[0])]
        
        # Execute integrator call
        final_momentum, final_state, _, _ = self.integrator(
            momentum_parts=initial_momentum,
            state_parts=initial_state
        )
        
        # Verify output shape
        self.assertEqual(final_momentum[0].shape, initial_momentum[0].shape)
        self.assertEqual(final_state[0].shape, initial_state[0].shape)
    
    def test_batch_integrate(self):
        """Test batch integration"""
        batch_size = 2
        p0_batch = tf.random.normal([batch_size, 1])
        
        # Execute batch integration
        current_momentum = [p0_batch]
        current_state = [tf.zeros_like(p0_batch)]
        positions = [current_state[0]]
        momenta = [current_momentum[0]]
        
        for _ in range(5):  
            next_momentum, next_state, _, _ = self.integrator(
                momentum_parts=current_momentum,
                state_parts=current_state
            )
            positions.append(next_state[0])
            momenta.append(next_momentum[0])
            current_state = next_state
            current_momentum = next_momentum
        
        # Verify trajectory
        positions_stack = tf.stack(positions, axis=0)
        momenta_stack = tf.stack(momenta, axis=0)
        
        self.assertEqual(positions_stack.shape, (6, batch_size, 1))  # Includes initial position, 6 positions total
        self.assertEqual(momenta_stack.shape, (6, batch_size, 1))  # Includes initial momentum, 6 momenta total


class TestHNNHMC(unittest.TestCase):
    """Test HNNHMC class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create parameters
        class Args:
            def __init__(self):
                self.input_dim = 2  # 1D_Gauss_mix requires 2D input (1D position + 1D momentum)
                self.hidden_dim = 16
                self.output_dim = 1
                self.latent_dim = 1  
                self.learning_rate = 0.001
                self.nonlinearity = 'tanh'  
                self.batch_size = 2
                self.test_batch_size = 2
                self.num_samples = 100
                self.seed = 42
                self.dist_name = '1D_Gauss_mix'  
        
        self.args = Args()
        tf.random.set_seed(self.args.seed)
        
        # Create model
        self.function_model = FunctionModel(self.args)
        
        nn_model = MLP(
            input_dim=self.args.input_dim,
            hidden_dim=self.args.hidden_dim,
            latent_dim=self.args.latent_dim,
            nonlinearity=self.args.nonlinearity
        )
        self.hnn_model = HNN(
            input_dim=self.args.input_dim,
            differentiable_model=nn_model
        )
        
        # Initialize model
        dummy_input = tf.zeros([1, self.args.input_dim])
        _ = self.hnn_model(dummy_input)
        
        # Set HMC parameters
        self.step_size = 0.1
        self.num_leapfrog_steps = 3
        
        # Target function
        self.target_fn = lambda z: get_target_log_prob(z, self.args.dist_name, self.args.input_dim)
        
        # Create HMC sampler
        self.hmc = HNNHMC(
            step_size=[self.step_size],  
            num_leapfrog_steps=self.num_leapfrog_steps,
            hnn_model=self.hnn_model,
            target_fn=self.target_fn
        )
        
        # Create initial state
        self.batch_size = 2
        self.initial_state = tf.random.normal([self.batch_size, 1])

    def test_hmc_properties(self):
        """Test HMC properties"""
        self.assertEqual(self.hmc.step_size, [self.step_size])  
        self.assertEqual(self.hmc.num_leapfrog_steps, self.num_leapfrog_steps)
        self.assertEqual(self.hmc.hnn_model, self.hnn_model)
        self.assertTrue(self.hmc.is_calibrated)

    def test_bootstrap_results(self):
        """Test bootstrap_results method"""
        kernel_results = self.hmc.bootstrap_results(self.initial_state)
        
        # Verify kernel_results contains necessary fields
        self.assertIn('log_accept_ratio', dir(kernel_results))
        self.assertIn('is_accepted', dir(kernel_results))
        self.assertIn('proposed_state', dir(kernel_results))

    def test_one_step(self):
        """Test one_step method"""
        # Get initial kernel_results
        kernel_results = self.hmc.bootstrap_results(self.initial_state)
        
        # Execute one sampling step
        next_state, next_kernel_results = self.hmc.one_step(
            current_state=self.initial_state,
            previous_kernel_results=kernel_results
        )
        
        # Verify output shape
        self.assertEqual(next_state.shape, self.initial_state.shape)
        
        # Verify kernel_results updates
        self.assertIn('log_accept_ratio', dir(next_kernel_results))
        self.assertIn('is_accepted', dir(next_kernel_results))


class TestRunSampling(unittest.TestCase):
    """Test using tfp.mcmc.run_sampling to wrap HNNHMC"""
    
    def setUp(self):
        """Set up test environment"""
        # Create parameters
        class Args:
            def __init__(self):
                self.input_dim = 2  # 1D_Gauss_mix requires 2D input (1D position + 1D momentum)
                self.hidden_dim = 16
                self.output_dim = 1
                self.latent_dim = 1  
                self.learning_rate = 0.001
                self.nonlinearity = 'tanh'  
                self.batch_size = 2
                self.test_batch_size = 2
                self.num_samples = 10  
                self.seed = 42
                self.dist_name = '1D_Gauss_mix'  
        
        self.args = Args()
        tf.random.set_seed(self.args.seed)
        
        # Create model
        self.function_model = FunctionModel(self.args)

        nn_model = MLP(
            input_dim=self.args.input_dim,
            hidden_dim=self.args.hidden_dim,
            latent_dim=self.args.latent_dim,
            nonlinearity=self.args.nonlinearity
        )
        self.hnn_model = HNN(
            input_dim=self.args.input_dim,
            differentiable_model=nn_model
        )
        
        # Use lambda to wrap HNN model, consistent with _figure2_reproduction.py
        self.wrapped_hnn_model = lambda x: hnn_wrapper(x, self.hnn_model)
        
        # Initialize model
        dummy_input = tf.zeros([1, self.args.input_dim])
        _ = self.hnn_model(dummy_input)
        
        # Set HMC parameters
        self.step_size = 0.1
        self.num_leapfrog_steps = 3
        
        # Target function
        self.target_fn = lambda z: get_target_log_prob(z, self.args.dist_name, self.args.input_dim)
        
        # Create HMC sampler
        self.hmc = HNNHMC(
            step_size=[self.step_size],  
            num_leapfrog_steps=self.num_leapfrog_steps,
            hnn_model=self.wrapped_hnn_model,  # Use wrapped HNN model
            target_fn=self.target_fn,
            state_gradients_are_stopped=False,
            store_parameters_in_results=True,
            name='hnn_hmc'
        )
        
        # Create initial state
        self.initial_state = tf.zeros([1], dtype=tf.float32)
        
        # Sampling parameters
        self.total_samples = 10
        self.burn_in = 2

    def test_run_sampling(self):
        """Test run_sampling function"""
        # Execute sampling
        samples, kernel_results = run_sampling(
            kernel=self.hmc,
            initial_state=self.initial_state,
            total_samples=self.total_samples,
            burn_in=self.burn_in
        )
        
        # Verify sample shape (dimensions should match initial state)
        self.assertEqual(samples.shape, (self.total_samples, 1))
        
        # Verify kernel_results contains necessary fields
        self.assertIn('log_accept_ratio', dir(kernel_results))
        self.assertIn('is_accepted', dir(kernel_results))


class TestUncalibratedHNNHMC(unittest.TestCase):
    """Test UncalibratedHNNHMC class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create parameters
        class Args:
            def __init__(self):
                self.input_dim = 2  # 1D_Gauss_mix requires 2D input (1D position + 1D momentum)
                self.hidden_dim = 16
                self.output_dim = 1
                self.latent_dim = 1  
                self.learning_rate = 0.001
                self.nonlinearity = 'tanh'  
                self.batch_size = 2
                self.test_batch_size = 2
                self.num_samples = 100
                self.seed = 42
                self.dist_name = '1D_Gauss_mix' 
        
        self.args = Args()
        tf.random.set_seed(self.args.seed)
        
        # Create model
        self.function_model = FunctionModel(self.args)
        
        nn_model = MLP(
            input_dim=self.args.input_dim,
            hidden_dim=self.args.hidden_dim,
            latent_dim=self.args.latent_dim,
            nonlinearity=self.args.nonlinearity
        )
        self.hnn_model = HNN(
            input_dim=self.args.input_dim,
            differentiable_model=nn_model
        )
        
        # Initialize model
        dummy_input = tf.zeros([1, self.args.input_dim])
        _ = self.hnn_model(dummy_input)
        
        # Set HMC parameters
        self.step_size = 0.1
        self.num_leapfrog_steps = 3
        
        # Target function
        self.target_fn = lambda z: get_target_log_prob(z, self.args.dist_name, self.args.input_dim)
        
        # Create UncalibratedHNNHMC
        self.uncalibrated_hmc = UncalibratedHNNHMC(
            step_size=[self.step_size], 
            num_leapfrog_steps=self.num_leapfrog_steps,
            hnn_model=self.hnn_model,
            target_fn=self.target_fn
        )
        
        # Create initial state
        self.batch_size = 2
        self.initial_state = tf.random.normal([self.batch_size, 1])

    def test_uncalibrated_hmc_properties(self):
        """Test UncalibratedHNNHMC properties"""
        self.assertEqual(self.uncalibrated_hmc.step_size, [self.step_size])  
        self.assertEqual(self.uncalibrated_hmc.num_leapfrog_steps, self.num_leapfrog_steps)
        self.assertEqual(self.uncalibrated_hmc.hnn_model, self.hnn_model)
        self.assertFalse(self.uncalibrated_hmc.is_calibrated)

    def test_bootstrap_results(self):
        """Test bootstrap_results method"""
        kernel_results = self.uncalibrated_hmc.bootstrap_results(self.initial_state)
        
        # Verify kernel_results contains necessary fields
        self.assertIn('target_log_prob', dir(kernel_results))
        self.assertIn('initial_momentum', dir(kernel_results))
        self.assertIn('step_size', dir(kernel_results))

    def test_one_step(self):
        """Test one_step method"""
        # Get initial kernel_results
        kernel_results = self.uncalibrated_hmc.bootstrap_results(self.initial_state)
        
        # Execute one sampling step
        next_state, next_kernel_results = self.uncalibrated_hmc.one_step(
            current_state=self.initial_state,
            previous_kernel_results=kernel_results
        )
        
        # Verify output shape
        self.assertEqual(next_state.shape, self.initial_state.shape)
        
        # Verify kernel_results updates
        self.assertIn('target_log_prob', dir(next_kernel_results))
        self.assertIn('final_momentum', dir(next_kernel_results))


class TestNoUTurnSampler(unittest.TestCase):
    """Test NoUTurnSampler class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create parameters
        class Args:
            def __init__(self):
                self.input_dim = 6  # Using simple version of nD_Rosenbrock (3D)
                self.hidden_dim = 16
                self.output_dim = 1
                self.latent_dim = 10    
                self.learning_rate = 0.001
                self.nonlinearity = 'tanh'
                self.batch_size = 2
                self.test_batch_size = 2
                self.num_samples = 10  
                self.seed = 42
                self.dist_name = 'nD_Rosenbrock'  
                self.nuts_step_size = 0.025
        
        self.args = Args()
        tf.random.set_seed(self.args.seed)
        
        # Create models
        self.function_model = FunctionModel(self.args)
        
        # Directly create HNN model
        nn_model = MLP(
            input_dim=self.args.input_dim,
            hidden_dim=self.args.hidden_dim,
            latent_dim=self.args.latent_dim,
            nonlinearity=self.args.nonlinearity
        )
        self.hnn_model = HNN(
            input_dim=self.args.input_dim,
            differentiable_model=nn_model
        )
        
        # Initialize models
        dummy_input = tf.zeros([1, self.args.input_dim])
        _ = self.hnn_model(dummy_input)
        _ = self.function_model(dummy_input)
        
        # Wrap functions
        self.wrapped_hnn = lambda x: hnn_wrapper(x, self.hnn_model)
        self.hamiltonian_func = lambda x: hamiltonian_wrapper(x, self.args, functions)
        self.target_log_prob = lambda x: get_target_log_prob(x, self.args.dist_name, self.args.input_dim)
        
        # Create traditional NUTS sampler
        self.nuts_traditional = NoUTurnSampler(
            hnn_model=self.hamiltonian_func,  # Use traditional Hamiltonian function
            hamiltonian_function=self.hamiltonian_func,
            target_log_prob_fn=self.target_log_prob,
            step_size=self.args.nuts_step_size,
            unrolled_leapfrog_steps=1
        )
        
        # Create HNN-NUTS sampler
        self.nuts_hnn = NoUTurnSampler(
            hnn_model=self.wrapped_hnn,  # Use HNN model
            hamiltonian_function=self.hamiltonian_func,
            target_log_prob_fn=self.target_log_prob,
            step_size=self.args.nuts_step_size,
            unrolled_leapfrog_steps=1,
            max_traditional_steps=5  
        )
        
        # Create initial state (position only, no momentum)
        self.initial_state = tf.zeros([1, self.args.input_dim//2], dtype=tf.float32)
        
        # Sampling parameters
        self.total_samples = 5  
        self.burn_in = 1

    def test_nuts_properties(self):
        """Test NUTS properties"""
        # Verify traditional NUTS sampler
        self.assertEqual(self.nuts_traditional.step_size, self.args.nuts_step_size)
        self.assertEqual(self.nuts_traditional.max_tree_depth, 10)  # Default value
        
        # Verify HNN-NUTS sampler
        self.assertEqual(self.nuts_hnn.step_size, self.args.nuts_step_size)
        self.assertEqual(self.nuts_hnn.max_tree_depth, 10)  # Default value


    def test_bootstrap_results(self):
        """Test bootstrap_results method"""
        # Test traditional NUTS
        kernel_results_trad = self.nuts_traditional.bootstrap_results(self.initial_state)
        
        # Verify kernel_results contains necessary fields
        self.assertIn('target_log_prob', dir(kernel_results_trad))
        self.assertIn('step_size', dir(kernel_results_trad))
        
        # Test HNN-NUTS
        kernel_results_hnn = self.nuts_hnn.bootstrap_results(self.initial_state)
        
        # Verify kernel_results contains necessary fields
        self.assertIn('target_log_prob', dir(kernel_results_hnn))
        self.assertIn('step_size', dir(kernel_results_hnn))
        self.assertIn('use_traditional', dir(kernel_results_hnn))  # HNN-specific field

    def test_one_step(self):
        """test one_step method (only test running, not verify results)"""

        
        # Test traditional NUTS
        kernel_results_trad = self.nuts_traditional.bootstrap_results(self.initial_state)
        next_state_trad, _ = self.nuts_traditional.one_step(
            current_state=self.initial_state,
            previous_kernel_results=kernel_results_trad
        )
        
        # Verify output shape
        self.assertEqual(next_state_trad.shape, self.initial_state.shape)
        
        # Test HNN-NUTS
        kernel_results_hnn = self.nuts_hnn.bootstrap_results(self.initial_state)
        next_state_hnn, _ = self.nuts_hnn.one_step(
            current_state=self.initial_state,
            previous_kernel_results=kernel_results_hnn
        )
        
        # Verify output shape
        self.assertEqual(next_state_hnn.shape, self.initial_state.shape)

    def test_run_sampling(self):
        """test run_sampling and NUTS"""
        
        # Test traditional NUTS
        samples_trad, kernel_results_trad = run_sampling(
            kernel=self.nuts_traditional,
            initial_state=self.initial_state,
            total_samples=self.total_samples,
            burn_in=self.burn_in
        )
        
        # Verify sample shape
        self.assertEqual(samples_trad.shape, (self.total_samples, 1, self.args.input_dim//2))
        
        
        samples_hnn, kernel_results_hnn = run_sampling(
            kernel=self.nuts_hnn,
            initial_state=self.initial_state,
            total_samples=self.total_samples,
            burn_in=self.burn_in
        )
        
        # Verify sample shape
        self.assertEqual(samples_hnn.shape, (self.total_samples, 1, self.args.input_dim//2))


if __name__ == '__main__':
    unittest.main()
