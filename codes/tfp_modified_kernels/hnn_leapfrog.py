import tensorflow.compat.v2 as tf
from tensorflow_probability.python.mcmc.internal import leapfrog_integrator as leapfrog_impl

def _one_step(hnn_model, target_fn, step_sizes, momentum_parts, state_parts, target, H_grad_parts):
    """Execute one leapfrog step."""
    with tf.name_scope('hnn_leapfrog_integrate_one_step'):
        # Get current state
        momentum = momentum_parts[0]  # shape: [batch_size]
        state = state_parts[0]        # shape: [batch_size]
        dt = step_sizes[0]
        M = tf.cast(hnn_model.M, dtype=state.dtype)

        # Update position
        next_state = state + dt / M * momentum - (dt ** 2) / (2 * M) * H_grad_parts[0]

        # Compute ∂H/∂q(t+∆t) using HNN for gradients
        z_next = tf.concat([next_state, momentum], axis=-1)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(z_next)
            next_H = hnn_model(z_next)  # shape: [batch_size,1]
        next_H_grads = tape.gradient(next_H, z_next)
        dim = state.shape[-1] 
        next_H_grad_parts = [next_H_grads[..., :dim]]  # ∂H/∂q(t+∆t)

        # Update momentum
        next_momentum = momentum - dt / 2 * (H_grad_parts[0] + next_H_grad_parts[0])

        next_target_return = target_fn(next_state)

        del tape

        return [
            [next_momentum],      # next_momentum_parts
            [next_state],         # next_state_parts
            next_target_return,        # next_target_log_prob
            next_H_grad_parts,    # next_target_grad_parts
        ]

def process_args(hnn_model, target_fn, momentum_parts, state_parts, H=None, H_grad_parts=None):
    """Process and validate input arguments."""
    with tf.name_scope('process_args'):
        # Convert to tensors
        momentum_parts = [
            tf.convert_to_tensor(v, dtype=tf.float32, name='momentum')
            for v in momentum_parts
        ]
        state_parts = [
            tf.convert_to_tensor(v, dtype=tf.float32, name='state')
            for v in state_parts
        ]

        # Get current state
        momentum = momentum_parts[0]  # shape: [batch_size]
        state = state_parts[0]        # shape: [batch_size]
        
        # obtain the target value
        target = target_fn(state)

        # Compute Hamiltonian and gradients if not provided
        if H is None or H_grad_parts is None:
            z = tf.concat([state, momentum], axis=-1)  # shape: [batch_size, 2]
            # Use HNN for gradients
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(z)
                H_for_grad = hnn_model(z)  # shape: [batch_size]
            grads = tape.gradient(H_for_grad, z)
            dim = state.shape[-1]
            H_grad_parts = [grads[..., :dim]]  # shape: [batch_size]
            
            
            del tape

        return momentum_parts, state_parts, target, H_grad_parts

class HNNLeapfrogIntegrator(leapfrog_impl.LeapfrogIntegrator):
    """Leapfrog integrator using HNN for Hamiltonian computation."""

    def __init__(self, hnn_model, target_fn, step_sizes, num_steps):
        """Initialize HNN Leapfrog integrator.

        Args:
            hnn_model: HNN model for computing gradients
            hamiltonian_function: Function for computing actual Hamiltonian values
            step_sizes: Step sizes for the leapfrog integrator
            num_steps: Number of leapfrog steps
        """
        self._hnn_model = hnn_model
        self._target_fn = target_fn
        self._step_sizes = step_sizes
        self._num_steps = num_steps
        if not hasattr(self._hnn_model, 'M'):
            self._hnn_model.M = 1.0

    @property
    def hnn_model(self):
        return self._hnn_model

    @property
    def target_fn(self):
        return self._target_fn

    @property
    def step_sizes(self):
        return self._step_sizes

    @property
    def num_steps(self):
        return self._num_steps

    def __call__(self, momentum_parts, state_parts, target=None,
                 target_grad_parts=None, kinetic_energy_fn=None, name=None):
        """Execute Leapfrog integration."""
        with tf.name_scope(name or 'hnn_leapfrog_integrate'):
            # Process input arguments
            [
                momentum_parts,
                state_parts,
                H,
                H_grad_parts,
            ] = process_args(
                self.hnn_model,
                self.target_fn,
                momentum_parts,
                state_parts,
                target,             
                target_grad_parts   # Hamiltonian gradients w.r.t. states
            )
            
            
            if H is None:
                H = self.target_fn(state_parts[0])
            
            
            if H_grad_parts is None or any(g is None for g in H_grad_parts):
                z = tf.concat([state_parts[0], momentum_parts[0]], axis=-1)
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(z)
                    H_val = self.hnn_model(z)
                grads = tape.gradient(H_val, z)
                dim = state_parts[0].shape[-1]
                H_grad_parts = [grads[..., :dim]]
                del tape
            
            
            num_steps = tf.constant(self.num_steps, dtype=tf.int32)
            
            # Multiple steps of leapfrog integration
            [
                _,
                next_momentum_parts,
                next_state_parts,
                next_target,
                next_H_grad_parts,
            ] = tf.while_loop(
                cond=lambda i, *_: i < num_steps,
                body=lambda i, *args: [i + 1] + list(_one_step(
                    self.hnn_model, self.target_fn, self.step_sizes, *args)),
                loop_vars=[
                    tf.zeros([], dtype=tf.int32, name='iter'), 
                    momentum_parts,
                    state_parts,
                    H,
                    H_grad_parts,
                ])

            return (
                next_momentum_parts,
                next_state_parts,
                next_target,
                next_H_grad_parts
            )
