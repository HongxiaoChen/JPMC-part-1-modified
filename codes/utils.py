import tensorflow as tf
import numpy as np
import pickle
import logging
import tensorflow_probability as tfp
from hnn import HNN
from nn_models import MLP
import sys
import datetime
from pathlib import Path


class Logger:
    def __init__(self, log_dir="logs"):
        # create log directory
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # create log file name (use current time)
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir) / f"hmc_sampling_{current_time}.log"

        # open log file
        self.terminal = sys.stdout
        self.log = open(log_file, "w", buffering=1)  # buffering=1 means line buffering

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # ensure real-time writing

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def dynamics_fn(function, z, args):
    """
    Compute the derivatives in a Hamiltonian system.

    Args:
        function: The Hamiltonian function.
        z (tf.Tensor): Current state [batch_size, dim].
        args: Configuration arguments.

    Returns:
        tf.Tensor: Derivatives [dq/dt, dp/dt] with shape [batch_size, dim].
    """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(z)
        H = function(z, args)  # [batch_size, 1]

    grads = tape.gradient(H, z)  # [batch_size, dim]

    dim = z.shape[1] // 2
    dH_dq = grads[:, :dim]  # ∂H/∂q = -dp/dt
    dH_dp = grads[:, dim:]  # ∂H/∂p = dq/dt

    del tape
    return tf.concat([dH_dp, -dH_dq], axis=1)  # return [dq/dt, dp/dt]


def traditional_leapfrog(function: object, z0: object, t_span: object, n_steps: object, args: object) -> object:
    """
    Traditional leapfrog integrator.

    Args:
        function: The Hamiltonian function.
        z0 (tf.Tensor): Initial state [batch_size, dim].
        t_span (list): Time range [t0, t1].
        n_steps (int): Number of integration steps.
        args: Configuration arguments.

    Returns:
        tuple: Trajectories (z) and derivatives (dz) with shapes:
               z: [n_steps+1, batch_size, dim],
               dz: [n_steps+1, batch_size, dim].
    """
    # check input with correct form
    if n_steps <= 0:
        raise tf.errors.InvalidArgumentError(None, None, "Number of steps must be positive")

    if len(z0.shape) == 1:
        z0 = tf.expand_dims(z0, 0)
    dt = (t_span[1] - t_span[0]) / n_steps
    n_steps = tf.cast(n_steps, tf.int32)
    t_span = tf.cast(t_span, tf.float32)

    # initialize storage
    z = tf.TensorArray(tf.float32, size=n_steps + 1, clear_after_read=False)
    dz = tf.TensorArray(tf.float32, size=n_steps + 1, clear_after_read=False)

    # Store initial values
    z = z.write(0, z0)
    dz0 = dynamics_fn(function, z0, args)
    dz = dz.write(0, dz0)

    # Main loop
    for i in tf.range(n_steps):
        z_curr = z.read(i)
        dim = z_curr.shape[1] // 2
        q = z_curr[:, :dim]
        p = z_curr[:, dim:]

        # Compute current gradients
        dz_curr = dynamics_fn(function, z_curr, args)  # z -> H, then return dH/dp = dq/dt, -dH/dq = dp/dt
        dH_dq = -dz_curr[:, dim:]  # dH/dq = -dp/dt

        # Update position
        q_next = q + dt * p - (dt ** 2) / 2 * dH_dq

        # Compute gradients at the new position
        z_temp = tf.concat([q_next, p], axis=1)
        dz_next = dynamics_fn(function, z_temp, args)
        dH_dq_next = -dz_next[:, dim:]

        # Update momentum
        p_next = p - dt / 2 * (dH_dq + dH_dq_next)

        # Store new state
        z_next = tf.concat([q_next, p_next], axis=1)
        z = z.write(i + 1, z_next)
        dz = dz.write(i + 1, dynamics_fn(function, z_next, args))

    return z.stack(), dz.stack()  # return a list of data and deriv #return [q,p], [dq/dt, dp/dt]


@tf.function(experimental_relax_shapes=True)
def L2_loss(u, v):
    """
    Compute the L2 loss.

    Args:
        u (tf.Tensor): Predicted values.
        v (tf.Tensor): Ground truth values.

    Returns:
        tf.Tensor: L2 loss.
    """
    return tf.reduce_mean(tf.square(u - v))


def to_pickle(thing, path):
    """
    Save an object to a pickle file.

    Args:
        obj: Object to save.
        path (str): File path to save the pickle.
    """
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)


def from_pickle(path):  # load something
    """
    Load an object from a pickle file.

    Args:
        path (str): File path to load the pickle.

    Returns:
        Object: The loaded object.
    """
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing


def compute_ess(samples, burn_in):
    """
    compute the effective sample size (ess) of each dimension

    Args:
        samples (tf.Tensor): samples [num_chains, num_samples, dim]
        burn_in (int): the number of burn-in samples

    Returns:
        list: the ess of each dimension
    """
    ess_values = []
    dim = samples.shape[-1]
    for d in range(dim):
        chain_samples = samples[0, burn_in:, d]
        ess = tfp.mcmc.effective_sample_size(chain_samples)
        ess_values.append(float(ess.numpy()))
    return ess_values


def hamiltonian_wrapper(coords, args, functions):
    """wrap the functions function, ensure the output dimension is (batch_size,)"""
    if len(coords.shape) == 1:
        coords = tf.expand_dims(coords, 0)  # add batch dimension
    H = functions(coords, args)  # [batch_size, 1]
    return 1.0 * tf.squeeze(H, axis=-1)  # [batch_size]


def hnn_wrapper(coords, hnn_model):
    """wrap the hnn model, ensure the output dimension is (batch_size,)"""
    # ensure the input is a 2D tensor
    if len(coords.shape) == 1:
        coords = tf.expand_dims(coords, 0)  # add batch dimension
    H = hnn_model(coords)  # [batch_size, 1]
    return 1.0 * tf.squeeze(H, axis=-1)  # [batch_size]


def create_hnn_model(args):
    """create the hnn model, but not load the weights"""
    nn_model = MLP(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        nonlinearity=args.nonlinearity
    )

    model = HNN(
        input_dim=args.input_dim,
        differentiable_model=nn_model
    )
    return model


def compute_metrics(samples, kernel_results, method_name, figure_num=None, burn_in=0):
    """
    compute the performance metrics of the sampling
    
    Parameters
    ----------
    samples : np.ndarray
        samples
    kernel_results : object
        the kernel results
    method_name : str
        the name of the sampling method
    figure_num : int, optional
        the figure number (for the log file name), by default None
    burn_in : int, optional
        the number of burn-in samples, by default 0
        
    Returns
    -------
    dict
        contains the performance metrics
    """
    # calculate the ess
    ess = compute_ess(samples, burn_in)

    # calculate the number of gradient computations
    leapfrogs_taken = kernel_results.leapfrogs_taken.numpy().sum()
    total_grads = leapfrogs_taken

    leapfrogs_taken_traditional = kernel_results.leapfrogs_taken_trad.numpy().sum()
    total_grads_traditional = leapfrogs_taken_traditional

    # create the log directory
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # create the log file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if figure_num is not None:
        log_file = logs_dir / f'figure{figure_num}_metrics_{method_name}_{timestamp}.log'
    else:
        log_file = logs_dir / f'metrics_{method_name}_{timestamp}.log'

    # record the metrics
    with open(log_file, 'w') as f:
        f.write(f"=== {method_name} Performance Metrics ===\n")
        f.write(f"ESS: ({', '.join(f'{x:.2f}' for x in ess)})\n")
        f.write(f"Total gradient computations: {total_grads}\n")
        f.write(f"Total gradient computations (traditional): {total_grads_traditional}\n")
        f.write(f"Average ESS: {np.mean(ess):.2f}\n")
        f.write(f"ESS per gradient: {np.mean(ess) / total_grads:.6f}\n")

    return {
        'ess': ess,
        'total_grads': total_grads,
        'avg_ess': tf.reduce_mean(ess),
        'ess_per_grad': tf.reduce_mean(ess) / total_grads
    }


@tf.function(experimental_relax_shapes=True)
def run_sampling(kernel, initial_state, total_samples, burn_in):
    """
    run the mcmc sampling
    
    Parameters
    ----------
    kernel : object
        mcmc kernel
    initial_state : tf.Tensor
        initial state
    total_samples : int
        total number of samples
    burn_in : int
        burn-in samples
        
    Returns
    -------
    tuple
        samples and kernel results
    """
    return tfp.mcmc.sample_chain(
        num_results=total_samples,
        current_state=initial_state,
        kernel=kernel,
        num_burnin_steps=burn_in,
        trace_fn=lambda _, pkr: pkr)


@tf.function(experimental_relax_shapes=True)
def process_samples(samples):
    """
    return the samples in the shape of [num_samples, num_chains, dim]
    
    Parameters
    ----------
    samples : tf.Tensor
        samples: [num_chains, num_samples, dim]
        
    Returns
    -------
    tf.Tensor
        samples: [num_samples, num_chains, dim]
    """
    return tf.transpose(samples, perm=[1, 0, 2])
