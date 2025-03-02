import tensorflow as tf
import numpy as np



@tf.function
def get_target_log_prob(state_parts, dist_name, input_dim):
    """
    Calculate the log probability of the target distribution.
    -----------
    state_parts : tf.Tensor
        Position coordinates, can be shape [dim] (single chain) or [batch_size, dim] (multiple chains)
    dist_name : str
        Distribution name
    input_dim : int
        Input dimension (total dimension of position + momentum). If only position dimension is used internally, can be simplified as needed.
    Returns:
        tf.Tensor: Log probability, shape=() for single chain or shape=[batch_size] for multiple chains
    """
    # convert state_parts to tensor (in case the input is numpy or other types)
    state_parts = tf.convert_to_tensor(state_parts, dtype=tf.float32)

    # determine the input dimension: if one-dimensional (dim,) is regarded as a single chain, it needs to be extended to (1, dim)
    is_1d_input = (state_parts.ndim == 1)
    if is_1d_input:
        state_parts = state_parts[tf.newaxis, :]  # shape (1, dim)

    # momentum is the same shape as state_parts
    momentum = tf.zeros_like(state_parts)  # shape (batch_size, dim)

    # concatenate position and momentum, get the shape (batch_size, 2*dim)
    coords = tf.concat([state_parts, momentum], axis=-1)

    # create a parameter object (if indeed needed to parse dist_name and input_dim inside functions)
    class Args:
        pass

    args = Args()
    args.dist_name = dist_name
    args.input_dim = input_dim

    # assume the function returns shape (batch_size,) or (batch_size, 1)
    # here take the negative of the Hamiltonian as the log probability: log_prob = -H
    # please replace "functions" with your actual calculation function
    hamiltonian = -functions(coords, args)  # shape (batch_size,) / (batch_size,1)

    # if hamiltonian is (batch_size, 1), squeeze the last dimension
    if hamiltonian.ndim == 2 and hamiltonian.shape[-1] == 1:
        hamiltonian = tf.squeeze(hamiltonian, axis=-1)  # -> shape (batch_size,)

    # if the original input is one-dimensional, then now hamiltonian shape=(1,); squeeze it to a scalar ()
    if is_1d_input:
        hamiltonian = tf.squeeze(hamiltonian, axis=0)  # -> shape ()

    return hamiltonian  # so it is log_prob


@tf.function
def f_tf(x, y):
    """
    calculate f(x,y)
    """
    term1 = 2 * tf.cos(2 * x) - (x + y) * 4 * tf.sin(2 * x)
    term2 = 2 * tf.cos(2 * y) - (x + y) * 4 * tf.sin(2 * y)
    return term1 + term2


@tf.function(experimental_relax_shapes=True)
def nearest_neighbor_derivative(x_samples, y_samples, g_values):
    """
    Compute partial derivatives using the nearest neighbor method.

    Parameters

    """
    n_points = tf.shape(x_samples)[1]

    # Extract data for batch dimension 1
    x = x_samples[0]  # Shape: [50]
    y = y_samples[0]  # Shape: [50]
    g = g_values[0]  # Shape: [50]

    # Compute distances between points
    x_diff = tf.expand_dims(x, 1) - tf.expand_dims(x, 0)  # Shape: [50, 50]
    y_diff = tf.expand_dims(y, 1) - tf.expand_dims(y, 0)  # Shape: [50, 50]
    g_diff = tf.expand_dims(g, 1) - tf.expand_dims(g, 0)  # Shape: [50, 50]

    # Find nearest neighbors (exclude self-points using a large mask)
    x_dist = tf.abs(x_diff) + tf.eye(n_points) * 1e8
    y_dist = tf.abs(y_diff) + tf.eye(n_points) * 1e8

    # Nearest neighbor indices
    x_nearest_idx = tf.argmin(x_dist, axis=1)  # Shape: [50]
    y_nearest_idx = tf.argmin(y_dist, axis=1)  # Shape: [50]

    # Compute differences for nearest neighbors
    x_nearest_diff = tf.gather(g_diff, x_nearest_idx, batch_dims=1)  # Shape: [50]
    y_nearest_diff = tf.gather(g_diff, y_nearest_idx, batch_dims=1)  # Shape: [50]

    dx = tf.gather(x_diff, x_nearest_idx, batch_dims=1)  # Shape: [50]
    dy = tf.gather(y_diff, y_nearest_idx, batch_dims=1)  # Shape: [50]

    # Compute partial derivatives
    d_g_dx = x_nearest_diff / (dx + 1e-8)  # Shape: [50]
    d_g_dy = y_nearest_diff / (dy + 1e-8)  # Shape: [50]

    # Add back batch dimension
    return tf.expand_dims(d_g_dx, 0), tf.expand_dims(d_g_dy, 0)  # Shapes: [1, 50], [1, 50]


@tf.function(experimental_relax_shapes=True)
def compute_f_hat_with_nearest_neighbor(x_samples, y_samples, q, u_x, u_y):
    """
    Compute predicted values f_hat using the nearest neighbor method
    """
    q = tf.clip_by_value(q, -10.0, 10.0)
    # Compute k(x, y) * du/dx and k(x, y) * du/dy
    ku_x = tf.clip_by_value(q * u_x, -1e6, 1e6)
    ku_y = tf.clip_by_value(q * u_y, -1e6, 1e6)

    # Compute derivatives
    d_ku_dx, _ = nearest_neighbor_derivative(x_samples, y_samples, ku_x)  # Shape: [1, 50]
    _, d_ku_dy = nearest_neighbor_derivative(x_samples, y_samples, ku_y)  # Shape: [1, 50]

    # Compute and clip f_hat
    f_hat = tf.clip_by_value(d_ku_dx + d_ku_dy, -200.0, 200.0)
    return f_hat  # Shape: [1, 50]


def f_obs():
    """
    Generate noisy observations f_obs for sampled positions.

    Returns
    -------
    tuple
        Noisy observations (f_samples_noisy), x_samples, and y_samples,
        each with shape [1, 50].
    """

    n_points = 50
    np.random.seed(40)
    x_samples = np.random.uniform(0, 3, n_points)

    np.random.seed(122)
    y_samples = np.random.uniform(0, 3, n_points)

    np.random.seed(36)
    noise = np.random.normal(0, 1, n_points)

    # Compute f(x, y) for the sampled positions
    x_samples_tf = tf.constant(x_samples, dtype=tf.float32)
    y_samples_tf = tf.constant(y_samples, dtype=tf.float32)
    f_samples = f_tf(x_samples_tf, y_samples_tf)

    # Add noise
    noise_tf = tf.constant(noise, dtype=tf.float32)
    f_samples_noisy = f_samples + noise_tf

    return (tf.expand_dims(f_samples_noisy, 0),
            tf.expand_dims(x_samples_tf, 0),
            tf.expand_dims(y_samples_tf, 0))


def functions(coords, args):
    """
    Compute the Hamiltonian H = U(q) + K(p).

    - U(q): Potential energy from the target density.
    - K(p): Kinetic energy = Σ(p² / 2m).

    Parameters
    ----------
    coords : tf.Tensor
        Position and momentum coordinates, shape [batch_size, input_dim].
    args : argparse.Namespace
        Arguments containing distribution name and input dimensions.

    Returns
    -------
    tf.Tensor
        Hamiltonian values with shape [batch_size, 1].
    """
    coords = tf.cast(coords, tf.float32)
    if len(coords.shape) == 1:
        coords = tf.expand_dims(coords, 0)  # [input_dim] -> [1, input_dim]

    # ******** 1D Gaussian Mixture #********
    if (args.dist_name == '1D_Gauss_mix'):
        # separate q and p
        q, p = tf.split(coords, 2, axis=1)  # q,p are all [batch_size, 1]

        # U(q)
        mu1, mu2 = 1.0, -1.0
        sigma = 0.35
        likelihood = 0.5 * (tf.exp(-(q - mu1) ** 2 / (2 * sigma ** 2)) +
                            tf.exp(-(q - mu2) ** 2 / (2 * sigma ** 2)))
        U = -tf.math.log(likelihood)  # [batch_size, 1]

        # K(p)
        K = 0.5 * tf.square(p)  # [batch_size, 1]

        return U + K  # [batch_size, 1]

    # ******** 2D Neal's Funnel #********
    elif (args.dist_name == '2D_Neal_funnel'):
        dim = args.input_dim // 2
        q = coords[:, :dim]  # [batch_size, 2]
        p = coords[:, dim:]  # [batch_size, 2]

        # separate q1,q2
        q1, q2 = tf.split(q, 2, axis=1)  # [batch_size, 1]
        p1, p2 = tf.split(p, 2, axis=1)  # [batch_size, 1]

        # potential energy
        U1 = 0.5 * tf.square(q1) / (3 ** 2)  # [batch_size, 1]
        U2 = 0.5 * tf.square(q2) / tf.exp(q1) + 0.5 * q1  # [batch_size, 1]
        U = U1 + U2  # [batch_size, 1]

        # K(p)
        K = 0.5 * (tf.square(p1) + tf.square(p2))  # [batch_size, 1]

        return U + K  # [batch_size, 1]

    # ******** 5D Ill-Conditioned Gaussian #********
    elif (args.dist_name == '5D_illconditioned_Gaussian'):
        # separate p and q
        dim = args.input_dim // 2
        q = coords[:, :dim]  # [batch_size, 5]
        p = coords[:, dim:]  # [batch_size, 5]

        # U(q)
        var = tf.constant([0.01, 0.1, 1.0, 10.0, 100.0])
        var = tf.reshape(var, [1, -1])  # [1, 5]
        U = 0.5 * tf.reduce_sum(tf.square(q) / var, axis=1, keepdims=True)  # [batch_size, 1]

        # K(p)
        K = 0.5 * tf.reduce_sum(tf.square(p), axis=1, keepdims=True)  # [batch_size, 1]

        return U + K  # [batch_size, 1]

    # ******** nD Rosenbrock #********
    elif (args.dist_name == 'nD_Rosenbrock'):
        dim = args.input_dim // 2
        q = coords[:, :dim]  # [batch_size, dim]
        p = coords[:, dim:]  # [batch_size, dim]

        # positions
        q_next = q[:, 1:]  # [batch_size, dim-1]
        q_curr = q[:, :-1]  # [batch_size, dim-1]

        # (q_{i+1} - q_i²)²
        term1 = 100.0 * tf.square(q_next - tf.square(q_curr))

        # (1-q_i)²
        term2 = tf.square(1.0 - q_curr)

        # U(q)
        U = tf.reduce_sum(term1 + term2, axis=1, keepdims=True) / 20.0

        # K(p)
        K = 0.5 * tf.reduce_sum(tf.square(p), axis=1, keepdims=True)

        return U + K  # [batch_size, 1]

    # ******** Allen-Cahn Stochastic PDE #********
    elif (args.dist_name == 'Allen_Cahn'):
        # separate p and q
        dim = args.input_dim // 2  # dim should be 25
        q = coords[:, :dim]  # [batch_size, 25]
        p = coords[:, dim:]  # [batch_size, 25]

        # interval length
        dx = 1.0 / dim

        # (u(i∆x + ∆x) − u(i∆x))²/(2∆x)
        q_next = q[:, 1:]  # [batch_size, dim-1]
        q_curr = q[:, :-1]  # [batch_size, dim-1]

        diff_term = tf.square(q_next - q_curr) / (2 * dx)

        # V(u) = (1-u²)²
        V = tf.square(1.0 - tf.square(q))

        # U(q)
        U = tf.reduce_sum(diff_term, axis=1, keepdims=True) + 0.5 * dx * tf.reduce_sum(V, axis=1, keepdims=True)

        # K(p)
        K = 0.5 * tf.reduce_sum(tf.square(p), axis=1, keepdims=True)

        return U + K  # [batch_size, 1]

        # ******** New Distribution: f_obs_mu ********
    elif (args.dist_name == 'Elliptic'):
        # separate p and q
        dim = args.input_dim // 2  # dim = 50
        q = coords[:, :dim]  # [1, 50]
        p = coords[:, dim:]  # [1, 50]

        # generate noise = f(x,y) + noise
        f_obs_values, x_samples, y_samples = f_obs()  # [1,50]

        # deriv of U
        u_x = tf.cos(2 * x_samples) * 2  # [1,50]
        u_y = tf.cos(2 * y_samples) * 2  # [1,50]

        f_hat = compute_f_hat_with_nearest_neighbor(
            x_samples, y_samples, q, u_x, u_y
        )  # [1,50]

        # U(q)
        diff = f_obs_values - f_hat  # [1,50]
        U = 0.5 * tf.reduce_sum(tf.square(diff), axis=1, keepdims=True)  # [1,1]

        # K(p)
        K = 0.5 * tf.reduce_sum(tf.square(p), axis=1, keepdims=True)  # [1,1]

        return U + K  # [1,1]
    else:
        raise ValueError(f"probability distribution name {args.dist_name} not recognized")


class FunctionModel(tf.keras.Model):
    """wrap the functions so that it has the same interface as HNN"""

    def __init__(self, args):
        super(FunctionModel, self).__init__()
        self.args = args
        self.M = 1.0  # unit mass matrix

    def call(self, x):
        """keep the same interface as HNN"""
        return functions(x, self.args)

    def compute_hamiltonian(self, x):
        """keep the same interface as HNN"""
        return functions(x, self.args)
