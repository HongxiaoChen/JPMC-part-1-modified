import tensorflow as tf
from functions import functions
from .utils import traditional_leapfrog, to_pickle, from_pickle, dynamics_fn
from .get_args import get_args
from pathlib import Path
import logging
from datetime import datetime
import time


def setup_logger(name):
    """
    Set up a logger for tracking data generation process.

    Parameters
    ----------
    name : str
        The name of the logger.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    # log files paths
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # log file name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'data_generation_{timestamp}.log'
    
    # configure logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Check if handlers already exist to avoid duplicates
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Set formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger

logger = setup_logger('HNN_Data_Generation')

def get_trajectory(t_span=None, y0=None, dt=None, args = None, **kwargs):
    """
    Generate trajectories using TensorFlow based on Hamiltonian dynamics.

    Parameters
    ----------
    t_span : list or tuple, optional
        Time span for trajectory generation as [start_time, end_time].
        Defaults to [0, args.len_sample].

    y0 : tf.Tensor or list, optional
        Initial state of the system. If None, it will be generated randomly.
    dt : float, optional
        Time step size. Defaults to 0.025.
    **kwargs : dict
        Additional arguments passed to the dynamics function.

    Returns
    -------
    tuple
        A tuple containing:
        - traj_split : list of tf.Tensor
            Trajectory splits for each dimension (e.g., q1, q2, p1, p2).
        - deriv_split : list of tf.Tensor
            Derivative splits for each dimension (e.g., dq1/dt, dq2/dt, dp1/dt, dp2/dt).
        - t_eval : tf.Tensor
            The evaluated time points.
    """
    # Set default parameters
    if args is None:
        args = get_args()
    if dt is None:
        dt = 0.025
    if t_span is None:
        t_span = [0, args.len_sample]

    # Compute number of steps
    n_steps = int((t_span[1] - t_span[0]) / dt)
    logger.info(f"Time parameters - dt: {dt}, n_steps: {n_steps}, t_span: {t_span}")

    

    if y0 is None:
        # Generate random initial state
        y0 = tf.concat([
            tf.random.normal([args.input_dim // 2], mean=0., stddev=1.),  # position q
            tf.zeros([args.input_dim // 2])  # momentum p
        ], axis=0)

    # Convert to TensorFlow tensor
    y0 = tf.convert_to_tensor(y0, dtype=tf.float32)
    if len(y0.shape) == 1:
        y0 = tf.expand_dims(y0, 0)  # Add a batch dimension if necessary

    logger.debug(f"Generating trajectory with initial state shape: {y0.shape}")

    trajectories, derivatives = traditional_leapfrog(
        functions,
        y0,
        t_span,
        n_steps, args
    ) # return dH/dp = dq/dt, -dH/dq = dp/dt

    # Split trajectories and derivatives into separate components
    trajectories = tf.transpose(trajectories, [1, 0, 2])  # [batch_size, time_steps, dim]
    derivatives = tf.transpose(derivatives, [1, 0, 2])  # [batch_size, time_steps, dim]

    traj_split = tf.split(trajectories, args.input_dim, axis=-1)
    deriv_split = tf.split(derivatives, args.input_dim, axis=-1)


    logger.debug(f"Generated trajectory splits with shapes: {[t.shape for t in traj_split]}")

    # Generate the time points for evaluation
    t_eval = tf.linspace(t_span[0], t_span[1], n_steps + 1)
    
    return traj_split, deriv_split, t_eval


def get_dataset(seed=0, samples=None, test_split=None, args = None, t_span = None, **kwargs):
    """
    Generate or load a dataset for training and testing Hamiltonian Neural Networks.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility. Defaults to 0.
    samples : int, optional
        Number of samples to generate. Defaults to args.num_samples.
    test_split : float, optional
        Fraction of the dataset reserved for testing. Defaults to (1 - args.test_fraction).
    **kwargs : dict
        Additional arguments.

    Returns
    -------
    dict
        A dictionary containing training and testing datasets:
        - 'coords' : tf.Tensor
            Training coordinates.
        - 'dcoords' : tf.Tensor
            Training derivatives.
        - 'test_coords' : tf.Tensor
            Testing coordinates.
        - 'test_dcoords' : tf.Tensor
            Testing derivatives.
    """
    logger.info("Starting dataset generation")
    start_time = time.time()
    if args is None:
        args = get_args()
    if samples is None:
        samples = args.num_samples
    if test_split is None:
        test_split = 1.0 - args.test_fraction

    # Check if the dataset should be loaded from a file
    if args.should_load:
        path = f'{args.load_dir}/{args.load_file_name + str(args.len_sample)}.pkl'
        logger.info(f"Loading existing dataset from {path}")
        data = from_pickle(path)
        logger.info("Dataset loaded successfully")
        return data

    tf.random.set_seed(seed)
    logger.info(f"Generating new dataset with {samples} samples")
    logger.info(f"Configuration: input_dim={args.input_dim}, len_sample={args.len_sample}")

    # Initialize lists to store generated trajectories and derivatives
    xs, dxs = [], []

    # Initialize the starting state
    y_init = tf.zeros(args.input_dim, dtype=tf.float32)
    momentum_indices = [[i] for i in range(args.input_dim // 2, args.input_dim)]
    momentum_values = tf.random.normal([args.input_dim // 2], mean=0., stddev=1.)
    y_init = tf.tensor_scatter_nd_update(y_init, momentum_indices, momentum_values)


    for s in range(samples):
        sample_start_time = time.time()
        logger.info(f'Generating sample {s + 1}/{samples}')

        # Generate a trajectory for the current sample
        traj_split, deriv_split, _ = get_trajectory(y0=y_init, args=args, t_span=t_span)

        # Combine trajectory splits and derivative splits
        traj_combined = tf.stack([tf.squeeze(t) for t in traj_split], axis=1)
        deriv_combined = tf.stack([tf.squeeze(d) for d in deriv_split], axis=1)

        xs.append(traj_combined)
        dxs.append(deriv_combined)

        # Update the initial state for the next sample
        y_init = tf.zeros(args.input_dim, dtype=tf.float32)

        # Update position coordinates
        position_indices = [[i] for i in range(args.input_dim // 2)]
        position_values = traj_combined[-1, :args.input_dim // 2]
        y_init = tf.tensor_scatter_nd_update(y_init, position_indices, position_values)


        # Update momentum coordinates
        momentum_values = tf.random.normal([args.input_dim // 2], mean=0., stddev=1.)
        y_init = tf.tensor_scatter_nd_update(y_init, momentum_indices, momentum_values)

        sample_time = time.time() - sample_start_time
        logger.info(f'Sample {s + 1} completed in {sample_time:.2f} seconds')

        # Calculate and display progress
        progress = (s + 1) / samples * 100
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / (s + 1) * samples
        remaining_time = estimated_total_time - elapsed_time
        
        logger.info(f'Progress: {progress:.1f}% - '
                   f'Elapsed: {elapsed_time:.1f}s - '
                   f'Remaining: {remaining_time:.1f}s')

    logger.info("Combining and processing dataset")

    # Combine all trajectories and derivatives into a single dataset
    data = {}
    data['coords'] = tf.concat(xs, axis=0)
    data['dcoords'] = tf.concat(dxs, axis=0)

    # Split the dataset into training and testing sets
    split_ix = int(len(data['coords']) * test_split)
    split_data = {}
    for k in ['coords', 'dcoords']:
        split_data[k] = data[k][:split_ix]
        split_data['test_' + k] = data[k][split_ix:]

    # Save the dataset to a file
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    path = f'{args.save_dir}/{args.dist_name + str(args.len_sample)}.pkl'
    logger.info(f"Saving dataset to {path}")
    to_pickle(split_data, path)

    total_time = time.time() - start_time
    logger.info(f"Dataset generation completed in {total_time:.2f} seconds")
    logger.info(f"Final dataset shapes - coords: {split_data['coords'].shape}, "
                f"dcoords: {split_data['dcoords'].shape}")

    return split_data

