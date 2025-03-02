import tensorflow as tf
import numpy as np
import os
from .nn_models import MLP
from .hnn import HNN
from .data import get_dataset
from .utils import L2_loss, to_pickle
from .get_args import get_args
from pathlib import Path
import logging
from datetime import datetime
import time
import gc
import multiprocessing as mp

def setup_logger(args):
    """
    Set up a logger that writes to both a file and the console.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'training_{timestamp}.log'

    logger = logging.getLogger('HNN_Training')
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info('Training started with following configuration:')
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')
    
    return logger


def train(args):
    # setup logger
    logger = setup_logger(args)
    start_time = time.time()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    logger.info('Initializing model...')
    # construct model
    latent_dim = args.latent_dim
    nn_model = MLP(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=latent_dim,
        nonlinearity=args.nonlinearity
    )
    model = HNN(args.input_dim, differentiable_model=nn_model)

    # initial leanring rate
    initial_learning_rate = args.learn_rate

    # define dacaying learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=args.decay_steps,
        decay_rate=args.decay_rate,
        staircase=True
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # prepare data
    data = get_dataset(seed=args.seed, args=args)
    x = tf.convert_to_tensor(data['coords'], dtype=tf.float32)  #return [q, p]
    dxdt = tf.convert_to_tensor(data['dcoords'], dtype=tf.float32)  #return [dq, dp]
    dataset = tf.data.Dataset.from_tensor_slices((x, dxdt))
    dataset = dataset.shuffle(buffer_size=10000).batch(args.batch_size)
    @tf.function
    def train_step(batch_x, batch_dxdt):
        with tf.GradientTape() as tape:
            with tf.GradientTape() as inner_tape:
                inner_tape.watch(batch_x)
                H = model.compute_hamiltonian(batch_x)

            # Hamiltonian gradients
            grads = inner_tape.gradient(H, batch_x)
            if grads is None:
                raise ValueError("Failed to compute gradients of Hamiltonian")

            dq_pred = grads[:, model.dim:]  # ∂H/∂p = dq/dt
            dp_pred = -grads[:, :model.dim]  # -∂H/∂q = dp/dt
            dxdt_hat = tf.concat([dq_pred, dp_pred], axis=1) # dq/dt, dp/dt from HNN
            del grads, dq_pred, dp_pred

            loss = L2_loss(batch_dxdt, dxdt_hat) #[dq dp] from traditional leapfrog, [dq, dp] from HNN

        # model gradients
        trainable_vars = model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        optimizer.apply_gradients(zip(gradients, trainable_vars))
        del tape, dxdt_hat, gradients
        return loss

    # training starts
    logger.info('Training HNN begins...')
    stats = {'train_loss': []}

    dataset_size = x.shape[0]
    for step in range(args.total_steps + 1):

        # obtain a batch from dataset
        for batch_x, batch_dxdt in dataset.take(1):
            loss = train_step(batch_x, batch_dxdt)
        stats['train_loss'].append(loss.numpy())
        if step % args.print_every == 0:
            elapsed_time = time.time() - start_time
            
            log_message = (f"Step {step}/{args.total_steps} "
                         f"({(step/args.total_steps)*100:.1f}%) - "
                         f"Loss: {loss:.4e} - "
                         f"Elapsed time: {elapsed_time:.2f}s")
            logger.info(log_message)

            # clean cache
            del batch_x, batch_dxdt, loss
            gc.collect()
            tf.keras.backend.clear_session()

        # clean cache as well
        if step % 1000 == 0:
            tf.keras.backend.clear_session()
            gc.collect()
            if hasattr(mp, 'set_start_method'):
                mp.set_start_method('spawn', force=True)

    total_time = time.time() - start_time
    logger.info(f'Training completed in {total_time:.2f} seconds')
    return model, stats


if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    # save the model weights
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.dist_name + str(args.len_sample))
    model.save_weights(save_path)

    logger = logging.getLogger('HNN_Training')
    logger.info(f'Model saved to {save_path}')