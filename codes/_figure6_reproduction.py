import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from get_args import get_args
from pathlib import Path
import datetime
from functions import functions, get_target_log_prob
from utils import hamiltonian_wrapper, hnn_wrapper, create_hnn_model, run_sampling, process_samples
from tfp_modified_kernels.tfp_hnn_nuts_online import NoUTurnSampler

def plot_figure6(errors_hnn, errors_lhnn):
    """
    generate figure 6: compare the error of hnn and lhnn
    
    Parameters
    ----------
    errors_hnn : np.ndarray
        the error of hnn sampling
    errors_lhnn : np.ndarray
        the error of lhnn sampling
    """
    plt.figure(figsize=(10, 4))

    # hnn error
    plt.subplot(121)
    plt.plot(errors_hnn[0], 'b-', linewidth=1)
    plt.axhline(y=10.0, color='r', linestyle='--', alpha=0.5)
    plt.title('hnn')
    plt.xlabel('Sample index')
    plt.ylabel('Error')
    plt.grid(True, alpha=0.3)

    # lhnn error
    plt.subplot(122)
    plt.plot(errors_lhnn[0], 'r-', linewidth=1)
    plt.axhline(y=10.0, color='r', linestyle='--', alpha=0.5)
    plt.title('lhnn')
    plt.xlabel('Sample index')
    plt.ylabel('Error')
    plt.grid(True, alpha=0.3)

    # save the figure
    figures_dir = Path("figures")
    if not figures_dir.exists():
        figures_dir.mkdir(parents=True)
        print(f"\nCreate directory: {figures_dir}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = figures_dir / f'figure_6_reproduction_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

def run_figure6():
    """run the experiment and generate figure 6"""
    # set the basic parameters
    args = get_args()
    args.dist_name = 'nD_Rosenbrock'
    args.input_dim = 6  # 3D Rosenbrock
    args.total_samples = 50
    args.burn_in = 0  
    args.nuts_step_size = 0.025
    args.hnn_error_threshold = 10.0
    args.save_dir = 'files'

    # create the hnn model
    print("\n===== run the hnn (latent_dim=1) =====")
    args.latent_dim = 1
    hnn_model = create_hnn_model(args)
    
    # call the model once to create variables
    dummy_input = tf.zeros([1, args.input_dim])
    _ = hnn_model(dummy_input)
    
    # load the weights
    hnn_model.load_weights(f"{args.save_dir}/hnn_3d_rosenbrock")
    
    # define the hamiltonian energy function
    hamiltonian_fn = lambda x: hamiltonian_wrapper(x, args, functions)

    # create the nuts kernel (with error monitoring)
    hnn_kernel = NoUTurnSampler(
        hnn_model=lambda x: hnn_wrapper(x, hnn_model),
        hamiltonian_function=lambda x: hamiltonian_wrapper(x, args, functions),
        target_log_prob_fn=lambda x: get_target_log_prob(x, args.dist_name, args.input_dim),
        step_size=args.nuts_step_size,
        max_hnn_error_threshold=args.hnn_error_threshold,
        unrolled_leapfrog_steps=1)

    # initial state
    initial_state = tf.zeros([1, args.input_dim//2], dtype=tf.float32)

    # run the hnn sampling
    samples_hnn, kernel_results_hnn = run_sampling(
        hnn_kernel, initial_state, args.total_samples, args.burn_in)

    # process the hnn samples
    processed_samples_hnn = process_samples(samples_hnn)
    
    # error from kernel results
    errors_hnn = kernel_results_hnn.hnn_error.numpy().reshape(1, -1)

    # create the lhnn model
    print("\n===== run the lhnn (latent_dim=100) =====")
    args.latent_dim = 100
    lhnn_model = create_hnn_model(args)
    
    # call the model once to create variables
    dummy_input = tf.zeros([1, args.input_dim])
    _ = lhnn_model(dummy_input)
    
    # load the weights
    lhnn_model.load_weights(f"{args.save_dir}/lhnn_3d_rosenbrock")

    # create the nuts kernel (with error monitoring)
    lhnn_kernel = NoUTurnSampler(
        hnn_model=lambda x: hnn_wrapper(x, lhnn_model),
        hamiltonian_function=lambda x: hamiltonian_wrapper(x, args, functions),
        target_log_prob_fn=lambda x: get_target_log_prob(x, args.dist_name, args.input_dim),
        step_size=args.nuts_step_size,
        max_hnn_error_threshold=args.hnn_error_threshold,
        unrolled_leapfrog_steps=1)

    # run the lhnn sampling
    samples_lhnn, kernel_results_lhnn = run_sampling(
        lhnn_kernel, initial_state, args.total_samples, args.burn_in)

    # process the lhnn samples
    processed_samples_lhnn = process_samples(samples_lhnn)
    
    # error from kernel results
    errors_lhnn = kernel_results_lhnn.hnn_error.numpy().reshape(1, -1)

    # generate the figure
    plot_figure6(errors_hnn, errors_lhnn)

    # print the statistics
    print("\nComparison statistics:")
    print(f"hnn average error: {np.mean(errors_hnn):.4f}")
    print(f"lhnn average error: {np.mean(errors_lhnn):.4f}")
    print(f"hnn acceptance rate: {tf.reduce_mean(tf.cast(kernel_results_hnn.is_accepted, tf.float32)):.4f}")
    print(f"lhnn acceptance rate: {tf.reduce_mean(tf.cast(kernel_results_lhnn.is_accepted, tf.float32)):.4f}")

if __name__ == "__main__":
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    try:
        run_figure6()
    except Exception as e:
        print("\nError occurred:")
        print(e)
        import traceback
        traceback.print_exc() 