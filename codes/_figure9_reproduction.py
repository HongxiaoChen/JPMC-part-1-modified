import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from get_args import get_args
from pathlib import Path
import datetime
from functions import functions, get_target_log_prob
from utils import hamiltonian_wrapper, hnn_wrapper, create_hnn_model, compute_metrics, run_sampling, process_samples
from tfp_modified_kernels.tfp_hnn_nuts_online import NoUTurnSampler
from scipy.stats import gaussian_kde


def plot_figure9_comparison(samples_nuts, samples_lhnn, burn_in):
    """
    generate Figure 9: compare the sampling results of traditional NUTS and L-HNN for 5D Gaussian distribution.

    Parameters
    ----------
    samples_nuts : np.ndarray
        samples generated by traditional NUTS
    samples_lhnn : np.ndarray
        samples generated by L-HNN
    burn_in : int
        number of burn-in samples to discard
    """
    # remove burn-in samples
    samples_nuts = samples_nuts[:, burn_in:, :]
    samples_lhnn = samples_lhnn[:, burn_in:, :]
    
    # scatter plot matrix
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))

    # iterate over all dimension combinations
    for i in range(5):
        for j in range(5):
            ax = axes[i, j]

            if i == j:
                # diagonal: kernel density estimation (KDE)
                nuts_data = samples_nuts[0, :, i]
                lhnn_data = samples_lhnn[0, :, i]

                # calculate x-axis range
                min_val = min(np.min(nuts_data), np.min(lhnn_data))
                max_val = max(np.max(nuts_data), np.max(lhnn_data))
                x = np.linspace(min_val, max_val, 200)

                # KDE calculation
                nuts_kde = gaussian_kde(nuts_data)
                lhnn_kde = gaussian_kde(lhnn_data)

                # plot KDE curves
                ax.plot(x, nuts_kde(x), color='blue', alpha=0.8, label='NUTS')
                ax.plot(x, lhnn_kde(x), color='red', alpha=0.8, label='L-HNN')
            else:
                # non-diagonal: scatter plots
                ax.scatter(samples_nuts[0, :, j], samples_nuts[0, :, i],
                           alpha=0.5, color='blue', s=1, label='NUTS')
                ax.scatter(samples_lhnn[0, :, j], samples_lhnn[0, :, i],
                           alpha=0.5, color='red', s=1, label='L-HNN')

            # set axis labels
            if i == 4:  # bottom row
                ax.set_xlabel(f'q{j + 1}')
            if j == 0:  # leftmost column
                ax.set_ylabel(f'q{i + 1}')

            # remove tick labels to reduce clutter
            ax.set_xticks([])
            ax.set_yticks([])

    # add legend (only in the first subplot)
    axes[0, 0].legend()

    # adjust subplot spacing
    plt.tight_layout()

    # save the figure
    figures_dir = Path("figures")
    if not figures_dir.exists():
        figures_dir.mkdir(parents=True)
        print(f"\nCreate directory: {figures_dir}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = figures_dir / f'figure_9_comparison_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def run_comparison():
    """Run the comparison experiment and generate the figure."""
    # set the basic parameters
    args = get_args()
    args.dist_name = '5D_illconditioned_Gaussian'
    args.input_dim = 10  # 5D Gaussian (position + momentum = 10 dimensions)
    args.latent_dim = 5
    args.total_samples = 100
    args.burn_in = 50
    args.nuts_step_size = 0.025
    args.save_dir = 'files'

    # create the traditional nuts kernel
    nuts_kernel = NoUTurnSampler(
        hnn_model=lambda x: hamiltonian_wrapper(x, args, functions),
        hamiltonian_function=lambda x: hamiltonian_wrapper(x, args, functions),
        target_log_prob_fn=lambda x: get_target_log_prob(x, args.dist_name, args.input_dim),
        step_size=args.nuts_step_size,
        unrolled_leapfrog_steps=1)

    # create the lhnn model
    hnn_model = create_hnn_model(args)

    # call the model once to create variables
    dummy_input = tf.zeros([1, args.input_dim])
    _ = hnn_model(dummy_input)

    # load the weights
    hnn_model.load_weights(f"{args.save_dir}/5D_illconditioned_Gaussian250")

    # create the lhnn kernel
    lhnn_kernel = NoUTurnSampler(
        hnn_model=lambda x: hnn_wrapper(x, hnn_model),
        hamiltonian_function=lambda x: hamiltonian_wrapper(x, args, functions),
        target_log_prob_fn=lambda x: get_target_log_prob(x, args.dist_name, args.input_dim),
        step_size=args.nuts_step_size,
        unrolled_leapfrog_steps=1)

    # initial state
    initial_state = tf.zeros([1, args.input_dim//2], dtype=tf.float32)

    # run the sampling
    print("\n===== run traditional NUTS =====")
    samples_nuts, kernel_results_nuts = run_sampling(
        nuts_kernel, initial_state, args.total_samples, args.burn_in)

    print("\n===== run L-HNN =====")
    samples_lhnn, kernel_results_lhnn = run_sampling(
        lhnn_kernel, initial_state, args.total_samples, args.burn_in)

    # process the samples
    samples_nuts = process_samples(samples_nuts)
    samples_lhnn = process_samples(samples_lhnn)

    # compute and record the performance metrics
    metrics_nuts = compute_metrics(samples_nuts.numpy(), kernel_results_nuts, "Traditional_NUTS", figure_num=9, burn_in=0)
    metrics_lhnn = compute_metrics(samples_lhnn.numpy(), kernel_results_lhnn, "LHNN", figure_num=9, burn_in=0)

    # generate the comparison figure
    plot_figure9_comparison(
        samples_nuts.numpy(), 
        samples_lhnn.numpy(), 
        burn_in=0
    )

if __name__ == "__main__":
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    try:
        run_comparison()
    except Exception as e:
        print("\nError occurred:")
        print(e)
        import traceback
        traceback.print_exc() 