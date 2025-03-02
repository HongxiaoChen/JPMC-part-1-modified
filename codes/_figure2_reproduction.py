import tensorflow as tf
import matplotlib.pyplot as plt
from get_args import get_args
from utils import Logger
import sys
import datetime
from functions import functions, FunctionModel, get_target_log_prob
from utils import hamiltonian_wrapper, hnn_wrapper, create_hnn_model, run_sampling
from pathlib import Path
from tfp_modified_kernels.hnn_leapfrog import HNNLeapfrogIntegrator
from tfp_modified_kernels.tfp_hnn_hmc import HNNHMC


def set_plot_style():
    """set the basic style of matplotlib"""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.color': '#CCCCCC',
        'grid.linestyle': '--',
        'grid.alpha': 0.3
    })


def batch_integrate(integrator, p0_batch):
    """run the batch integration"""
    current_momentum = [p0_batch]
    current_state = [tf.zeros_like(p0_batch)]
    positions = [current_state[0]]
    momenta = [current_momentum[0]]

    for _ in range(100):
        next_momentum, next_state, _, _ = integrator(
            momentum_parts=current_momentum,
            state_parts=current_state
        )
        positions.append(next_state[0])
        momenta.append(next_momentum[0])
        current_state = next_state
        current_momentum = next_momentum

    return tf.stack(positions, axis=0), tf.stack(momenta, axis=0)


def plot_all_figures(args, logger):
    """plot all figures on one figure"""
    print("\nstart to plot all figures...", file=logger)

    # create a big figure, containing three subfigures
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # the first subfigure: phase space trajectory
    step_size = 0.05
    num_steps = 1

    # Initialize models
    function_model = FunctionModel(args)

    # create HNN model
    hnn_model = create_hnn_model(args)

    # call model once to create variables
    dummy_input = tf.zeros([1, args.input_dim])
    _ = hnn_model(dummy_input)

    # load weights
    print(f"\nload weights from: {args.save_dir}", file=logger)
    hnn_model.load_weights(args.save_dir)

    traditional_integrator = HNNLeapfrogIntegrator(
        hnn_model=function_model,
        target_fn=lambda x: get_target_log_prob(x[:, :1], args.dist_name, args.input_dim),
        step_sizes=[step_size],
        num_steps=num_steps
    )

    hnn_integrator = HNNLeapfrogIntegrator(
        hnn_model=hnn_model,
        target_fn=lambda x: get_target_log_prob(x[:, :1], args.dist_name, args.input_dim),
        step_sizes=[step_size],
        num_steps=num_steps
    )

    initial_momenta = tf.constant([[0.5], [1.0], [1.5], [2.0], [2.5]], dtype=tf.float32)

    trad_pos, trad_mom = batch_integrate(traditional_integrator, initial_momenta)
    hnn_pos, hnn_mom = batch_integrate(hnn_integrator, initial_momenta)

    for i in range(initial_momenta.shape[0]):
        ax1.plot(trad_pos[:, i, 0], trad_mom[:, i, 0],
                 'b-', alpha=0.7, linewidth=1.0)
        ax1.plot(hnn_pos[:, i, 0], hnn_mom[:, i, 0],
                 'r--', alpha=0.7, linewidth=1.0)

    ax1.plot([], [], 'b-', label='Traditional HMC')
    ax1.plot([], [], 'r--', label='L-HNN')
    ax1.set_xlabel('Position (q)')
    ax1.set_ylabel('Momentum (p)')
    ax1.set_title('Phase Space')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-2.5, 2.5])
    ax1.set_ylim([-4, 4])

    # the second subfigure: HNN distribution

    hnn_kernel = HNNHMC(
        step_size=[0.05],
        num_leapfrog_steps=20,
        hnn_model=lambda x: hnn_wrapper(x, hnn_model),
        target_fn=lambda x: get_target_log_prob(x, args.dist_name, args.input_dim),
        state_gradients_are_stopped=False,
        store_parameters_in_results=True,
        name='hnn_hmc'
    )

    initial_state = tf.zeros([1], dtype=tf.float32)

    samples, kernel_results = run_sampling(hnn_kernel, initial_state, args.hmc_samples, args.num_burnin)

    ax2.hist(samples.numpy().flatten(), bins=25, density=True,
             color='red', alpha=0.7, label='HNN')
    ax2.set_xlabel('Position (q)')
    ax2.set_ylabel('Density')
    ax2.set_title('HNN HMC using HNNHMC kernel')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-2, 2])

    # the third subfigure: traditional HMC distribution
    hnn_kernel = HNNHMC(
        step_size=[0.05],
        num_leapfrog_steps=20,
        hnn_model=lambda x: hamiltonian_wrapper(x, args, functions),
        target_fn=lambda x: get_target_log_prob(x, args.dist_name, args.input_dim),
        state_gradients_are_stopped=False,
        store_parameters_in_results=True,
        name='traditional_hmc'
    )

    samples, kernel_results = run_sampling(hnn_kernel, initial_state, args.hmc_samples, args.num_burnin)

    ax3.hist(samples.numpy().flatten(), bins=25, density=True,
             color='blue', alpha=0.7, label='Traditional')
    ax3.set_xlabel('Position (q)')
    ax3.set_ylabel('Density')
    ax3.set_title('Traditional HMC using HNNHMC kernel')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([-2, 2])

    # adjust the space between subfigures
    plt.tight_layout()

    # save the figure
    save_figure(fig, "comparison", logger)
    return fig


def save_figure(fig, name, logger):
    """save the figure to file"""
    figures_dir = Path("figures")
    if not figures_dir.exists():
        figures_dir.mkdir(parents=True)
        print(f"\ncreate the directory: {figures_dir}", file=logger)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = figures_dir / f'figure_2_{name}_{timestamp}.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nfigure has been saved to {filename}", file=logger)


if __name__ == "__main__":
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    logger = Logger()
    sys.stdout = logger

    try:
        # get and configure the parameters
        args = get_args()

        # configure the necessary parameters
        args.dist_name = "1D_Gauss_mix"
        args.save_dir = 'files/1D_Gauss_mix'
        args.input_dim = 2  # 1D position + 1D momentum
        args.hidden_dim = 100
        args.latent_dim = 100
        args.nonlinearity = 'sine'
        args.hmc_samples = 100
        args.num_burnin = 0

        # set the style of matplotlib
        set_plot_style()

        # generate all figures
        all_figs = plot_all_figures(args, logger)

        plt.show()

    finally:
        logger.close()
        sys.stdout = logger.terminal 