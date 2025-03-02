import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from get_args import get_args
from utils import Logger
import sys
import datetime
from functions import functions, FunctionModel, get_target_log_prob
from pathlib import Path
from tfp_modified_kernels.hnn_leapfrog import HNNLeapfrogIntegrator
from utils import create_hnn_model


def plot_figure_3_negative_time(args, logger):
    print("\nGenerating Figure 3 using negative time steps...", file=logger)

    # Change figure size for 2x2 layout
    fig = plt.figure(figsize=(12, 10))

    # Setup parameters
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

    # Create integrators
    forward_traditional = HNNLeapfrogIntegrator(
        hnn_model=function_model,
        target_fn=lambda x: get_target_log_prob(x[:, :1], args.dist_name, args.input_dim),
        step_sizes=[step_size],
        num_steps=num_steps
    )

    backward_traditional = HNNLeapfrogIntegrator(
        hnn_model=function_model,
        target_fn=lambda x: get_target_log_prob(x[:, :1], args.dist_name, args.input_dim),
        step_sizes=[-step_size],
        num_steps=num_steps
    )

    forward_hnn = HNNLeapfrogIntegrator(
        hnn_model=hnn_model,
        target_fn=lambda x: get_target_log_prob(x[:, :1], args.dist_name, args.input_dim),
        step_sizes=[step_size],
        num_steps=num_steps
    )

    backward_hnn = HNNLeapfrogIntegrator(
        hnn_model=hnn_model,
        target_fn=lambda x: get_target_log_prob(x[:, :1], args.dist_name, args.input_dim),
        step_sizes=[-step_size],
        num_steps=num_steps
    )

    # ** Subplot 1: Traditional Time Reversibility **
    print("\nPlotting traditional time reversibility...", file=logger)
    ax1 = fig.add_subplot(221)

    initial_momenta = [0.25, -0.5]
    forward_time_points = np.linspace(0, 5, 101)  # foward sample time:0 -> 5
    backward_time_points = np.linspace(5, 0, 101)  # backward sample time:5 -> 0

    # First generate all forward trajectories
    forward_trajectories = []

    for p0 in initial_momenta:
        # Forward trajectory
        current_momentum = [tf.constant([[p0]], dtype=tf.float32)]
        current_state = [tf.constant([[0.0]], dtype=tf.float32)]

        forward_positions = []

        # Generate forward trajectory
        for _ in range(100):
            forward_positions.append(current_state[0].numpy())
            next_momentum, next_state, _, _ = forward_traditional(
                momentum_parts=current_momentum,
                state_parts=current_state
            )
            current_state = next_state
            current_momentum = next_momentum

        forward_positions.append(current_state[0].numpy())

        # Store trajectory and final state/momentum
        forward_trajectories.append({
            'positions': np.concatenate(forward_positions, axis=0).flatten(),
            'final_state': current_state,
            'final_momentum': current_momentum
        })

        # Plot forward trajectory with forward time
        ax1.plot(forward_time_points, forward_trajectories[-1]['positions'], 'b-', alpha=0.7)

    # Generate backward trajectories from each forward trajectory's endpoint
    for traj in forward_trajectories:
        current_state = traj['final_state']
        current_momentum = traj['final_momentum']

        backward_positions = []

        # Generate backward trajectory
        for _ in range(100):
            backward_positions.append(current_state[0].numpy())
            next_momentum, next_state, _, _ = backward_traditional(
                momentum_parts=current_momentum,
                state_parts=current_state
            )
            current_state = next_state
            current_momentum = next_momentum

        backward_positions.append(current_state[0].numpy())

        # Plot backward trajectory with backward time
        backward_positions = np.concatenate(backward_positions, axis=0).flatten()
        ax1.plot(backward_time_points, backward_positions, 'r--', alpha=0.7)

    ax1.plot([], [], 'b-', label='Forward time')
    ax1.plot([], [], 'r--', label='Reverse time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Position (q)')
    ax1.set_title('(a) Traditional Time Reversibility')
    ax1.legend()
    ax1.grid(False)
    ax1.set_xlim([0, 5])

    # ** Subplot 2: HNN Time Reversibility **
    print("\nPlotting HNN time reversibility...", file=logger)
    ax2 = fig.add_subplot(222)

    # First generate all forward trajectories
    forward_trajectories = []

    for p0 in initial_momenta:
        # Forward trajectory
        current_momentum = [tf.constant([[p0]], dtype=tf.float32)]
        current_state = [tf.constant([[0.0]], dtype=tf.float32)]

        forward_positions = []

        # Generate forward trajectory
        for _ in range(100):
            forward_positions.append(current_state[0].numpy())
            next_momentum, next_state, _, _ = forward_hnn(
                momentum_parts=current_momentum,
                state_parts=current_state
            )
            current_state = next_state
            current_momentum = next_momentum

        forward_positions.append(current_state[0].numpy())

        # Store trajectory and final state/momentum
        forward_trajectories.append({
            'positions': np.concatenate(forward_positions, axis=0).flatten(),
            'final_state': current_state,
            'final_momentum': current_momentum
        })

        # Plot forward trajectory with forward time
        ax2.plot(forward_time_points, forward_trajectories[-1]['positions'], 'b-', alpha=0.7)

    # Generate backward trajectories from each forward trajectory's endpoint
    for traj in forward_trajectories:
        current_state = traj['final_state']
        current_momentum = traj['final_momentum']

        backward_positions = []

        # Generate backward trajectory
        for _ in range(100):
            backward_positions.append(current_state[0].numpy())
            next_momentum, next_state, _, _ = backward_hnn(
                momentum_parts=current_momentum,
                state_parts=current_state
            )
            current_state = next_state
            current_momentum = next_momentum

        backward_positions.append(current_state[0].numpy())

        # Plot backward trajectory with backward time
        backward_positions = np.concatenate(backward_positions, axis=0).flatten()
        ax2.plot(backward_time_points, backward_positions, 'r--', alpha=0.7)

    ax2.plot([], [], 'b-', label='Forward time')
    ax2.plot([], [], 'r--', label='Reverse time')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Position (q)')
    ax2.set_title('(b) HNN Time Reversibility')
    ax2.legend()
    ax2.grid(False)
    ax2.set_xlim([0, 5])

    # ** Subplot 3: Hamiltonian Conservation (Traditional Function) **
    print("\nPlotting Hamiltonian conservation with traditional function...", file=logger)
    ax3 = fig.add_subplot(223)

    initial_momenta = [0.25, 0.5, 1.0, 1.25, 1.5]

    for p0 in initial_momenta:
        # Traditional trajectory
        current_momentum = [tf.constant([[p0]], dtype=tf.float32)]
        current_state = [tf.constant([[0.0]], dtype=tf.float32)]

        trad_states = []
        trad_momenta = []

        for _ in range(101):
            trad_states.append(current_state[0].numpy())
            trad_momenta.append(current_momentum[0].numpy())

            if _ < 100:
                next_momentum, next_state, _, _ = forward_traditional(
                    momentum_parts=current_momentum,
                    state_parts=current_state
                )
                current_state = next_state
                current_momentum = next_momentum

        # HNN trajectory
        current_momentum = [tf.constant([[p0]], dtype=tf.float32)]
        current_state = [tf.constant([[0.0]], dtype=tf.float32)]

        hnn_states = []
        hnn_momenta = []

        for _ in range(101):
            hnn_states.append(current_state[0].numpy())
            hnn_momenta.append(current_momentum[0].numpy())

            if _ < 100:
                next_momentum, next_state, _, _ = forward_hnn(
                    momentum_parts=current_momentum,
                    state_parts=current_state
                )
                current_state = next_state
                current_momentum = next_momentum

        # Compute Hamiltonians using traditional function
        trad_H = []
        hnn_H = []

        for q, p in zip(trad_states, trad_momenta):
            state = tf.constant([[q[0][0], p[0][0]]], dtype=tf.float32)
            trad_H.append(float(functions(state, args)))

        for q, p in zip(hnn_states, hnn_momenta):
            state = tf.constant([[q[0][0], p[0][0]]], dtype=tf.float32)
            hnn_H.append(float(functions(state, args)))

        # Plot Hamiltonians using traditional function
        time_points = np.linspace(0, 5, 101)
        ax3.plot(time_points, trad_H, 'b-', alpha=0.7)
        ax3.plot(time_points, hnn_H, 'r--', alpha=0.7)

    ax3.plot([], [], 'b-', label='Traditional HMC')
    ax3.plot([], [], 'r--', label='L-HNN')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Hamiltonian')
    ax3.set_title('(c) Hamiltonian Conservation\n(Traditional Function)')
    ax3.legend()
    ax3.grid(False)

    # ** Subplot 4: Hamiltonian Conservation (HNN Function) **
    print("\nPlotting Hamiltonian conservation with HNN function...", file=logger)
    ax4 = fig.add_subplot(224)

    for p0 in initial_momenta:
        # Traditional trajectory
        current_momentum = [tf.constant([[p0]], dtype=tf.float32)]
        current_state = [tf.constant([[0.0]], dtype=tf.float32)]

        trad_states = []
        trad_momenta = []

        for _ in range(101):
            trad_states.append(current_state[0].numpy())
            trad_momenta.append(current_momentum[0].numpy())

            if _ < 100:
                next_momentum, next_state, _, _ = forward_traditional(
                    momentum_parts=current_momentum,
                    state_parts=current_state
                )
                current_state = next_state
                current_momentum = next_momentum

        # HNN trajectory
        current_momentum = [tf.constant([[p0]], dtype=tf.float32)]
        current_state = [tf.constant([[0.0]], dtype=tf.float32)]

        hnn_states = []
        hnn_momenta = []

        for _ in range(101):
            hnn_states.append(current_state[0].numpy())
            hnn_momenta.append(current_momentum[0].numpy())

            if _ < 100:
                next_momentum, next_state, _, _ = forward_hnn(
                    momentum_parts=current_momentum,
                    state_parts=current_state
                )
                current_state = next_state
                current_momentum = next_momentum

        # Compute Hamiltonians using HNN model
        trad_H = []
        hnn_H = []

        for q, p in zip(trad_states, trad_momenta):
            state = tf.constant([[q[0][0], p[0][0]]], dtype=tf.float32)
            trad_H.append(float(hnn_model(state)))

        for q, p in zip(hnn_states, hnn_momenta):
            state = tf.constant([[q[0][0], p[0][0]]], dtype=tf.float32)
            hnn_H.append(float(hnn_model(state)))

        # Plot Hamiltonians
        time_points = np.linspace(0, 5, 101)
        ax4.plot(time_points, trad_H, 'b-', alpha=0.7)
        ax4.plot(time_points, hnn_H, 'r--', alpha=0.7)

    ax4.plot([], [], 'b-', label='Traditional HMC')
    ax4.plot([], [], 'r--', label='L-HNN')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Hamiltonian')
    ax4.set_title('(d) Hamiltonian Conservation\n(HNN Function)')
    ax4.legend()
    ax4.grid(False)

    # save figure
    plt.tight_layout()

    figures_dir = Path("figures")
    if not figures_dir.exists():
        figures_dir.mkdir(parents=True)
        print(f"\nCreate directory: {figures_dir}", file=logger)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = figures_dir / f'figure_3_negative_time_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nImage saved as {filename}", file=logger)

    return fig, (ax1, ax2, ax3, ax4)


if __name__ == "__main__":
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    logger = Logger()
    sys.stdout = logger

    try:
        # get and configure parameters
        args = get_args()
        args.dist_name = "1D_Gauss_mix"
        args.save_dir = 'files/1D_Gauss_mix'
        args.input_dim = 2  # 1D position + 1D momentum
        args.hidden_dim = 100
        args.latent_dim = 100
        args.nonlinearity = 'sine'

        # generate image
        fig, (ax1, ax2, ax3, ax4) = plot_figure_3_negative_time(args, logger)
        plt.show()

    finally:
        logger.close()
        sys.stdout = logger.terminal 