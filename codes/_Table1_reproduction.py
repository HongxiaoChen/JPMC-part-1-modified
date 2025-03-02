import tensorflow as tf
import numpy as np
from pathlib import Path
import datetime
import time
from get_args import get_args
import tensorflow_probability as tfp

# Import necessary modules
from functions import functions, FunctionModel, get_target_log_prob
from utils import hamiltonian_wrapper, hnn_wrapper, create_hnn_model, run_sampling
from tfp_modified_kernels.tfp_hnn_hmc import HNNHMC
from tfp_modified_kernels.tfp_hnn_nuts import NoUTurnSampler


def compute_metrics(samples, kernel_results, method_name, args, step_size, num_leapfrog_steps):
    """Calculate sampling performance metrics without writing to log file"""
    print(f"\ncalculate {method_name} performance metrics...")

    # Calculate ESS
    try:
        ess = tfp.mcmc.effective_sample_size(samples)

        # Convert to scalar
        if hasattr(ess, "numpy"):
            ess_value = ess.numpy()
            if isinstance(ess_value, np.ndarray):
                if ess_value.size == 1:
                    ess_value = float(ess_value)
                else:
                    ess_value = float(np.mean(ess_value))
        else:
            ess_value = float(ess)
    except Exception as e:
        print(f"error when calculating ess: {e}")
        ess_value = 1.0  # Default value if calculation fails

    print(f"ESS: {ess_value:.2f}")

    # Calculate leapfrog count
    if "L-HNN" in method_name:
        # L-HNN methods use fixed 8000 leapfrog count
        total_leapfrogs = 8000
        print(f"{method_name} uses fixed leapfrog count: {total_leapfrogs}")
    elif 'NUTS' in method_name:
        # Traditional NUTS leapfrog count from kernel_results.leapfrogs_taken
        leapfrogs_taken = tf.reduce_sum(kernel_results.leapfrogs_taken).numpy()
        total_leapfrogs = int(leapfrogs_taken)
        print(f"NUTS leapfrogs_taken: {total_leapfrogs}")
    else:
        # Traditional HMC leapfrog count = total_samples × leapfrog_steps
        total_leapfrogs = args.hmc_samples * num_leapfrog_steps
        print(f"HMC leapfrogs: {args.hmc_samples} × {num_leapfrog_steps} = {total_leapfrogs}")

    # Calculate acceptance rate
    if hasattr(kernel_results, 'is_accepted'):
        acceptance_rate = tf.reduce_mean(tf.cast(kernel_results.is_accepted, tf.float32)).numpy()
        print(f"acceptance rate: {acceptance_rate:.4f}")
    else:
        acceptance_rate = 0.0
        print("warning: cannot get acceptance rate")

    # Calculate ESS/leapfrog efficiency
    ess_per_leapfrog = ess_value / total_leapfrogs if total_leapfrogs > 0 else 0
    print(f"ESS/leapfrog: {ess_per_leapfrog:.6f}")

    return {
        'ess': ess_value,
        'total_leapfrogs': total_leapfrogs,
        'acceptance_rate': acceptance_rate,
        'ess_per_leapfrog': ess_per_leapfrog
    }


def compare_sampling_methods():
    """Compare performance of different sampling methods"""
    # Set parameters
    args = get_args()
    args.dist_name = "1D_Gauss_mix"
    args.save_dir = 'files/1D_Gauss_mix'
    args.input_dim = 2  # 1D position + 1D momentum
    args.hidden_dim = 100
    args.latent_dim = 100
    args.nonlinearity = 'sine'
    args.hmc_samples = 100
    args.num_burnin = 0

    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create log file (only record the results summary)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"table1_results_{timestamp}.log"
    log_file = open(log_path, 'w')

    print("===== Sampling methods performance comparison =====")
    print(f"distribution: {args.dist_name}")
    print(f"sampling count: {args.hmc_samples}")
    print(f"burn-in count: {args.num_burnin}")

    # Initialize models
    print("\ninitialize models...")

    # Initialize function model (for traditional methods)
    function_model = FunctionModel(args)

    # create HNN model
    hnn_model = create_hnn_model(args)

    # call model once to create variables
    dummy_input = tf.zeros([1, args.input_dim])
    _ = hnn_model(dummy_input)

    # load weights
    print(f"\nload weights from: {args.save_dir}...")
    try:
        hnn_model.load_weights(args.save_dir)
    except Exception as e:
        print(f"warning: failed to load weights - {e}")

    # Initial state
    initial_state = tf.zeros([1], dtype=tf.float32)

    # Sampling parameters
    step_size = 0.05
    num_leapfrog_steps = 20

    results = {}

    # ==================== 1. Traditional HMC ====================
    print("\n1. run traditional HMC...")
    start_time = time.time()

    trad_hmc_kernel = HNNHMC(
        step_size=[step_size],
        num_leapfrog_steps=num_leapfrog_steps,
        hnn_model=lambda x: hamiltonian_wrapper(x, args, functions),
        target_fn=lambda x: get_target_log_prob(x, args.dist_name, args.input_dim),
        state_gradients_are_stopped=False,
        store_parameters_in_results=True,
        name='traditional_hmc'
    )

    trad_hmc_samples, trad_hmc_kernel_results = run_sampling(
        trad_hmc_kernel, initial_state, args.hmc_samples, args.num_burnin
    )

    trad_hmc_time = time.time() - start_time
    print(f"running time: {trad_hmc_time:.2f} seconds")

    trad_hmc_metrics = compute_metrics(
        trad_hmc_samples, trad_hmc_kernel_results, "Traditional HMC",
        args, step_size, num_leapfrog_steps)

    results['Traditional HMC'] = {
        'samples': trad_hmc_samples,
        'metrics': trad_hmc_metrics,
        'time': trad_hmc_time
    }

    # ==================== 2. L-HNN HMC ====================
    print("\n2. run L-HNN HMC...")
    start_time = time.time()

    lhnn_hmc_kernel = HNNHMC(
        step_size=[step_size],
        num_leapfrog_steps=num_leapfrog_steps,
        hnn_model=lambda x: hnn_wrapper(x, hnn_model),
        target_fn=lambda x: get_target_log_prob(x, args.dist_name, args.input_dim),
        state_gradients_are_stopped=False,
        store_parameters_in_results=True,
        name='lhnn_hmc'
    )

    lhnn_hmc_samples, lhnn_hmc_kernel_results = run_sampling(
        lhnn_hmc_kernel, initial_state, args.hmc_samples, args.num_burnin
    )

    lhnn_hmc_time = time.time() - start_time
    print(f"running time: {lhnn_hmc_time:.2f} seconds")

    lhnn_hmc_metrics = compute_metrics(
        lhnn_hmc_samples, lhnn_hmc_kernel_results, "L-HNN HMC",
        args, step_size, num_leapfrog_steps)

    results['L-HNN HMC'] = {
        'samples': lhnn_hmc_samples,
        'metrics': lhnn_hmc_metrics,
        'time': lhnn_hmc_time
    }

    # ==================== 3. Traditional NUTS ====================
    print("\n3. run traditional NUTS...")
    start_time = time.time()

    trad_nuts_kernel = NoUTurnSampler(
        hnn_model=lambda x: hamiltonian_wrapper(x, args, functions),
        target_log_prob_fn=lambda x: get_target_log_prob(x, args.dist_name, args.input_dim),
        step_size=step_size,
        unrolled_leapfrog_steps=1
    )

    trad_nuts_samples, trad_nuts_kernel_results = run_sampling(
        trad_nuts_kernel, initial_state, args.hmc_samples, args.num_burnin
    )

    trad_nuts_time = time.time() - start_time
    print(f"running time: {trad_nuts_time:.2f} seconds")

    trad_nuts_metrics = compute_metrics(
        trad_nuts_samples, trad_nuts_kernel_results, "Traditional NUTS",
        args, step_size, num_leapfrog_steps)

    results['Traditional NUTS'] = {
        'samples': trad_nuts_samples,
        'metrics': trad_nuts_metrics,
        'time': trad_nuts_time
    }

    # ==================== 4. L-HNN NUTS ====================
    print("\n4. run L-HNN NUTS...")
    start_time = time.time()

    lhnn_nuts_kernel = NoUTurnSampler(
        hnn_model=lambda x: hnn_wrapper(x, hnn_model),
        target_log_prob_fn=lambda x: get_target_log_prob(x, args.dist_name, args.input_dim),
        step_size=step_size,
        unrolled_leapfrog_steps=1
    )

    lhnn_nuts_samples, lhnn_nuts_kernel_results = run_sampling(
        lhnn_nuts_kernel, initial_state, args.hmc_samples, args.num_burnin
    )

    lhnn_nuts_time = time.time() - start_time
    print(f"running time: {lhnn_nuts_time:.2f} seconds")

    lhnn_nuts_metrics = compute_metrics(
        lhnn_nuts_samples, lhnn_nuts_kernel_results, "L-HNN NUTS",
        args, step_size, num_leapfrog_steps)

    results['L-HNN NUTS'] = {
        'samples': lhnn_nuts_samples,
        'metrics': lhnn_nuts_metrics,
        'time': lhnn_nuts_time
    }

    # ==================== Results Summary ====================
    print("\n===== Results Summary =====")
    log_file.write("===== Sampling methods performance comparison results =====\n")
    log_file.write(f"distribution: {args.dist_name}\n")
    log_file.write(f"sampling count: {args.hmc_samples}\n")
    log_file.write(f"burn-in count: {args.num_burnin}\n\n")

    # Header
    header = "method              ESS      total leapfrog count    ESS/leapfrog      running time(seconds)    acceptance rate"
    print(header)
    log_file.write(header + "\n")

    print("-" * 80)
    log_file.write("-" * 80 + "\n")

    # Print results for all methods
    methods = ['Traditional HMC', 'L-HNN HMC', 'Traditional NUTS', 'L-HNN NUTS']
    for method in methods:
        metrics = results[method]['metrics']
        run_time = results[method]['time']

        result_line = f"{method:15s}  {metrics['ess']:7.2f}   {metrics['total_leapfrogs']:10d}   {metrics['ess_per_leapfrog']:.6f}   {run_time:8.2f}   {metrics['acceptance_rate']:.4f}"
        print(result_line)
        log_file.write(result_line + "\n")

    log_file.write("\nExperiment completed!\n")
    log_file.close()
    print(f"\nResults saved to: {log_path}\n")

    return results


if __name__ == "__main__":
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    print("\n===== Start sampling methods performance comparison =====")

    try:
        # Run comparison experiment
        results = compare_sampling_methods()

        print("\nExperiment completed!")

    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback

        traceback.print_exc()