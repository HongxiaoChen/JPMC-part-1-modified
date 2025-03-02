

import argparse
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.path.join(THIS_DIR, 'files')
#***** Names of coded probability distribution functions: *****
# - '1D_Gauss_mix' 2 input
# - '2D_Neal_funnel'   4 input
# - '5D_illconditioned_Gaussian'  10 input
# - 'nD_Rosenbrock' 6 or 20 input
# - 'Allen_Cahn'  50 input
# - 'Elliptic' 100 input
def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=6, type=int, help='dimensionality of input tensor (postion + momentum variables)')
    parser.add_argument('--num_samples', default=5, type=int, help='number of training samples simulated using Hamiltonian Monte Carlo')
    parser.add_argument('--len_sample', default=5, type=int, help='length of Hamiltonian trajectory for each training sample')
    parser.add_argument('--dist_name', default='nD_Rosenbrock', type=str, help='name of the probability distribution function')
    parser.add_argument('--save_dir', default=FILES_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--load_dir', default=FILES_DIR, type=str, help='where to load the training data from')
    parser.add_argument('--should_load', default=False, type=bool, help='should load training data?')
    parser.add_argument('--load_file_name', default='nD_Rosenbrock', type=str, help='if load training data, the file name (.pkl format)')

    parser.add_argument('--total_steps', default=100000, type=int, help='number of gradient steps')
    parser.add_argument('--hidden_dim', default=100, type=int, help='hidden dimension of mlp')
    parser.add_argument('--latent_dim', default=1, type=int, help='latent dimension of mlp')
    parser.add_argument('--num_layers', default=3, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=5e-4, type=float, help='initial learning rate')
    parser.add_argument('--batch_size', default=512, type=int, help='batch_size')
    parser.add_argument('--nonlinearity', default='sine', type=str, help='neural net nonlinearity')
    parser.add_argument('--test_fraction', default=0., type=float, help='fraction of testing samples')
    parser.add_argument('--step_size', default=0.025, type=int, help='step size for time integration')
    parser.add_argument('--print_every', default=25, type=int, help='number of gradient steps between prints')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.set_defaults(feature=True)
    
    parser.add_argument('--num_chains', default=1, type=int,
                       help='number of Markov chains')
    parser.add_argument('--hmc_samples', default=5000, type=int,
                       help='number of samples per chain')
    parser.add_argument('--trajectory_length', default=5, type=int,
                       help='length of Hamiltonian trajectory')
    parser.add_argument('--num_burnin', default=1000, type=int,
                       help='number of burn-in samples')
    parser.add_argument('--hmc_step_size', default=0.05, type=float,
                       help='step size for leapfrog integration')


   
    parser.add_argument('--decay_steps', default = 1000, type=int, 
                        help = 'steps of each decay in lr')
    parser.add_argument('--decay_rate', default = 0.96, type=float,
                        help = 'learning rate decay rate')
    


    # NUTS
    parser.add_argument('--total_samples', default=5000, type=int,
                       help='number of total NUTS samples')
    parser.add_argument('--burn_in', default=1000, type=int,
                       help='number of burn-in samples')
    parser.add_argument('--nuts_step_size', default=0.025, type=float,
                       help='step size for NUTS integration')
    parser.add_argument('--n_cooldown', default=20, type=int,
                       help='number of cool-down samples when switching back to HNN')
    parser.add_argument('--hnn_error_threshold', default=10.0, type=float,
                       help='error threshold for HNN integration, -float(\'inf\') mutes hnn')
    parser.add_argument('--leapfrog_error_threshold', default=1000.0, type=float,
                       help='error threshold for traditional leapfrog integration')


    if __name__ == "__main__":
        return parser.parse_args()
    else:
        # if perform in IDE, neglect unknown
        args, unknown = parser.parse_known_args()

    return args
