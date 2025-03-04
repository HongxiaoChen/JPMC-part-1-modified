# HNN and HMC

This repository contains code samples for the **2025 Machine Learning Center of Excellence Summer Associate** projects. Specifically, it replicates the results from the article *"Efficient Bayesian inference with latent Hamiltonian neural networks in No-U-Turn Sampling"* by **Somayajulu L.N. Dhulipala, Yifeng Che, and Michael D. Shields**.

Codes and reports for part 2 of this task are in [Pseudo-Marginal HNN HMC](https://github.com/HongxiaoChen/JPMC-part-2).

## Environment Setup

The project requires:
- Python 3.11
- TensorFlow 2.15  
- TensorFlow Probability 0.23.0

## Repository Structure

The repository is organized into two main directories:
- `codes/`: Contains the core implementation files
- `tests/`: Contains unit tests

A `run.bash` script is provided in the root directory to reproduce all results.

A `Report1.pdf` file summaries the replication results.

## Core Implementation (`codes/`)

### Result Reproduction Scripts
Run the following to reproduce all replication results:
- `_Table1_reproduction.py`
- `_figure2_reproduction.py` through `_figure11_reproduction.py`

### TFP Modified Kernels
Located in `codes/tfp_modified_kernels/`:
- `hnn_leapfrog.py`: Inherits from `tensorflow_probability.python.mcmc.internal.leapfrog_integrator.LeapfrogIntegrator`. It uses the one-step leapfrog as in the paper of Dhulipala (2023), and takes HNN as an extra input.
- `tfp_hnn_hmc.py`: Includes a new transition kernel that inherits from `kernel.TransitionKernel`, implements LHNN HMC sampling, takes HNN as an extra input, and uses `hnn_leapfrog.py` to produce integrator.
- `tfp_hnn_nuts.py`: Includes a new transition kernel that inherits from `kernel.TransitionKernel`, implements LHNN NUTS sampling, takes HNN as an extra input, and uses `hnn_leapfrog.py` to produce integrator.
- `tfp_hnn_nuts_online.py`: Includes a new transition kernel that inherits from `kernel.TransitionKernel`, implements LHNN NUTS with online monitoring, takes both HNN and Hamiltonian function as extra inputs, and uses `hnn_leapfrog.py` to produce integrator.
- All samplings are run via `tfp.mcmc.sample_chain` that lies in `run_sampling` function in `utils.py`, which is decorated with `@tf.function`

### Core Files
- `hnn.py` & `nn_models.py`: Structures for Hamiltonian Neural Networks
- `data.py`: Data generation for training, which is used in `train_hnn.py`
- `train_hnn.py`: Training script that trains HNN weights
- `get_args.py`: Parameter configurations
- `utils.py`: Utility functions
- `functions.py`: Contains all Hamiltonian functions and target log probabilities

### Output Directories
- `codes/logs/`: Contains all execution logs
- `codes/figures/`: Stores generated figures
- `codes/files/`: Contains model weights

## Testing

The `tests/` directory contains comprehensive unit tests for all components. Run tests using:

```bash
python -m unittest discover -v
```
from the root directory.
