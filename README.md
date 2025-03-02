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

## Core Implementation (`codes/`)

### Output Directories
- `codes/logs/`: Contains all execution logs
- `codes/figures/`: Stores generated figures
- `codes/files/`: Contains model weights

### TFP Modified Kernels
Located in `codes/tfp_modified_kernels/`:
- `hnn_leapfrog.py`: Inherits from `tensorflow_probability.python.mcmc.internal.leapfrog_integrator.LeapfrogIntegrator`
- `tfp_hnn_hmc.py`: Inherits from `kernel_base.TransitionKernel`, implements LHNN HMC sampling using `hnn_leapfrog.py`
- `tfp_hnn_nuts.py`: Inherits from `kernel_base.TransitionKernel`, implements LHNN NUTS sampling using `hnn_leapfrog.py`
- `tfp_hnn_nuts_online.py`: Inherits from `kernel_base.TransitionKernel`, implements LHNN NUTS with online monitoring using `hnn_leapfrog.py`

### Core Files
- `hnn.py` & `nn_models.py`: Build Hamiltonian Neural Networks
- `data.py`: Data generation for training
- `train_hnn.py`: Training script using `data.py` to train HNN weights
- `get_args.py`: Parameter configurations
- `utils.py`: Utility functions

### Result Reproduction Scripts
- `_Table1_reproduction.py`
- `_figure2_reproduction.py` through `_figure11_reproduction.py`

## Testing

The `tests/` directory contains comprehensive unit tests for all components. Run tests using:

```bash
python -m unittest discover -v
