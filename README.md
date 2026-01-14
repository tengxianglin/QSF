# QuAIR QSF: Entropy and Fidelity Estimation

This repository provides code for estimating
Von Neumann entropy and quantum state fidelity
via trained parameterized quantum circuits,
as presented in the paper:

https://arxiv.org/abs/2412.01696

## Maintainer

Tengxiang Lin  
tengxianglin23@gmail.com

## Overview

- Implements Taylor‐series‐based estimators
  using circuit‐training and simulated measurements.
- Generates filled‐error‐bar plots to compare
  estimation methods against theoretical values.

## Requirements

- Python >= 3.8
- NumPy, Matplotlib, PyTorch, tqdm
- QuAIRKit (set to complex128 dtype)

## Usage

1. Adjust parameters in `qsf_plot.py`.
2. Run the script:  
   `python qsf_plot.py`
3. Output plots: `entropy_plots.pdf/svg`,
   `fidelity_plots.pdf/svg`.

