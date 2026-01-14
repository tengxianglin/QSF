"""
This script estimates Von Neumann entropy and quantum state fidelity by training
parameterized quantum circuits to approximate Taylor‐series expansions of these
quantities. It computes theoretical values, optimizes circuits to generate
coefficients for entropy/fidelity estimators, runs simulated measurement
experiments to collect statistics, and produces filled‐error‐bar plots
for direct comparison of methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from tqdm import tqdm  # progress bar
from quairkit.database.state import completely_mixed_computational
from quairkit.database import random_state
from quairkit.qinfo import *
from math import comb
from quairkit.circuit import Circuit
import quairkit
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from typing import List

quairkit.set_dtype("complex128")

# Set the random seed for reproducibility.
quairkit.set_seed(seed := 856104 or random.randint(0, 1e6))
# Optionally set the computation device.
# quairkit.set_device("cuda:1")

# Define the loss function for training the circuit. The loss is based on 1-State Fidelity.


def loss_func(circuit: Circuit, target_state_normalized: torch.Tensor) -> torch.Tensor:
    """
    Compute the loss for the circuit based on 1-State Fidelity.

    Args:
        circuit (Circuit): The quantum circuit.
        target_state_normalized (torch.Tensor): The normalized target state.

    Returns:
        torch.Tensor: The computed loss.
    """
    return 1 - torch.abs(circuit().bra @ target_state_normalized).real ** 2


# Train the circuit to generate a state proportional to the given coefficient list.


def optimize_circuit(aj_list: list, j: int = 0) -> List[int]:
    """
    Optimize a quantum circuit to generate a state proportional to the given coefficients.

    Args:
        aj_list (list): List of coefficients.
        j (int, optional): Index of the coefficient to focus on. Defaults to 0.

    Returns:
        list: The optimized state as a list of real values.
    """
    # Calculate the next power of 2 for the length
    original_length = len(aj_list)
    target_length = 2 ** int(np.ceil(np.log2(original_length)))

    # Make a copy to avoid modifying the original list
    padded_list = aj_list.copy()

    # Pad with zeros to reach the next power of 2
    padded_list.extend([0] * (target_length - original_length))

    target_state = torch.tensor(padded_list, dtype=quairkit.get_dtype()).reshape(
        [-1, 1]
    )
    target_state_normalized = target_state / torch.norm(target_state)

    # Calculate the number of qubits needed
    num_qubits = int(np.log2(target_length))

    cir = Circuit(num_qubits)
    # cir.real_entangled_layer(list(range(num_qubits)), 2)
    cir.universal_three_qubits(range(num_qubits))
    opt = torch.optim.Adam(cir.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min")

    for epoch in range(num_epoch := 1000):
        opt.zero_grad()
        loss = loss_func(cir, target_state_normalized)
        loss.backward()
        opt.step()
        scheduler.step(loss)
        if epoch % (num_epoch // 10) == 0 or epoch == num_epoch - 1:
            print(
                f'Epoch {epoch}: Loss = {loss.item()}, Learning Rate = {opt.param_groups[0]["lr"]}'
            )

    output_state = (
        cir().ket
        * torch.exp(-1j * torch.angle(cir().ket[j, 0]))
        * torch.norm(target_state)
    )
    # return output_state.real.squeeze().tolist()[:original_length]
    return output_state.real.squeeze().tolist()


# Define a unified function to compute coefficients for entropy and fidelity


def compute_coefficients(mode: str, max_n: int) -> list:
    """
    Compute coefficients for entropy or fidelity based on the mode.

    Args:
        mode (str): Either 'entropy' or 'fidelity'.
        max_n (int): The maximum degree for the coefficients.

    Returns:
        list: List of coefficients.
    """
    coefficients = []
    if mode == "entropy":
        for j in range(max_n + 1):
            aj = sum((-1) ** j * comb(k, j) / k for k in range(1, max_n + 1)) / np.log(
                2
            )
            coefficients.append(aj)
    elif mode == "fidelity":
        for n in range(max_n):
            if n == 0:
                coefficients.append(12155 / 65536)  # c_0
            elif n == 1:
                coefficients.append(109395 / 65536)  # c_1
            elif n == 2:
                coefficients.append(-36465 / 16384)  # c_2
            elif n == 3:
                coefficients.append(51051 / 16384)  # c_3
            elif n == 4:
                coefficients.append(-109395 / 32768)  # c_4
            elif n == 5:
                coefficients.append(85085 / 32768)  # c_5
            elif n == 6:
                coefficients.append(-23205 / 16384)  # c_6
            elif n == 7:
                coefficients.append(8415 / 16384)  # c_7
            elif n == 8:
                coefficients.append(-7293 / 65536)  # c_8
            elif n == 9:
                coefficients.append(715 / 65536)  # c_9
    return coefficients


# Define a unified function for theoretical calculations
def theoretical_function(mode: str, mat: torch.Tensor) -> float:
    """
    Compute the theoretical value for entropy or fidelity.

    Args:
        mode (str): Either 'entropy' or 'fidelity'.
        rho (torch.Tensor): The density matrix.
        sigma (torch.Tensor, optional): The second density matrix for fidelity. Defaults to None.

    Returns:
        float: The computed theoretical value.
    """
    if mode == "entropy":
        return -trace(mat @ logm(mat)).real.item() / np.log(2)
    elif mode == "fidelity":
        return trace(sqrtm(mat)).real.item() ** 2


# Define the measurement experiment:
# 1. Map measurement values from the range [-1, 1] to [0, 1].
# 2. Perform measurements to obtain `num_samples` values and compute their mean.
# 3. Repeat step 2 to obtain multiple sets of measurement values.
# 4. Compute the mean and variance of the averages from multiple sets.


def simulate_measurement_stats(
    approx_values: float, coeff: float, n_values: int, num_simulations: int = 10
) -> tuple:
    """
    Simulate measurement experiments and compute the mean and standard deviation.

    Args:
        approx_values (float): Approximation value in the range [-1, 1].
        n_values (int): Number of measurement samples.
        coeff (float): Coefficient for scaling the approximation value.
        num_simulations (int, optional): Number of simulations to perform. Defaults to 10.

    Returns:
        tuple: Mean and standard deviation of the measurements.
    """
    # Scale approx_values by dividing with coeff
    scaled_values = approx_values / coeff

    # Convert scaled_values to a scalar if it's an array
    if isinstance(scaled_values, (list, np.ndarray)):
        scaled_values = np.mean(scaled_values)

    # scaled_values is between -1 and 1, so we need to convert it to a probability
    P_plus = (scaled_values + 1) / 2
    measure_means = []

    # Simulate the measurement num_simulations times
    for _ in range(num_simulations):
        measure_values = [
            1 if random.random() <= P_plus else -1 for _ in range(n_values)
        ]
        measure_mean = np.mean(measure_values)
        measure_means.append(measure_mean)

    # Scale the results back by multiplying with coeff
    mean_result = np.mean(measure_means) * coeff
    stddev_result = np.std(measure_means) * coeff

    return mean_result, stddev_result


def plot_with_fill(
    ax: plt.Axes,
    x_values: list,
    y_values: list,
    y_stddev: list,
    color: str,
    label: str,
    marker: str,
)-> plt.Line2D:
    """
    Plot a line with a filled area representing the standard deviation.

    Args:
        ax (plt.Axes): The axis to plot on.
        x_values (list): X-axis values.
        y_values (list): Y-axis values.
        y_stddev (list): Standard deviation for Y-axis values.
        color (str): Line and fill color.
        label (str): Label for the line.
        marker (str): Marker style for the line.
    """
    (line,) = ax.plot(
        x_values,
        y_values,
        color=color,
        label=label,
        linewidth=2.5,
        marker=marker,
        markersize=8,
    )
    ax.fill_between(
        x_values,
        np.array(y_values) - np.array(y_stddev),
        np.array(y_values) + np.array(y_stddev),
        color=color,
        alpha=0.3,
    )
    return line


def format_entropy_plot(
    ax: plt.Axes,
    num_copy_list: list,
    # actual_entropy_value: float,
    legend_handles: list,
    legend_labels: list,
    lines: list,
    # labels: list,
):
    """
    Format and style the entropy plot.

    Args:
        ax (plt.Axes): The axis to format.
        num_copy_list (list): List of x-axis values (number of copies).
        actual_entropy_value (float): The actual entropy value.
        legend_handles (list): List to collect unique legend handles.
        legend_labels (list): List to collect unique legend labels.
        lines (list): List of line objects to add to the legend.
        labels (list): List of labels corresponding to the lines.
    """
    ax.set_xscale("log")
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlim(num_copy_list[0], num_copy_list[-1])
    ax.set_xticks(
        [
            10**i
            for i in range(
                int(np.log10(num_copy_list[0])), int(np.log10(num_copy_list[-1])) + 1
            )
        ]
    )

    # # Add the actual entropy value as a horizontal line
    # theorectical_line = ax.axhline(
    #     y=actual_entropy_value,
    #     color="black",
    #     linestyle="--",
    #     label=r"$S(\rho)$",
    #     linewidth=1.5,
    # )
    # lines.append(theorectical_line)
    # labels.append(r"$S(\rho)$")

    # Only add unique legend handles and labels
    for line in lines:
        label = line.get_label()
        if label not in legend_labels:
            legend_handles.append(line)
            legend_labels.append(label)

    # Set x-axis labels to scientific notation
    ax.set_xticklabels(
        [f"$10^{{{i}}}$" for i in range(3, 7)],
        fontsize=20,
    )

    # Add labels
    ax.set_xlabel("No. of Copies", fontsize=20)
    ax.set_ylabel("Von Neumann Entropy", fontsize=20) if ax == axes[0] else None

    # Remove grid lines
    ax.grid(False)

    # Set black border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.5)

    # Adjust ticks
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(
        axis="both",
        which="both",
        length=5,
        color="black",
        direction="out",
        labelsize=20,
    )
    # Set y-axis to integer ticks
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


entropy_state: torch.Tensor = torch.load("entropy_state_mat.pt")
fidelity_states: torch.Tensor = torch.load("fidelity_states_mat.pt")
completely_mixed_state: torch.Tensor = completely_mixed_computational(1).density_matrix

# Step-1: Get the coefficients from the trained circuit
degree = 5
# Add a loop to handle both entropy and fidelity calculations and plots
for mode, mats in zip(
    ["entropy", "fidelity"],
    [
        [entropy_state, completely_mixed_state],
        [
            fidelity_states[0] @ fidelity_states[1],
            fidelity_states[0] @ completely_mixed_state,
        ],
    ],
):
    # Create subplots for side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 14 / ((1 + 5**0.5))))

    # Create an empty list to collect unique legend handles and labels
    legend_handles = []
    legend_labels = []

    aj_list = compute_coefficients(mode, degree)
    # Coefficient list for eff_max and eff_norm
    aj_list_trained = optimize_circuit(aj_list)
    # Coefficient list for general SWAP Test
    single_aj_list_trained_list = []
    for j, aj in enumerate(aj_list):
        single_term_list = [0] * len(aj_list)
        single_term_list[j] = aj
        single_aj_list_trained = optimize_circuit(single_term_list, j)
        single_aj_list_trained_list.append(single_aj_list_trained)
    # Compute the coefficients for eff_max and eff_norm
    eff_max = max(aj_list_trained) * degree
    eff_norm = np.linalg.norm(aj_list_trained)

    # Define the degree and the range of the number of samples
    num_copy_list = np.logspace(3, 6, num=10, dtype=int)

    # Process each input state and plot results
    for ax, mat in zip(axes, mats):
        theoretical_value = theoretical_function(mode, mat)

        # Step-2: Compute the estimation value
        # Compute the estimation value for eff_max and eff_norm
        estimation_value = sum(
            aj * torch.trace(torch.matrix_power(mat, j + 1 if mode == "entropy" else j))
            for j, aj in enumerate(aj_list_trained)
        ).real.item()
        # Compute the estimation value for general SWAP Test
        single_estimation_values = []
        for j, aj in enumerate(aj_list):
            single_aj_list_trained = single_aj_list_trained_list[j]
            single_estimation_value = (
                single_aj_list_trained[j]
                * torch.trace(
                    torch.matrix_power(mat, j + 1 if mode == "entropy" else j)
                ).abs()
                * np.sign(aj_list[j])
            )
            single_estimation_values.append(single_estimation_value)

        measurements = {
            "eff_max": {"values": [], "stddev": []},
            "eff_norm": {"values": [], "stddev": []},
            "general_swap": {"values": [], "stddev": []},
        }

        # Process general_swap
        for num_copy in tqdm(num_copy_list, desc=f"Processing num_copy ({mode})"):
            # Compute the number of samples for eff_max and eff_norm
            num_sample = num_copy // degree
            for coeff, label in zip([eff_max, eff_norm], ["eff_max", "eff_norm"]):
                taylor_measures, taylor_std = simulate_measurement_stats(
                    estimation_value, coeff, num_sample
                )
                measurements[label]["values"].append(taylor_measures)
                measurements[label]["stddev"].append(taylor_std)

            # Compute the number of samples for the general SWAP Test
            num_sample = num_copy // (degree * len(aj_list_trained))
            general_swap_result = 0
            general_swap_variance = 0  # Accumulate variance
            for single_estimation_value in single_estimation_values:
                single_term_measures, single_term_std = simulate_measurement_stats(
                    single_estimation_value, 1, num_sample
                )
                general_swap_result += single_term_measures
                general_swap_variance += single_term_std**2  # Accumulate variance

            general_swap_stddev = np.sqrt(general_swap_variance)

            measurements["general_swap"]["values"].append(general_swap_result)
            measurements["general_swap"]["stddev"].append(general_swap_stddev)

        # Define plot parameters
        plot_params = [
            ("eff_max", "#FFA756", "o", "variant"),
            ("eff_norm", "#4C72B0", "s", "QSF"),
            ("general_swap", "#55A868", "^", "GSWAP"),
            # ("theoretical", "#000000", "-", "")
        ]

        lines = []
        symbol_str = "S" if mode == "entropy" else "F"
        for method, color, marker, label_suffix in plot_params:
            line = plot_with_fill(
                ax,
                num_copy_list,
                measurements[method]["values"],
                measurements[method]["stddev"],
                color=color,
                label=rf"$\hat{{{symbol_str}}}_{{{degree+1}}}(\rho)_{{{label_suffix}}}$",
                marker=marker,
            )
            lines.append(line)

        theorectical_line = ax.axhline(
            y=theoretical_value,
            color="#000000",
            linestyle="--",
            label=rf"${symbol_str}(\rho)$",
            linewidth=1.5,
        )
        lines.append(theorectical_line)
        # Format the plot
        format_entropy_plot(
            ax,
            num_copy_list,
            legend_handles,
            legend_labels,
            lines,
            # [
            #     rf"$\hat{{{symbol_str}}}_{{{degree+1}}}(\rho)_{{variant}}$",
            #     rf"$\hat{{{symbol_str}}}_{{{degree+1}}}(\rho)_{{QSF}}$",
            #     rf"$\hat{{{symbol_str}}}_{{{degree+1}}}(\rho)_{{GSWAP}}$",
            #     rf"${symbol_str}(\rho)$",
            # ],
        )

    # Add a unified legend to the right side of the entire plot
    fig.legend(
        handles=legend_handles, 
        loc="upper left", bbox_to_anchor=(0.86, 1), fontsize=18
    )

    # Adjust layout to remove excess whitespace
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.1)
    plt.savefig(f"{mode}_plots.pdf", format="pdf", bbox_inches="tight")  # Save as PDF
    plt.savefig(f"{mode}_plots.svg", format="svg", bbox_inches="tight")  # Save as SVG
    plt.show()

    # Print the seed used for reproducibility
    print(f"Seed used: {quairkit.get_seed()}")
