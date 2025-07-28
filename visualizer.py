# visualizer.py

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from typing import List, Callable, Optional, Dict
from models import WaveVectorData, FittingFunctions

class Visualizer:
    """Handles all plotting and data visualization tasks."""
    def __init__(self, wave_data: WaveVectorData):
        self.wave_data = wave_data
        plt.style.use('default')

    def plot_raw_data(self, S1xyz: np.ndarray, S2xyz: np.ndarray, Moxyz: np.ndarray):
        """Plots the initial raw coordinate data."""
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        datasets = {"S1xyz": S1xyz, "S2xyz": S2xyz, "Moxyz": Moxyz}

        for ax, (title, data) in zip(axs, datasets.items()):
            ax.set_title(f"{title} – Raw Coordinates", fontsize=14)
            ax.set_xlabel("x-coordinate")
            ax.grid(True, linestyle='--', alpha=0.6)
            for i in range(data.shape[0]):
                ax.plot(data[i, :, 0], data[i, :, 2], marker='o', markersize=2, linestyle='-', label=f"Row {i}")

        axs[0].set_ylabel("z-coordinate")
        axs[0].legend(ncol=2, fontsize="small")
        plt.tight_layout()
        plt.show()

    def plot_wave_vector(self):
        """Plots the calculated wave vector kk."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.wave_data.kk, marker='o', linestyle='-', color='b', linewidth=2, markersize=6)
        plt.title("Wave Vector (kk) vs. Configuration Index", fontsize=14, fontweight='bold')
        plt.xlabel("Configuration Index", fontsize=12)
        plt.ylabel("Wave Vector Value (kk)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def plot_fits(self, x_data: np.ndarray, y_data: np.ndarray, fit_func: Callable,
                  params_list: List[np.ndarray], title: str, num_to_plot: Optional[int] = 6):
        """Plots raw data against their fitted functions."""
        num_to_plot = min(num_to_plot, len(x_data))
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
        axes = axes.flatten()

        for i in range(num_to_plot):
            ax = axes[i]
            kz, kx = self.wave_data.kk[i], 2 * self.wave_data.kk[i]
            FittingFunctions.set_k_values(kz, kx)

            ax.scatter(x_data[i], y_data[i], alpha=0.7, s=30, label='Data')
            x_fit = np.linspace(np.min(x_data[i]), np.max(x_data[i]), 200)
            y_fit = fit_func(x_fit, *params_list[i])
            ax.plot(x_fit, y_fit, 'r-', linewidth=2.5, label='Fit')

            ax.set_title(f'Configuration {i}', fontweight='bold')
            ax.set_xlabel('rx')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)

        for i in range(num_to_plot, 6):
            axes[i].set_visible(False)

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def plot_symbolic_functions(self, expr: sp.Expr, params_dict: Dict[str, np.ndarray],
                               x_data: np.ndarray, title: str):
        """Plots functions derived from solved symbolic expressions."""
        plt.figure(figsize=(12, 7))
        param_names = list(params_dict.keys())
        num_plots = len(params_dict[param_names[0]])

        x_sym, k_sym = sp.Symbol('x'), sp.Symbol('k')

        for i in range(num_plots):
            subs_dict = {sp.Symbol(name): values[i] for name, values in params_dict.items()}
            subs_dict[k_sym] = self.wave_data.kk[i + 1]

            try:
                func = sp.lambdify(x_sym, expr.subs(subs_dict), 'numpy')
                x_vals = np.linspace(np.min(x_data[i + 1]), np.max(x_data[i + 1]), 200)
                y_vals = func(x_vals)
                plt.plot(x_vals, y_vals, label=f'Config {i + 1}', linewidth=2)
            except Exception as e:
                print(f"Failed to plot symbolic function for config {i+1}: {e}")

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('x', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()
