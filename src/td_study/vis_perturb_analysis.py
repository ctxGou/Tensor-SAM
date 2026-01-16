import numpy as np
import matplotlib.pyplot as plt
import os

# plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['ps.fonttype'] = 42
# plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams["pdf.fonttype"] = "truetype"

plt.rcParams.update({
    'font.family': 'DejaVu Sans',     # or 'Arial', 'Helvetica', etc.
    'font.size': 14,                  # Adjust size to match visual density
    'legend.fontsize': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

colors = {
    'adam': '#8abb6c',  # Blue
    'sam': '#932f67',   # Orange
    'das': '#d92c54',   # Green
    'hooi': '#dddeab',   # Red
}

def plot_comparison(configurations, plot_filename):
    """
    Loads data for a given list of configurations and plots them on the same figure.

    Args:
        configurations (list of dict): A list of dictionaries, where each dict
                                       defines a run via 'optimization', 'seed', and 'rho'.
        plot_filename (str): The name of the file to save the plot to (e.g., 'comparison.png').
    """
    # Set up the plots with two subplots (one for loss, one for R2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.7))
    
    print(f"\n--- Generating plot: {plot_filename} ---")

    # --- Plot 1: Loss vs. d ---
    # ax1.set_title('Training Loss vs. Random Perturbation Scale (d)')
    ax1.set_xlabel(' ')
    ax1.set_yscale('log')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='-', alpha=0.6)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax1.spines[spine].set_color('black')
        ax1.spines[spine].set_linewidth(2)
    ax1.tick_params(axis='both', colors='gray', length=0)
    ax1.xaxis.label.set_color('gray')
    ax1.yaxis.label.set_color('gray')
    # --- Plot 2: R^2 vs. d ---
    # ax2.set_title('R² Score vs. Random Perturbation Scale (d)')
    # ax2.set_xlabel('Perturbation Scale (d)')
    ax2.set_ylabel('R² Score')
    ax2.set_ylim(0, 1)  # R² scores are between 0 and 1
    ax2.grid(True, linestyle='-', alpha=0.6)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax2.spines[spine].set_color('black')
        ax2.spines[spine].set_linewidth(2)
    ax2.tick_params(axis='both', colors='gray', length=0)
    ax2.xaxis.label.set_color('gray')
    ax2.yaxis.label.set_color('gray')
    found_any_data = False

    # Loop through each configuration to plot
    for config in configurations:
        opt = config['optimization']
        seed = config['seed']
        rho = config['rho']
        alpha = config['alpha']
        

        # Construct the unique identifier and filenames
        run_identifier = f"{opt}_{seed}_{rho}_{alpha}"
        loss_file = f"./npy/losses_vs_d_{run_identifier}.npy"
        r2_file = f"./npy/r2_vs_d_{run_identifier}.npy"
        d_file = f"./npy/d_values_{run_identifier}.npy"

        if opt.lower() == 'adam-sam':
            opt = 'SAM'
        elif opt.lower() == 'adam-bar':
            opt = 'DAS'

        # Check if all necessary files exist before trying to plot
        if not all(os.path.exists(f) for f in [loss_file, r2_file, d_file]):
            print(f"⚠️  Warning: Data for run '{run_identifier}' not found. Skipping.")
            continue

        found_any_data = True
        print(f"  - Plotting run: {run_identifier}")

        # Load the data from the saved .npy files
        losses = np.load(loss_file)
        r2_scores = np.load(r2_file)
        d_values = np.load(d_file)
        
        # Add the data to the plots
        label = f"{opt.upper()}"
        if rho != 0.0: label += f", ρ={rho}"
        if alpha != 0.0: label += f", α={alpha}"
        ax1.plot(d_values, losses, label=label, color=colors.get(opt.lower(), 'gray'))
        ax2.plot(d_values, r2_scores, label=label, color=colors.get(opt.lower(), 'gray'))

    # Finalize and save the plot
    if found_any_data:
        handles, labels = ax1.get_legend_handles_labels()
        legend = fig.legend(
            handles, labels,
            loc='lower center',
            ncol=len(labels),
            frameon=True, facecolor='white', edgecolor='gray',
            bbox_to_anchor=(0.5, -0.02),
            columnspacing=0.8,      # reduce horizontal space between entries
            handletextpad=0.4       # reduce space between line and label
        )
        legend.get_frame().set_linewidth(0.5)
        for text in legend.get_texts():
            text.set_color('gray')
        fig.subplots_adjust(bottom=0.5)
        # fig.suptitle('Perturbation Analysis of Final Model', fontsize=20, y=1.02)
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to '{plot_filename}'")
    else:
        print("❌ No data found for any configuration. No plot was generated.")
    
    plt.close(fig) # Close the figure to free up memory


if __name__ == '__main__':
    # --- Define the sets of runs you want to compare ---
    
    # EXAMPLE 1: Compare SGD and SAM with a fixed seed and rho=0.0 for SGD
    # comparison_1 = [
    #     {'optimization': 'SGD', 'seed': 42, 'rho': 0.0},
    #     {'optimization': 'SAM', 'seed': 42, 'rho': 0.05},
    # ]
    # plot_comparison(configurations=comparison_1, plot_filename='plot_SGD_vs_SAM_seed42.png')

    # # EXAMPLE 2: Compare SAM with different rho values
    # comparison_2 = [
    #     {'optimization': 'SAM', 'seed': 42, 'rho': 0.01},
    #     {'optimization': 'SAM', 'seed': 42, 'rho': 0.05},
    #     {'optimization': 'SAM', 'seed': 42, 'rho': 0.1},
    # ]
    # plot_comparison(configurations=comparison_2, plot_filename='plot_SAM_rho_comparison_seed42.png')
    
    # # EXAMPLE 3: Compare results from a different seed
    # comparison_3 = [
    #     {'optimization': 'SGD', 'seed': 99, 'rho': 0.0},
    #     {'optimization': 'SAM', 'seed': 99, 'rho': 0.05},
    # ]
    # plot_comparison(configurations=comparison_3, plot_filename='plot_SGD_vs_SAM_seed99.png')

    # You can add as many comparisons as you like here.
    # Just define a new list of configurations and call plot_comparison().

    comparison = [
        {'optimization': 'Adam', 'seed': 42, 'rho': 0.0, 'alpha':0.0},
        # {'optimization': 'SAM', 'seed': 42, 'rho': 0.5, 'alpha':0.0},
        # {'optimization': 'BAR', 'seed': 42, 'rho': 0.0, 'alpha':0.0001},
        # {'optimization': 'BAR', 'seed': 42, 'rho': 0.0, 'alpha':0.001},
        # {'optimization': 'BAR', 'seed': 42, 'rho': 0.0, 'alpha':0.01},
        # {'optimization': 'BAR', 'seed': 42, 'rho': 0.0, 'alpha':0.1},
        # {'optimization': 'BAr', 'seed': 42, 'rho': 0.0, 'alpha':2.0},
        # # {'optimization': 'BAr', 'seed': 42, 'rho': 0.0, 'alpha':1.0},
        # {'optimization': 'Bar', 'seed': 42, 'rho': 0.0, 'alpha':0.5},
        # {'optimization': 'Bar', 'seed': 42, 'rho': 0.0, 'alpha':1.0},
        # {'optimization': 'Bar', 'seed': 42, 'rho': 0.0, 'alpha':2.0},
        {'optimization': 'ADAM-SAM', 'seed': 42, 'rho': 0.01, 'alpha':0.0},
        {'optimization': 'ADAM-BAR', 'seed': 42, 'rho': 0.0, 'alpha':0.001},
        # {'optimization': 'ADAM-BAR', 'seed': 42, 'rho': 0.0, 'alpha':0.002},
        # {'optimization': 'ADAM-BAR', 'seed': 42, 'rho': 0.0, 'alpha':0.005},
        {'optimization': 'HOOI', 'seed': 42, 'rho': 0.0, 'alpha':0.0}
    ]
    plot_comparison(configurations=comparison, plot_filename='./fig/tucker-covid-loss.pdf')