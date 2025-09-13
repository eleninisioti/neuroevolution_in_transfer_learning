import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Import the Coolors palette
# Palette from: https://coolors.co/palette/ff595e-ffca3a-8ac926-1982c4-6a4c93
coolors_palette = [
    '#ff595e',  # Vibrant red
    '#ffca3a',  # Bright yellow
    '#8ac926',  # Fresh green
    '#1982c4',  # Deep blue
    '#6a4c93'   # Rich purple
]

def load_task_data(task):
    """Load data for a single task from no_noise directory"""
    num_trials = 10
    methods = ["ppo", "ga", "openes", "cma_es"]
    method_labels = {"ppo": "PPO", "ga": "GA", "openes": "OpenES", "cma_es": "CMA-ES"}
    all_methods_data = {}

    for method in methods:
        all_trials = []
        
        for trial in range(num_trials):
            csv_path = f"scripts/alife_2025/data/no_noise/{task}/{method}/trial{trial}.csv"
            
            # Check if file exists, skip if it doesn't
            if not os.path.exists(csv_path):
                print(f"Warning: {csv_path} not found, skipping trial {trial}")
                continue
                
            try:
                df = pd.read_csv(csv_path)
                df['trial'] = trial  # Add trial column
                df['method'] = method  # Add method column
                all_trials.append(df)
            except Exception as e:
                print(f"Warning: Error reading {csv_path}: {e}, skipping trial {trial}")
                continue

        # Concatenate all trials for this method (only if we have some trials)
        if all_trials:
            combined_df = pd.concat(all_trials, ignore_index=True)
            all_methods_data[method] = combined_df
            
            print(f"{task.upper()} - {method.upper()} data loaded successfully!")
            print(f"Combined shape: {combined_df.shape}")
            print(f"Columns: {combined_df.columns.tolist()}")
            print(f"Number of trials: {combined_df['trial'].nunique()}")
            print("\nFirst few rows:")
            print(combined_df.head())
            print("\n" + "="*50 + "\n")
        else:
            print(f"Warning: No trials found for {task.upper()} - {method.upper()}")

    return all_methods_data, method_labels

def make_no_noise_visuals(task):
    """Create visualization for a single task from no_noise data, showing only steps up to 400"""
    # Set the color palette for matplotlib
    plt.style.use('default')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=coolors_palette)

    # Load data
    all_methods_data, method_labels = load_task_data(task)
    
    # Check if we have any data
    if not all_methods_data:
        print(f"No data found for {task}, skipping visualization")
        return
        
    methods = list(all_methods_data.keys())  # Only use methods that have data

    # Create line plot with confidence intervals for each method
    plt.figure(figsize=(3.15*0.7, 1.97))  # 8 cm x 5 cm
    
    # Plot each method with different colors
    for i, method in enumerate(methods):
        method_df = all_methods_data[method]
        
        # Filter data to only include steps up to 400
        method_df = method_df[method_df['Step'] <= 400]
        
        # Get the fitness column name (it might be different for different methods)
        fitness_cols = [col for col in method_df.columns if 'fitness' in col.lower()]
        if fitness_cols:
            fitness_col = fitness_cols[0]  # Use the first fitness column found
        else:
            # Fallback to a common column name pattern
            fitness_col = method_df.columns[1]  # Usually the second column
        
        # Calculate mean and confidence intervals for each step
        steps = method_df['Step'].unique()
        means = []
        lower_bounds = []
        upper_bounds = []

        for step in steps:
            step_data = method_df[method_df['Step'] == step][fitness_col]
            mean_val = step_data.mean()
            std_val = step_data.std()
            
            # 95% confidence interval (approximately 1.96 * std / sqrt(n))
            ci_95 = 1.96 * std_val / np.sqrt(len(step_data))
            
            means.append(mean_val)
            lower_bounds.append(mean_val - ci_95)
            upper_bounds.append(mean_val + ci_95)

        # Plot the mean line using different colors from the palette
        plt.plot(steps, means, color=coolors_palette[i], linewidth=2, 
                label=method_labels[method])
        
        # Plot confidence intervals (without adding to legend)
        plt.fill_between(steps, lower_bounds, upper_bounds, alpha=0.3, 
                        color=coolors_palette[i])

    # Customize the plot with black axis labels
    plt.xlabel('Steps', fontsize=7, color='black')  # Black text
    plt.ylabel('Episode reward', fontsize=7, color='black')  # Black text
   
    # Grid and background
    plt.grid(True, alpha=0.3, color=coolors_palette[2])  # Green grid
    plt.gca().set_facecolor('white')  # White background
    
    # Set x-axis limits to show only up to step 400
    plt.xlim(0, 400)
    
    # Set x-axis ticks for steps up to 400
    plt.xticks(range(0, 401, 200))

    # Apply tight layout first
    plt.tight_layout()

    # Force tick label font sizes AFTER layout
    plt.tick_params(axis="both", labelsize=7)

    # Show the plot
    plt.show()

    # Save the main plot (after applying tick size)
    plt.savefig(f'scripts/alife_2025/{task}_no_noise_comparison.png', dpi=300, bbox_inches='tight', 
                facecolor='white')  # Save with white background
    plt.savefig(f'scripts/alife_2025/{task}_no_noise_comparison.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white')  # Save with white background
    print(f"\nPlot saved as '{task}_no_noise_comparison.png' and '{task}_no_noise_comparison.pdf'!")

    # Create and save legend as a separate figure
    fig_legend = plt.figure(figsize=(4, 1))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis('off')
    
    legend_elements = [
        plt.Line2D([0], [0], color=coolors_palette[i], linewidth=2, label=method_labels[method])
        for i, method in enumerate(methods)
    ]
    
    ax_legend.legend(
        handles=legend_elements,
        frameon=True,
        facecolor='white',
        edgecolor=coolors_palette[4],
        loc='center',
        fontsize=9,
        ncol=4
    )
    
    # Save legend figure
    fig_legend.savefig(f'scripts/alife_2025/{task}_no_noise_legend.png', dpi=300, bbox_inches='tight', 
                       facecolor='white', pad_inches=0.1)
    fig_legend.savefig(f'scripts/alife_2025/{task}_no_noise_legend.pdf', dpi=300, bbox_inches='tight', 
                       facecolor='white', pad_inches=0.1)
    plt.close(fig_legend)

    print(f"Legend saved as '{task}_no_noise_legend.png' and '{task}_no_noise_legend.pdf'!")

if __name__ == "__main__":
    # Create individual plots for each task
    tasks = ["acrobot", "cartpole", "mountaincar"]
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Creating visualization for {task.upper()} (no_noise data, steps 0-400)")
        print(f"{'='*60}")
        make_no_noise_visuals(task=task)
