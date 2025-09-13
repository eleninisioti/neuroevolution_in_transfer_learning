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
    """Load data for a single task"""
    num_trials = 10
    methods = ["ppo", "ga", "openes", "cma_es"]
    method_labels = {"ppo": "PPO", "ga": "GA", "openes": "OpenES", "cma_es": "CMA-ES"}
    all_methods_data = {}

    for method in methods:
        all_trials = []
        
        for trial in range(num_trials):
            csv_path = f"scripts/alife_2025/data/noise_2/{task}/{method}/trial{trial}.csv"
            df = pd.read_csv(csv_path)
            df['trial'] = trial  # Add trial column
            df['method'] = method  # Add method column
            all_trials.append(df)

        # Concatenate all trials for this method
        combined_df = pd.concat(all_trials, ignore_index=True)
        all_methods_data[method] = combined_df
        
        print(f"{task.upper()} - {method.upper()} data loaded successfully!")
        print(f"Combined shape: {combined_df.shape}")
        print(f"Columns: {combined_df.columns.tolist()}")
        print(f"Number of trials: {combined_df['trial'].nunique()}")
        print("\nFirst few rows:")
        print(combined_df.head())
        print("\n" + "="*50 + "\n")

    return all_methods_data, method_labels

def make_visuals(task):
    """Original function for single task visualization"""
    # Set the color palette for matplotlib
    plt.style.use('default')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=coolors_palette)

    # Load data
    all_methods_data, method_labels = load_task_data(task)
    methods = ["ppo", "ga", "openes", "cma_es"]

    # Create line plot with confidence intervals for each method
    plt.figure(figsize=(3.15*1.5, 1.97))  # 8 cm x 5 cm
    # Plot each method with different colors
    for i, method in enumerate(methods):
        method_df = all_methods_data[method]
        
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
            
            # 95% confidence interval (approximately 1.96 * std)
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
    plt.xlabel('Steps', fontsize=9, color='black')  # Black text
    plt.ylabel('Episode reward', fontsize=9, color='black')  # Black text
   
    # Remove legend from main plot
    plt.grid(True, alpha=0.3, color=coolors_palette[2])  # Green grid

    # Set background color to white for better contrast with vibrant colors
    plt.gca().set_facecolor('white')

    # Create and save legend as a separate file
    fig_legend = plt.figure(figsize=(4, 1))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis('off')
    
    # Create legend with the same styling
    legend_elements = []
    for i, method in enumerate(methods):
        legend_elements.append(plt.Line2D([0], [0], color=coolors_palette[i], linewidth=2, label=method_labels[method]))
    
    legend_fig = ax_legend.legend(handles=legend_elements, frameon=True, 
                                 facecolor='white', edgecolor=coolors_palette[4],
                                 loc='center', fontsize=9, ncol=4)  # 4 columns for 4 methods
    
    # Save legend figure
    fig_legend.savefig(f'scripts/alife_2025/{task}_legend.png', dpi=300, bbox_inches='tight', 
                       facecolor='white', pad_inches=0.1)
    fig_legend.savefig(f'scripts/alife_2025/{task}_legend.pdf', dpi=300, bbox_inches='tight', 
                       facecolor='white', pad_inches=0.1)
    plt.close(fig_legend)  # Close the legend figure to free memory

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Save the main plot
    plt.savefig(f'scripts/alife_2025/{task}_comparison.png', dpi=300, bbox_inches='tight', 
                facecolor='white')  # Save with white background
    plt.savefig(f'scripts/alife_2025/{task}_comparison.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white')  # Save with white background
    print(f"\nPlot saved as '{task}_comparison.png' and '{task}_comparison.pdf'!")
    print(f"Legend saved as '{task}_legend.png' and '{task}_legend.pdf'!")

def make_combined_visuals():
    """Create combined visualization with 3 subplots stacked vertically"""
    tasks = ["acrobot", "cartpole", "mountaincar"]
    methods = ["ppo", "ga", "openes", "cma_es"]
    method_labels = {"ppo": "PPO", "ga": "GA", "openes": "OpenES", "cma_es": "CMA-ES"}
    
    # Set the color palette for matplotlib
    plt.style.use('default')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=coolors_palette)
    
    # Create figure with 3 subplots stacked vertically
    fig, axes = plt.subplots(3, 1, figsize=(3.15*1.5, 1.97*3), sharex=True)
    
    # Load data for all tasks
    all_tasks_data = {}
    for task in tasks:
        all_tasks_data[task], _ = load_task_data(task)
    
    # Determine the maximum steps across all tasks for vertical lines and ticks
    all_max_steps = []
    for task in tasks:
        all_methods_data = all_tasks_data[task]
        task_max_steps = max([max(method_df['Step']) for method_df in all_methods_data.values()])
        all_max_steps.append(task_max_steps)
    global_max_steps = max(all_max_steps)
    
    # Define vertical lines and x-ticks
    vertical_steps = list(range(0, int(global_max_steps) + 1, 200))
    # Ensure 2000 is included if it exists in the data
    if int(global_max_steps) >= 2000 and 2000 not in vertical_steps:
        vertical_steps.append(2000)
    vertical_steps.sort()
    
    # Plot each task
    for task_idx, task in enumerate(tasks):
        ax = axes[task_idx]
        all_methods_data = all_tasks_data[task]
        
        # Plot each method with different colors
        for i, method in enumerate(methods):
            method_df = all_methods_data[method]
            
            # Get the fitness column name
            fitness_cols = [col for col in method_df.columns if 'fitness' in col.lower()]
            if fitness_cols:
                fitness_col = fitness_cols[0]
            else:
                fitness_col = method_df.columns[1]
            
            # Calculate mean and confidence intervals for each step
            steps = method_df['Step'].unique()
            means = []
            lower_bounds = []
            upper_bounds = []

            for step in steps:
                step_data = method_df[method_df['Step'] == step][fitness_col]
                mean_val = step_data.mean()
                std_val = step_data.std()
                
                # 95% confidence interval
                ci_95 = 1.96 * std_val / np.sqrt(len(step_data))
                
                means.append(mean_val)
                lower_bounds.append(mean_val - ci_95)
                upper_bounds.append(mean_val + ci_95)

            # Plot the mean line using different colors from the palette
            ax.plot(steps, means, color=coolors_palette[i], linewidth=2, 
                   label=method_labels[method])
            
            # Plot confidence intervals
            ax.fill_between(steps, lower_bounds, upper_bounds, alpha=0.3, 
                           color=coolors_palette[i])

        # Add vertical dashed lines at steps 0, 200, 400, 600, etc. to each subplot
        for step in vertical_steps:
            ax.axvline(x=step, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        # Customize each subplot
        ax.set_ylabel('Episode reward', fontsize=9, color='black')
        ax.set_title(task.title(), fontsize=10, color='black', fontweight='bold')
        ax.grid(True, alpha=0.3, color=coolors_palette[2])
        ax.set_facecolor('white')
        
        # Set x-axis ticks for all subplots with smaller font size
        ax.set_xticks(vertical_steps)
        ax.tick_params(axis="x", labelsize=7)  # Reduce tick label size by 2 points (from 9 to 7)
        
        # Only show x-axis label on bottom subplot
        if task_idx == len(tasks) - 1:
            ax.set_xlabel('Steps', fontsize=9, color='black')

    # Create and save legend as a separate file
    fig_legend = plt.figure(figsize=(4, 1))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis('off')
    
    # Create legend with the same styling
    legend_elements = []
    for i, method in enumerate(methods):
        legend_elements.append(plt.Line2D([0], [0], color=coolors_palette[i], linewidth=2, label=method_labels[method]))
    
    legend_fig = ax_legend.legend(handles=legend_elements, frameon=True, 
                                 facecolor='white', edgecolor=coolors_palette[4],
                                 loc='center', fontsize=9, ncol=4)
    
    # Save legend figure
    fig_legend.savefig('scripts/alife_2025/combined_legend.png', dpi=300, bbox_inches='tight', 
                       facecolor='white', pad_inches=0.1)
    fig_legend.savefig('scripts/alife_2025/combined_legend.pdf', dpi=300, bbox_inches='tight', 
                       facecolor='white', pad_inches=0.1)
    plt.close(fig_legend)

    # Adjust layout and show
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)  # Add some space between subplots
    plt.show()

    # Save the combined plot
    fig.savefig('scripts/alife_2025/combined_comparison.png', dpi=300, bbox_inches='tight', 
                facecolor='white')
    fig.savefig('scripts/alife_2025/combined_comparison.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white')
    print(f"\nCombined plot saved as 'combined_comparison.png' and 'combined_comparison.pdf'!")
    print(f"Legend saved as 'combined_legend.png' and 'combined_legend.pdf'!")

if __name__ == "__main__":
    # Create combined visualization
    make_combined_visuals()
    
    # Also create individual plots if needed
    # tasks = ["acrobot", "cartpole", "mountaincar"]
    # for task in tasks:
    #     make_visuals(task=task)
