import pandas as pd
import numpy as np
import yaml
import os
from scipy import stats
from scipy.stats import f_oneway, kruskal, ttest_ind, mannwhitneyu
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

def load_task_data(task):
    """Load data for a single task"""
    num_trials = 10
    methods = ["ppo", "ga", "openes", "cma_es"]
    method_labels = {"ppo": "PPO", "ga": "GA", "openes": "OpenES", "cma_es": "CMA-ES"}
    all_methods_data = {}

    for method in methods:
        all_trials = []
        
        for trial in range(num_trials):
            csv_path = f"scripts/alife_2025/data/{task}/{method}/trial{trial}.csv"
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

def compute_cumulative_fitness_at_specific_steps(task_data, method_labels):
    """Compute cumulative fitness by summing fitness values at specific steps (199, 399, 599, etc.)"""
    methods = ["ppo", "ga", "openes", "cma_es"]
    results = {}
    
    # Define specific steps to sum
    specific_steps = [199, 399, 599, 799, 999, 1199, 1399, 1599, 1799, 1999]
    
    for method in methods:
        method_df = task_data[method]
        
        # Get the fitness column name
        fitness_cols = [col for col in method_df.columns if 'fitness' in col.lower()]
        if fitness_cols:
            fitness_col = fitness_cols[0]
        else:
            fitness_col = method_df.columns[1]  # Usually the second column
        
        # Compute cumulative fitness for each trial by summing specific steps
        trial_cumulative_fitness = []
        
        for trial in range(10):
            trial_data = method_df[method_df['trial'] == trial].copy()
            trial_data = trial_data.sort_values('Step')  # Ensure proper ordering
            
            # Sum fitness values only at specific steps
            cumulative_fitness = 0
            for step in specific_steps:
                step_data = trial_data[trial_data['Step'] == step]
                if not step_data.empty:
                    fitness_value = step_data[fitness_col].iloc[0]
                    cumulative_fitness += fitness_value
            
            trial_cumulative_fitness.append(cumulative_fitness)
        
        # Average across trials
        mean_cumulative_fitness = np.mean(trial_cumulative_fitness)
        std_cumulative_fitness = np.std(trial_cumulative_fitness)
        
        results[method_labels[method]] = {
            'cumulative_fitness_at_specific_steps': float(mean_cumulative_fitness),
            'std': float(std_cumulative_fitness),
            'steps_summed': specific_steps,
            'trial_values': trial_cumulative_fitness  # Keep individual trial values for statistical testing
        }
    
    return results

def perform_statistical_tests(task_results):
    """Perform group-level and pairwise statistical tests"""
    methods = list(task_results.keys())
    trial_values = [task_results[method]['trial_values'] for method in methods]
    
    # Check normality using Shapiro-Wilk test
    normality_results = {}
    for i, method in enumerate(methods):
        stat, p_value = stats.shapiro(trial_values[i])
        normality_results[method] = {'statistic': stat, 'p_value': p_value, 'normal': p_value > 0.05}
    
    # Group-level tests
    # ANOVA (parametric)
    f_stat, anova_p = f_oneway(*trial_values)
    
    # Kruskal-Wallis (non-parametric)
    h_stat, kw_p = kruskal(*trial_values)
    
    # Pairwise comparisons
    pairwise_results = []
    method_pairs = list(combinations(methods, 2))
    
    for method1, method2 in method_pairs:
        values1 = task_results[method1]['trial_values']
        values2 = task_results[method2]['trial_values']
        
        # Check if both groups are normal
        both_normal = normality_results[method1]['normal'] and normality_results[method2]['normal']
        
        if both_normal:
            # Use t-test
            t_stat, t_p = ttest_ind(values1, values2)
            test_type = 't-test'
        else:
            # Use Mann-Whitney U test
            u_stat, u_p = mannwhitneyu(values1, values2, alternative='two-sided')
            t_stat, t_p = u_stat, u_p
            test_type = 'Mann-Whitney U'
        
        pairwise_results.append({
            'method1': method1,
            'method2': method2,
            'test_type': test_type,
            'statistic': float(t_stat),
            'p_value': float(t_p),
            'mean1': float(np.mean(values1)),
            'mean2': float(np.mean(values2)),
            'std1': float(np.std(values1)),
            'std2': float(np.std(values2))
        })
    
    # Apply Bonferroni correction for multiple comparisons
    num_comparisons = len(pairwise_results)
    bonferroni_alpha = 0.05 / num_comparisons
    
    for result in pairwise_results:
        result['p_value_corrected'] = min(result['p_value'] * num_comparisons, 1.0)
        result['significant_uncorrected'] = result['p_value'] < 0.05
        result['significant_corrected'] = result['p_value_corrected'] < 0.05
    
    return {
        'normality_tests': normality_results,
        'group_tests': {
            'anova': {'f_statistic': float(f_stat), 'p_value': float(anova_p)},
            'kruskal_wallis': {'h_statistic': float(h_stat), 'p_value': float(kw_p)}
        },
        'pairwise_tests': pairwise_results,
        'bonferroni_alpha': bonferroni_alpha,
        'num_comparisons': num_comparisons
    }

def generate_yaml_report(all_results, all_statistical_results):
    """Generate YAML report with specific steps results and statistical tests"""
    yaml_data = {
        'cumulative_fitness_analysis': {
            'description': 'Cumulative fitness by summing fitness values at specific steps (199, 399, 599, etc.) averaged across 10 trials',
            'tasks': {}
        }
    }
    
    for task, task_results in all_results.items():
        yaml_data['cumulative_fitness_analysis']['tasks'][task] = {
            'cumulative_fitness': {},
            'statistical_tests': all_statistical_results[task]
        }
        
        for method, data in task_results.items():
            yaml_data['cumulative_fitness_analysis']['tasks'][task]['cumulative_fitness'][method] = {
                'cumulative_fitness_at_specific_steps': data['cumulative_fitness_at_specific_steps'],
                'std': data['std'],
                'steps_summed': data['steps_summed']
            }
    
    # Save YAML file
    with open('scripts/alife_2025/cumulative_fitness_results.yaml', 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, indent=2)
    
    print("YAML report saved as 'cumulative_fitness_results.yaml'")

def generate_markdown_table(all_results, all_statistical_results):
    """Generate markdown table with statistical results"""
    markdown_content = "# Cumulative Fitness Results with Statistical Analysis\n\n"
    markdown_content += "Cumulative fitness by summing fitness values at steps 199, 399, 599, 799, 999, 1199, 1399, 1599, 1799, 1999.\n\n"
    
    for task, task_results in all_results.items():
        markdown_content += f"## {task.title()}\n\n"
        
        # Cumulative fitness table
        markdown_content += "### Cumulative Fitness Results\n\n"
        markdown_content += "| Method | Cumulative Fitness (Specific Steps) | Std Dev |\n"
        markdown_content += "|--------|-----------------------------------|----------|\n"
        
        for method, data in task_results.items():
            cumulative_fitness = data['cumulative_fitness_at_specific_steps']
            std = data['std']
            markdown_content += f"| {method} | {cumulative_fitness:.2f} | {std:.2f} |\n"
        
        markdown_content += "\n"
        
        # Statistical tests
        stats_results = all_statistical_results[task]
        
        # Group-level tests
        markdown_content += "### Group-Level Statistical Tests\n\n"
        markdown_content += "| Test | Statistic | p-value | Significant (α=0.05) |\n"
        markdown_content += "|------|-----------|---------|---------------------|\n"
        
        anova = stats_results['group_tests']['anova']
        kw = stats_results['group_tests']['kruskal_wallis']
        
        markdown_content += f"| ANOVA | F={anova['f_statistic']:.3f} | {anova['p_value']:.3f} | {'Yes' if anova['p_value'] < 0.05 else 'No'} |\n"
        markdown_content += f"| Kruskal-Wallis | H={kw['h_statistic']:.3f} | {kw['p_value']:.3f} | {'Yes' if kw['p_value'] < 0.05 else 'No'} |\n"
        
        markdown_content += "\n"
        
        # Pairwise tests
        markdown_content += "### Pairwise Comparisons\n\n"
        markdown_content += "| Method 1 | Method 2 | Test | Statistic | p-value | p-value (Bonferroni) | Significant |\n"
        markdown_content += "|----------|----------|------|-----------|---------|---------------------|-------------|\n"
        
        for result in stats_results['pairwise_tests']:
            sig_uncorrected = "Yes" if result['significant_uncorrected'] else "No"
            sig_corrected = "Yes" if result['significant_corrected'] else "No"
            markdown_content += f"| {result['method1']} | {result['method2']} | {result['test_type']} | {result['statistic']:.3f} | {result['p_value']:.3f} | {result['p_value_corrected']:.3f} | {sig_corrected} |\n"
        
        markdown_content += "\n"
    
    # Save markdown file
    with open('scripts/alife_2025/cumulative_fitness_table.md', 'w') as f:
        f.write(markdown_content)
    
    print("Markdown table saved as 'cumulative_fitness_table.md'")

def generate_latex_table_with_significance_lines(all_results, all_statistical_results):
    """Generate beautiful LaTeX table with significance lines"""
    latex_content = "\\documentclass[12pt]{article}\n"
    latex_content += "\\usepackage[utf8]{inputenc}\n"
    latex_content += "\\usepackage{booktabs}\n"
    latex_content += "\\usepackage{multirow}\n"
    latex_content += "\\usepackage{array}\n"
    latex_content += "\\usepackage{geometry}\n"
    latex_content += "\\usepackage{xcolor}\n"
    latex_content += "\\usepackage{colortbl}\n"
    latex_content += "\\usepackage{longtable}\n"
    latex_content += "\\usepackage{tikz}\n"
    latex_content += "\\usetikzlibrary{calc}\n\n"
    
    # Define colors for methods
    latex_content += "% Define custom colors for methods\n"
    latex_content += "\\definecolor{ppo}{HTML}{ff595e}\n"
    latex_content += "\\definecolor{ga}{HTML}{ffca3a}\n"
    latex_content += "\\definecolor{openes}{HTML}{8ac926}\n"
    latex_content += "\\definecolor{cmaes}{HTML}{1982c4}\n\n"
    
    latex_content += "% Page setup for better table display\n"
    latex_content += "\\geometry{margin=1in}\n"
    latex_content += "\\pagestyle{empty}\n\n"
    
    latex_content += "\\begin{document}\n\n"
    
    # Title section
    latex_content += "\\begin{center}\n"
    latex_content += "\\Large\\textbf{Cumulative Fitness Results with Statistical Analysis}\n\\vspace{0.5cm}\n"
    latex_content += "\\normalsize Cumulative fitness by summing fitness values at specific steps with statistical testing.\n\\vspace{0.5cm}\n"
    latex_content += "\\end{center}\n\n"
    
    for task, task_results in all_results.items():
        task_name = task.title()
        stats_results = all_statistical_results[task]
        
        # Create significance mapping
        method_order = ["PPO", "GA", "OpenES", "CMA-ES"]
        significance_matrix = {}
        
        # Initialize matrix
        for i, method1 in enumerate(method_order):
            for j, method2 in enumerate(method_order):
                if i != j:
                    significance_matrix[(method1, method2)] = False
        
        # Fill in significant comparisons
        for result in stats_results['pairwise_tests']:
            if result['significant_corrected']:
                significance_matrix[(result['method1'], result['method2'])] = True
                significance_matrix[(result['method2'], result['method1']] = True
        
        # Cumulative fitness table with significance lines
        latex_content += f"\\section{{{task_name}}}\n\n"
        latex_content += "\\begin{table}[h!]\n"
        latex_content += "\\centering\n"
        latex_content += f"\\caption{{{task_name} - Cumulative Fitness Results with Statistical Significance}}\n"
        
        # Create table with extra columns for significance lines
        latex_content += "\\begin{tabular}{|l|l|r|r|"
        for i in range(len(method_order) - 1):
            latex_content += "c|"
        latex_content += "}\n"
        
        latex_content += "\\hline\n"
        
        # Header row
        header = "\\textbf{Task} & \\textbf{Method} & \\textbf{Cumulative Fitness} & \\textbf{Std Dev} &"
        for i in range(len(method_order) - 1):
            header += f" \\textbf{{{method_order[i+1]}}} &"
        header = header.rstrip('&') + " \\\\\n"
        latex_content += header
        latex_content += "\\hline\\hline\n"
        
        # Data rows
        methods = list(task_results.keys())
        for i, (method, data) in enumerate(task_results.items()):
            cumulative_fitness = data['cumulative_fitness_at_specific_steps']
            std = data['std']
            
            # Get color for this method
            color_name = method.lower().replace('-', '')
            
            if i == 0:
                row = f"\\multirow{{{len(methods)}}}{{*}}{{\\textbf{{{task_name}}}}} & \\rowcolor{{{color_name}!20}}{method} & {cumulative_fitness:,.2f} & {std:,.2f} &"
            else:
                row = f" & \\rowcolor{{{color_name}!20}}{method} & {cumulative_fitness:,.2f} & {std:,.2f} &"
            
            # Add significance indicators
            for j, other_method in enumerate(method_order[1:], 1):
                if significance_matrix.get((method, other_method), False):
                    row += " $*$ &"
                else:
                    row += " &"
            
            row = row.rstrip('&') + " \\\\\n"
            latex_content += row
            
            if i < len(methods) - 1:
                latex_content += "\\cline{2-" + str(len(method_order) + 2) + "}\n"
        
        latex_content += "\\hline\n"
        latex_content += "\\end{tabular}\n"
        latex_content += "\\end{table}\n\n"
        
        # Add legend
        latex_content += "\\begin{center}\n"
        latex_content += "\\small\n"
        latex_content += "\\textbf{Legend:} $*$ indicates significant difference (p < 0.05, Bonferroni corrected)\n"
        latex_content += "\\end{center}\n\n"
        
        # Statistical summary
        latex_content += "\\begin{table}[h!]\n"
        latex_content += "\\centering\n"
        latex_content += f"\\caption{{{task_name} - Statistical Summary}}\n"
        latex_content += "\\begin{tabular}{|l|r|r|c|}\n"
        latex_content += "\\hline\n"
        latex_content += "\\textbf{Test} & \\textbf{Statistic} & \\textbf{p-value} & \\textbf{Significant} \\\\\n"
        latex_content += "\\hline\\hline\n"
        
        anova = stats_results['group_tests']['anova']
        kw = stats_results['group_tests']['kruskal_wallis']
        
        latex_content += f"ANOVA & F={anova['f_statistic']:.3f} & {anova['p_value']:.3f} & {'Yes' if anova['p_value'] < 0.05 else 'No'} \\\\\n"
        latex_content += f"Kruskal-Wallis & H={kw['h_statistic']:.3f} & {kw['p_value']:.3f} & {'Yes' if kw['p_value'] < 0.05 else 'No'} \\\\\n"
        
        latex_content += "\\hline\n"
        latex_content += "\\end{tabular}\n"
        latex_content += "\\end{table}\n\n"
    
    latex_content += "\\end{document}\n"
    
    # Save LaTeX file
    with open('scripts/alife_2025/cumulative_fitness_table.tex', 'w') as f:
        f.write(latex_content)
    
    print("Beautiful LaTeX table with significance lines saved as 'cumulative_fitness_table.tex'")

def main():
    """Main function to compute cumulative fitness and generate reports"""
    tasks = ["acrobot", "cartpole", "mountaincar"]
    all_results = {}
    all_statistical_results = {}
    
    print("Computing cumulative fitness at specific steps for all tasks and methods...")
    print("="*60)
    
    for task in tasks:
        print(f"\nProcessing {task.upper()}...")
        print("-" * 40)
        
        # Load data for this task
        task_data, method_labels = load_task_data(task)
        
        # Compute cumulative fitness at specific steps
        task_results = compute_cumulative_fitness_at_specific_steps(task_data, method_labels)
        all_results[task] = task_results
        
        # Perform statistical tests
        statistical_results = perform_statistical_tests(task_results)
        all_statistical_results[task] = statistical_results
        
        print(f"\n{task.upper()} Results:")
        for method, data in task_results.items():
            print(f"  {method}: Cumulative fitness (specific steps) = {data['cumulative_fitness_at_specific_steps']:.2f} ± {data['std']:.2f}")
        
        print(f"\n{task.upper()} Statistical Tests:")
        print(f"  ANOVA: F={statistical_results['group_tests']['anova']['f_statistic']:.3f}, p={statistical_results['group_tests']['anova']['p_value']:.3f}")
        print(f"  Kruskal-Wallis: H={statistical_results['group_tests']['kruskal_wallis']['h_statistic']:.3f}, p={statistical_results['group_tests']['kruskal_wallis']['p_value']:.3f}")
        
        print(f"\n{task.upper()} Pairwise Comparisons (Bonferroni corrected):")
        for result in statistical_results['pairwise_tests']:
            sig = "***" if result['significant_corrected'] else ""
            print(f"  {result['method1']} vs {result['method2']}: p={result['p_value_corrected']:.3f} {sig}")
    
    print("\n" + "="*60)
    print("Generating reports...")
    
    # Generate reports
    generate_yaml_report(all_results, all_statistical_results)
    generate_markdown_table(all_results, all_statistical_results)
    generate_latex_table_with_significance_lines(all_results, all_statistical_results)
    
    print("\nAll reports generated successfully!")
    print("Files created:")
    print("- cumulative_fitness_results.yaml")
    print("- cumulative_fitness_table.md")
    print("- cumulative_fitness_table.tex")

if __name__ == "__main__":
    main()
