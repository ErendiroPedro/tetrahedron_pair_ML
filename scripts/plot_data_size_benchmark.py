"""
Plot benchmark results: 2D Heatmap of Training Data Size vs Model Size with Mean Accuracy as color.

X-Axis: Training Data Size (Log scale)
Y-Axis: Model Size (Number of Parameters)
Color: Mean Accuracy over the 5 test datasets
"""

import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path
from datetime import datetime


# Data sizes and labels (must match benchmark script)
DATA_SIZES = [10_000, 50_000, 100_000, 500_000, 1_000_000]
DATA_SIZE_LABELS = ['10k', '50k', '100k', '500k', '1M']

# Model order (smallest to largest)
MODEL_ORDER = ['Smallest', 'Medium', 'Largest']


def load_results(results_dir):
    """Load benchmark results from individual JSON files."""
    results = []
    
    # Try loading individual result files
    for json_file in sorted(results_dir.glob('*_results.json')):
        if 'combined' not in json_file.name:
            with open(json_file, 'r') as f:
                results.append(json.load(f))
    
    # If no individual files, try combined file
    if not results:
        combined_file = results_dir / 'data_size_benchmark_combined.json'
        if combined_file.exists():
            with open(combined_file, 'r') as f:
                data = json.load(f)
                results = data.get('results', [])
    
    return results


def plot_heatmap(results, output_dir):
    """
    Create 2D heatmap: Model Size (y-axis) vs Training Data Size (x-axis), color = Mean Accuracy.
    Smallest models at bottom, largest at top.
    """
    # Filter successful results
    successful = [r for r in results if 'error' not in r and 'mean_accuracy' in r]
    
    if not successful:
        print("No successful results to plot!")
        return None
    
    # Get unique model names and sort by parameter count
    model_data = {}
    for r in successful:
        name = r['model_name']
        if name not in model_data:
            model_data[name] = {
                'params': r['total_parameters'],
                'results': {}
            }
        model_data[name]['results'][r['data_size']] = r['mean_accuracy']
    
    # Sort models by parameter count (smallest first for bottom of heatmap)
    sorted_models = sorted(model_data.keys(), key=lambda x: model_data[x]['params'])
    
    # Build accuracy matrix (rows = models, cols = data sizes)
    accuracy_matrix = []
    model_params = []
    
    for model_name in sorted_models:
        row = []
        for data_size in DATA_SIZES:
            accuracy = model_data[model_name]['results'].get(data_size, np.nan)
            row.append(accuracy)
        accuracy_matrix.append(row)
        model_params.append(model_data[model_name]['params'])
    
    accuracy_matrix = np.array(accuracy_matrix)
    
    # Flip matrix so smallest model (row 0) appears at bottom
    accuracy_matrix_flipped = accuracy_matrix[::-1]
    model_params_flipped = model_params[::-1]
    sorted_models_flipped = sorted_models[::-1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap with custom colormap
    # Use a diverging colormap centered around a good accuracy value
    vmin = max(0.5, np.nanmin(accuracy_matrix) - 0.05)
    vmax = min(1.0, np.nanmax(accuracy_matrix) + 0.02)
    
    im = ax.imshow(accuracy_matrix_flipped, cmap='RdYlGn', aspect='auto',
                   vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Mean Accuracy', fontsize=14)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2%}'))
    
    # Set axis labels
    ax.set_xlabel('Training Data Size', fontsize=14)
    ax.set_ylabel('Model Size (Parameters)', fontsize=14)
    ax.set_title('TetrahedronPairNet: Accuracy Heatmap\n', 
                 fontsize=16, fontweight='bold')
    
    # Set x-axis ticks (data sizes)
    ax.set_xticks(range(len(DATA_SIZE_LABELS)))
    ax.set_xticklabels(DATA_SIZE_LABELS, fontsize=11)
    
    # Set y-axis ticks (model sizes - reversed to match flipped matrix)
    y_labels = [f"{sorted_models_flipped[i]}\n({model_params_flipped[i]:,})" 
                for i in range(len(sorted_models_flipped))]
    ax.set_yticks(range(len(sorted_models_flipped)))
    ax.set_yticklabels(y_labels, fontsize=10)
    
    # Add accuracy values as text annotations
    for i in range(len(sorted_models_flipped)):
        for j in range(len(DATA_SIZE_LABELS)):
            accuracy = accuracy_matrix_flipped[i, j]
            if not np.isnan(accuracy):
                # Choose text color based on background brightness
                text_color = 'white' if accuracy < 0.7 else 'black'
                ax.text(j, i, f'{accuracy:.1%}', 
                       ha='center', va='center', fontsize=11, 
                       color=text_color, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = output_dir / f'data_size_accuracy_heatmap_{timestamp}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved to: {plot_path}")
    
    # Also save as PDF
    pdf_path = output_dir / f'data_size_accuracy_heatmap_{timestamp}.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")
    
    plt.show()
    
    return plot_path


def plot_accuracy_curves(results, output_dir):
    """
    Create line plots showing accuracy vs data size for each model.
    """
    # Filter successful results
    successful = [r for r in results if 'error' not in r and 'mean_accuracy' in r]
    
    if not successful:
        print("No successful results to plot!")
        return None
    
    # Organize data by model
    model_data = {}
    for r in successful:
        name = r['model_name']
        if name not in model_data:
            model_data[name] = {
                'params': r['total_parameters'],
                'data_sizes': [],
                'accuracies': [],
                'std_accuracies': []
            }
        model_data[name]['data_sizes'].append(r['data_size'])
        model_data[name]['accuracies'].append(r['mean_accuracy'])
        model_data[name]['std_accuracies'].append(r['evaluation'].get('std_accuracy', 0))
    
    # Sort models by parameter count
    sorted_models = sorted(model_data.keys(), key=lambda x: model_data[x]['params'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
    markers = ['o', 's', '^']
    
    for idx, model_name in enumerate(sorted_models):
        data = model_data[model_name]
        
        # Sort by data size
        sorted_indices = np.argsort(data['data_sizes'])
        x = np.array(data['data_sizes'])[sorted_indices]
        y = np.array(data['accuracies'])[sorted_indices]
        yerr = np.array(data['std_accuracies'])[sorted_indices]
        
        label = f"{model_name} ({data['params']:,} params)"
        
        ax.errorbar(x, y, yerr=yerr, marker=markers[idx], markersize=10,
                   linewidth=2, capsize=5, capthick=2, color=colors[idx],
                   label=label)
    
    ax.set_xscale('log')
    ax.set_xlabel('Training Data Size', fontsize=14)
    ax.set_ylabel('Mean Accuracy', fontsize=14)
    ax.set_title('TetrahedronPairNet: Accuracy vs Training Data Size\n(by Model Size)', 
                 fontsize=16, fontweight='bold')
    
    # Set x-axis ticks
    ax.set_xticks(DATA_SIZES)
    ax.set_xticklabels(DATA_SIZE_LABELS, fontsize=11)
    
    # Set y-axis range
    ax.set_ylim(0.5, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
    
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = output_dir / f'data_size_accuracy_curves_{timestamp}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Curves plot saved to: {plot_path}")
    
    plt.show()
    
    return plot_path


def print_summary_table(results):
    """Print a formatted summary table."""
    successful = [r for r in results if 'error' not in r and 'mean_accuracy' in r]
    
    if not successful:
        print("No successful results!")
        return
    
    # Organize by model
    model_data = {}
    for r in successful:
        name = r['model_name']
        if name not in model_data:
            model_data[name] = {'params': r['total_parameters'], 'results': {}}
        model_data[name]['results'][r['data_size']] = r
    
    sorted_models = sorted(model_data.keys(), key=lambda x: model_data[x]['params'])
    
    print("\n" + "="*100)
    print("BENCHMARK RESULTS SUMMARY: Mean Accuracy across 5 Test Datasets")
    print("="*100)
    
    # Header
    header = f"{'Model':>12} | {'Params':>10} |"
    for label in DATA_SIZE_LABELS:
        header += f" {label:>8} |"
    print(header)
    print("-"*100)
    
    for model_name in sorted_models:
        data = model_data[model_name]
        row = f"{model_name:>12} | {data['params']:>10,} |"
        
        for data_size in DATA_SIZES:
            if data_size in data['results']:
                accuracy = data['results'][data_size]['mean_accuracy']
                row += f" {accuracy:>7.2%} |"
            else:
                row += f" {'N/A':>7} |"
        
        print(row)
    
    print("-"*100)
    
    # Find best configuration
    best_accuracy = 0
    best_config = None
    for r in successful:
        if r['mean_accuracy'] > best_accuracy:
            best_accuracy = r['mean_accuracy']
            best_config = r
    
    if best_config:
        print(f"\nBest Configuration:")
        print(f"  Model: {best_config['model_name']} ({best_config['total_parameters']:,} params)")
        print(f"  Training Data: {best_config['data_size_label']} ({best_config['data_size']:,} samples)")
        print(f"  Mean Accuracy: {best_accuracy:.2%}")


def main():
    """Main plotting function."""
    base_dir = Path('/home/sei/tetrahedron_pair_ML')
    results_dir = base_dir / 'artifacts' / 'data_size_benchmark_results'
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Please run benchmark_data_size_vs_model_size.py first.")
        return
    
    print(f"Loading results from: {results_dir}")
    results = load_results(results_dir)
    
    print(f"Loaded {len(results)} experiment results")
    
    # Print summary
    print_summary_table(results)
    
    # Create heatmap
    plot_heatmap(results, results_dir)
    
    # Create line plots
    plot_accuracy_curves(results, results_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
