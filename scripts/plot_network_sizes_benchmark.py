"""
Plot benchmark results: 2D Heatmap of Model Size vs Batch Size with Throughput as color.
Run this after benchmark_network_sizes.py completes.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
from datetime import datetime


# Batch sizes used in benchmark
BATCH_SIZES = [128, 256, 512, 1024, 2048]


def load_results(results_dir):
    """Load benchmark results from individual JSON files."""
    results = []
    for json_file in sorted(results_dir.glob('config_*_results.json')):
        with open(json_file, 'r') as f:
            results.append(json.load(f))
    return results


def plot_heatmap(results, output_dir):
    """
    Create 2D heatmap: Number of Parameters (y-axis) vs Batch Size (x-axis), color = Throughput.
    Smallest models at bottom, largest at top.
    """
    # Filter successful results (no errors)
    successful = [r for r in results if 'error' not in r and 'inference_by_batch_size' in r]
    
    if not successful:
        print("No successful results with batch size data to plot!")
        return None
    
    # Sort by parameters (smallest first)
    successful.sort(key=lambda x: x['total_parameters'])
    
    # Build throughput matrix (rows = model configs, cols = batch sizes)
    throughput_matrix = []
    for r in successful:
        row = []
        for bs in BATCH_SIZES:
            # Handle both string and int keys
            bs_data = r['inference_by_batch_size'].get(str(bs)) or r['inference_by_batch_size'].get(bs, {})
            throughput = bs_data.get('throughput_samples_per_sec', 0)
            row.append(throughput)
        throughput_matrix.append(row)
    
    throughput_matrix = np.array(throughput_matrix)
    
    # Flip matrix so smallest params (row 0) appears at bottom
    throughput_matrix_flipped = throughput_matrix[::-1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap with flipped data
    im = ax.imshow(throughput_matrix_flipped, cmap='plasma', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Throughput (samples/sec)', fontsize=14)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}k'))
    
    # Set axis labels
    ax.set_xlabel('Batch Size', fontsize=14)
    ax.set_ylabel('Number of Parameters', fontsize=14)
    ax.set_title('TetrahedronPairNet: Throughput Heatmap', 
                 fontsize=18, fontweight='bold')
    
    # Set x-axis ticks (batch sizes)
    ax.set_xticks(range(len(BATCH_SIZES)))
    ax.set_xticklabels(BATCH_SIZES)
    
    # Set y-axis ticks - reversed to match flipped matrix (largest at top, smallest at bottom)
    y_labels = [f"{r['total_parameters']:,}" for r in successful[::-1]]  # Reverse to match flipped matrix
    ax.set_yticks(range(len(successful)))
    ax.set_yticklabels(y_labels, fontsize=9)
    
    # Add throughput values as text annotations (use flipped matrix)
    for i in range(len(successful)):
        for j in range(len(BATCH_SIZES)):
            throughput = throughput_matrix_flipped[i, j]
            # Choose text color based on background brightness
            text_color = 'white' if throughput < throughput_matrix.mean() else 'black'
            ax.text(j, i, f'{throughput/1000:.0f}k', 
                   ha='center', va='center', fontsize=8, color=text_color)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = output_dir / f'throughput_heatmap_{timestamp}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved to: {plot_path}")
    
    # Also save as PDF
    pdf_path = output_dir / f'throughput_heatmap_{timestamp}.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")
    
    plt.show()
    
    return plot_path


def print_summary_table(results):
    """Print a formatted summary table."""
    successful = [r for r in results if 'error' not in r and 'inference_by_batch_size' in r]
    successful.sort(key=lambda x: x['model_size_mb'])
    
    print("\n" + "="*110)
    print("BENCHMARK RESULTS SUMMARY: Throughput (samples/sec)")
    print("="*110)
    
    # Header
    header = f"{'Config':>6} | {'Params':>10} | {'Size(MB)':>8} |"
    for bs in BATCH_SIZES:
        header += f" BS={bs:>4} |"
    print(header)
    print("-"*110)
    
    for r in successful:
        row = f"{r['config_id']:>6} | {r['total_parameters']:>10,} | {r['model_size_mb']:>8.4f} |"
        for bs in BATCH_SIZES:
            bs_data = r['inference_by_batch_size'].get(str(bs)) or r['inference_by_batch_size'].get(bs, {})
            throughput = bs_data.get('throughput_samples_per_sec', 0)
            row += f" {throughput/1000:>6.1f}k |"
        print(row)
    
    print("-"*110)
    
    # Find best throughput overall
    best_throughput = 0
    best_config = None
    best_batch = None
    for r in successful:
        for bs in BATCH_SIZES:
            bs_data = r['inference_by_batch_size'].get(str(bs)) or r['inference_by_batch_size'].get(bs, {})
            throughput = bs_data.get('throughput_samples_per_sec', 0)
            if throughput > best_throughput:
                best_throughput = throughput
                best_config = r['config_id']
                best_batch = bs
    
    print(f"\nBest throughput: {best_throughput:,.0f} samples/sec")
    print(f"  Config #{best_config}, Batch Size {best_batch}")


def main():
    """Main plotting function."""
    base_dir = Path('/home/sei/tetrahedron_pair_ML')
    results_dir = base_dir / 'artifacts' / 'benchmark_results'
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Please run benchmark_network_sizes.py first.")
        return
    
    print(f"Loading results from: {results_dir}")
    results = load_results(results_dir)
    
    print(f"Loaded {len(results)} configurations")
    
    # Print summary
    print_summary_table(results)
    
    # Create heatmap
    plot_heatmap(results, results_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
