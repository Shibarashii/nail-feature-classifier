"""
Visualization script for comparing model performance across strategies and models.
Automatically generates comparison graphs from CSV files.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from typing import List, Optional


def plot_model_strategies_comparison(
    csv_path: Path,
    metrics: List[str] = None,
    output_path: Optional[Path] = None,
    target_line: Optional[float] = None
):
    """
    Plot comparison of all strategies for a single model.

    Parameters
    ----------
    csv_path : Path
        Path to the comparison CSV file
    metrics : List[str], optional
        List of metrics to plot. If None, uses default metrics.
    output_path : Path, optional
        Path to save the plot. If None, saves next to CSV.
    target_line : float, optional
        Add a horizontal target line at this value
    """
    if metrics is None:
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

    # Load data
    df = pd.read_csv(csv_path)

    if df.empty:
        print(f"Warning: Empty CSV file: {csv_path}")
        return

    # Get model name from data
    model_name = df['model_name'].iloc[0] if 'model_name' in df.columns else 'Unknown'

    # Group by strategy and compute mean (in case of multiple runs)
    grouped = df.groupby('strategy')[metrics].mean().reset_index()

    strategies = grouped['strategy'].tolist()
    metric_labels = {
        'accuracy': 'Accuracy',
        'precision_macro': 'Precision',
        'recall_macro': 'Recall',
        'f1_macro': 'F1-Score',
        'precision_weighted': 'Precision (W)',
        'recall_weighted': 'Recall (W)',
        'f1_weighted': 'F1-Score (W)',
        'binary_sensitivity': 'Sensitivity (Binary)',
        'binary_specificity': 'Specificity (Binary)',
        'binary_ppv': 'PPV (Binary)',
        'binary_npv': 'NPV (Binary)',
    }

    # Prepare data
    metric_data = []
    labels = []
    for metric in metrics:
        if metric in grouped.columns:
            metric_data.append(grouped[metric].tolist())
            labels.append(metric_labels.get(metric, metric))
        else:
            print(f"Warning: Metric '{metric}' not found in CSV")

    if not metric_data:
        print(f"Error: No valid metrics found in CSV")
        return

    # Use default color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    x = np.arange(len(strategies))
    width = 0.8 / len(metric_data)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot grouped bars
    for i, (data, label) in enumerate(zip(metric_data, labels)):
        bars = ax.bar(
            x + i * width - (len(metric_data) - 1) * width / 2,
            data,
            width,
            label=label,
            alpha=0.8,
            color=colors[i % len(colors)]
        )

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.005,
                f"{height:.4f}",
                ha='center',
                va='bottom',
                fontsize=8,
                color='#444444'
            )

    # Add target line if specified
    if target_line is not None:
        ax.axhline(
            y=target_line,
            color='red',
            linestyle='--',
            linewidth=1,
            label=f'Target ({target_line})'
        )

    # Formatting
    ax.set_title(f"{model_name} - Strategy Comparison", fontsize=14, pad=10)
    ax.set_xlabel("Strategy", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))
    ax.grid(axis='y', alpha=0.3)
    ax.legend(frameon=False, loc='lower right')

    plt.tight_layout()

    # Save plot
    if output_path is None:
        output_path = csv_path.parent / f"{csv_path.stem}_plot.png"

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plot: {output_path}")


def plot_all_models_comparison(
    csv_path: Path,
    metrics: List[str] = None,
    output_path: Optional[Path] = None,
    target_line: Optional[float] = None,
    strategy_filter: Optional[str] = None
):
    """
    Plot comparison of all models (optionally filtered by strategy).

    Parameters
    ----------
    csv_path : Path
        Path to the comparison CSV file
    metrics : List[str], optional
        List of metrics to plot. If None, uses default metrics.
    output_path : Path, optional
        Path to save the plot. If None, saves next to CSV.
    target_line : float, optional
        Add a horizontal target line at this value
    strategy_filter : str, optional
        Only include results from this strategy
    """
    if metrics is None:
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

    # Load data
    df = pd.read_csv(csv_path)

    if df.empty:
        print(f"Warning: Empty CSV file: {csv_path}")
        return

    # Filter by strategy if specified
    if strategy_filter:
        df = df[df['strategy'] == strategy_filter]
        if df.empty:
            print(f"Warning: No data found for strategy: {strategy_filter}")
            return

    # Group by model and compute mean (in case of multiple runs/strategies)
    grouped = df.groupby('model_name')[metrics].mean().reset_index()

    models = grouped['model_name'].tolist()
    metric_labels = {
        'accuracy': 'Accuracy',
        'precision_macro': 'Precision',
        'recall_macro': 'Recall',
        'f1_macro': 'F1-Score',
        'precision_weighted': 'Precision (W)',
        'recall_weighted': 'Recall (W)',
        'f1_weighted': 'F1-Score (W)',
        'binary_sensitivity': 'Sensitivity (Binary)',
        'binary_specificity': 'Specificity (Binary)',
        'binary_ppv': 'PPV (Binary)',
        'binary_npv': 'NPV (Binary)',
    }

    # Prepare data
    metric_data = []
    labels = []
    for metric in metrics:
        if metric in grouped.columns:
            metric_data.append(grouped[metric].tolist())
            labels.append(metric_labels.get(metric, metric))
        else:
            print(f"Warning: Metric '{metric}' not found in CSV")

    if not metric_data:
        print(f"Error: No valid metrics found in CSV")
        return

    # Use default color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    x = np.arange(len(models))
    width = 0.8 / len(metric_data)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot grouped bars
    for i, (data, label) in enumerate(zip(metric_data, labels)):
        bars = ax.bar(
            x + i * width - (len(metric_data) - 1) * width / 2,
            data,
            width,
            label=label,
            alpha=0.8,
            color=colors[i % len(colors)]
        )

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.005,
                f"{height:.4f}",
                ha='center',
                va='bottom',
                fontsize=8,
                color='#444444'
            )

    # Add target line if specified
    if target_line is not None:
        ax.axhline(
            y=target_line,
            color='red',
            linestyle='--',
            linewidth=1,
            label=f'Target ({target_line})'
        )

    # Formatting
    title = "Model Performance Comparison"
    if strategy_filter:
        title += f" ({strategy_filter})"
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))
    ax.grid(axis='y', alpha=0.3)
    ax.legend(frameon=False, loc='lower right')

    plt.tight_layout()

    # Save plot
    if output_path is None:
        suffix = f"_{strategy_filter}" if strategy_filter else ""
        output_path = csv_path.parent / f"{csv_path.stem}_plot{suffix}.png"

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plot: {output_path}")


def plot_binary_metrics_comparison(
    csv_path: Path,
    output_path: Optional[Path] = None,
    comparison_type: str = 'models'
):
    """
    Plot binary disease detection metrics (sensitivity, specificity, PPV, NPV).

    Parameters
    ----------
    csv_path : Path
        Path to the comparison CSV file
    output_path : Path, optional
        Path to save the plot
    comparison_type : str
        'models' or 'strategies'
    """
    metrics = ['binary_sensitivity',
               'binary_specificity', 'binary_ppv', 'binary_npv']

    # Load data
    df = pd.read_csv(csv_path)

    if df.empty:
        print(f"Warning: Empty CSV file: {csv_path}")
        return

    # Check if binary metrics exist
    if not all(m in df.columns for m in metrics):
        print(f"Warning: Binary metrics not found in CSV")
        return

    # Group appropriately
    if comparison_type == 'strategies':
        group_col = 'strategy'
        title_prefix = df['model_name'].iloc[0] if 'model_name' in df.columns else 'Model'
    else:
        group_col = 'model_name'
        title_prefix = "All Models"

    grouped = df.groupby(group_col)[metrics].mean().reset_index()
    groups = grouped[group_col].tolist()

    metric_labels = {
        'binary_sensitivity': 'Sensitivity',
        'binary_specificity': 'Specificity',
        'binary_ppv': 'PPV',
        'binary_npv': 'NPV',
    }

    # Prepare data
    metric_data = [grouped[m].tolist() for m in metrics]
    labels = [metric_labels[m] for m in metrics]

    # Use default color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    x = np.arange(len(groups))
    width = 0.8 / len(metric_data)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot grouped bars
    for i, (data, label) in enumerate(zip(metric_data, labels)):
        bars = ax.bar(
            x + i * width - (len(metric_data) - 1) * width / 2,
            data,
            width,
            label=label,
            alpha=0.8,
            color=colors[i % len(colors)]
        )

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.005,
                f"{height:.4f}",
                ha='center',
                va='bottom',
                fontsize=8,
                color='#444444'
            )

    # Formatting
    ax.set_title(
        f"{title_prefix} - Binary Disease Detection Metrics", fontsize=14, pad=10)
    ax.set_xlabel(group_col.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))
    ax.grid(axis='y', alpha=0.3)
    ax.legend(frameon=False, loc='lower right')

    plt.tight_layout()

    # Save plot
    if output_path is None:
        output_path = csv_path.parent / f"{csv_path.stem}_binary_metrics.png"

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved binary metrics plot: {output_path}")


def auto_visualize_all(output_dir: Path, target_line: Optional[float] = None):
    """
    Automatically find and visualize all comparison CSV files.

    Parameters
    ----------
    output_dir : Path
        Base output directory
    target_line : float, optional
        Target performance line
    """
    print("\n🎨 Auto-generating visualizations...")

    # Find model-specific comparison files
    model_csvs = list(output_dir.glob("*/comparison_*_all_strategies.csv"))

    for csv_path in model_csvs:
        print(f"\n📊 Visualizing: {csv_path.name}")
        plot_model_strategies_comparison(csv_path, target_line=target_line)
        plot_binary_metrics_comparison(csv_path, comparison_type='strategies')

    # Find all-model comparison files
    all_model_csvs = list(output_dir.glob("comparison_all_models_*.csv"))

    for csv_path in all_model_csvs:
        print(f"\n📊 Visualizing: {csv_path.name}")
        plot_all_models_comparison(csv_path, target_line=target_line)
        plot_binary_metrics_comparison(csv_path, comparison_type='models')

    # Find comprehensive comparison
    comprehensive_csv = output_dir / "comparison_all_experiments.csv"
    if comprehensive_csv.exists():
        print(f"\n📊 Visualizing: {comprehensive_csv.name}")
        plot_all_models_comparison(comprehensive_csv, target_line=target_line)

        # Also create per-strategy comparisons
        df = pd.read_csv(comprehensive_csv)
        strategies = df['strategy'].unique()

        for strategy in strategies:
            print(f"  Creating comparison for strategy: {strategy}")
            plot_all_models_comparison(
                comprehensive_csv,
                target_line=target_line,
                strategy_filter=strategy
            )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize model comparison results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Visualize specific comparison file
  python -m src.visualize_comparisons --csv outputs/convnexttiny/comparison_convnexttiny_all_strategies.csv
  
  # Visualize with custom metrics
  python -m src.visualize_comparisons --csv outputs/comparison_all_models_baseline.csv --metrics accuracy f1_macro binary_ppv
  
  # Auto-visualize all comparisons
  python -m src.visualize_comparisons --output-dir outputs --auto
  
  # With target line
  python -m src.visualize_comparisons --output-dir outputs --auto --target 0.8
        """
    )

    parser.add_argument(
        '--csv',
        type=str,
        help='Path to comparison CSV file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='src/output',
        help='Base output directory (default: outputs)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output path for the plot'
    )

    parser.add_argument(
        '--metrics',
        nargs='+',
        help='Metrics to plot (default: accuracy precision_macro recall_macro f1_macro)'
    )

    parser.add_argument(
        '--target',
        type=float,
        help='Add a target performance line at this value'
    )

    parser.add_argument(
        '--auto',
        action='store_true',
        help='Automatically visualize all comparison files in output-dir'
    )

    parser.add_argument(
        '--type',
        choices=['models', 'strategies', 'auto'],
        default='auto',
        help='Type of comparison (auto-detect by default)'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.auto:
        # Auto-visualize all
        auto_visualize_all(output_dir, args.target)
    elif args.csv:
        # Visualize specific CSV
        csv_path = Path(args.csv)

        if not csv_path.exists():
            print(f"Error: CSV file not found: {csv_path}")
            return

        output_path = Path(args.output) if args.output else None

        # Determine comparison type
        if args.type == 'auto':
            if 'all_strategies' in csv_path.name:
                comp_type = 'strategies'
            elif 'all_models' in csv_path.name or 'all_experiments' in csv_path.name:
                comp_type = 'models'
            else:
                comp_type = 'models'  # default
        else:
            comp_type = args.type

        print(f"\n📊 Visualizing {comp_type} comparison: {csv_path.name}")

        if comp_type == 'strategies':
            plot_model_strategies_comparison(
                csv_path,
                metrics=args.metrics,
                output_path=output_path,
                target_line=args.target
            )
            plot_binary_metrics_comparison(csv_path, output_path, 'strategies')
        else:
            plot_all_models_comparison(
                csv_path,
                metrics=args.metrics,
                output_path=output_path,
                target_line=args.target
            )
            plot_binary_metrics_comparison(csv_path, output_path, 'models')
    else:
        print("Error: Please specify either --csv or --auto")
        parser.print_help()

    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()
