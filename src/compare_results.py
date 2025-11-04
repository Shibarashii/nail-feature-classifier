"""
Comparison script for analyzing experiment results across models and strategies.
Generates CSV and JSON files that can be easily converted to tables.
"""
import json
import pandas as pd
from pathlib import Path
import argparse
from typing import List, Dict, Optional
import numpy as np


def load_metrics(metrics_path: Path) -> Optional[Dict]:
    """Load metrics from a JSON file."""
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {metrics_path}: {e}")
        return None


def extract_key_metrics(metrics: Dict) -> Dict:
    """Extract key metrics from the full metrics dictionary."""
    ml = metrics.get('ml_metrics', {})
    med = metrics.get('medical_metrics', {})

    result = {
        'model_name': metrics.get('model_name', 'unknown'),
        'strategy': metrics.get('strategy', 'unknown'),
        'num_samples': metrics.get('num_samples', 0),
        'num_classes': metrics.get('num_classes', 0),
        'lowest_val_loss': metrics.get('lowest_val_loss', np.nan),
        'epoch_of_lowest_val_loss': metrics.get('epoch_of_lowest_val_loss', 0),
        'accuracy': ml.get('accuracy', np.nan),
        'precision_macro': ml.get('precision_macro', np.nan),
        'recall_macro': ml.get('recall_macro', np.nan),
        'f1_macro': ml.get('f1_macro', np.nan),
        'precision_weighted': ml.get('precision_weighted', np.nan),
        'recall_weighted': ml.get('recall_weighted', np.nan),
        'f1_weighted': ml.get('f1_weighted', np.nan),
    }

    # Add medical metrics (macro-averaged if multi-class)
    if 'macro_avg' in med:
        result.update({
            'sensitivity_macro': med['macro_avg'].get('sensitivity', np.nan),
            'specificity_macro': med['macro_avg'].get('specificity', np.nan),
            'ppv_macro': med['macro_avg'].get('ppv', np.nan),
            'npv_macro': med['macro_avg'].get('npv', np.nan),
        })
    else:
        result.update({
            'sensitivity': med.get('sensitivity', np.nan),
            'specificity': med.get('specificity', np.nan),
            'ppv': med.get('ppv', np.nan),
            'npv': med.get('npv', np.nan),
        })

    return result


def find_all_experiments(base_path: Path) -> List[Path]:
    """Find all experiment directories containing metrics.json."""
    experiments = []
    for metrics_file in base_path.rglob("evaluation/metrics.json"):
        experiments.append(metrics_file.parent.parent)
    return sorted(experiments)


def compare_same_model(output_dir: Path, model_name: str) -> pd.DataFrame:
    """
    Compare all experiments (strategies) for a single model.

    Parameters
    ----------
    output_dir : Path
        Base output directory containing all experiments
    model_name : str
        Name of the model to compare (e.g., 'convnexttiny', 'efficientnetv2s')

    Returns
    -------
    pd.DataFrame
        Comparison table with all metrics
    """
    model_path = output_dir / model_name

    if not model_path.exists():
        print(f"Warning: Model directory not found: {model_path}")
        return pd.DataFrame()

    experiments = find_all_experiments(model_path)

    if not experiments:
        print(f"Warning: No experiments found for model: {model_name}")
        return pd.DataFrame()

    results = []
    for exp_path in experiments:
        metrics_path = exp_path / "evaluation" / "metrics.json"
        metrics = load_metrics(metrics_path)

        if metrics:
            key_metrics = extract_key_metrics(metrics)
            key_metrics['experiment_path'] = str(
                exp_path.relative_to(output_dir))
            key_metrics['timestamp'] = exp_path.name
            results.append(key_metrics)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Sort by strategy and timestamp
    df = df.sort_values(['strategy', 'timestamp'])

    # Save to CSV and JSON in the model's directory
    save_path_csv = model_path / f"comparison_{model_name}_all_strategies.csv"
    save_path_json = model_path / \
        f"comparison_{model_name}_all_strategies.json"

    df.to_csv(save_path_csv, index=False, float_format='%.4f')
    df.to_json(save_path_json, orient='records', indent=2)

    print(f"✓ Saved comparison for {model_name}: {save_path_csv}")

    return df


def compare_same_strategy(output_dir: Path, strategy: str) -> pd.DataFrame:
    """
    Compare all models for a single strategy.

    Parameters
    ----------
    output_dir : Path
        Base output directory containing all experiments
    strategy : str
        Name of the strategy to compare (e.g., 'baseline', 'full_finetune', 'gradual_unfreeze')

    Returns
    -------
    pd.DataFrame
        Comparison table with all metrics
    """
    experiments = find_all_experiments(output_dir)

    # Filter experiments by strategy
    strategy_experiments = [
        exp for exp in experiments
        if exp.parent.name == strategy
    ]

    if not strategy_experiments:
        print(f"Warning: No experiments found for strategy: {strategy}")
        return pd.DataFrame()

    results = []
    for exp_path in strategy_experiments:
        metrics_path = exp_path / "evaluation" / "metrics.json"
        metrics = load_metrics(metrics_path)

        if metrics:
            key_metrics = extract_key_metrics(metrics)
            key_metrics['experiment_path'] = str(
                exp_path.relative_to(output_dir))
            key_metrics['timestamp'] = exp_path.name
            results.append(key_metrics)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Sort by model_name and timestamp
    df = df.sort_values(['model_name', 'timestamp'])

    # Save to CSV and JSON
    save_path_csv = output_dir / f"comparison_all_models_{strategy}.csv"
    save_path_json = output_dir / f"comparison_all_models_{strategy}.json"

    df.to_csv(save_path_csv, index=False, float_format='%.4f')
    df.to_json(save_path_json, orient='records', indent=2)

    print(f"✓ Saved comparison for strategy '{strategy}': {save_path_csv}")

    return df


def compare_all_experiments(output_dir: Path) -> pd.DataFrame:
    """
    Compare all experiments across all models and strategies.

    Parameters
    ----------
    output_dir : Path
        Base output directory containing all experiments

    Returns
    -------
    pd.DataFrame
        Comprehensive comparison table with all metrics
    """
    experiments = find_all_experiments(output_dir)

    if not experiments:
        print(f"Warning: No experiments found in: {output_dir}")
        return pd.DataFrame()

    results = []
    for exp_path in experiments:
        metrics_path = exp_path / "evaluation" / "metrics.json"
        metrics = load_metrics(metrics_path)

        if metrics:
            key_metrics = extract_key_metrics(metrics)
            key_metrics['experiment_path'] = str(
                exp_path.relative_to(output_dir))
            key_metrics['timestamp'] = exp_path.name
            results.append(key_metrics)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Sort by model_name, strategy, and timestamp
    df = df.sort_values(['model_name', 'strategy', 'timestamp'])

    # Save to CSV and JSON
    save_path_csv = output_dir / "comparison_all_experiments.csv"
    save_path_json = output_dir / "comparison_all_experiments.json"

    df.to_csv(save_path_csv, index=False, float_format='%.4f')
    df.to_json(save_path_json, orient='records', indent=2)

    print(f"✓ Saved comprehensive comparison: {save_path_csv}")

    return df


def generate_summary_statistics(output_dir: Path) -> pd.DataFrame:
    """
    Generate summary statistics for each model-strategy combination.
    Aggregates multiple runs of the same model-strategy pair.

    Parameters
    ----------
    output_dir : Path
        Base output directory containing all experiments

    Returns
    -------
    pd.DataFrame
        Summary statistics table
    """
    experiments = find_all_experiments(output_dir)

    if not experiments:
        print(f"Warning: No experiments found in: {output_dir}")
        return pd.DataFrame()

    results = []
    for exp_path in experiments:
        metrics_path = exp_path / "evaluation" / "metrics.json"
        metrics = load_metrics(metrics_path)

        if metrics:
            key_metrics = extract_key_metrics(metrics)
            results.append(key_metrics)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Group by model and strategy
    grouped = df.groupby(['model_name', 'strategy'])

    # Calculate statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    summary_stats = []
    for (model, strategy), group in grouped:
        stats = {
            'model_name': model,
            'strategy': strategy,
            'num_runs': len(group),
        }

        for col in numeric_cols:
            if col not in ['num_samples', 'num_classes', 'epoch_of_lowest_val_loss']:
                stats[f'{col}_mean'] = group[col].mean()
                stats[f'{col}_std'] = group[col].std()
                stats[f'{col}_min'] = group[col].min()
                stats[f'{col}_max'] = group[col].max()

        summary_stats.append(stats)

    summary_df = pd.DataFrame(summary_stats)

    # Sort by model_name and strategy
    summary_df = summary_df.sort_values(['model_name', 'strategy'])

    # Save to CSV and JSON
    save_path_csv = output_dir / "summary_statistics.csv"
    save_path_json = output_dir / "summary_statistics.json"

    summary_df.to_csv(save_path_csv, index=False, float_format='%.4f')
    summary_df.to_json(save_path_json, orient='records', indent=2)

    print(f"✓ Saved summary statistics: {save_path_csv}")

    return summary_df


def generate_best_results(output_dir: Path, metric: str = 'accuracy') -> pd.DataFrame:
    """
    Generate a table showing the best result for each model-strategy combination.

    Parameters
    ----------
    output_dir : Path
        Base output directory containing all experiments
    metric : str
        Metric to use for determining "best" (default: 'accuracy')

    Returns
    -------
    pd.DataFrame
        Best results table
    """
    experiments = find_all_experiments(output_dir)

    if not experiments:
        print(f"Warning: No experiments found in: {output_dir}")
        return pd.DataFrame()

    results = []
    for exp_path in experiments:
        metrics_path = exp_path / "evaluation" / "metrics.json"
        metrics = load_metrics(metrics_path)

        if metrics:
            key_metrics = extract_key_metrics(metrics)
            key_metrics['experiment_path'] = str(
                exp_path.relative_to(output_dir))
            key_metrics['timestamp'] = exp_path.name
            results.append(key_metrics)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Get best result for each model-strategy combination
    if metric not in df.columns:
        print(
            f"Warning: Metric '{metric}' not found. Using 'accuracy' instead.")
        metric = 'accuracy'

    # Group by model and strategy, get the row with max metric value
    idx = df.groupby(['model_name', 'strategy'])[metric].idxmax()
    best_df = df.loc[idx].copy()

    # Sort by model_name and strategy
    best_df = best_df.sort_values(['model_name', 'strategy'])

    # Save to CSV and JSON
    save_path_csv = output_dir / f"best_results_by_{metric}.csv"
    save_path_json = output_dir / f"best_results_by_{metric}.json"

    best_df.to_csv(save_path_csv, index=False, float_format='%.4f')
    best_df.to_json(save_path_json, orient='records', indent=2)

    print(f"✓ Saved best results by {metric}: {save_path_csv}")

    return best_df


def main():
    parser = argparse.ArgumentParser(
        description="Compare experiment results across models and strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Compare all strategies for a specific model
  python -m src.compare_results --output-dir outputs --model convnexttiny
  
  # Compare all models for a specific strategy
  python -m src.compare_results --output-dir outputs --strategy baseline
  
  # Compare all experiments
  python -m src.compare_results --output-dir outputs --all
  
  # Generate summary statistics
  python -m src.compare_results --output-dir outputs --summary
  
  # Find best results by accuracy
  python -m src.compare_results --output-dir outputs --best accuracy
        """
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='src/output',
        help='Base output directory containing experiment results (default: outputs)'
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Compare all strategies for a specific model (e.g., convnexttiny)'
    )

    parser.add_argument(
        '--strategy',
        type=str,
        help='Compare all models for a specific strategy (e.g., baseline, full_finetune)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Compare all experiments across all models and strategies'
    )

    parser.add_argument(
        '--summary',
        action='store_true',
        help='Generate summary statistics for all model-strategy combinations'
    )

    parser.add_argument(
        '--best',
        type=str,
        metavar='METRIC',
        help='Generate best results table by specified metric (e.g., accuracy, f1_macro)'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        return

    # Execute requested comparisons
    if args.model:
        print(f"\n📊 Comparing all strategies for model: {args.model}")
        compare_same_model(output_dir, args.model)

    if args.strategy:
        print(f"\n📊 Comparing all models for strategy: {args.strategy}")
        compare_same_strategy(output_dir, args.strategy)

    if args.all:
        print(f"\n📊 Comparing all experiments")
        compare_all_experiments(output_dir)

    if args.summary:
        print(f"\n📊 Generating summary statistics")
        generate_summary_statistics(output_dir)

    if args.best:
        print(f"\n📊 Generating best results by {args.best}")
        generate_best_results(output_dir, args.best)

    # If no specific comparison requested, do everything
    if not any([args.model, args.strategy, args.all, args.summary, args.best]):
        print(f"\n📊 No specific comparison requested. Generating all comparisons...")

        # Get all unique models and strategies
        experiments = find_all_experiments(output_dir)
        models = set()
        strategies = set()

        for exp in experiments:
            metrics_path = exp / "evaluation" / "metrics.json"
            metrics = load_metrics(metrics_path)
            if metrics:
                models.add(metrics.get('model_name', 'unknown'))
                strategies.add(metrics.get('strategy', 'unknown'))

        # Compare each model
        for model in sorted(models):
            print(f"\n📊 Comparing all strategies for model: {model}")
            compare_same_model(output_dir, model)

        # Compare each strategy
        for strategy in sorted(strategies):
            print(f"\n📊 Comparing all models for strategy: {strategy}")
            compare_same_strategy(output_dir, strategy)

        # All experiments
        print(f"\n📊 Comparing all experiments")
        compare_all_experiments(output_dir)

        # Summary statistics
        print(f"\n📊 Generating summary statistics")
        generate_summary_statistics(output_dir)

        # Best results
        print(f"\n📊 Generating best results")
        generate_best_results(output_dir, 'accuracy')
        generate_best_results(output_dir, 'f1_macro')

    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()
