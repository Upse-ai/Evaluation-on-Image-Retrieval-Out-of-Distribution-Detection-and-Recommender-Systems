"""
Evaluation Metrics for Similarity Search

This module provides evaluation functions for comparing approximate
nearest neighbor algorithms against exact search.

Metrics:
- Recall@k: Fraction of true nearest neighbors retrieved
- Query time: Time to perform searches
- Index build time: Time to construct the search index
- Amortized time: Total time normalized by number of queries
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import time


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    algorithm: str
    metric: str
    k: int
    recall: float
    query_time: float
    avg_query_time: float
    index_build_time: float
    amortized_time: float
    n_queries: int
    params: Dict[str, Any]


def compute_recall_at_k(predicted_indices: np.ndarray,
                        ground_truth_indices: np.ndarray) -> float:
    """
    Compute recall@k for predicted neighbors vs ground truth.

    Recall@k = (# of true neighbors in top k predictions) / k

    Args:
        predicted_indices: Predicted neighbor indices, shape (n_queries, k)
        ground_truth_indices: True neighbor indices, shape (n_queries, k)

    Returns:
        Average recall@k across all queries
    """
    n_queries = predicted_indices.shape[0]
    k = predicted_indices.shape[1]

    recalls = []
    for i in range(n_queries):
        pred_set = set(predicted_indices[i])
        true_set = set(ground_truth_indices[i])

        # Count how many true neighbors are in predictions
        intersection = len(pred_set & true_set)
        recall = intersection / k
        recalls.append(recall)

    return np.mean(recalls)


def compute_precision_at_k(predicted_indices: np.ndarray,
                           ground_truth_indices: np.ndarray) -> float:
    """
    Compute precision@k (same as recall when comparing k predictions to k truth).

    Args:
        predicted_indices: Predicted neighbor indices, shape (n_queries, k)
        ground_truth_indices: True neighbor indices, shape (n_queries, k)

    Returns:
        Average precision@k across all queries
    """
    # For kNN where we compare k predictions to k ground truth,
    # precision@k equals recall@k
    return compute_recall_at_k(predicted_indices, ground_truth_indices)


def compute_mean_reciprocal_rank(predicted_indices: np.ndarray,
                                  ground_truth_indices: np.ndarray) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    MRR considers the rank of the first correct result.

    Args:
        predicted_indices: Predicted neighbor indices, shape (n_queries, k)
        ground_truth_indices: True neighbor indices, shape (n_queries, k)

    Returns:
        Mean reciprocal rank
    """
    n_queries = predicted_indices.shape[0]
    reciprocal_ranks = []

    for i in range(n_queries):
        true_set = set(ground_truth_indices[i])

        # Find rank of first correct prediction
        for rank, pred in enumerate(predicted_indices[i], 1):
            if pred in true_set:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    return np.mean(reciprocal_ranks)


def evaluate_algorithm(approx_result: Dict[str, Any],
                       exact_result: Dict[str, Any],
                       k: int) -> EvaluationResult:
    """
    Evaluate an approximate algorithm against exact search.

    Args:
        approx_result: Results from approximate algorithm
        exact_result: Results from exact kNN (ground truth)
        k: Number of neighbors evaluated

    Returns:
        EvaluationResult with all metrics
    """
    approx_indices = approx_result['k_results'][k]['indices']
    exact_indices = exact_result['k_results'][k]['indices']

    recall = compute_recall_at_k(approx_indices, exact_indices)
    query_time = approx_result['k_results'][k]['query_time']
    avg_query_time = approx_result['k_results'][k]['avg_query_time']
    index_build_time = approx_result['index_build_time']
    n_queries = approx_result['n_queries']

    # Amortized time: (build time + total query time) / n_queries
    amortized_time = (index_build_time + query_time) / n_queries

    return EvaluationResult(
        algorithm=approx_result['algorithm'],
        metric=approx_result['metric'],
        k=k,
        recall=recall,
        query_time=query_time,
        avg_query_time=avg_query_time,
        index_build_time=index_build_time,
        amortized_time=amortized_time,
        n_queries=n_queries,
        params=approx_result['params']
    )


def run_full_evaluation(database: np.ndarray,
                        queries: np.ndarray,
                        algorithms: List[str],
                        metrics: List[str],
                        k_values: List[int] = [1, 5, 10, 30],
                        algorithm_params: Optional[Dict[str, Dict]] = None) -> pd.DataFrame:
    """
    Run full evaluation across multiple algorithms, metrics, and k values.

    Args:
        database: Database vectors
        queries: Query vectors
        algorithms: List of algorithm names
        metrics: List of distance metrics
        k_values: List of k values to evaluate
        algorithm_params: Optional algorithm-specific parameters

    Returns:
        DataFrame with all evaluation results
    """
    from .similarity_search import run_experiment

    if algorithm_params is None:
        algorithm_params = {}

    all_results = []
    max_k = max(k_values)

    for metric in metrics:
        # First, get ground truth from exact search
        print(f"\nRunning exact kNN ({metric})...")
        exact_result = run_experiment(
            database, queries, 'exact', metric, k_values=[max_k]
        )

        for algorithm in algorithms:
            params = algorithm_params.get(algorithm, {})
            print(f"Running {algorithm} ({metric})...")

            approx_result = run_experiment(
                database, queries, algorithm, metric,
                k_values=[max_k], **params
            )

            for k in k_values:
                # Truncate results to k
                approx_k_result = {
                    'algorithm': approx_result['algorithm'],
                    'metric': approx_result['metric'],
                    'params': approx_result['params'],
                    'index_build_time': approx_result['index_build_time'],
                    'n_queries': approx_result['n_queries'],
                    'k_results': {
                        k: {
                            'indices': approx_result['k_results'][max_k]['indices'][:, :k],
                            'distances': approx_result['k_results'][max_k]['distances'][:, :k],
                            'query_time': approx_result['k_results'][max_k]['query_time'],
                            'avg_query_time': approx_result['k_results'][max_k]['avg_query_time']
                        }
                    }
                }

                exact_k_result = {
                    'k_results': {
                        k: {
                            'indices': exact_result['k_results'][max_k]['indices'][:, :k]
                        }
                    }
                }

                eval_result = evaluate_algorithm(approx_k_result, exact_k_result, k)
                all_results.append(asdict(eval_result))

    return pd.DataFrame(all_results)


def create_summary_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table from evaluation results.

    Args:
        results_df: DataFrame with evaluation results

    Returns:
        Pivoted summary table
    """
    summary = results_df.pivot_table(
        values=['recall', 'avg_query_time', 'index_build_time', 'amortized_time'],
        index=['algorithm', 'metric'],
        columns='k',
        aggfunc='mean'
    )

    return summary


def plot_recall_vs_k(results_df: pd.DataFrame,
                     metric: str = 'euclidean',
                     save_path: Optional[Path] = None):
    """
    Plot recall@k curves for different algorithms.

    Args:
        results_df: DataFrame with evaluation results
        metric: Distance metric to plot
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))

    filtered_df = results_df[results_df['metric'] == metric]

    for algorithm in filtered_df['algorithm'].unique():
        algo_data = filtered_df[filtered_df['algorithm'] == algorithm]
        algo_data = algo_data.sort_values('k')

        plt.plot(algo_data['k'], algo_data['recall'],
                marker='o', label=algorithm, linewidth=2, markersize=8)

    plt.xlabel('k', fontsize=12)
    plt.ylabel('Recall@k', fontsize=12)
    plt.title(f'Recall@k vs k ({metric.capitalize()} Distance)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()


def plot_recall_vs_time(results_df: pd.DataFrame,
                        k: int = 10,
                        save_path: Optional[Path] = None):
    """
    Plot recall vs query time trade-off.

    Args:
        results_df: DataFrame with evaluation results
        k: k value to plot
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))

    filtered_df = results_df[results_df['k'] == k]

    for metric in filtered_df['metric'].unique():
        metric_data = filtered_df[filtered_df['metric'] == metric]

        for algorithm in metric_data['algorithm'].unique():
            algo_data = metric_data[metric_data['algorithm'] == algorithm]

            plt.scatter(algo_data['avg_query_time'] * 1000,
                       algo_data['recall'],
                       label=f"{algorithm} ({metric})",
                       s=100)

    plt.xlabel('Average Query Time (ms)', fontsize=12)
    plt.ylabel(f'Recall@{k}', fontsize=12)
    plt.title(f'Recall@{k} vs Query Time Trade-off', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()


def plot_timing_comparison(results_df: pd.DataFrame,
                           save_path: Optional[Path] = None):
    """
    Create bar chart comparing timing metrics across algorithms.

    Args:
        results_df: DataFrame with evaluation results
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Average over k values
    timing_df = results_df.groupby(['algorithm', 'metric']).agg({
        'index_build_time': 'mean',
        'avg_query_time': 'mean',
        'amortized_time': 'mean'
    }).reset_index()

    # Build time
    ax = axes[0]
    pivot = timing_df.pivot(index='algorithm', columns='metric', values='index_build_time')
    pivot.plot(kind='bar', ax=ax)
    ax.set_ylabel('Time (s)')
    ax.set_title('Index Build Time')
    ax.legend(title='Metric')
    ax.tick_params(axis='x', rotation=45)

    # Query time
    ax = axes[1]
    pivot = timing_df.pivot(index='algorithm', columns='metric', values='avg_query_time')
    pivot.plot(kind='bar', ax=ax)
    ax.set_ylabel('Time (s)')
    ax.set_title('Average Query Time')
    ax.legend(title='Metric')
    ax.tick_params(axis='x', rotation=45)

    # Amortized time
    ax = axes[2]
    pivot = timing_df.pivot(index='algorithm', columns='metric', values='amortized_time')
    pivot.plot(kind='bar', ax=ax)
    ax.set_ylabel('Time (s)')
    ax.set_title('Amortized Time per Query')
    ax.legend(title='Metric')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()


def save_results(results_df: pd.DataFrame,
                 output_dir: Path,
                 experiment_name: str = 'experiment'):
    """
    Save evaluation results to files.

    Args:
        results_df: DataFrame with evaluation results
        output_dir: Output directory
        experiment_name: Name prefix for output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = output_dir / f'{experiment_name}_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    # Save JSON summary
    summary = {
        'experiment': experiment_name,
        'algorithms': results_df['algorithm'].unique().tolist(),
        'metrics': results_df['metric'].unique().tolist(),
        'k_values': sorted(results_df['k'].unique().tolist()),
        'summary_stats': {}
    }

    for algorithm in results_df['algorithm'].unique():
        algo_data = results_df[results_df['algorithm'] == algorithm]
        summary['summary_stats'][algorithm] = {
            'mean_recall': algo_data['recall'].mean(),
            'mean_query_time': algo_data['avg_query_time'].mean(),
            'index_build_time': algo_data['index_build_time'].iloc[0]
        }

    json_path = output_dir / f'{experiment_name}_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {json_path}")


def print_results_table(results_df: pd.DataFrame):
    """
    Print a formatted results table.

    Args:
        results_df: DataFrame with evaluation results
    """
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    for metric in results_df['metric'].unique():
        print(f"\n{metric.upper()} DISTANCE")
        print("-" * 60)

        metric_df = results_df[results_df['metric'] == metric]

        # Create pivot table for display
        pivot = metric_df.pivot_table(
            values='recall',
            index='algorithm',
            columns='k'
        )

        print("\nRecall@k:")
        print(pivot.round(4).to_string())

        print("\nAverage Query Time (ms):")
        time_pivot = metric_df.pivot_table(
            values='avg_query_time',
            index='algorithm',
            columns='k'
        ) * 1000  # Convert to ms
        print(time_pivot.round(2).to_string())

        print("\nIndex Build Time (s):")
        build_times = metric_df.groupby('algorithm')['index_build_time'].first()
        for algo, time in build_times.items():
            print(f"  {algo}: {time:.4f}s")


if __name__ == "__main__":
    # Example usage demonstration
    print("Evaluation Module for Similarity Search")
    print("=" * 50)

    # Simulate some results for demonstration
    np.random.seed(42)

    # Example: Create mock evaluation results
    predicted = np.array([[1, 2, 3, 4, 5], [10, 11, 12, 13, 14]])
    ground_truth = np.array([[1, 2, 3, 6, 7], [10, 11, 15, 16, 17]])

    recall = compute_recall_at_k(predicted, ground_truth)
    print(f"\nExample recall@5: {recall:.4f}")

    mrr = compute_mean_reciprocal_rank(predicted, ground_truth)
    print(f"Example MRR: {mrr:.4f}")
