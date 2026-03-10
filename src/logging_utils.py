import os

import numpy as np
import pandas as pd


def setup_dirs(results_dir='results'):
    """Create results/csv and results/figures directories."""
    csv_dir     = os.path.join(results_dir, 'csv')
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(csv_dir,     exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    return csv_dir, figures_dir


def build_epoch_df(method, run_id, seed, epoch_records):
    """Return a DataFrame with per-epoch metrics for a single run."""
    df = pd.DataFrame(epoch_records)
    df.insert(0, 'seed',   seed)
    df.insert(0, 'run_id', run_id)
    df.insert(0, 'method', method)
    return df


def build_run_record(method, run_id, seed, result):
    """Return a dict summarising a single run."""
    epoch_records  = result['epoch_records']
    eval_times     = [r['eval_time']           for r in epoch_records
                      if not np.isnan(r['eval_time'])]
    epoch_times    = [r['train_epoch_time']    for r in epoch_records]
    sampling_times = [r['train_sampling_time'] for r in epoch_records]

    return {
        'method':                  method,
        'run_id':                  run_id,
        'seed':                    seed,
        'best_val_auc':            result['best_val_auc'],
        'best_test_auc':           result['best_test_auc'],
        'final_val_auc':           result['final_val_auc'],
        'final_test_auc':          result['final_test_auc'],
        'mean_train_sampling_time': np.mean(sampling_times),
        'mean_train_epoch_time':   np.mean(epoch_times),
        'mean_eval_time':          np.mean(eval_times) if eval_times else float('nan'),
        'total_run_time':          result['total_run_time'],
    }


def save_epoch_metrics(epoch_dfs, csv_dir):
    """Concatenate per-run epoch DataFrames and write epoch_metrics.csv."""
    df   = pd.concat(epoch_dfs, ignore_index=True)
    path = os.path.join(csv_dir, 'epoch_metrics.csv')
    df.to_csv(path, index=False)
    print(f'Saved: {path}')
    return df


def save_run_summary(run_records, csv_dir):
    """Write one row per run to run_summary.csv."""
    df   = pd.DataFrame(run_records)
    path = os.path.join(csv_dir, 'run_summary.csv')
    df.to_csv(path, index=False)
    print(f'Saved: {path}')
    return df


def compute_aggregate(run_records):
    """Return a dict of aggregated statistics across runs."""
    df     = pd.DataFrame(run_records)
    method = df['method'].iloc[0] if len(df) > 0 else 'unknown'
    n_runs = len(df)
    return {
        'method':                 method,
        'n_runs':                 n_runs,
        'mean_best_val_auc':      df['best_val_auc'].mean(),
        'std_best_val_auc':       df['best_val_auc'].std(ddof=1),
        'mean_best_test_auc':     df['best_test_auc'].mean(),
        'std_best_test_auc':      df['best_test_auc'].std(ddof=1),
        'mean_train_epoch_time':  df['mean_train_epoch_time'].mean(),
        'std_train_epoch_time':   df['mean_train_epoch_time'].std(ddof=1),
        'mean_eval_time':         df['mean_eval_time'].mean(),
        'std_eval_time':          df['mean_eval_time'].std(ddof=1),
        'mean_total_run_time':    df['total_run_time'].mean(),
        'std_total_run_time':     df['total_run_time'].std(ddof=1),
    }


def save_aggregate_summary(run_records, csv_dir):
    """Write one row of aggregated statistics to aggregate_summary.csv."""
    agg  = compute_aggregate(run_records)
    df   = pd.DataFrame([agg])
    path = os.path.join(csv_dir, 'aggregate_summary.csv')
    df.to_csv(path, index=False)
    print(f'Saved: {path}')
    return df
