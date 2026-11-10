"""Plot recorded training and full-validation curves without reevaluating test data."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/single_da_02_t2m_dim32_main'
OUT = RUN / 'diagnostics'
OUT.mkdir(exist_ok=True)


def read(path):
    return pd.DataFrame([json.loads(line) for line in path.read_text().splitlines() if line.strip()])


def main():
    colors = {'da': '#2563eb', 'neural_correction': '#e87920'}
    names = {'da': 'DA', 'neural_correction': 'Neural Correction'}
    validations, report = {}, {}
    fig, axes = plt.subplots(3, 1, figsize=(10, 10.5), constrained_layout=True)
    for key in colors:
        train = read(RUN / key / 'training.jsonl')
        val = read(RUN / key / 'validation.jsonl').drop_duplicates('step', keep='last').sort_values('step')
        validations[key] = val
        epoch = train['step'] / 4384
        axes[0].plot(epoch, train['train_loss'], color=colors[key], alpha=.08, linewidth=.5)
        axes[0].plot(epoch, train['train_loss'].rolling(51, min_periods=1, center=True).mean(),
                     color=colors[key], label=names[key], linewidth=1.6)
        axes[1].plot(val['step'] / 4384, val['rmse_K'], '-o', color=colors[key], label=names[key], markersize=4)
        best = val.loc[val['rmse_K'].idxmin()]
        axes[1].scatter(best['step'] / 4384, best['rmse_K'], marker='*', s=130, color=colors[key], zorder=5)
        logged_epoch_means = train.groupby('epoch')['train_loss'].mean()
        report[key] = {'best_step': int(best['step']), 'best_validation_rmse_K': float(best['rmse_K']),
                       'last_validation_rmse_K': float(val.iloc[-1]['rmse_K']),
                       'last_minus_best_K': float(val.iloc[-1]['rmse_K'] - best['rmse_K']),
                       'logged_training_epoch_means_approx': {str(int(k)): float(v) for k, v in logged_epoch_means.items()},
                       'last_learning_rate': float(val.iloc[-1]['learning_rate'])}
    da = validations['da'].set_index('step')
    nc = validations['neural_correction'].set_index('step')
    common = da.index.intersection(nc.index)
    difference = (da.loc[common, 'rmse_K'] - nc.loc[common, 'rmse_K']) * 1000
    axes[2].plot(common / 4384, difference, '-o', color='#6d28d9', markersize=4)
    axes[2].axhline(0, color='#64748b', linewidth=1)
    axes[1].axhline(da.iloc[0]['baseline0_rmse_K'], linestyle='--', color='#64748b', label='Baseline 0')
    axes[0].set(title='Training loss: logged samples and 51-record moving mean', ylabel='Area-weighted normalized MSE')
    axes[1].set(title='Full 2021 validation: 1,460 cycles at each epoch', ylabel='Pooled area-weighted RMSE (K)')
    axes[2].set(title='Paired validation gap: negative means DA is better', ylabel='DA minus Neural Correction (mK)', xlabel='Training epoch')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.set_xlim(0, 10)
    axes[0].legend(loc='upper right')
    axes[1].legend(loc='upper right')
    fig.suptitle('Single-cycle DA v1 | seed 42 | validation-selected checkpoints', fontsize=14)
    fig.savefig(OUT / 'training_validation_curves.png', dpi=180)
    fig.savefig(OUT / 'training_validation_curves.pdf')
    plt.close(fig)
    report['scope'] = 'Training curve is subsampled logging, not exact epoch training loss; validation is full-year. Ten epochs do not prove convergence.'
    (OUT / 'curve_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

