import json

with open('notebooks/main_experiment.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell24_new = '''# ── 2.2 OoD Detection: AUC-ROC Comparison — Bar Chart ───────────────────────
import ast, numpy as np
import pandas as pd

ood_res = pd.read_csv(RESULTS_DIR / 'ood_results.csv')
ood_res['hp'] = ood_res['hyperparams'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x != '{}' else {})
ood_res['ef_search'] = ood_res['hp'].apply(lambda x: x.get('ef_search', None))

strategy_labels = {
    'dnn':             'Deep Nearest\\nNeighbors (DNN)',
    'nn_guide_msp':    'NN-Guide\\n(MSP)',
    'nn_guide_energy': 'NN-Guide\\n(Energy)',
}
strategies = list(strategy_labels.keys())

# Global best/worst efSearch across all HNSW rows
hnsw_all = ood_res[ood_res['algorithm'] == 'hnsw']
best_ef  = hnsw_all['ef_search'].max()
worst_ef = hnsw_all['ef_search'].min()

# Build bar values per strategy
rows = []
for strat in strategies:
    sub = ood_res[ood_res['strategy'] == strat]
    exact_auc  = sub[sub['algorithm'] == 'exact']['auroc'].mean()
    random_auc = sub[sub['algorithm'] == 'random']['auroc'].mean()
    hnsw_sub   = sub[sub['algorithm'] == 'hnsw']
    best_hnsw  = hnsw_sub[hnsw_sub['ef_search'] == best_ef]['auroc'].mean() if len(hnsw_sub) > 0 else float('nan')
    worst_hnsw = hnsw_sub[hnsw_sub['ef_search'] == worst_ef]['auroc'].mean() if len(hnsw_sub) > 0 else float('nan')
    rows.append({'strategy': strat, 'exact': exact_auc, 'best_hnsw': best_hnsw,
                 'worst_hnsw': worst_hnsw, 'random': random_auc})

df_bar = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle('OoD Detection: AUC-ROC by Method and Strategy', fontsize=13, fontweight='bold')

x = np.arange(len(strategies))
width = 0.2
bars = [
    ('exact',      'steelblue', 'Exact kNN'),
    ('best_hnsw',  '#2ca02c',   f'HNSW best (efS={int(best_ef) if not np.isnan(best_ef) else "?"})',),
    ('worst_hnsw', '#ff7f0e',   f'HNSW worst (efS={int(worst_ef) if not np.isnan(worst_ef) else "?"})',),
    ('random',     'gray',      'Random'),
]

for i, (col, color, label) in enumerate(bars):
    offset = (i - 1.5) * width
    vals = df_bar[col].values
    rects = ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)
    for rect, v in zip(rects, vals):
        if not np.isnan(v):
            ax.text(rect.get_x() + rect.get_width()/2, v + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7)

ax.set_ylabel('AUC-ROC')
ax.set_ylim(0, 1.08)
ax.set_xticks(x)
ax.set_xticklabels([strategy_labels[s] for s in strategies])
ax.legend(fontsize=9)
ax.axhline(1.0, color='black', linestyle='--', alpha=0.2)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'ood_auroc_vs_time.png', dpi=150)
plt.show()
print('Saved: ood_auroc_vs_time.png')
'''

nb['cells'][24]['source'] = [cell24_new]

with open('notebooks/main_experiment.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Cell 24 updated.')
