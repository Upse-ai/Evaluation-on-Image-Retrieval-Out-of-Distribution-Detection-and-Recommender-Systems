import json

with open('notebooks/main_experiment.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ── CELL 14: Speed-accuracy scatter → curves ──────────────────────────────
cell14_new = '''# ── Plot 3: Recall@10 vs Query Time — Curves (Speed-Accuracy Trade-off) ──────
import ast

img_res = pd.read_csv(RESULTS_DIR / 'image_search_results.csv')
img_res['hp'] = img_res['hyperparams'].apply(lambda x: ast.literal_eval(x) if x != '{}' else {})
img_res['ef_search'] = img_res['hp'].apply(lambda x: x.get('ef_search', None))
img_res['nbits']     = img_res['hp'].apply(lambda x: x.get('nbits', None))
img_res['M']         = img_res['hp'].apply(lambda x: x.get('M', None))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Speed-Accuracy Trade-off: Recall@10 vs Avg Query Time', fontsize=13, fontweight='bold')

for ax, (config, metric) in zip(axes, [('conf1', 'cosine'), ('conf2', 'euclidean')]):
    sub = img_res[(img_res['config'] == config) &
                  (img_res['metric'] == metric) &
                  (img_res['query_set'] == 'InDQuery') &
                  (img_res['k'] == 10)]

    # Exact — single star marker
    exact = sub[sub['algorithm'] == 'exact']
    if len(exact) > 0:
        ax.scatter(exact['avg_query_time'].mean() * 1000, exact['recall'].mean(),
                   color='steelblue', s=180, zorder=5, label='Exact', marker='*')

    # HNSW — one curve per M value (average over efC, sort by efSearch)
    hnsw = sub[sub['algorithm'] == 'hnsw'].copy()
    colors_hnsw = ['#ff7f0e', '#d62728', '#9467bd']
    for idx, M_val in enumerate(sorted(hnsw['M'].dropna().unique())):
        m_sub = (hnsw[hnsw['M'] == M_val]
                 .groupby('ef_search')
                 .agg(recall=('recall','mean'), time=('avg_query_time','mean'))
                 .reset_index()
                 .sort_values('ef_search'))
        ax.plot(m_sub['time'] * 1000, m_sub['recall'], marker='o',
                color=colors_hnsw[idx % len(colors_hnsw)],
                label=f'HNSW M={int(M_val)}', linewidth=2)

    # LSH — single curve sorted by nbits, annotated
    lsh = sub[sub['algorithm'] == 'lsh'].copy()
    if len(lsh) > 0:
        lsh_curve = (lsh.groupby('nbits')
                     .agg(recall=('recall','mean'), time=('avg_query_time','mean'))
                     .reset_index()
                     .sort_values('nbits'))
        ax.plot(lsh_curve['time'] * 1000, lsh_curve['recall'], marker='s',
                color='green', label='LSH', linestyle='--', linewidth=2)
        for _, row in lsh_curve.iterrows():
            ax.annotate(str(int(row['nbits'])), (row['time']*1000, row['recall']),
                        textcoords='offset points', xytext=(4, 4), fontsize=7)

    ax.set_xlabel('Avg Query Time (ms)')
    ax.set_ylabel('Recall@10')
    ax.set_title(f'{config} — {metric}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'recall_vs_querytime.png', dpi=150)
plt.show()

# ── Plot 4: HNSW — Effect of efSearch on Recall@10 ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('HNSW: Effect of efSearch on Recall@10', fontsize=13, fontweight='bold')

for ax, (config, metric) in zip(axes, [('conf1', 'cosine'), ('conf2', 'euclidean')]):
    sub = img_res[(img_res['config'] == config) &
                  (img_res['metric'] == metric) &
                  (img_res['algorithm'] == 'hnsw') &
                  (img_res['query_set'] == 'InDQuery') &
                  (img_res['k'] == 10)]
    if len(sub) == 0:
        ax.set_title(f'{config} — {metric} (no data)'); continue
    for M_val in sorted(sub['M'].dropna().unique()):
        m_sub = sub[sub['M'] == M_val].groupby('ef_search')['recall'].mean().reset_index()
        ax.plot(m_sub['ef_search'], m_sub['recall'], marker='o', label=f'M={int(M_val)}')
    ax.set_xlabel('efSearch'); ax.set_ylabel('Recall@10')
    ax.set_title(f'{config} — {metric}')
    ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'hnsw_efsearch_recall.png', dpi=150)
plt.show()

# ── Plot 5: LSH — Effect of nbits on Recall@10 ──────────────────────────────
fig, axes = plt.subplots(1, 1, figsize=(7, 5))
sub = img_res[(img_res['algorithm'] == 'lsh') &
              (img_res['query_set'] == 'InDQuery') &
              (img_res['k'] == 10)]
for config in ['conf1', 'conf2']:
    cs = sub[sub['config'] == config].groupby('nbits')['recall'].mean().reset_index()
    axes.plot(cs['nbits'], cs['recall'], marker='o', label=config)
axes.set_xlabel('nbits'); axes.set_ylabel('Recall@10')
axes.set_title('LSH: Effect of nbits on Recall@10')
axes.legend(); axes.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'lsh_nbits_recall.png', dpi=150)
plt.show()
print('Saved: recall_vs_querytime.png, hnsw_efsearch_recall.png, lsh_nbits_recall.png')
'''

nb['cells'][14]['source'] = [cell14_new]
print('Cell 14 updated')

# ── CELL 17: MovieLens — load from CSV, fix efSearch parsing, per-M curves ──
cell17_new = '''# ── MovieLens Visualizations (load from CSV for robustness) ─────────────────
import ast, pandas as pd

ml_results = pd.read_csv(RESULTS_DIR / 'movielens_search_results.csv')
ml_results['hp'] = ml_results['hyperparams'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x != '{}' else {})
ml_results['ef_search'] = ml_results['hp'].apply(lambda x: x.get('ef_search', None))
ml_results['nbits_val'] = ml_results['hp'].apply(lambda x: x.get('nbits', None))
ml_results['M_val']     = ml_results['hp'].apply(lambda x: x.get('M', None))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('MovieLens — Similarity Search Results', fontsize=13, fontweight='bold')

# Left: Recall@k
ax = axes[0]
for algo in ['lsh', 'hnsw']:
    sub = ml_results[(ml_results['algorithm'] == algo) & (ml_results['n_queries'] == 1000)]
    mean_rec = sub.groupby('k')['recall'].mean()
    ax.plot(mean_rec.index, mean_rec.values, marker='o', label=algo, linewidth=2)
ax.set_xlabel('k'); ax.set_ylabel('Recall@k')
ax.set_title('Recall@k (Cosine, n_queries=1000)')
ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.3)

# Middle: Amortized time vs n_queries
ax = axes[1]
for algo in ['exact', 'lsh', 'hnsw']:
    sub = ml_results[(ml_results['algorithm'] == algo) & (ml_results['k'] == 10)]
    mean_am = sub.groupby('n_queries')['amortized_time'].mean()
    ax.plot(mean_am.index, mean_am.values, marker='o', label=algo, linewidth=2)
ax.set_xlabel('Number of Queries'); ax.set_ylabel('Amortized Time (s/query)')
ax.set_title('Amortized Time vs Query Count')
ax.legend(); ax.grid(True, alpha=0.3)

# Right: HNSW efSearch vs Recall@10 — one curve per M value
ax = axes[2]
hnsw_sub = ml_results[(ml_results['algorithm'] == 'hnsw') & (ml_results['k'] == 10)].copy()
for M_val in sorted(hnsw_sub['M_val'].dropna().unique()):
    m_sub = (hnsw_sub[hnsw_sub['M_val'] == M_val]
             .groupby('ef_search')['recall'].mean().reset_index()
             .sort_values('ef_search'))
    ax.plot(m_sub['ef_search'], m_sub['recall'], marker='o', label=f'M={int(M_val)}')
ax.set_xlabel('efSearch'); ax.set_ylabel('Recall@10')
ax.set_title('HNSW: Effect of efSearch on Recall@10')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'movielens_results.png', dpi=150)
plt.show()
print('Saved: movielens_results.png')
'''

nb['cells'][17]['source'] = [cell17_new]
print('Cell 17 updated')

# ── CELL 24: Task 2 — replace scatter with AUC-ROC vs efSearch + bar chart ──
cell24_new = '''# ── 2.2 OoD Detection: Approximate Search vs Exact (AUC-ROC comparison) ───────
import ast, pandas as pd

ood_res = pd.read_csv(RESULTS_DIR / 'ood_results.csv')
img_res2 = pd.read_csv(RESULTS_DIR / 'image_search_results.csv')
img_res2['hp'] = img_res2['hyperparams'].apply(lambda x: ast.literal_eval(x) if x != '{}' else {})
img_res2['ef_search'] = img_res2['hp'].apply(lambda x: x.get('ef_search', None))
img_res2['M']         = img_res2['hp'].apply(lambda x: x.get('M', None))

# Join OoD AUC-ROC with query times from image search
# DNN uses conf2/euclidean; NN-Guide uses conf1/cosine
config_map = {'dnn': ('conf2','euclidean'), 'nn_guide_msp': ('conf1','cosine'), 'nn_guide_energy': ('conf1','cosine')}
strategy_labels = {'dnn': 'Deep Nearest Neighbors', 'nn_guide_msp': 'NN-Guide (MSP)', 'nn_guide_energy': 'NN-Guide (Energy)'}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('OoD Detection: AUC-ROC vs Avg Query Time', fontsize=13, fontweight='bold')

for ax, strategy in zip(axes, ['dnn', 'nn_guide_msp', 'nn_guide_energy']):
    cfg, met = config_map[strategy]
    ood_sub = ood_res[ood_res['strategy'] == strategy].groupby(['algorithm','hyperparams'])['auroc'].mean().reset_index()

    # Get query times for each (algorithm, hyperparams) from image results
    img_sub = (img_res2[(img_res2['config'] == cfg) &
                        (img_res2['metric'] == met) &
                        (img_res2['query_set'] == 'InDQuery') &
                        (img_res2['k'] == 10)]
               .groupby(['algorithm','hyperparams'])['avg_query_time'].mean().reset_index())

    merged = ood_sub.merge(img_sub, on=['algorithm','hyperparams'], how='left')

    for algo in ['hnsw', 'exact', 'random']:
        a = merged[merged['algorithm'] == algo]
        if len(a) == 0:
            continue
        if algo == 'exact':
            ax.scatter(a['avg_query_time'] * 1000, a['auroc'],
                       color='steelblue', s=180, zorder=5, label='Exact', marker='*')
        elif algo == 'random':
            ax.axhline(a['auroc'].mean(), color='gray', linestyle=':', linewidth=1.5, label='Random')
        else:
            # HNSW — parse ef_search and M, make curves per M
            a2 = a.copy()
            a2['hp_dict'] = a2['hyperparams'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x != '{}' else {})
            a2['ef_search'] = a2['hp_dict'].apply(lambda x: x.get('ef_search', 0))
            a2['M_v'] = a2['hp_dict'].apply(lambda x: x.get('M', 0))
            colors_h = ['#ff7f0e', '#d62728', '#9467bd']
            for idx, M_val in enumerate(sorted(a2['M_v'].unique())):
                m_sub2 = a2[a2['M_v'] == M_val].sort_values('ef_search')
                ax.plot(m_sub2['avg_query_time'] * 1000, m_sub2['auroc'],
                        marker='o', color=colors_h[idx % 3],
                        label=f'HNSW M={int(M_val)}', linewidth=2)

    ax.set_xlabel('Avg Query Time (ms)')
    ax.set_ylabel('AUC-ROC')
    ax.set_title(strategy_labels[strategy])
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'ood_auroc_vs_time.png', dpi=150)
plt.show()
print('Saved: ood_auroc_vs_time.png')
'''

nb['cells'][24]['source'] = [cell24_new]
print('Cell 24 updated')

# ── CELL 34: Task 3 — fix right graph: lock k=10, one point per config ───────
cell34_new = '''## 3.4 Recommender System Visualizations

import ast
rec_res = pd.read_csv(RESULTS_DIR / 'recommender_results.csv')
ml_res  = pd.read_csv(RESULTS_DIR / 'movielens_search_results.csv')

# Parse hyperparams for both
for df in [rec_res, ml_res]:
    df['hp'] = df['hyperparams'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x != '{}' else {})
    df['ef_search'] = df['hp'].apply(lambda x: x.get('ef_search', None))
    df['nbits_v']   = df['hp'].apply(lambda x: x.get('nbits', None))
    df['M_val']     = df['hp'].apply(lambda x: x.get('M', None))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Recommender System: Item-Based kNN (MovieLens 10m)', fontsize=13, fontweight='bold')

# Plot 1: Hit Rate vs k per algorithm
ax = axes[0]
for algo in ['exact', 'hnsw', 'lsh', 'random']:
    sub = rec_res[rec_res['algorithm'] == algo].groupby('k')['hit_rate'].mean().reset_index()
    if len(sub) == 0:
        continue
    ls = '--' if algo == 'random' else '-'
    ax.plot(sub['k'], sub['hit_rate'], marker='o', label=algo, linewidth=2, linestyle=ls)
ax.set_xlabel('k'); ax.set_ylabel('Hit Rate@k')
ax.set_title('Hit Rate vs k')
ax.legend(); ax.grid(True, alpha=0.3)

# Plot 2: HNSW efSearch effect on Hit Rate@10
ax = axes[1]
hnsw_sub = rec_res[(rec_res['algorithm'] == 'hnsw') & (rec_res['k'] == 10)].copy()
for M_val in sorted(hnsw_sub['M_val'].dropna().unique()):
    m_sub = (hnsw_sub[hnsw_sub['M_val'] == M_val]
             .groupby('ef_search')['hit_rate'].mean().reset_index()
             .sort_values('ef_search'))
    ax.plot(m_sub['ef_search'], m_sub['hit_rate'], marker='o', label=f'M={int(M_val)}')
exact_hr10 = rec_res[(rec_res['algorithm'] == 'exact') & (rec_res['k'] == 10)]['hit_rate'].mean()
ax.axhline(exact_hr10, color='gray', linestyle='--', label='Exact kNN')
ax.set_xlabel('efSearch'); ax.set_ylabel('Hit Rate@10')
ax.set_title('HNSW: efSearch vs Hit Rate@10')
ax.legend(); ax.grid(True, alpha=0.3)

# Plot 3: Recall@10 (Task 1) vs Hit Rate@10 (Task 3) — fixed k=10
# Each point = one algorithm config; lines connect points by efSearch/nbits
ax = axes[2]
ml_k10  = ml_res[(ml_res['n_queries'] == 1000) & (ml_res['k'] == 10)]
rec_k10 = rec_res[rec_res['k'] == 10]

# Exact — single point
exact_recall = ml_k10[ml_k10['algorithm'] == 'exact']['recall'].mean()
exact_hr     = rec_k10[rec_k10['algorithm'] == 'exact']['hit_rate'].mean()
ax.scatter([exact_recall], [exact_hr], color='steelblue', s=180, zorder=5, marker='*', label='Exact')

# HNSW — one curve per M, sorted by efSearch
colors_h = ['#ff7f0e', '#d62728', '#9467bd']
for idx, M_val in enumerate(sorted(ml_k10[ml_k10['algorithm']=='hnsw']['M_val'].dropna().unique())):
    r_sub = (ml_k10[(ml_k10['algorithm']=='hnsw') & (ml_k10['M_val']==M_val)]
             .groupby('ef_search')['recall'].mean().reset_index())
    h_sub = (rec_k10[(rec_k10['algorithm']=='hnsw') & (rec_k10['M_val']==M_val)]
             .groupby('ef_search')['hit_rate'].mean().reset_index())
    merged = r_sub.merge(h_sub, on='ef_search').sort_values('ef_search')
    if len(merged) > 0:
        ax.plot(merged['recall'], merged['hit_rate'], marker='o',
                color=colors_h[idx % 3], label=f'HNSW M={int(M_val)}', linewidth=2)

# LSH — one curve sorted by nbits
r_lsh = (ml_k10[ml_k10['algorithm']=='lsh']
         .groupby('nbits_v')['recall'].mean().reset_index())
h_lsh = (rec_k10[rec_k10['algorithm']=='lsh']
         .groupby('nbits_v')['hit_rate'].mean().reset_index())
merged_lsh = r_lsh.merge(h_lsh, on='nbits_v').sort_values('nbits_v')
if len(merged_lsh) > 0:
    ax.plot(merged_lsh['recall'], merged_lsh['hit_rate'], marker='s',
            color='green', linestyle='--', label='LSH', linewidth=2)
    for _, row in merged_lsh.iterrows():
        ax.annotate(str(int(row['nbits_v'])), (row['recall'], row['hit_rate']),
                    textcoords='offset points', xytext=(4, 4), fontsize=7)

ax.set_xlabel('Recall@10 (Task 1 search quality)')
ax.set_ylabel('Hit Rate@10 (Task 3 recommendation)')
ax.set_title('Search Quality vs Recommendation Quality (k=10)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'recommender_results.png', dpi=150)
plt.show()
print('Saved: recommender_results.png')
'''

nb['cells'][34]['source'] = [cell34_new]
print('Cell 34 updated')

with open('notebooks/main_experiment.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('All cells saved to notebook.')
