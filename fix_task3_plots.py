import json

with open('notebooks/main_experiment.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell34_new = '''## 3.4 Recommender System Visualizations

import ast
rec_res = pd.read_csv(RESULTS_DIR / 'recommender_results.csv')
ml_res  = pd.read_csv(RESULTS_DIR / 'movielens_search_results.csv')

for df in [rec_res, ml_res]:
    df['hp'] = df['hyperparams'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x != '{}' else {})
    df['ef_search'] = df['hp'].apply(lambda x: x.get('ef_search', None))
    df['nbits_v']   = df['hp'].apply(lambda x: x.get('nbits', None))
    df['M_val']     = df['hp'].apply(lambda x: x.get('M', None))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Recommender System: Item-Based kNN (MovieLens 10m)', fontsize=13, fontweight='bold')

# ── Plot 1: Hit Rate vs k — per-M curves for HNSW ───────────────────────────
ax = axes[0]
# Exact
sub = rec_res[rec_res['algorithm'] == 'exact'].groupby('k')['hit_rate'].mean().reset_index()
ax.plot(sub['k'], sub['hit_rate'], marker='*', color='steelblue', label='Exact', linewidth=2, markersize=10)

# HNSW — one curve per M (averaged over efC/efS)
colors_h = ['#ff7f0e', '#d62728', '#9467bd']
hnsw_r = rec_res[rec_res['algorithm'] == 'hnsw']
for idx, M_val in enumerate(sorted(hnsw_r['M_val'].dropna().unique())):
    sub = hnsw_r[hnsw_r['M_val'] == M_val].groupby('k')['hit_rate'].mean().reset_index()
    ax.plot(sub['k'], sub['hit_rate'], marker='o', color=colors_h[idx % 3],
            label=f'HNSW M={int(M_val)}', linewidth=2)

# LSH
sub = rec_res[rec_res['algorithm'] == 'lsh'].groupby('k')['hit_rate'].mean().reset_index()
ax.plot(sub['k'], sub['hit_rate'], marker='s', color='green', label='LSH', linewidth=2, linestyle='--')

# Random
sub = rec_res[rec_res['algorithm'] == 'random'].groupby('k')['hit_rate'].mean().reset_index()
ax.plot(sub['k'], sub['hit_rate'], marker='x', color='gray', label='Random', linewidth=1.5, linestyle=':')

ax.set_xlabel('k'); ax.set_ylabel('Hit Rate@k')
ax.set_title('Hit Rate vs k (per M-value)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── Plot 2: HNSW efSearch effect — multiple k values ────────────────────────
ax = axes[1]
hnsw_sub = rec_res[rec_res['algorithm'] == 'hnsw'].copy()
# Use best M (M=32) to keep it clean
best_M = hnsw_sub['M_val'].max()
hnsw_bestM = hnsw_sub[hnsw_sub['M_val'] == best_M]

k_colors = {1: '#1f77b4', 5: '#ff7f0e', 10: '#2ca02c', 30: '#d62728'}
for k_val in sorted(rec_res['k'].unique()):
    m_sub = (hnsw_bestM[hnsw_bestM['k'] == k_val]
             .groupby('ef_search')['hit_rate'].mean().reset_index()
             .sort_values('ef_search'))
    ax.plot(m_sub['ef_search'], m_sub['hit_rate'], marker='o',
            color=k_colors.get(k_val, 'black'), label=f'k={k_val}', linewidth=2)
    # Add exact kNN baseline as horizontal marker
    exact_hr = rec_res[(rec_res['algorithm'] == 'exact') & (rec_res['k'] == k_val)]['hit_rate'].mean()
    ax.axhline(exact_hr, color=k_colors.get(k_val, 'black'), linestyle='--', alpha=0.4, linewidth=1)

ax.set_xlabel('efSearch'); ax.set_ylabel('Hit Rate@k')
ax.set_title(f'HNSW (M={int(best_M)}): efSearch vs Hit Rate\\n(dashed = Exact kNN baseline per k)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── Plot 3: Search Quality vs Recommendation Quality — curves varying k ──────
# For each algorithm (best config), plot Recall@k vs Hit Rate@k as k varies 1→30
ax = axes[2]
ml_base = ml_res[ml_res['n_queries'] == 1000]

# Exact: recall=1 at all k, so just plot hit rate vs k
exact_rec = ml_base[ml_base['algorithm'] == 'exact'].groupby('k')['recall'].mean().reset_index()
exact_hr  = rec_res[rec_res['algorithm'] == 'exact'].groupby('k')['hit_rate'].mean().reset_index()
merged_e  = exact_rec.merge(exact_hr, on='k')
ax.plot(merged_e['recall'], merged_e['hit_rate'], marker='*', color='steelblue',
        label='Exact', linewidth=2, markersize=10)
for _, row in merged_e.iterrows():
    ax.annotate(f'k={int(row["k"])}', (row['recall'], row['hit_rate']),
                textcoords='offset points', xytext=(5, 3), fontsize=7, color='steelblue')

# HNSW — best config (max M, max efSearch) for each k
hnsw_ml = ml_base[ml_base['algorithm'] == 'hnsw']
hnsw_rc = rec_res[rec_res['algorithm'] == 'hnsw']
for idx, M_val in enumerate(sorted(hnsw_ml['M_val'].dropna().unique())):
    # Use highest efSearch per k
    best_ef = hnsw_ml['ef_search'].max()
    r_sub = (hnsw_ml[(hnsw_ml['M_val'] == M_val) & (hnsw_ml['ef_search'] == best_ef)]
             .groupby('k')['recall'].mean().reset_index())
    h_sub = (hnsw_rc[(hnsw_rc['M_val'] == M_val) & (hnsw_rc['ef_search'] == best_ef)]
             .groupby('k')['hit_rate'].mean().reset_index())
    merged = r_sub.merge(h_sub, on='k').sort_values('k')
    if len(merged) > 0:
        ax.plot(merged['recall'], merged['hit_rate'], marker='o',
                color=colors_h[idx % 3], label=f'HNSW M={int(M_val)} (best efS)', linewidth=2)

# LSH — best nbits
lsh_ml = ml_base[ml_base['algorithm'] == 'lsh']
lsh_rc = rec_res[rec_res['algorithm'] == 'lsh']
best_nb = lsh_ml['nbits_v'].max()
r_lsh = lsh_ml[lsh_ml['nbits_v'] == best_nb].groupby('k')['recall'].mean().reset_index()
h_lsh = lsh_rc[lsh_rc['nbits_v'] == best_nb].groupby('k')['hit_rate'].mean().reset_index()
merged_lsh = r_lsh.merge(h_lsh, on='k').sort_values('k')
if len(merged_lsh) > 0:
    ax.plot(merged_lsh['recall'], merged_lsh['hit_rate'], marker='s',
            color='green', linestyle='--', label=f'LSH nbits={int(best_nb)}', linewidth=2)

ax.set_xlabel('Recall@k (Task 1 search quality)')
ax.set_ylabel('Hit Rate@k (Task 3 recommendation)')
ax.set_title('Search Quality vs Recommendation Quality\\n(best config per algorithm, k varies 1→30)')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'recommender_results.png', dpi=150)
plt.show()
print('Saved: recommender_results.png')
'''

nb['cells'][34]['source'] = [cell34_new]

with open('notebooks/main_experiment.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Cell 34 updated with improved all-3-graphs version.')
