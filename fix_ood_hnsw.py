import json

with open('notebooks/main_experiment.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell21_new = '''# ── Strategy 1: Deep Nearest Neighbors (Conf2 + Euclidean) ───────────────────
ood_rows = []

for k in K_VALUES:
    # Exact
    auroc, bt, qt = ood_experiment_dnn(k,
        lambda db: build_exact_index(db, 'euclidean'),
        index_kwargs={}, config='conf2'
    )
    ood_rows.append({'strategy': 'dnn', 'algorithm': 'exact', 'k': k,
                     'hyperparams': {}, 'auroc': auroc})

    # Random baseline
    rnd_auroc = ood_random_baseline(k, None, None, 'dnn', 'conf2')
    ood_rows.append({'strategy': 'dnn', 'algorithm': 'random', 'k': k,
                     'hyperparams': {}, 'auroc': rnd_auroc})

    # HNSW Euclidean
    hp = HYPERPARAM_RANGES['images']['hnsw']
    for M in hp['M']:
        for ef_c in hp['ef_construction']:
            for ef_s in hp['ef_search']:
                auroc, bt, qt = ood_experiment_dnn(k,
                    lambda db, M=M, ef_c=ef_c, ef_s=ef_s: build_hnsw_index(db, 'euclidean', M, ef_c, ef_s),
                    index_kwargs={}, config='conf2'
                )
                ood_rows.append({'strategy': 'dnn', 'algorithm': 'hnsw', 'k': k,
                                 'hyperparams': {'M': M, 'ef_construction': ef_c, 'ef_search': ef_s},
                                 'auroc': auroc})

# ── Strategy 2: NN-Guide (Conf1 + Cosine) ────────────────────────────────────
for conf_measure in ['msp', 'energy']:
    for k in K_VALUES:
        # Exact
        auroc, bt, qt = ood_experiment_nn_guide(k,
            lambda db: build_exact_index(db, 'cosine'),
            index_kwargs={}, confidence_measure=conf_measure, config='conf1'
        )
        ood_rows.append({'strategy': f'nn_guide_{conf_measure}', 'algorithm': 'exact',
                         'k': k, 'hyperparams': {}, 'auroc': auroc})

        # Random baseline
        rnd_auroc = ood_random_baseline(k, None, None, 'nn_guide', 'conf1',
                                        confidence_measure=conf_measure)
        ood_rows.append({'strategy': f'nn_guide_{conf_measure}', 'algorithm': 'random',
                         'k': k, 'hyperparams': {}, 'auroc': rnd_auroc})

        # HNSW Cosine
        hp = HYPERPARAM_RANGES['images']['hnsw']
        for M in hp['M']:
            for ef_c in hp['ef_construction']:
                for ef_s in hp['ef_search']:
                    auroc, bt, qt = ood_experiment_nn_guide(k,
                        lambda db, M=M, ef_c=ef_c, ef_s=ef_s: build_hnsw_index(db, 'cosine', M, ef_c, ef_s),
                        index_kwargs={}, confidence_measure=conf_measure, config='conf1'
                    )
                    ood_rows.append({'strategy': f'nn_guide_{conf_measure}', 'algorithm': 'hnsw',
                                     'k': k, 'hyperparams': {'M': M, 'ef_construction': ef_c, 'ef_search': ef_s},
                                     'auroc': auroc})

ood_results = pd.DataFrame(ood_rows)
ood_results.to_csv(RESULTS_DIR / 'ood_results.csv', index=False)
print(f"Saved {len(ood_results)} OoD result rows")
'''

nb['cells'][21]['source'] = [cell21_new]

with open('notebooks/main_experiment.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Cell 21 updated with HNSW for NN-Guide.')
