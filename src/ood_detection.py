"""
Out-of-Distribution (OoD) Detection - Task 2

Implements two kNN-based OoD detection strategies:

1. Deep Nearest Neighbors [Sun et al. 2022]
   - Uses Conf2 features (contrastively trained) + Euclidean distance
   - Score: S_InD(q) = -d_L2(q, kth-NN)   (higher = more in-distribution)

2. NN-Guide [Park et al. 2023]
   - Uses Conf1 features (standard ImageNet) + Cosine similarity
   - Score: S_InD(q) = conf(q) * sim(q)
     where conf(q) = classification confidence (MSP or -Energy)
     and   sim(q)  = confidence-weighted average cosine similarity to kNNs

Both methods require L2-normalized feature vectors.

Evaluation:
- AUC-ROC (primary metric)
- Optional: FPR@TPR95 (requires a threshold calibrated on InDSearch)

Random baseline:
- Query's kNNs are selected uniformly at random from InDSearch
- Required by the project to separate true search benefit from coincidental OoD results
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Tuple, Optional
import time


# ─── Confidence measures for NN-Guide ─────────────────────────────────────────

def maximum_softmax_probability(logits: np.ndarray) -> np.ndarray:
    """
    MSP: max over class probabilities after softmax.
    conf(q) = max_c softmax(logits)_c   ∈ (0, 1]
    """
    exp_shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)
    return probs.max(axis=1)


def negative_energy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    –Energy score: -T * log(sum_c exp(logits_c / T)).
    Higher value → more in-distribution.
    conf(q) = -Energy(logits)
    """
    exp_shifted = np.exp(logits / temperature)
    return -temperature * np.log(exp_shifted.sum(axis=1))


# ─── Strategy 1: Deep Nearest Neighbors ───────────────────────────────────────

def deep_nearest_neighbors_score(knn_distances: np.ndarray) -> np.ndarray:
    """
    Compute in-distribution score using Deep Nearest Neighbors method.

    Uses the Euclidean distance to the k-th nearest neighbor.
    Per the spec: S_InD(q) = -d_L2,k(q)
    where d_L2,k(q) is the Euclidean distance to the k-th NN.

    NOTE: Vectors must be L2-normalized before the kNN search (see project spec).

    Args:
        knn_distances: Euclidean distances [n_queries, k]
                       (output of FAISS IndexFlatL2 or IndexHNSWFlat with METRIC_L2)

    Returns:
        In-distribution scores [n_queries]  (higher = more InD)
    """
    # Use the distance to the k-th (last) nearest neighbor
    dist_to_kth = knn_distances[:, -1]
    return -dist_to_kth


# ─── Strategy 2: NN-Guide ─────────────────────────────────────────────────────

def nn_guide_score(query_logits: np.ndarray,
                   neighbor_logits: np.ndarray,
                   knn_cosine_similarities: np.ndarray,
                   confidence_measure: str = 'msp') -> np.ndarray:
    """
    Compute in-distribution score using NN-Guide method.

    S_InD(q) = conf(q) * sim(q)

    where:
      conf(q) = classification confidence of query (MSP or -Energy)
      sim(q)  = (1/k) * sum_{x in kNNs} conf(x) * cosine_sim(q, x)

    The kNN search is performed with UNWEIGHTED cosine similarity first
    (i.e., standard inner product on normalized vectors), then confidence
    weights are applied when averaging similarities.

    NOTE: All vectors must be L2-normalized before the kNN search.
    NOTE: Use Conf1 features for this method (non-contrastive training).

    Args:
        query_logits:             Logits for query images   [n_queries, 1000]
        neighbor_logits:          Logits for each query's kNNs [n_queries, k, 1000]
        knn_cosine_similarities:  Inner product similarities [n_queries, k]
                                  (FAISS IndexFlatIP output with normalized vectors)
        confidence_measure:       'msp' or 'energy'

    Returns:
        In-distribution scores [n_queries]
    """
    if confidence_measure == 'msp':
        conf_fn = maximum_softmax_probability
    elif confidence_measure == 'energy':
        conf_fn = negative_energy
    else:
        raise ValueError(f"confidence_measure must be 'msp' or 'energy', got '{confidence_measure}'")

    n_queries, k = knn_cosine_similarities.shape

    # conf(q): scalar confidence for each query
    conf_q = conf_fn(query_logits)  # [n_queries]

    # conf(x): confidence for each neighbor
    # neighbor_logits shape: [n_queries, k, 1000]
    neighbor_logits_flat = neighbor_logits.reshape(-1, neighbor_logits.shape[-1])
    conf_neighbors_flat = conf_fn(neighbor_logits_flat)   # [n_queries * k]
    conf_neighbors = conf_neighbors_flat.reshape(n_queries, k)  # [n_queries, k]

    # sim(q) = (1/k) * sum_j conf(x_j) * cosine_sim(q, x_j)
    # cosine_sim is already the inner product (normalized vectors)
    weighted_sims = conf_neighbors * knn_cosine_similarities  # [n_queries, k]
    sim_q = weighted_sims.mean(axis=1)  # [n_queries]

    return conf_q * sim_q


# ─── Random baseline ──────────────────────────────────────────────────────────

def random_knn_indices(n_queries: int, n_database: int,
                       k: int, random_state: int = 42) -> np.ndarray:
    """
    Select k random 'neighbors' from the database for each query.

    This is the required random baseline from the project spec.
    Provides a comparison for when kNN search provides no benefit.

    Returns:
        Random indices [n_queries, k]
    """
    rng = np.random.default_rng(random_state)
    return rng.integers(0, n_database, size=(n_queries, k))


# ─── AUC-ROC evaluation ───────────────────────────────────────────────────────

def compute_auroc(ind_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """
    Compute Area Under the ROC Curve (AUC-ROC).

    InDQuery  = Positive class (InD)
    OoDQuery  = Negative class (OoD)

    Higher AUC-ROC = better OoD detection.
    AUC-ROC = 0.5 means random performance.

    Args:
        ind_scores: S_InD scores for InDQuery vectors   [n_ind]
        ood_scores: S_InD scores for OoDQuery vectors   [n_ood]

    Returns:
        AUC-ROC scalar in [0, 1]
    """
    # Label 1 = InD (positive), 0 = OoD (negative)
    labels = np.concatenate([
        np.ones(len(ind_scores)),
        np.zeros(len(ood_scores))
    ])
    scores = np.concatenate([ind_scores, ood_scores])
    return roc_auc_score(labels, scores)


def compute_fpr_at_tpr95(ind_scores: np.ndarray,
                          ood_scores: np.ndarray,
                          reference_scores: np.ndarray) -> float:
    """
    Compute False Positive Rate at 95% True Positive Rate.

    Threshold λ is calibrated on a reference set (subsample of InDSearch),
    NOT on the test sets, to avoid data leakage.

    Args:
        ind_scores:       S_InD for InDQuery  [n_ind]
        ood_scores:       S_InD for OoDQuery  [n_ood]
        reference_scores: S_InD for InDSearch subsample (to calibrate λ)

    Returns:
        FPR at TPR=95%
    """
    # Find threshold that gives TPR=95% on the reference (InDSearch) set
    threshold = np.percentile(reference_scores, 5)  # 5th percentile → 95% TPR

    # Apply threshold to test sets
    tpr = (ind_scores >= threshold).mean()
    fpr = (ood_scores >= threshold).mean()

    return float(fpr)


# ─── Full OoD pipeline ────────────────────────────────────────────────────────

def run_ood_pipeline(strategy: str,
                     ind_knn_indices: np.ndarray,
                     ood_knn_indices: np.ndarray,
                     knn_distances_or_sims_ind: np.ndarray,
                     knn_distances_or_sims_ood: np.ndarray,
                     # For NN-Guide only:
                     ind_query_logits: Optional[np.ndarray] = None,
                     ood_query_logits: Optional[np.ndarray] = None,
                     neighbor_logits_ind: Optional[np.ndarray] = None,
                     neighbor_logits_ood: Optional[np.ndarray] = None,
                     confidence_measure: str = 'msp') -> dict:
    """
    Run the OoD detection pipeline for one strategy and return AUC-ROC.

    Args:
        strategy:        'dnn' (Deep Nearest Neighbors) or 'nn_guide'
        ind_knn_indices: kNN indices for InDQuery   [n_ind, k]
        ood_knn_indices: kNN indices for OoDQuery   [n_ood, k]
        knn_distances_or_sims_ind: For DNN → Euclidean distances [n_ind, k]
                                   For NN-Guide → cosine sims     [n_ind, k]
        knn_distances_or_sims_ood: Same for OoDQuery

    Returns:
        dict with 'auroc', 'ind_scores', 'ood_scores'
    """
    if strategy == 'dnn':
        ind_scores = deep_nearest_neighbors_score(knn_distances_or_sims_ind)
        ood_scores = deep_nearest_neighbors_score(knn_distances_or_sims_ood)

    elif strategy == 'nn_guide':
        if any(x is None for x in [ind_query_logits, ood_query_logits,
                                    neighbor_logits_ind, neighbor_logits_ood]):
            raise ValueError("nn_guide requires query and neighbor logits")

        ind_scores = nn_guide_score(
            ind_query_logits, neighbor_logits_ind,
            knn_distances_or_sims_ind, confidence_measure
        )
        ood_scores = nn_guide_score(
            ood_query_logits, neighbor_logits_ood,
            knn_distances_or_sims_ood, confidence_measure
        )
    else:
        raise ValueError(f"strategy must be 'dnn' or 'nn_guide', got '{strategy}'")

    auroc = compute_auroc(ind_scores, ood_scores)

    return {
        'strategy': strategy,
        'auroc': auroc,
        'ind_scores': ind_scores,
        'ood_scores': ood_scores
    }


def collect_neighbor_logits(all_logits: np.ndarray,
                             knn_indices: np.ndarray) -> np.ndarray:
    """
    Gather logits for the retrieved neighbors.

    Args:
        all_logits:  All feature/logit vectors in InDSearch [n_database, 1000]
        knn_indices: Indices of kNNs [n_queries, k]

    Returns:
        Neighbor logits [n_queries, k, 1000]
    """
    n_queries, k = knn_indices.shape
    neighbor_logits = all_logits[knn_indices.reshape(-1)]  # [n_queries*k, 1000]
    return neighbor_logits.reshape(n_queries, k, -1)


if __name__ == "__main__":
    print("OoD Detection Module")
    print("=" * 50)
    print("\nStrategy 1: Deep Nearest Neighbors")
    print("  - Uses Conf2 features + Euclidean distance")
    print("  - S_InD(q) = -d_L2(q, k-th NN)")
    print("\nStrategy 2: NN-Guide")
    print("  - Uses Conf1 features + Cosine similarity")
    print("  - S_InD(q) = conf(q) * sim(q)")
    print("\nEvaluation: AUC-ROC")
    print("Random baseline: kNNs chosen uniformly at random from InDSearch")
