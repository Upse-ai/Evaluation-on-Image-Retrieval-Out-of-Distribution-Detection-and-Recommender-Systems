# Approximate Similarity Search - Bachelor Project

This project experimentally tests and compares the behavior of different Approximate Nearest Neighbors (ANN) algorithms, evaluating the trade-off between efficiency and effectiveness.

## Project Structure

```
├── data/
│   ├── raw/           # Raw datasets (images, MovieLens)
│   ├── processed/     # Extracted feature vectors
│   └── results/       # Experiment results and metrics
├── notebooks/
│   └── main_experiment.ipynb
├── src/
│   ├── feature_extraction.py   # WideResNet-50-2 feature extraction
│   ├── movielens_preprocessing.py  # MovieLens sparse vector preprocessing
│   ├── similarity_search.py    # FAISS implementations (kNN, LSH, HNSW)
│   └── evaluation.py           # Evaluation metrics
├── requirements.txt
└── README.md
```

## Task 1: Similarity Search

### Datasets

1. **Image Dataset**: Feature vectors extracted using WideResNet-50-2
   - **Conf1**: Features from penultimate layer (2048-dim dense vectors)
   - **Conf2**: Features from last convolutional layer with spatial pooling

2. **MovieLens Dataset**: Sparse user-item rating vectors

### Distance Metrics

- **Images**: Cosine distance and Euclidean distance
- **MovieLens**: Cosine distance only

### ANN Algorithms (FAISS)

1. **Exact kNN** (IndexFlatL2 / IndexFlatIP)
2. **LSH** (Locality-Sensitive Hashing)
3. **HNSW** (Hierarchical Navigable Small World graphs)

### Evaluation Metrics

- **Recall@k**: Fraction of true k-nearest neighbors found
- **Query time**: Time to perform a single query
- **Index build time**: Time to construct the index
- **Amortized time**: Total time / number of queries

### Parameters

- k ∈ {1, 5, 10, 30}
- Image preprocessing: Resize to 224×224

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place your image dataset in `data/raw/`
2. Download MovieLens dataset to `data/raw/`
3. Run the main experiment notebook: `notebooks/main_experiment.ipynb`

## Requirements

- Python 3.8+
- PyTorch 2.0+
- FAISS
- See `requirements.txt` for full list
