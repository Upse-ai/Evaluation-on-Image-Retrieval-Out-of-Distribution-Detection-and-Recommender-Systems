"""
MovieLens Data Preprocessing

Creates sparse MOVIE vectors for similarity search.
Each MOVIE is represented as a sparse vector over users:
  - coordinate i = rating given by user i (0 if not rated)

Primary dataset: ml-10m (10M ratings, 10,677 movies, 69,878 users)
Backup:         ml-1m  (only if ml-10m causes memory overflow, >16GB RAM)

Note on MovieLens querying (from project spec):
  The entire dataset is used as the search set. For each movie query,
  we search for k+1 neighbors and remove the query itself from the result.
  There is NO separate train/test split.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Union
import urllib.request
import zipfile


# Dataset specifications
DATASETS = {
    '10m': {
        'url': 'https://files.grouplens.org/datasets/movielens/ml-10m.zip',
        'ratings_file': 'ml-10M100K/ratings.dat',
        'separator': '::',
        'columns': ['user_id', 'item_id', 'rating', 'timestamp'],
        'description': '10M ratings, 10677 movies, 69878 users'
    },
    '1m': {
        'url': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
        'ratings_file': 'ml-1m/ratings.dat',
        'separator': '::',
        'columns': ['user_id', 'item_id', 'rating', 'timestamp'],
        'description': '1M ratings, 3706 movies, 6040 users (backup only)'
    }
}


class MovieLensPreprocessor:
    """
    Preprocessor for MovieLens dataset.

    IMPORTANT: Vectors represent MOVIES (not users).
    Each movie's vector has one coordinate per user.
    Non-zero values = rating given by that user.
    Zero = user has not rated the movie.

    The entire dataset is used as both the database and the query pool.
    """

    def __init__(self, data_dir: Union[str, Path], version: str = '10m'):
        """
        Args:
            data_dir: Directory for raw data storage
            version: '10m' (primary) or '1m' (backup if RAM overflow)
        """
        if version not in DATASETS:
            raise ValueError(f"version must be '10m' or '1m', got '{version}'")

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.version = version
        self.info = DATASETS[version]

        self.ratings_df: Optional[pd.DataFrame] = None
        self.movie_matrix: Optional[np.ndarray] = None  # shape: (n_movies, n_users)
        self.movie_ids: Optional[np.ndarray] = None
        self.user_ids: Optional[np.ndarray] = None

        print(f"MovieLens {version}: {self.info['description']}")

    def download(self, force: bool = False) -> None:
        """Download and extract the dataset."""
        zip_path = self.data_dir / f"ml-{self.version}.zip"

        if not zip_path.exists() or force:
            print(f"Downloading MovieLens {self.version}...")
            urllib.request.urlretrieve(self.info['url'], zip_path)
            print(f"Downloaded to {zip_path}")
        else:
            print(f"Already downloaded: {zip_path}")

        ratings_path = self.data_dir / self.info['ratings_file']
        if not ratings_path.exists() or force:
            print("Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(self.data_dir)
            print("Extraction complete.")

    def load_ratings(self, ratings_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Load the ratings.dat file."""
        if ratings_path is None:
            ratings_path = self.data_dir / self.info['ratings_file']

        ratings_path = Path(ratings_path)
        if not ratings_path.exists():
            self.download()

        print(f"Loading ratings from {ratings_path}...")

        self.ratings_df = pd.read_csv(
            ratings_path,
            sep=self.info['separator'],
            names=self.info['columns'],
            engine='python',
            usecols=['user_id', 'item_id', 'rating']
        )

        n_ratings = len(self.ratings_df)
        n_users = self.ratings_df['user_id'].nunique()
        n_movies = self.ratings_df['item_id'].nunique()

        print(f"Loaded {n_ratings:,} ratings")
        print(f"  Movies (items): {n_movies:,}")
        print(f"  Users:          {n_users:,}")

        return self.ratings_df

    def create_movie_vectors(self,
                             binary: bool = False,
                             normalize: bool = True) -> np.ndarray:
        """
        Create dense movie-user matrix.

        Each ROW is a movie, each COLUMN is a user.
        Entry (i, j) = rating of movie i by user j (0 if not rated).

        Args:
            binary: If True, replace ratings with 1 (interaction only)
            normalize: If True, L2-normalize each row (required for cosine)

        Returns:
            Dense matrix of shape (n_movies, n_users), dtype float32
        """
        if self.ratings_df is None:
            self.load_ratings()

        # Build consistent index mappings
        self.movie_ids = np.sort(self.ratings_df['item_id'].unique())
        self.user_ids = np.sort(self.ratings_df['user_id'].unique())

        movie_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}
        user_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}

        n_movies = len(self.movie_ids)
        n_users = len(self.user_ids)

        print(f"Building movie-user matrix: ({n_movies}, {n_users})")

        # Estimate memory
        mem_gb = n_movies * n_users * 4 / (1024 ** 3)
        print(f"Estimated dense matrix size: {mem_gb:.2f} GB")
        if mem_gb > 14:
            print("WARNING: Matrix may exceed 16GB RAM. Consider using ml-1m as backup.")

        # Build matrix
        self.movie_matrix = np.zeros((n_movies, n_users), dtype=np.float32)

        rows = self.ratings_df['item_id'].map(movie_to_idx).values
        cols = self.ratings_df['user_id'].map(user_to_idx).values

        if binary:
            self.movie_matrix[rows, cols] = 1.0
            print("Using binary representation (1 = interaction, 0 = no interaction)")
        else:
            self.movie_matrix[rows, cols] = self.ratings_df['rating'].values.astype(np.float32)
            print("Using rating values")

        density = self.ratings_df.shape[0] / (n_movies * n_users)
        print(f"  Density: {density:.4%}")

        if normalize:
            self.movie_matrix = self._normalize_rows(self.movie_matrix)
            print("  Applied L2 row normalization")

        return self.movie_matrix

    def _normalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        """L2-normalize each row."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return matrix / norms

    def select_query_movies(self, n_queries: int, random_state: int = 42) -> np.ndarray:
        """
        Randomly select movie indices to use as queries.

        Per the project spec: start with 100, explore up to 500, 1000, 2000.
        The entire movie matrix is the search set (no separate split).

        Args:
            n_queries: Number of movies to query
            random_state: Random seed

        Returns:
            Array of movie indices to use as queries
        """
        if self.movie_matrix is None:
            self.create_movie_vectors()

        rng = np.random.default_rng(random_state)
        n_total = self.movie_matrix.shape[0]
        n_queries = min(n_queries, n_total)

        query_indices = rng.choice(n_total, size=n_queries, replace=False)
        return query_indices

    def save_processed(self, output_dir: Union[str, Path], tag: str = ''):
        """Save the movie matrix and metadata."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        suffix = f"_{tag}" if tag else ""
        np.save(output_dir / f"ml{self.version}_movie_matrix{suffix}.npy", self.movie_matrix)
        np.save(output_dir / f"ml{self.version}_movie_ids.npy", self.movie_ids)
        np.save(output_dir / f"ml{self.version}_user_ids.npy", self.user_ids)

        print(f"Saved movie matrix to {output_dir}")
        print(f"  Shape: {self.movie_matrix.shape}")

    def load_processed(self, input_dir: Union[str, Path], tag: str = ''):
        """Load previously saved movie matrix."""
        input_dir = Path(input_dir)
        suffix = f"_{tag}" if tag else ""

        self.movie_matrix = np.load(input_dir / f"ml{self.version}_movie_matrix{suffix}.npy")
        self.movie_ids = np.load(input_dir / f"ml{self.version}_movie_ids.npy")
        self.user_ids = np.load(input_dir / f"ml{self.version}_user_ids.npy")

        print(f"Loaded movie matrix: {self.movie_matrix.shape}")


def search_movielens_knn(database: np.ndarray,
                         query_indices: np.ndarray,
                         index,
                         k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search k nearest neighbors for MovieLens queries.

    Per the project spec: search for k+1 neighbors and remove the
    query movie itself from the results, so we never return a movie
    as its own nearest neighbor.

    Args:
        database: Full movie matrix [n_movies, n_users]
        query_indices: Indices of movies to query
        index: Built FAISS index (over full database)
        k: Number of neighbors to retrieve

    Returns:
        (filtered_indices [n_queries, k], filtered_distances [n_queries, k])
    """
    queries = database[query_indices]  # shape: (n_queries, n_users)
    queries = np.ascontiguousarray(queries, dtype=np.float32)

    # Search k+1 to account for self-retrieval
    distances, indices = index.search(queries, k + 1)

    # Remove self from each result
    filtered_indices = np.zeros((len(query_indices), k), dtype=np.int64)
    filtered_distances = np.zeros((len(query_indices), k), dtype=np.float32)

    for i, qidx in enumerate(query_indices):
        row_idx = indices[i]
        row_dist = distances[i]

        # Remove the query itself
        mask = row_idx != qidx
        clean_idx = row_idx[mask][:k]
        clean_dist = row_dist[mask][:k]

        # Pad if fewer than k results remain (edge case)
        if len(clean_idx) < k:
            pad = k - len(clean_idx)
            clean_idx = np.concatenate([clean_idx, np.full(pad, -1, dtype=np.int64)])
            clean_dist = np.concatenate([clean_dist, np.full(pad, np.inf)])

        filtered_indices[i] = clean_idx
        filtered_distances[i] = clean_dist

    return filtered_indices, filtered_distances


if __name__ == "__main__":
    print("MovieLens Preprocessing Module")
    print("=" * 50)
    print("\nPrimary dataset: ml-10m")
    print("  - 10M ratings, 10,677 movies, 69,878 users")
    print("  - Each movie = 69,878-dimensional sparse vector")
    print("\nBackup: ml-1m (only if RAM overflow > 16GB)")
    print("\nUsage:")
    print("  preprocessor = MovieLensPreprocessor(data_dir, version='10m')")
    print("  preprocessor.download()")
    print("  movie_matrix = preprocessor.create_movie_vectors(normalize=True)")
