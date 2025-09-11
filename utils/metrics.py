import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

# === Functions ===
def mean_js_divergence(X_src, X_trg, num_bins=100):
    """
    Computes mean Jensen-Shannon divergence across all features.

    Args:
        X_src (np.ndarray): Source samples (N_src, features)
        X_trg (np.ndarray): Target samples (N_trg, features)
        num_bins (int): Number of bins for histogram estimation (default 100)

    Returns:
        float: Mean JS divergence over all features
    """
    n_features = X_src.shape[1]
    js_scores = []

    for f in range(n_features):
        src_feat = X_src[:, f]
        trg_feat = X_trg[:, f]

        # Combine to get common bin edges
        combined = np.concatenate([src_feat, trg_feat])
        bins = np.linspace(np.min(combined), np.max(combined), num_bins)

        src_hist, _ = np.histogram(src_feat, bins=bins, density=True)
        trg_hist, _ = np.histogram(trg_feat, bins=bins, density=True)

        # Add small epsilon to avoid zeros
        src_hist = src_hist + 1e-8
        trg_hist = trg_hist + 1e-8

        # Normalize
        src_hist /= np.sum(src_hist)
        trg_hist /= np.sum(trg_hist)

        # Compute JS divergence
        js = jensenshannon(src_hist, trg_hist, base=2)  # base=2 → JS in [0,1]
        js_scores.append(js)

    return np.mean(js_scores)

def mean_wasserstein(X, Y):
    """
    Compute the mean Wasserstein (Earth Mover's) distance across all feature dimensions.

    This function calculates the 1D Wasserstein distance (also known as Earth Mover's Distance)
    between corresponding features (columns) of two input datasets and returns the average distance.

    Args:
        X (np.ndarray): Array of shape (n_samples, n_features), representing the first dataset.
        Y (np.ndarray): Array of shape (n_samples, n_features), representing the second dataset.

    Returns:
        float: The mean Wasserstein distance across all feature dimensions.
    """
    distances = [
        wasserstein_distance(X[:, i], Y[:, i])
        for i in range(X.shape[1])
    ]
    return np.mean(distances)
