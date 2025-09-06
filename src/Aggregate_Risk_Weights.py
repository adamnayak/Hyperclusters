import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import requests
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec
import scipy.stats as stats
from esda.getisord import G_Local
from libpysal.weights import Queen
from matplotlib.colors import SymLogNorm
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from scipy.stats import ttest_ind
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import ListedColormap, BoundaryNorm

def subjective_weighting_AHP(pairwise_matrix, ri_map=None, max_cr=0.1):
    """
    Compute AHP weights from a pairwise comparison matrix.
    Enforces non-negativity and unit-sum; checks Consistency Ratio (CR).

    Parameters
    ----------
    pairwise_matrix : (n, n) np.ndarray
        Saaty AHP pairwise matrix (positive reciprocal).
    ri_map : dict or None
        Random Index by n. Defaults to Saaty (1980) for n<=10.
    max_cr : float
        Maximum acceptable consistency ratio.

    Returns
    -------
    SW : (n,) np.ndarray
        Subjective weights, >=0 and sum to 1.
    CR : float
        Consistency Ratio.
    """
    if ri_map is None:
        ri_map = {1:0.00, 2:0.00, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}

    eigvals, eigvecs = np.linalg.eig(pairwise_matrix)
    idx = np.argmax(eigvals.real)
    max_eig = eigvals[idx].real
    w = eigvecs[:, idx].real
    w = np.maximum(w, 0)  # enforce non-negativity
    if w.sum() == 0:
        w = np.ones_like(w)
    SW = w / w.sum()

    n = pairwise_matrix.shape[0]
    CI = (max_eig - n) / (n - 1) if n > 2 else 0.0
    RI = ri_map.get(n, 1.49)  # conservative fallback for n>10
    CR = 0.0 if RI == 0 else CI / RI
    if CR > max_cr:
        raise ValueError(f"AHP Consistency Ratio too high (CR={CR:.3f} > {max_cr}). Revisit pairwise matrix.")
    return SW, CR


def objective_weighting_entropy(df, metric_cols, log_transform=True, eps=1e-9):
    """
    Compute entropy-based objective weights over metrics.

    Steps: (optional) log1p -> column-normalize -> entropy H_i -> weight ~ (1-H_i)

    Parameters
    ----------
    df : pd.DataFrame
    metric_cols : list[str]
        Columns with non-negative metric values (by county/row).
    log_transform : bool
        Apply log1p to stabilize heavy tails before entropy.
    eps : float
        Numerical safeguard.

    Returns
    -------
    OW : (n_metrics,) np.ndarray
        Objective weights, sum to 1.
    norm_for_entropy : (n_rows, n_metrics) np.ndarray
        Column-normalized matrix used for entropy.
    """
    X = df[metric_cols].to_numpy(dtype=float)
    if log_transform:
        X = np.log1p(np.maximum(X, 0) + eps)

    # Column-normalize across counties (judgment matrix b_ij -> f_ij)
    col_sums = X.sum(axis=0, keepdims=True) + eps
    F = X / col_sums

    # Entropy per metric
    m = F.shape[0]
    H = -(F * (np.log(F + eps))).sum(axis=0) / np.log(m + eps)

    # Diversity and weights
    D = 1 - H
    D = np.clip(D, 0, None)
    if D.sum() == 0:
        OW = np.ones_like(D) / len(D)
    else:
        OW = D / D.sum()

    return OW, F


def combine_weights_game_theory(SW, OW, nonneg=True, normalize=True):
    """
    Combine L=2 weight vectors via LS/Nash-style coefficients α.

    Parameters
    ----------
    SW, OW : array-like
        Subjective and objective weight vectors (length n_metrics).
    nonneg : bool, default=True
        Clip α >= 0 before normalizing.
    normalize : bool, default=True
        If True, normalize final combined weights (CW) to sum=1.
        If False, return CW as computed (may not sum to 1).

    Returns
    -------
    CW : (n_metrics,) np.ndarray
        Combined weights.
    alpha : (2,) np.ndarray
        Normalized combination coefficients used on [SW, OW].
    """
    SW = np.asarray(SW, dtype=float).ravel()
    OW = np.asarray(OW, dtype=float).ravel()
    assert SW.shape == OW.shape, "SW and OW must have same length."

    # Solve for α
    W = np.vstack([SW, OW])           # (2, n)
    rhs = W @ np.ones(W.shape[1])     # (2,)
    A = W @ W.T                       # (2, 2)
    alpha = np.linalg.solve(A, rhs)

    if nonneg:
        alpha = np.clip(alpha, 0, None)
    if alpha.sum() == 0:
        alpha = np.ones_like(alpha)
    if normalize:
        alpha = alpha / alpha.sum()

    # Combined weights
    CW = (W.T @ alpha).ravel()
    if nonneg:
        CW = np.maximum(CW, 0)

    if normalize:
        s = CW.sum()
        CW = CW / s if s > 0 else np.ones_like(CW) / len(CW)

    return CW, alpha


# Function to assign risk levels based on threshold bins
def assign_risk_category(score, thresholds, categories):
    bin_index = np.digitize(score, thresholds, right=True)  # Finds the appropriate bin index
    return categories[min(bin_index, len(categories) - 1)]  # Ensure it does not exceed max index


def return_weights(
    df,
    metric_cols,
    pairwise_matrix,
    *,
    normalize=True,
    non_neg=True,
    log_for_entropy=True,
    standardize_for_composite=True,
    num_bins=5,
    risk_cats = ["lowest", "lower", "medium", "higher", "highest"],
    eps=1e-9):
    """
    Full workflow:
      1) SW via AHP
      2) OW via entropy
      3) CW via game-theoretic combination (SW + OW)
      4) Build composite risk scores with four weight schemes:
         - EW (equal weights baseline)
         - SW-only
         - OW-only
         - CW (combined)
      5) Calculate aggregated CW risk scores
      6) Extract K-means binning thresholds for discrete categories
      7) Calculate aggregated CW risk categories and return updated gdf

    Returns
    -------
    out : dict
        {
          'weights': {'EW':..., 'SW':..., 'OW':..., 'CW':..., 'alpha':...},
          'scores' : {'EW':..., 'SW':..., 'OW':..., 'CW':...},  # pd.Series,
          'gdf': gpd.Geodataframe containing 'Risk_Score' and 'Risk_Category'
    """
    # 1) SW
    SW, CR = subjective_weighting_AHP(pairwise_matrix)

    # 2) OW
    OW, _ = objective_weighting_entropy(df, metric_cols, log_transform=log_for_entropy)

    # 3) CW
    CW, alpha = combine_weights_game_theory(SW, OW, nonneg=non_neg, normalize=normalize)

    # 4) Build four composite scores
    n = len(metric_cols)
    EW = np.ones(n) / n

    # Optionally standardize metrics for composition (0-1 per metric)
    X = df[metric_cols].to_numpy(dtype=float)
    if log_for_entropy:
        X = np.log1p(np.maximum(X, 0) + eps)
    if standardize_for_composite:
        scaler = MinMaxScaler()
        X_std = scaler.fit_transform(X)
    else:
        X_std = X
    
    # 5) Aggregate Risk Scores with CW
    risk_scores = X_std @ CW  # Weighted sum of standardized metrics
    df['Risk_Score'] = risk_scores.flatten()
    
    # 6) Determine Binning using K-means
    kmeans = KMeans(n_clusters=num_bins-1, random_state=42)
    risk_scores = risk_scores.reshape(-1, 1)  # Reshape for K-means
    kmeans.fit(risk_scores)
    threshold_standards = np.sort(kmeans.cluster_centers_.flatten())  # Sorted cluster centers as thresholds
    
    # 7) Apply binning to the risk scores
    df["Risk_Category"] = df["Risk_Score"].apply(lambda x: assign_risk_category(x, threshold_standards, risk_cats))

    scores = {
        'EW': pd.Series(X_std @ EW, index=df.index, name='score_EW'),
        'SW': pd.Series(X_std @ SW, index=df.index, name='score_SW'),
        'OW': pd.Series(X_std @ OW, index=df.index, name='score_OW'),
        'CW': pd.Series(X_std @ CW, index=df.index, name='score_CW'),
    }

    weights = {'EW': EW, 'SW': SW, 'OW': OW, 'CW': CW, 'Thresholds': threshold_standards}

    return {'weights': weights, 'scores': scores, 'gdf': df}