"""Statistics for the evaluation: exact paired Wilcoxon test, Holm correction, clustered bootstrap.

Pure numpy. The bootstrap resamples whole clusters first (overlap groups of tiles), then tiles'
ships, because ships within a tile share scene and noise and are not independent.
"""
import numpy as np


def _midranks(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    ranks = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def exact_wilcoxon(x, y=None) -> dict:
    """Two-sided exact signed-rank test on paired samples (zero differences dropped, ties get
    midranks). The p-value comes from all 2^n sign assignments, so it is exact for n <= 20 (we have
    16 tiles). Returns p, n, w_plus and the rank-biserial correlation (effect size in [-1, 1])."""
    d = np.asarray(x, dtype=float) - (0.0 if y is None else np.asarray(y, dtype=float))
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return {"p": 1.0, "n": 0, "w_plus": 0.0, "rank_biserial": 0.0}
    if n > 20:
        raise ValueError(f"exact test enumerates 2^n sign patterns; n={n} > 20")
    ranks = _midranks(np.abs(d))
    total = ranks.sum()
    w_plus = ranks[d > 0].sum()
    signs = (np.arange(2**n)[:, None] >> np.arange(n)) & 1
    w_all = signs @ ranks
    p = float(np.mean(np.abs(w_all - total / 2) >= abs(w_plus - total / 2) - 1e-9))
    return {"p": p, "n": n, "w_plus": float(w_plus), "rank_biserial": float((2 * w_plus - total) / total)}


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def hierarchical_bootstrap_ci(values_by_tile: dict, cluster_of: dict | None = None, n_boot: int = 10000,
                              alpha: float = 0.05, seed: int = 0):
    """Percentile CI of the pooled mean over ships. values_by_tile: tile -> 1-D array of one value
    per ship (or per ship difference between two methods). cluster_of: tile -> cluster id (tiles
    that overlap spatially share one); default every tile is its own cluster. Each replicate
    resamples clusters with replacement, takes all their tiles, then resamples the ships within each
    tile with replacement. Returns (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    tiles = sorted(values_by_tile)
    cluster_of = cluster_of or {t: t for t in tiles}
    clusters: dict = {}
    for t in tiles:
        clusters.setdefault(cluster_of[t], []).append(t)
    keys = list(clusters)
    vals = {t: np.asarray(values_by_tile[t], dtype=float) for t in tiles}

    stats = np.empty(n_boot)
    for b in range(n_boot):
        total, count = 0.0, 0
        for k in rng.integers(0, len(keys), len(keys)):
            for t in clusters[keys[k]]:
                v = vals[t]
                if len(v):
                    total += v[rng.integers(0, len(v), len(v))].sum()
                    count += len(v)
        stats[b] = total / count
    pooled = np.concatenate([vals[t] for t in tiles])
    return float(pooled.mean()), float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


# Two-sided 95% Student-t critical values, df = 1..30 (normal 1.96 beyond).
_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
        2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048,
        2.045, 2.042]


def cluster_robust_ci(values_by_tile: dict, cluster_of: dict | None = None):
    """95% cluster-robust confidence interval for the pooled mean over ships (ratio estimator with
    clusters = overlap groups of tiles, t distribution with G-1 degrees of freedom). Preferred main
    interval when there are few clusters; the percentile bootstrap is the sensitivity check. On
    simulated data shaped like ours (13 clusters, one holding a quarter of the ships) it covers
    about 90%, the tile-resampling bootstrap about 87-89%, and a naive ship-level bootstrap 14%."""
    tiles = sorted(values_by_tile)
    cluster_of = cluster_of or {t: t for t in tiles}
    sums: dict = {}
    counts: dict = {}
    for t in tiles:
        v = np.asarray(values_by_tile[t], dtype=float)
        sums[cluster_of[t]] = sums.get(cluster_of[t], 0.0) + v.sum()
        counts[cluster_of[t]] = counts.get(cluster_of[t], 0) + len(v)
    keys = list(sums)
    g = len(keys)
    if g < 2:
        raise ValueError("need at least two clusters")
    y = np.array([sums[k] for k in keys])
    n = np.array([counts[k] for k in keys], dtype=float)
    mean = y.sum() / n.sum()
    resid = y - mean * n
    se = np.sqrt(g / (g - 1) * np.sum(resid**2)) / n.sum()
    crit = _T95[g - 2] if g - 1 <= len(_T95) else 1.96
    return float(mean), float(mean - crit * se), float(mean + crit * se)


def naive_bootstrap_ci(values, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """Plain bootstrap over ships, ignoring tiles. Kept only to show how much too narrow it is."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    stats = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))
