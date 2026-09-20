"""Verify src.eval.stats: exact Wilcoxon on hand-computable cases, Holm, and bootstrap coverage on
clustered data (the clustered bootstrap should cover about 95%, the naive one far less)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.stats import cluster_robust_ci, exact_wilcoxon, hierarchical_bootstrap_ci, holm, naive_bootstrap_ci

failures = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        failures.append(name)


# Wilcoxon: cases whose exact p-values follow from counting sign patterns by hand.
check("wilcoxon n=5 all positive, p = 2/32", abs(exact_wilcoxon([1, 2, 3, 4, 5])["p"] - 2 / 32) < 1e-12)
check("wilcoxon [1,2,3], p = 2/8", abs(exact_wilcoxon([1, 2, 3])["p"] - 0.25) < 1e-12)
check("wilcoxon [-1,2,3], p = 4/8", abs(exact_wilcoxon([-1, 2, 3])["p"] - 0.5) < 1e-12)
check("wilcoxon balanced, p = 1", abs(exact_wilcoxon([1, -1, 2, -2])["p"] - 1.0) < 1e-12)
check("wilcoxon paired form equals difference form",
      exact_wilcoxon([3, 5, 9, 4], [1, 2, 3, 6])["p"] == exact_wilcoxon([2, 3, 6, -2])["p"])
check("wilcoxon drops zero differences", exact_wilcoxon([1, 1, 3, 3], [1, 1, 1, 1])["n"] == 2)
check("wilcoxon rank-biserial +1 when all positive", exact_wilcoxon([1, 2, 3, 4])["rank_biserial"] == 1.0)
rng = np.random.default_rng(0)
p16 = exact_wilcoxon(rng.normal(0.4, 1, 16))
check("wilcoxon n=16 runs, p in [0,1]", 0 <= p16["p"] <= 1, f"(p={p16['p']:.4f})")

# Holm: sorted p = .01,.03,.04 with m=3 -> .03, .06, .06.
check("holm example", np.allclose(holm([0.04, 0.01, 0.03]), [0.06, 0.03, 0.06]))
check("holm caps at 1 and is monotone", np.allclose(holm([0.5, 0.9]), [1.0, 1.0]))

# Bootstrap coverage on clustered data: 16 tiles, ship counts like the real ones, tile effect sd 1,
# ship noise sd 0.3, true mean 0 (mean of the tile effects). Tiles 0-1 and 2-4 form overlap clusters
# whose members share one effect (as near-duplicate scenes would).
counts = [146, 66, 69, 42, 119, 122, 33, 50, 66, 5, 5, 5, 8, 15, 39, 263]
cluster = {f"t{i:02d}": f"c{c}" for i, c in enumerate([0, 1, 2, 3, 4, 5, 5, 5, 6, 7, 7, 8, 9, 10, 11, 12])}
hit_h = hit_n = hit_r = 0
reps = 400
rng = np.random.default_rng(1)
for r in range(reps):
    cl_eff = {c: rng.normal(0, 1) for c in set(cluster.values())}
    data = {f"t{i:02d}": cl_eff[cluster[f"t{i:02d}"]] + rng.normal(0, 0.3, n) for i, n in enumerate(counts)}
    _, lo, hi = hierarchical_bootstrap_ci(data, cluster, n_boot=300, seed=r)
    hit_h += lo <= 0 <= hi
    _, lo, hi = cluster_robust_ci(data, cluster)
    hit_r += lo <= 0 <= hi
    _, lo, hi = naive_bootstrap_ci(np.concatenate(list(data.values())), n_boot=300, seed=r)
    hit_n += lo <= 0 <= hi
cov_h, cov_r, cov_n = hit_h / reps, hit_r / reps, hit_n / reps
print(f"     coverage of the true mean over {reps} simulations (nominal 0.95): "
      f"cluster-robust t {cov_r:.2f}, cluster bootstrap {cov_h:.2f}, naive ship-level bootstrap {cov_n:.2f}")
print("     (13 clusters, one holding a quarter of the ships: clustered methods still under-cover by a few points)")
check("cluster-robust interval covers at least 0.85", cov_r >= 0.85)
check("cluster bootstrap covers at least 0.80", cov_h >= 0.80)
check("naive bootstrap under-covers badly (< 0.6)", cov_n < 0.6)
mm, lo, hi = cluster_robust_ci({"a": [1.0, 1.0], "b": [3.0], "c": [2.0, 2.0]})
check("cluster-robust mean is the pooled mean", abs(mm - 9 / 5) < 1e-12 and lo < mm < hi)
m, lo, hi = hierarchical_bootstrap_ci({"a": [1.0, 1.0], "b": [3.0]}, n_boot=500)
check("bootstrap mean is the pooled mean", abs(m - 5 / 3) < 1e-12 and lo <= m <= hi)

if failures:
    print("\nFAILED:", failures)
    sys.exit(1)
print("\nall statistics checks passed")
