"""Tile-level cross-validation splits that never separate spatially overlapping tiles."""
import itertools
import random
import statistics


def overlap_groups(bounds: dict[str, list[float]], min_overlap_m: float = 1.0) -> list[list[str]]:
    """Connected components of tiles whose bounding boxes [left, bottom, right, top] overlap by
    more than min_overlap_m in both axes (tiles that only touch are independent)."""
    parent = {t: t for t in bounds}

    def find(t: str) -> str:
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    for a, b in itertools.combinations(sorted(bounds), 2):
        la, ba, ra, ta = bounds[a]
        lb, bb, rb, tb = bounds[b]
        if min(ra, rb) - max(la, lb) > min_overlap_m and min(ta, tb) - max(ba, bb) > min_overlap_m:
            parent[find(a)] = find(b)

    comps: dict[str, list[str]] = {}
    for t in sorted(bounds):
        comps.setdefault(find(t), []).append(t)
    return sorted(comps.values())


def _partitions(groups: list[list[str]], n_bins: int, cap: int):
    """Yield every way to pack whole groups into n_bins bins of exactly cap tiles (bins unlabeled)."""
    groups = sorted(groups, key=lambda g: (-len(g), g))
    bins: list[list[list[str]]] = [[] for _ in range(n_bins)]
    sizes = [0] * n_bins

    def rec(i: int):
        if i == len(groups):
            if all(s == cap for s in sizes):
                yield [[t for g in b for t in g] for b in bins]
            return
        opened_empty = False
        for k in range(n_bins):
            if sizes[k] + len(groups[i]) > cap:
                continue
            if sizes[k] == 0:
                if opened_empty:
                    continue
                opened_empty = True
            bins[k].append(groups[i])
            sizes[k] += len(groups[i])
            yield from rec(i + 1)
            bins[k].pop()
            sizes[k] -= len(groups[i])

    yield from rec(0)


def _cv(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def _pick_val(pool_groups, polygons, target, val_uses, n_val, rng):
    """Choose whole groups totalling n_val tiles whose ship count is near target, spreading use
    across folds (each earlier use of a tile as val costs 25 polygons of closeness)."""
    best, best_score = None, None
    for r in range(1, n_val + 1):
        for combo in itertools.combinations(pool_groups, r):
            tiles = [t for g in combo for t in g]
            if len(tiles) != n_val:
                continue
            score = (
                abs(sum(polygons[t] for t in tiles) - target)
                + 25 * sum(val_uses[t] for t in tiles)
                + rng.random() * 1e-6
            )
            if best_score is None or score < best_score:
                best, best_score = tiles, score
    if best is None:
        raise ValueError("no val set of whole overlap groups with the requested size")
    return sorted(best)


def make_folds(tiles, bounds, polygons, ship_pixels, n_folds=4, n_val=2, seed=0):
    """Return (folds, groups). Every tile is a test tile exactly once. Test sets are the packing of
    overlap groups into equal-size bins that best balances ship polygons and ship pixels."""
    tiles = sorted(tiles)
    if len(tiles) % n_folds:
        raise ValueError(f"{len(tiles)} tiles not divisible into {n_folds} test sets")
    groups = overlap_groups({t: bounds[t] for t in tiles})
    cap = len(tiles) // n_folds

    best, best_score = None, None
    for bins in _partitions(groups, n_folds, cap):
        score = _cv([sum(polygons[t] for t in b) for b in bins]) + _cv([sum(ship_pixels[t] for t in b) for b in bins])
        if best_score is None or score < best_score - 1e-12:
            best, best_score = bins, score
    if best is None:
        raise ValueError("overlap groups cannot be packed into equal-size test sets")

    rng = random.Random(seed)
    test_sets = [sorted(b) for b in best]
    rng.shuffle(test_sets)

    group_of = {t: tuple(g) for g in groups for t in g}
    target = sum(polygons[t] for t in tiles) / (2 * n_folds)
    val_uses = {t: 0 for t in tiles}
    folds = []
    for k, test in enumerate(test_sets):
        pool = [g for g in groups if not set(g) & set(test)]
        val = _pick_val(pool, polygons, target, val_uses, n_val, rng)
        for t in val:
            val_uses[t] += 1
        train = sorted(set(tiles) - set(test) - set(val))
        folds.append({"fold": k, "train": train, "val": val, "test": test})
    validate_folds(folds, tiles, groups)
    return folds, groups


def validate_folds(folds, tiles, groups):
    tiles = sorted(tiles)
    tested = sorted(t for f in folds for t in f["test"])
    assert tested == tiles, "every tile must be a test tile exactly once"
    for f in folds:
        parts = [set(f["train"]), set(f["val"]), set(f["test"])]
        assert sum(len(p) for p in parts) == len(tiles), f"fold {f['fold']}: splits overlap"
        assert set().union(*parts) == set(tiles), f"fold {f['fold']}: splits do not cover all tiles"
        for g in groups:
            homes = {i for i, p in enumerate(parts) if p & set(g)}
            assert len(homes) == 1, f"fold {f['fold']}: overlap group {g} split across train/val/test"
