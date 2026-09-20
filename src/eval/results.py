"""Results databases: create, record runs and metrics, and merge per-person shards."""
import sqlite3
from pathlib import Path

RESULTS_SCHEMA = Path(__file__).resolve().parents[2] / "db" / "results_schema.sql"


def connect(path) -> sqlite3.Connection:
    """Open a results database, creating it from the schema if it does not exist."""
    path = Path(path)
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    if is_new:
        conn.executescript(RESULTS_SCHEMA.read_text())
    return conn


def add_run(conn, name: str, model: str, fold=None, seed=None, config_json=None, git_commit=None, notes=None) -> int:
    with conn:
        cur = conn.execute(
            "INSERT INTO runs (name, model, fold, seed, config_json, git_commit, notes) VALUES (?,?,?,?,?,?,?)",
            (name, model, fold, seed, config_json, git_commit, notes),
        )
    return cur.lastrowid


def add_metrics(conn, run_id: int, tile: str, noise_type: str, snr_db: float, rows: list[dict]) -> None:
    """rows: dicts with region, metric, band (or None), value, n_pixels, as returned by tile_metrics."""
    with conn:
        conn.executemany(
            "INSERT INTO metrics (run_id, tile, noise_type, snr_db, region, metric, band, value, n_pixels)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            [(run_id, tile, noise_type, snr_db, r["region"], r["metric"], r["band"], r["value"], r["n_pixels"])
             for r in rows],
        )


def add_instance_metrics(conn, run_id: int, noise_type: str, snr_db: float, metric: str, values: dict) -> None:
    """values: instance_id -> float (NaN values are skipped)."""
    with conn:
        conn.executemany(
            "INSERT INTO instance_metrics VALUES (?,?,?,?,?,?)",
            [(run_id, iid, noise_type, snr_db, metric, float(v)) for iid, v in values.items() if v == v],
        )


def merge(dst_path, src_path) -> dict:
    """Copy every run (with its metrics) from src into dst, assigning new run ids. Raises if a run
    name already exists in dst, so shards from different people cannot silently overwrite each other."""
    dst = connect(dst_path)
    dst.execute("ATTACH DATABASE ? AS src", (str(src_path),))
    clash = dst.execute("SELECT s.name FROM src.runs s JOIN main.runs d ON d.name = s.name").fetchall()
    if clash:
        dst.close()
        raise ValueError(f"run names already present in {dst_path}: {[c[0] for c in clash]}")
    counts = {"runs": 0, "metrics": 0, "instance_metrics": 0}
    with dst:
        for row in dst.execute("SELECT run_id, name, model, fold, seed, config_json, git_commit, status, "
                               "created_at, finished_at, notes FROM src.runs ORDER BY run_id").fetchall():
            old = row[0]
            cur = dst.execute(
                "INSERT INTO main.runs (name, model, fold, seed, config_json, git_commit, status, created_at,"
                " finished_at, notes) VALUES (?,?,?,?,?,?,?,?,?,?)", row[1:])
            new = cur.lastrowid
            counts["runs"] += 1
            counts["metrics"] += dst.execute(
                "INSERT INTO main.metrics (run_id, tile, noise_type, snr_db, region, metric, band, value, n_pixels)"
                " SELECT ?, tile, noise_type, snr_db, region, metric, band, value, n_pixels FROM src.metrics"
                " WHERE run_id = ?", (new, old)).rowcount
            counts["instance_metrics"] += dst.execute(
                "INSERT INTO main.instance_metrics SELECT ?, instance_id, noise_type, snr_db, metric, value"
                " FROM src.instance_metrics WHERE run_id = ?", (new, old)).rowcount
    dst.execute("DETACH DATABASE src")
    dst.close()
    return counts
