-- Results database: runs and their metrics. One file per person or batch, merged later with
-- src/eval/results.py. Tiles and ships are referenced by name (they exist in db/reference.sqlite).
-- Run identity is `name`, which must be unique across all shards.

CREATE TABLE runs (
    run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL UNIQUE,
    model        TEXT NOT NULL,
    fold         INTEGER,
    seed         INTEGER,
    config_json  TEXT,
    git_commit   TEXT,
    status       TEXT NOT NULL DEFAULT 'created',
    created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at  TEXT,
    notes        TEXT
);

-- Per-tile and per-region metrics in long format. region: 'all', 'ship', 'water', 'background'.
-- band is NULL for metrics aggregated over bands, otherwise the band name (per-band RMSE, PSNR, SSIM).
CREATE TABLE metrics (
    metric_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(run_id),
    tile        TEXT NOT NULL,
    noise_type  TEXT NOT NULL,
    snr_db      REAL NOT NULL,
    region      TEXT NOT NULL,
    metric      TEXT NOT NULL,
    band        TEXT,
    value       REAL NOT NULL,
    n_pixels    INTEGER
);
CREATE UNIQUE INDEX idx_metrics_key
    ON metrics(run_id, tile, noise_type, snr_db, region, metric, COALESCE(band, ''));

-- One value per ship (COCO polygon, ids from reference.sqlite ship_instances) and metric, for
-- per-instance analysis and cluster-robust intervals.
CREATE TABLE instance_metrics (
    run_id       INTEGER NOT NULL REFERENCES runs(run_id),
    instance_id  TEXT NOT NULL,
    noise_type   TEXT NOT NULL,
    snr_db       REAL NOT NULL,
    metric       TEXT NOT NULL,
    value        REAL NOT NULL,
    PRIMARY KEY (run_id, instance_id, noise_type, snr_db, metric)
);
