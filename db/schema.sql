-- Reference database for the SNR-conditioned Sentinel-2 reconstruction project.
-- Everything here is deterministic and rebuilt from files by scripts/build_db.py, so it can always
-- be overwritten. Run and metric results live in separate results databases (db/results_schema.sql),
-- one per person or batch, which are merged later (src/eval/results.py).

CREATE TABLE tiles (
    tile                          TEXT PRIMARY KEY,
    date                          TEXT NOT NULL,
    height                        INTEGER NOT NULL,
    width                         INTEGER NOT NULL,
    crs                           TEXT NOT NULL,
    bounds_left                   REAL NOT NULL,
    bounds_bottom                 REAL NOT NULL,
    bounds_right                  REAL NOT NULL,
    bounds_top                    REAL NOT NULL,
    ship_pixels                   INTEGER NOT NULL,
    ship_instances                INTEGER NOT NULL,   -- COCO segmentation polygons (1,053 in total)
    water_fraction                REAL NOT NULL,
    ship_on_water_fraction        REAL,
    overlap_group                 TEXT                -- tiles that overlap spatially share a group; NULL if independent
);

-- Signal statistics that define SNR (P_b = mean square over valid pixels: not 0, not saturated).
CREATE TABLE tile_band_stats (
    tile            TEXT NOT NULL REFERENCES tiles(tile),
    band            TEXT NOT NULL,
    power           REAL NOT NULL,      -- P_b
    mean            REAL NOT NULL,      -- mean signal over valid pixels (sets the Poisson gain)
    valid_fraction  REAL NOT NULL,
    power_naive     REAL NOT NULL,      -- mean square over all pixels, for reference only
    peak            REAL NOT NULL,      -- 99.9th percentile of valid pixels: the PSNR peak
    PRIMARY KEY (tile, band)
);

-- Ship visibility: mean ship-pixel value minus mean value of nearby open water (3-10 px ring around
-- ship pixels, inside the water mask). CNR at a given SNR is |contrast_dn| / sigma_b. NULL when fewer
-- than 30 background pixels exist (the delivered water mask misses the river in 'rome').
CREATE TABLE tile_ship_contrast (
    tile         TEXT NOT NULL REFERENCES tiles(tile),
    band         TEXT NOT NULL,
    contrast_dn  REAL,
    ship_mean    REAL,
    bg_mean      REAL,
    n_ship_px    INTEGER NOT NULL,
    n_bg_px      INTEGER NOT NULL,
    PRIMARY KEY (tile, band)
);

-- Cross-validation folds: every tile is in test exactly once.
CREATE TABLE folds (
    fold   INTEGER NOT NULL,
    tile   TEXT NOT NULL REFERENCES tiles(tile),
    split  TEXT NOT NULL CHECK (split IN ('train', 'val', 'test')),
    PRIMARY KEY (fold, tile)
);

CREATE TABLE patches (
    patch_id        TEXT PRIMARY KEY,
    tile            TEXT NOT NULL REFERENCES tiles(tile),
    y               INTEGER NOT NULL,
    x               INTEGER NOT NULL,
    ship_pixels     INTEGER NOT NULL,
    ship_instances  INTEGER NOT NULL,   -- ships whose centroid is inside; a ship appears in several overlapping patches
    water_frac      REAL NOT NULL,
    sat_any_frac    REAL NOT NULL,
    sat_b09_frac    REAL NOT NULL,
    zero_any_frac   REAL NOT NULL,
    UNIQUE (tile, y, x)
);
CREATE INDEX idx_patches_tile ON patches(tile);

-- One row per ship (COCO polygon). home_patch_id is the single patch a ship is counted in for
-- per-instance statistics: the patch containing its centroid whose centre is nearest.
CREATE TABLE ship_instances (
    instance_id    TEXT PRIMARY KEY,
    tile           TEXT NOT NULL REFERENCES tiles(tile),
    cx             REAL NOT NULL,
    cy             REAL NOT NULL,
    home_patch_id  TEXT NOT NULL REFERENCES patches(patch_id)
);
CREATE INDEX idx_ship_instances_tile ON ship_instances(tile);

-- The fixed test degradations: one noisy realisation per (tile, noise type, SNR), shared by every
-- model; test patches are crops of it (src.degrade.noise.test_noisy_tile). params_json is the full
-- parameter record returned by noise_params (per-band SNR, sigma, Poisson gain, read sigma,
-- correlation sd, shot fraction). Training noise is drawn on the fly and not logged per sample.
CREATE TABLE degradations (
    degradation_id  INTEGER PRIMARY KEY,
    tile            TEXT NOT NULL REFERENCES tiles(tile),
    noise_type      TEXT NOT NULL CHECK (noise_type IN ('gaussian', 'poisson_gaussian', 'correlated_gaussian')),
    snr_db          REAL NOT NULL,
    seed            INTEGER NOT NULL,
    shot_fraction   REAL NOT NULL,
    params_json     TEXT NOT NULL,
    UNIQUE (tile, noise_type, snr_db)
);
CREATE INDEX idx_degradations_level ON degradations(noise_type, snr_db);

