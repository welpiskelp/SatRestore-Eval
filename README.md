# SatRestore-Eval

SNR-Conditioned, Spectrally Aware, Ship-Preserving Reconstruction of Sentinel-2 Maritime Imagery under Controlled Noise Degradation.

## Phase 1: data acquisition + audit

1. Create/activate the project virtual environment and install dependencies:
   ```
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```
2. Authenticate with EOTDL (interactive, requires a phone-verified account — opens a browser login):
   ```
   eotdl auth login
   ```
3. Download the S2-SHIPS dataset:
   ```
   eotdl datasets get S2-SHIPS -p data
   ```
4. Run the data audit:
   ```
   .venv\Scripts\python scripts\audit.py
   ```
   This checks all 16 tiles (band count, shape, dtype, per-band stats, cloud/nodata estimate, mask/annotation presence) and writes `reports/audit_report.json`.

Preprocessing, the degradation engine, the SQLite reference DB, and model/training code are later phases, not yet implemented.
