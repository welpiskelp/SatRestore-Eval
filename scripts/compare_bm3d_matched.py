"""BM3D vs conditioned (full_medium) vs blind, on the SAME tiles and SAME 9 SNR levels, gaussian
noise, region=all PSNR -- apples-to-apples, unlike the original 6-tile/5-SNR baseline_bm3d run.
Only reports tiles where the matched-protocol BM3D run (db/results_bm3d_matched.sqlite, unioned
across every run name starting with "baseline_bm3d_matched" since each batch round uses a new
run name) has all 9 SNR levels -- partial tiles are excluded rather than padded.

Run: python scripts/compare_bm3d_matched.py
"""
import sqlite3
from pathlib import Path

MAIN_DB = "db/results_pilot_full.sqlite"
BM3D_DB = "db/results_bm3d_matched.sqlite"
OUT = Path("reports/partial_results")
OUT.mkdir(parents=True, exist_ok=True)

bm3d_con = sqlite3.connect(BM3D_DB)
main_con = sqlite3.connect(MAIN_DB)

complete_tiles = sorted(t for (t,) in bm3d_con.execute(
    "select tile from metrics where metric='psnr' group by tile having count(distinct snr_db) >= 9"))

def bm3d_psnr(tile, snr):
    return bm3d_con.execute(
        "select avg(value) from metrics where tile=? and snr_db=? and metric='psnr' and region='all' "
        "and band is null", (tile, snr)).fetchone()[0]

def model_psnr(run_like, tile, snr):
    return main_con.execute(
        "select avg(m.value) from metrics m join runs r on m.run_id=r.run_id "
        "where r.name like ? and r.name not like '%_offset%' and m.tile=? and m.snr_db=? "
        "and m.metric='psnr' and m.region='all' and m.band is null and m.noise_type='gaussian'",
        (run_like, tile, snr)).fetchone()[0]

SNR_LEVELS = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

lines = ["# BM3D vs conditioned vs blind, matched protocol (apples-to-apples)", ""]
lines.append(f"Tiles with all 9 SNR levels in the matched BM3D run so far: {complete_tiles} "
              f"({len(complete_tiles)}/16). Partial tiles excluded. Gaussian noise, PSNR, region=all.")
lines.append("")
lines.append("| SNR (dB) | BM3D | Conditioned (full_medium) | Blind (e1_blind) | BM3D vs Conditioned | Winner |")
lines.append("|---|---|---|---|---|---|")

crossover_notes = []
prev_winner = None
for snr in SNR_LEVELS:
    bm3d_vals, cond_vals, blind_vals = [], [], []
    for t in complete_tiles:
        b = bm3d_psnr(t, snr)
        c = model_psnr("full_medium_fold%_seed%", t, snr)
        bl = model_psnr("e1_blind_fold%_seed%", t, snr)
        if b is not None: bm3d_vals.append(b)
        if c is not None: cond_vals.append(c)
        if bl is not None: blind_vals.append(bl)
    bm3d_mean = sum(bm3d_vals) / len(bm3d_vals) if bm3d_vals else float("nan")
    cond_mean = sum(cond_vals) / len(cond_vals) if cond_vals else float("nan")
    blind_mean = sum(blind_vals) / len(blind_vals) if blind_vals else float("nan")
    gap = bm3d_mean - cond_mean
    winner = "BM3D" if bm3d_mean > cond_mean else "conditioned"
    if prev_winner and winner != prev_winner:
        crossover_notes.append(f"{snr:g} dB")
    prev_winner = winner
    lines.append(f"| {snr:g} | {bm3d_mean:.2f} | {cond_mean:.2f} | {blind_mean:.2f} | {gap:+.2f} | {winner} |")

lines.append("")
if crossover_notes:
    lines.append(f"**Crossover point(s): {', '.join(crossover_notes)}** -- BM3D and the conditioned "
                  "model swap which one wins around here.")
else:
    lines.append("**No crossover in this SNR range** -- one method wins at every level tested so far.")
lines.append("")
lines.append(f"Caveat: only {len(complete_tiles)}/16 tiles so far (more BM3D batches still running); "
              "this table will be regenerated as more tiles land. These tiles are not a random "
              "sample of the 16 -- rotterdam1/2/3 are the overlap-group trio and toulon has the "
              "most ships -- so don't treat this as the final 16-tile answer yet.")

report = "\n".join(lines)
(OUT / "11_bm3d_matched_comparison.md").write_text(report, encoding="utf-8")
print(report)
print("\nsaved reports/partial_results/11_bm3d_matched_comparison.md")
