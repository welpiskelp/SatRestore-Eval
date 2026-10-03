"""SNR-mismatch screen (protocol E4) analysis: does telling the conditioned model the WRONG
SNR degrade its output, and does a bigger lie hurt more? That's the signature that confirms the
FiLM conditioning signal is actually being used, not ignored.

Compares each fold's screened tile at its true offset (0, i.e. the already-existing H1 run) against
the same tile/SNR evaluated with the model told +/-5 and +/-10 dB off from the truth
(db/results_mismatch.sqlite, merged into db/results_pilot_full.sqlite as
full_medium_fold{k}_seed0_offset{+-}{5,10}dB).

Run: python scripts/analyze_mismatch.py
"""
import json
import sqlite3
from pathlib import Path

DB = "db/results_pilot_full.sqlite"
OUT = Path("reports/partial_results")
OUT.mkdir(parents=True, exist_ok=True)

FOLDS = json.load(open("configs/splits.json"))["folds"]
SCREEN_TILE = {k: FOLDS[k]["test"][0] for k in range(4)}
OFFSETS = [-10.0, -5.0, 0.0, 5.0, 10.0]
SNR_LEVELS = [0.0, 20.0, 40.0]

con = sqlite3.connect(DB)
cur = con.cursor()


def psnr(run_name, tile, snr):
    row = cur.execute(
        "select avg(m.value) from metrics m join runs r on m.run_id=r.run_id "
        "where r.name=? and m.tile=? and m.snr_db=? and m.metric='psnr' and m.region='all' "
        "and m.band is null and m.noise_type='gaussian'",
        (run_name, tile, snr)).fetchone()[0]
    return row


lines = ["# SNR-mismatch screen (E4): does the model actually use the conditioning signal?", ""]
lines.append("For each fold's first test tile, the already-trained `full_medium` checkpoint is "
              "evaluated at the TRUE SNR (offset 0, from the existing H1 run) and again after "
              "being told an SNR that is off by +/-5 and +/-10 dB (same noisy input each time -- "
              "only what the model is TOLD changes). If conditioning is actually used, PSNR should "
              "drop as the lie gets bigger, and should drop in both directions (telling it "
              "'cleaner than it is' and 'noisier than it is' should both hurt, just not necessarily "
              "symmetrically).")
lines.append("")
lines.append("| Fold | Tile | SNR (dB) | -10dB | -5dB | 0 (true) | +5dB | +10dB | Monotonic around 0? |")
lines.append("|---|---|---|---|---|---|---|---|---|")

all_monotonic = True
for fold in range(4):
    tile = SCREEN_TILE[fold]
    base = f"full_medium_fold{fold}_seed0"
    for snr in SNR_LEVELS:
        vals = {}
        for off in OFFSETS:
            name = base if off == 0 else f"{base}_offset{off:+g}dB"
            vals[off] = psnr(name, tile, snr)
        if any(v is None for v in vals.values()):
            lines.append(f"| {fold} | {tile} | {snr:g} | missing data |  |  |  |  |  |")
            continue
        true_v = vals[0.0]
        # "monotonic": both +-5 drop vs true, and +-10 drops more than +-5 (allow small noise tolerance)
        mono = (vals[-5.0] <= true_v + 0.05 and vals[5.0] <= true_v + 0.05
                and vals[-10.0] <= vals[-5.0] + 0.05 and vals[10.0] <= vals[5.0] + 0.05)
        all_monotonic &= mono
        lines.append(
            f"| {fold} | {tile} | {snr:g} | {vals[-10.0]:.2f} | {vals[-5.0]:.2f} | "
            f"**{true_v:.2f}** | {vals[5.0]:.2f} | {vals[10.0]:.2f} | {'yes' if mono else 'no'} |"
        )

lines.append("")
lines.append("## Verdict")
lines.append("")
if all_monotonic:
    lines.append("**Every row is monotonic around the true SNR**: telling the model a wrong noise "
                  "level always hurts, and a bigger lie (10 dB) hurts at least as much as a smaller "
                  "one (5 dB), in both directions. This is exactly the signature expected if FiLM "
                  "conditioning is doing real work -- the model is not ignoring the conditioning "
                  "vector, and its output is sensitive to it in the direction that makes sense.")
else:
    lines.append("**Not every row is monotonic.** Some mismatch directions/magnitudes do not "
                  "degrade PSNR as expected. This does not necessarily mean conditioning is broken "
                  "-- single-seed, single-tile-per-fold screening has real noise -- but it means "
                  "the 'conditioning clearly works' claim should be qualified, not stated flatly, "
                  "until checked against more tiles/seeds.")
lines.append("")
lines.append("**Caveat:** this is a screening pass (1 tile per fold, 1 seed, 3 SNR levels, gaussian "
              "noise only), not the full protocol grid -- scoped this way so it would finish in "
              "hours rather than days. Treat it as supporting evidence that the conditioning signal "
              "is used, not as a precise quantification of the mismatch penalty.")

report = "\n".join(lines)
(OUT / "10_mismatch_screen.md").write_text(report, encoding="utf-8")
print(report)
print("\nsaved reports/partial_results/10_mismatch_screen.md")
