"""DESIGN §7.6: TVTP-MSGJ2 (lagged log CDS) on real data, the checks that are not in phase4 / event_scoring output.

  LR      TVTP vs static MSGJ2 on the burn-in fit (2008-08 .. 2012-12), chi2_2 at 1%. The TVTP fit is warm-started from
          the static one with delta = 0 (as in §6.1b, so LR >= 0); the Phase 4 first-block fit is kept if better.
  signs   median over Phase 4 refits of delta(calm->stress) > 0 and delta(stress->calm) < 0
  DM      log score vs static MSGJ2 and vs GJR-t, per target (> -1.96)
  CDS     smoothed-probability-weighted mean CDS by regime, validation only, last refit's params

Needs experiments/results/phase4/{MSGJ2,GJR-t,TVTP-MSGJ2}.pkl.   Run: python3 experiments/tvtp_real.py
"""
import pickle
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np
import pandas as pd
from scipy.stats import chi2

from fit import default_sig_min, fit
from model import hamilton, smooth
from phase4_bench import LADDER, TARGETS, dm, load, load_x

P4 = ROOT / "experiments/results/phase4"


def main():
    y, dates, i0 = load()
    x = load_x(dates)
    R = {m: pickle.load(open(P4 / f"{m}.pkl", "rb")) for m in ("MSGJ2", "GJR-t", "TVTP-MSGJ2")}
    ps, pt = R["MSGJ2"]["params"][0], R["TVTP-MSGJ2"]["params"][0]  # both fitted on y[:i0]
    s = replace(LADDER["TVTP-MSGJ2"], sig_min=default_sig_min(y[:i0]))
    warm, ll_warm, _ = fit(y[:i0], s, n_starts=1, init=ps | {"delta": np.zeros((2, 2, 1))}, x=x[:i0])
    ll_s = float(hamilton(ps, y[:i0])[0])
    ll_p4 = float(hamilton(pt, y[:i0], x=x[:i0])[0])
    ll_t, pb = max((ll_warm, warm), (ll_p4, pt), key=lambda r: r[0])
    LR = 2 * (ll_t - ll_s)
    pval = chi2.sf(LR, 2)

    D = np.array([p["delta"][..., 0] for p in R["TVTP-MSGJ2"]["params"]])  # (refits, 2, 2); regime 1 = stress
    up, down = np.median(D[:, 0, 1]), np.median(D[:, 1, 0])

    dmt = {(ref, k): dm(R["TVTP-MSGJ2"]["targets"][k]["ls"] - R[ref]["targets"][k]["ls"])
           for ref in ("MSGJ2", "GJR-t") for k in TARGETS}

    pl = R["TVTP-MSGJ2"]["params"][-1]
    _, filt, pred = hamilton(pl, y, x=x)
    sm = smooth(pl, filt, pred, x=x)[i0:]
    cds = pd.read_parquet(ROOT / "data/processed/covariates.parquet").reindex(dates)["cds"].to_numpy()[i0:]
    cds_by = (sm * cds[:, None]).sum(0) / sm.sum(0)

    ok = {"LR at 1%": pval < 0.01, "signs": up > 0 > down, "DM > -1.96": min(dmt.values()) > -1.96,
          "CDS stress > calm": cds_by[1] > cds_by[0]}
    L = ["# TVTP-MSGJ2 on real data, §7.6 checks (validation 2013-2019; detection is in event_scoring.md)", "",
         f"LR (burn-in fit): static ll {ll_s:.1f}, TVTP ll {ll_t:.1f} (warm {ll_warm:.1f}, Phase 4 first block {ll_p4:.1f})"
         f" -> LR {LR:.2f}, p {pval:.4f}  [burn-in delta: up {pb['delta'][0, 1, 0]:.2f}, down {pb['delta'][1, 0, 0]:.2f}]",
         f"delta over {len(D)} refits: calm->stress median {up:.2f} (q10-q90 {np.quantile(D[:, 0, 1], .1):.2f}..{np.quantile(D[:, 0, 1], .9):.2f}),"
         f" stress->calm median {down:.2f} (q10-q90 {np.quantile(D[:, 1, 0], .1):.2f}..{np.quantile(D[:, 1, 0], .9):.2f})",
         "DM log score (TVTP minus ref; + = TVTP better): "
         + ", ".join(f"vs {r} {k} {v:.2f}" for (r, k), v in dmt.items()),
         f"Mean CDS by regime (smoothed, validation): calm {cds_by[0]:.0f} bp, stress {cds_by[1]:.0f} bp;"
         f" share of days P(stress) > 0.5: {(sm[:, 1] > .5).mean():.1%}", "",
         "Pass/fail: " + ", ".join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in ok.items())]
    txt = "\n".join(L)
    (ROOT / "experiments/results/tvtp_real_summary.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
