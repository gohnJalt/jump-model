"""Chart history and charts for the daily report. reports/history.csv is the data the charts (and the website) use.

One row per session since the first validation refit (2013-01-02), all real-time:
  rE, rF, rUSD                  realized log returns of the session
  VaR/ES{a}_{E,F,USD}           1-day MSGJ1-t forecast for the session, made at the previous close (as in daily.csv,
                                where the row date is the as-of close instead)
  pct, flag, sd_usd             MSGJ1 stress gauge at the session's close (expanding percentile, flag rule of §7.8)
  pj_E, pj_F, pj_C              P(equity-only / FX-only / co-jump on the session | its return), MSGJ1-t block params
  fx_share                      share of the session's forecast XU100-in-USD variance due to USDTRY (Euler)
  period                        validation (blind) / holdout (seen) / live
Past rows never change (real-time forecasts), except the last REDO sessions, recomputed each run so revised vendor
data (e.g. a late BFIX file) flows through.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from event_scoring import flag, gjr_sd, pct_expanding
from model import _terms, garch_sig, predictive
from phase4_bench import ALPHAS, TARGETS, MixT

HIST = ROOT / "reports/history.csv"
REDO = 21
HOLD0, LIVE0 = "2020-01-01", "2026-09-26"  # holdout opened on the 2026-09-25 data; later sessions are live


def _block(p, y, b0, b1):
    """Real-time rows for sessions b0..b1-1 from one refit's params."""
    sig = np.asarray(garch_sig(p, y[:b1]))[b0:b1]  # (T,1,2): row t uses y up to t-1
    out = {}
    for k, w in TARGETS.items():
        mix = MixT(*predictive(p, np.ones((b1 - b0, 1)), w, sig), p["nu"])
        for a in ALPHAS:
            v = mix.var(a)
            out[f"VaR{a}_{k}"], out[f"ES{a}_{k}"] = v, mix.es(a, v)
    lt, n = _terms(p, y[:b1])
    lt = np.asarray(lt)[b0:b1, 0]
    post = np.exp(lt - logsumexp(lt, axis=1, keepdims=True))
    for i, k in enumerate("EFC"):
        out[f"pj_{k}"] = post[:, n[:, i] > 0].sum(1)
    lam, s = np.asarray(p["lam"])[0], sig[:, 0]
    jv = [lam[i] * (p["jm"][i] ** 2 + p["js"][i] ** 2) + lam[2] * (p["mC"][i] ** 2 + p["sC"][i] ** 2) for i in range(2)]
    vE, vF = s[:, 0] ** 2 + jv[0], s[:, 1] ** 2 + jv[1]
    cv = p["rho"][0] * s[:, 0] * s[:, 1] + lam[2] * (p["rhoC"] * p["sC"][0] * p["sC"][1] + p["mC"][0] * p["mC"][1])
    out["fx_share"] = (vF - cv) / (vE + vF - 2 * cv)
    return pd.DataFrame(out)


def history(y, d, st, fl, tau, m, path=HIST):
    """Update and return the history (default reports/history.csv) for sessions st['starts'][0] .. len(y)-1."""
    assert st["starts"] == fl["starts"], "VaR and flag refits must share blocks"
    n, i0 = len(y), st["starts"][0]
    old = pd.read_csv(path, index_col="date", parse_dates=["date"]) if path.exists() else None
    t0 = i0 if old is None else max(i0, min(int(d.index.get_indexer([old.index[-1]])[0]) + 1, n) - REDO)
    starts = [b for b in st["starts"] if b < n]
    parts = []
    for k, b0 in enumerate(starts):
        b1 = starts[k + 1] if k + 1 < len(starts) else n
        if b1 > t0:
            parts.append(_block(st["params"][k], y, max(b0, t0), b1))
    new = pd.concat(parts, ignore_index=True)
    new.index = d.index[t0:n].rename("date")
    g = gjr_sd(y, list(zip(starts, starts[1:] + [n])), fl["params"][:len(starts)])[:, 0]  # row t: after close of t
    pct = pct_expanding(g, i0)
    new["pct"], new["flag"], new["sd_usd"] = pct[t0 - i0:], flag(pct, tau, m)[t0 - i0:], g[t0:]
    new["rE"], new["rF"] = y[t0:, 0], y[t0:, 1]
    new["rUSD"] = new["rE"] - new["rF"]
    new["period"] = np.where(new.index < HOLD0, "validation", np.where(new.index < LIVE0, "holdout", "live"))
    h = new if old is None else pd.concat([old[old.index < new.index[0]], new])
    h.to_csv(path, float_format="%.6g")
    return h


# ---------- charts (PNG, for the markdown report) ----------

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3
INK, INK2, GRID, SHADE, DOTS = "#0b0b0b", "#52514e", "#e4e3df", "#f1f0ec", "#b9b8b2"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titlelocation": "left", "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8.5,
                         "savefig.bbox": "tight", "savefig.dpi": 150})
    return plt


def _periods(ax, h):
    """Shade the seen holdout and mark where live forecasting starts."""
    ax.axvspan(pd.Timestamp(HOLD0), pd.Timestamp(LIVE0), color=SHADE, lw=0, zorder=0)
    ax.axvline(pd.Timestamp(LIVE0), color=INK2, lw=0.8, ls="--")


def charts(h, f, out, tau):
    """Four PNGs in out/: stress gauge, 1-day VaR vs returns, jump probability, FX share of USD risk."""
    plt = _style()
    out.mkdir(parents=True, exist_ok=True)
    ev = pd.read_csv(ROOT / "data/events.csv", parse_dates=["onset"])
    ev = ev[ev.onset >= h.index[0]]
    last = h.index[-1]

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(h.index, 100 * h["pct"], color=BLUE, lw=0.7, label="volatility percentile, XU100 in USD")
    ax.axhline(100 * tau, color=INK2, lw=0.8, ls=":", label=f"flag threshold ({tau:.1%}, 3 sessions)")
    on = h.index[h["flag"].astype(bool)]
    ax.scatter(on, np.full(len(on), 102), marker="|", s=20, color=ORANGE, label="flag on")
    for o in ev.onset:
        ax.axvline(o, color=AQUA, lw=0.8)
    ax.plot([], [], color=AQUA, label="listed event")
    ax.scatter([last], [100 * h["pct"].iat[-1]], s=30, color=INK, zorder=5)
    ax.annotate(f"{last.date()}: {h['pct'].iat[-1]:.0%}", (last, 100 * h["pct"].iat[-1]), xytext=(-8, 8),
                textcoords="offset points", ha="right", fontsize=8.5)
    _periods(ax, h)
    ax.set_ylim(0, 105)
    ax.set_ylabel("percentile")
    ax.set_title("Stress gauge since 2013 (shaded: holdout, seen; dashed: live forecasting starts)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.35), ncol=4)
    fig.savefig(out / "stress.png")
    plt.close(fig)

    lab = {"E": "XU100 in TRY", "USD": "XU100 in USD", "F": "USDTRY (long USD)"}
    fig, axs = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)
    for ax, k in zip(axs, ("E", "USD", "F")):
        r, v = 100 * h[f"r{k}"], 100 * h[f"VaR0.01_{k}"]
        hit = r <= v
        ax.scatter(h.index[~hit], r[~hit], s=1.2, color=DOTS, lw=0)
        ax.plot(h.index, v, color=BLUE, lw=0.8)
        ax.scatter(h.index[hit], r[hit], s=8, color=ORANGE, lw=0)
        ax.scatter([last + pd.Timedelta(days=1)], [100 * f[f"VaR0.01_{k}"]], s=30, color=INK, zorder=5)
        ax.set_title(f"{lab[k]}: 1% VaR (blue), breaches (orange) {hit.sum()} of {len(hit)} = {hit.mean():.2%}; "
                     f"black dot = tomorrow {100 * f[f'VaR0.01_{k}']:.2f}%", fontsize=9)
        ax.set_ylim(max(r.min(), -15) - 1, min(max(r.max(), 5), 10) + 1)  # KKM day (+26% USD) would squash the rest
        ax.set_ylabel("%")
        _periods(ax, h)
    fig.tight_layout()
    fig.savefig(out / "var.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 2.8))
    for k, c, name in (("E", BLUE, "equity-only"), ("F", ORANGE, "FX-only"), ("C", AQUA, "co-jump")):
        big = h[f"pj_{k}"] >= 0.2
        ax.vlines(h.index[big], 0, h.loc[big, f"pj_{k}"], color=c, lw=1.1, label=name)
    for o in ev.onset:
        ax.axvline(o, color=INK2, lw=0.6, ls=":")
    _periods(ax, h)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(jump | return)")
    ax.set_title("Probability that a session contained a jump (sessions with P ≥ 0.2; dotted: listed events)")
    ax.legend(loc="upper left", ncol=3)
    fig.savefig(out / "jumps.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.plot(h.index, 100 * h["fx_share"], color=BLUE, lw=0.8)
    _periods(ax, h)
    ax.set_ylim(0, 100)
    ax.set_ylabel("%")
    ax.set_title(f"Share of XU100-in-USD risk coming from USDTRY (latest {h['fx_share'].iat[-1]:.0%})")
    fig.savefig(out / "fx_share.png")
    plt.close(fig)
    return ["stress.png", "var.png", "jumps.png", "fx_share.png"]
