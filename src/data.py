"""Phase 1 data pipeline: fetch -> align -> tag -> describe.

Vendor files (primary, dropped in data/raw/ by hand, CSV, first column = date, second = close):
  bbg_xu100.csv   XU100 Index PX_LAST
  USDTRY.csv      USDTRY sampled at the BIST close (BFIX or intraday), NOT a 24h close
Fallbacks when a vendor file is missing: yfinance XU100.IS; EVDS indicative USD (mid), then yfinance TRY=X.
Keys from env or ./.env: EVDS_API_KEY.

Run: python3 src/data.py [--refresh]
"""
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
RAW, OUT = ROOT / "data/raw", ROOT / "data/processed"
START = "1997-01-01"
FLOAT_DATE = pd.Timestamp("2001-02-22")  # D11: pre-float indicator
BIG_MOVE = 0.08                          # DESIGN §5.2 anomaly threshold
# Level breaks where the new unit = old / factor. Applied only if the data actually shows the break.
BREAKS = {"xu100": [("2020-07-27", 100)], "usdtry": [("2005-01-03", 1e6)]}
# DESIGN §8.1 (redrawn 2026-09-25 for the BFIX-only sample): descriptive stats never look at the holdout.
PERIODS = {"burn-in": ("2008-08-01", "2012-12-31"), "validation": ("2013-01-01", "2019-12-31")}
HOLDOUT = "2020-01-01"


# ---------- fetch (immutable raw snapshots, one parquet per source per day) ----------

def _snapshot(name, fetch, refresh):
    RAW.mkdir(parents=True, exist_ok=True)
    have = sorted(RAW.glob(f"{name}_*.parquet"))
    if have and not refresh:
        return pd.read_parquet(have[-1])["close"]
    s = fetch().dropna().rename("close")
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize().rename("date")
    path = RAW / f"{name}_{date.today():%Y%m%d}.parquet"
    if not path.exists():  # never overwrite a raw snapshot
        s.to_frame().to_parquet(path)
    return s


def _yf(ticker):
    import yfinance as yf
    return lambda: yf.download(ticker, start=START, progress=False, auto_adjust=False)["Close"].squeeze()


def _evds_usd():
    from evds import evdsAPI
    api = evdsAPI(os.environ["EVDS_API_KEY"])
    # EVDS silently returns only the LAST 1000 rows of a request -> fetch one calendar year at a time
    df = pd.concat([api.get_data(["TP.DK.USD.A.YTL", "TP.DK.USD.S.YTL"],
                                 startdate=f"01-01-{y}", enddate=f"31-12-{y}")
                    for y in range(pd.Timestamp(START).year, date.today().year + 1)])
    cols = [c for c in df.columns if c.startswith("TP")]
    s = df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)  # mid of buying/selling
    s.index = pd.to_datetime(df["Tarih"], dayfirst=True)
    return s


def _vendor(fname):
    p = RAW / fname
    if not p.exists():
        return None
    df = pd.read_csv(p)
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
                  index=pd.to_datetime(df.iloc[:, 0]).rename("date"), name="close").dropna().sort_index()
    snap = RAW / f"bbg_{p.stem}_{date.today():%Y%m%d}.parquet"  # vendor files get overwritten; keep what we used
    if not snap.exists():
        s.to_frame().to_parquet(snap)
    return s


def load(refresh=False):
    """Return {name: (series, source)}; vendor first, then fallbacks."""
    out = {}
    eq = _vendor("bbg_xu100.csv")
    out["xu100"] = (eq, "bbg") if eq is not None else (_snapshot("yf_XU100.IS", _yf("XU100.IS"), refresh), "yf")
    fx = _vendor("USDTRY.csv")
    if fx is not None:
        out["usdtry"] = (fx, "bbg")
    elif os.environ.get("EVDS_API_KEY"):
        # EVDS rate dated t is fixed ~15:30 on the previous publication day (measured: xcorr with TRY=X peaks at lag 1)
        out["usdtry"] = (_snapshot("evds_usd", _evds_usd, refresh).shift(-1).dropna(), "evds")
    else:
        print("WARN: no USDTRY.csv and no EVDS_API_KEY -> yfinance TRY=X, which starts 2005", file=sys.stderr)
        out["usdtry"] = (_snapshot("yf_TRY=X", _yf("TRY=X"), refresh), "yf")
    # cross-check copies, never used for modelling
    out["xcheck_fx_yf"] = (_snapshot("yf_TRY=X", _yf("TRY=X"), refresh), "yf")
    return out


# ---------- clean / align / tag ----------

def fix_breaks(s, breaks):
    """Rescale history before a known unit break, but only if the jump is in the data (±3 sessions)."""
    s = s.copy()
    for d, factor in breaks:
        r = np.log(s).diff()
        win = r[pd.Timestamp(d) - pd.Timedelta(days=7): pd.Timestamp(d) + pd.Timedelta(days=7)]
        hit = win[(win + np.log(factor)).abs() < 0.5]
        if len(hit):
            s[s.index < hit.index[0]] /= factor
    return s


def align(eq, fx, max_lag_days=4):
    """BIST sessions define the calendar; FX is the last print on or before each session (<= max_lag_days old)."""
    e = eq.rename("xu100").rename_axis("date").reset_index().astype({"date": "datetime64[ns]"})
    f = fx.rename("usdtry").rename_axis("date").reset_index().astype({"date": "datetime64[ns]"})
    f["fx_date"] = f["date"]
    m = pd.merge_asof(e, f, on="date", tolerance=pd.Timedelta(days=max_lag_days))
    m["fx_lag_days"] = (m["date"] - m["fx_date"]).dt.days
    return m.drop(columns="fx_date").set_index("date")


def build(refresh=False):
    src = load(refresh)
    eq = fix_breaks(src["xu100"][0], BREAKS["xu100"])
    fx = fix_breaks(src["usdtry"][0], BREAKS["usdtry"])
    lv = align(eq, fx)
    df = pd.DataFrame(index=lv.index[1:])
    df["rE"] = np.log(lv["xu100"]).diff().iloc[1:]
    df["rF"] = np.log(lv["usdtry"]).diff().iloc[1:]
    df["rUSD"] = df["rE"] - df["rF"]  # exact identity, DESIGN §2
    df["gap_days"] = lv.index.to_series().diff().dt.days.iloc[1:].astype(int)
    df["fx_lag_days"] = lv["fx_lag_days"].iloc[1:]
    df["pre_float"] = df.index < FLOAT_DATE
    df["eq_src"], df["fx_src"] = src["xu100"][1], src["usdtry"][1]
    if src["usdtry"][1] == "bbg":  # D7 (revised): start where time-matched BFIX FX is continuous (after its last >10-day hole)
        gaps = fx.index.to_series().diff().dt.days
        df = df[df.index > gaps[gaps > 10].index.max() if (gaps > 10).any() else fx.index[0]]
    df["unexplained_big_move"] = _unexplained(df)
    tags = RAW.parent / "tags.csv"  # optional manual tags: date,tag (halts, short bans, ...)
    if tags.exists():
        t = pd.read_csv(tags, parse_dates=["date"]).groupby("date")["tag"].agg(";".join)
        df["tag"] = t.reindex(df.index)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "returns.parquet")
    df[df["unexplained_big_move"]].to_csv(OUT / "anomalies.csv")
    return df, src


def covariates(df):
    """TVTP covariates on the BIST session calendar (§12 Q11). Row t = what is public before the close of day t:
    the CDS snap time is unknown and probably after the 18:00 BIST close, so CDS is lagged one session.
    z_cds = log CDS standardized on burn-in stats only; P_t (row t) drives the t -> t+1 regime move."""
    cds = _vendor("TR_CDS_5Y.csv")
    x = align(df["rE"], cds).rename(columns={"usdtry": "cds", "fx_lag_days": "cds_lag_days"}).drop(columns="xu100")
    x = x.shift(1)  # one-session publication lag
    x["cds"] = x["cds"].ffill().bfill()  # vendor holes (ANOMALIES.md): hold the last known print; bfill only row 0
    lc, b = np.log(x["cds"]), slice(*PERIODS["burn-in"])
    x["z_cds"] = (lc - lc[b].mean()) / lc[b].std()
    x.to_parquet(OUT / "covariates.parquet")
    return x


def _unexplained(df):
    ev = pd.read_csv(RAW.parent / "events.csv", parse_dates=["onset"])["onset"]
    # ponytail: fixed window (-7d, +90d) around onsets; crude, it only builds a review list, never a fix
    near = np.zeros(len(df), bool)
    for o in ev:
        near |= (df.index >= o - pd.Timedelta(days=7)) & (df.index <= o + pd.Timedelta(days=90))
    big = (df["rE"].abs() > BIG_MOVE) | (df["rF"].abs() > BIG_MOVE)
    return big & ~near


# ---------- describe ----------

def _stats(r):
    from statsmodels.stats.diagnostic import acorr_ljungbox
    r = r.dropna()
    if len(r) < 60:
        return pd.Series({"n": len(r)})
    z =r / r.rolling(60).std().shift(1)  # rough jump screen; real jump tests need intraday (DESIGN §7.1)
    return pd.Series({
        "n": len(r), "mean_ann": r.mean() * 252, "vol_ann": r.std() * np.sqrt(252),
        "skew": r.skew(), "exkurt": r.kurt(), "min": r.min(), "max": r.max(),
        "q01": r.quantile(.01), "q99": r.quantile(.99),
        "acf1_r": r.autocorr(1), "acf1_r2": (r**2).autocorr(1),
        "LB10_r_p": acorr_ljungbox(r, [10])["lb_pvalue"].iloc[0],
        "LB10_r2_p": acorr_ljungbox(r**2, [10])["lb_pvalue"].iloc[0],
        "jump_days_4sd": int((z.abs() > 4).sum()),
    })


def describe(df, src):
    lines = [f"sources: eq={df['eq_src'].iat[0]} fx={df['fx_src'].iat[0]}   sample {df.index[0]:%Y-%m-%d} .. {df.index[-1]:%Y-%m-%d}",
             "", "== data QA (full sample, incl. holdout) ==",
             f"sessions {len(df)}, FX missing {df['rF'].isna().sum()}, FX stale (lag>0d) {(df['fx_lag_days'] > 0).sum()}",
             f"zero returns: equity {(df['rE'] == 0).sum()}, FX {(df['rF'] == 0).sum()} (stale fixings), gaps >4 calendar days {(df['gap_days'] > 4).sum()}",
             f"unexplained |r|>{BIG_MOVE:.0%}: {int(df['unexplained_big_move'].sum())} -> data/processed/anomalies.csv"]
    # FX timing check: which lag of the modelling FX best matches yfinance TRY=X?
    xf = np.log(src["xcheck_fx_yf"][0]).diff()
    if df["fx_src"].iat[0] != "yf":
        c = {k: df["rF"].corr(xf.reindex(df.index).shift(k)) for k in (-1, 0, 1)}
        lines.append("corr(rF, yf TRY=X shifted k): " + ", ".join(f"k={k}: {v:.2f}" for k, v in c.items())
                     + "   <- peak off k=0 means a date-convention mismatch")
    lines += ["", "== descriptive stats (burn-in + validation only; holdout untouched, DESIGN §8.1) =="]
    for name, (a, b) in PERIODS.items():
        p = df[a:b]
        if p.empty:
            continue
        t = pd.DataFrame({c: _stats(p[c]) for c in ("rE", "rF", "rUSD")})
        lines += [f"-- {name} ({a} .. {b}), corr(rE,rF) = {p['rE'].corr(p['rF']):.2f}", t.round(4).to_string(), ""]
    txt = "\n".join(lines)
    (OUT / "describe.txt").write_text(txt)
    return txt


if __name__ == "__main__":
    df, src = build(refresh="--refresh" in sys.argv)
    covariates(df)
    print(describe(df, src))
