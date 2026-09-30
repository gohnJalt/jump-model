#!/bin/zsh
# Scheduled daily run (launchd: ~/Library/LaunchAgents/com.jumpmodel.daily.plist, weekdays 19:30 and 22:30).
# Refreshes XU100 (yfinance), rebuilds returns, then writes the report only if the last session's USDTRY print is
# current: a stale BFIX file would put a fake 0 FX return into the forecast. Otherwise it notifies and skips.
# A session that already has a report is skipped (so the 22:30 run is a no-op after a good 19:30 one).
# Manual run after dropping data/raw/USDTRY.csv: scripts/daily.sh   (rebuild an existing report: --force)
cd "${0:A:h}/.." || exit 1
PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
notify() { osascript -e "display notification \"$1\" with title \"XU100 risk\""; }
echo "== $(date '+%F %T')"
$PY src/data.py --refresh || { notify "data.py failed: see reports/daily.log"; exit 1; }
read day lag <<< "$($PY -c "
import pandas as pd
d = pd.read_parquet('data/processed/returns.parquet').iloc[-1]
print(d.name.date(), 99 if pd.isna(d.fx_lag_days) else int(d.fx_lag_days))")"
if [[ -f reports/daily/$day.md && $1 != --force ]]; then
  echo "skipped $day: report exists (use --force to rebuild it)"
  exit 0
fi
if [[ $lag != 0 ]]; then
  notify "$day: USDTRY is stale ($lag day lag). Drop the BFIX export into data/raw/USDTRY.csv, then run scripts/daily.sh"
  echo "skipped $day: FX lag $lag"
  exit 0
fi
$PY src/daily.py || { notify "daily.py failed: see reports/daily.log"; exit 1; }
# Website feed: derived numbers only (src/site_export.py). The push triggers the site's deploy workflow.
# A failure here never fails the run: the report is already written.
SITE=../bubble-methodology
if [[ -d $SITE/.git ]]; then
  { $PY src/site_export.py && git -C $SITE add site/risk.json &&
    { git -C $SITE diff --cached --quiet || git -C $SITE commit -qm "Risk readings after the $day close" -- site/risk.json; } &&
    git -C $SITE push -q; } || notify "$day: report written, but the site update failed: see reports/daily.log"
fi
notify "$day: $(grep -o 'Stress flag: [A-Za-z]*' reports/daily/$day.md). Report in reports/daily/$day.md"
