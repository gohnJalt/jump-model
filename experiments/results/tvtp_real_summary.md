# TVTP-MSGJ2 on real data, §7.6 checks (validation 2013-2019; detection is in event_scoring.md)

LR (burn-in fit): static ll 7217.1, TVTP ll 7219.1 (warm 7218.8, Phase 4 first block 7219.1) -> LR 3.94, p 0.1395  [burn-in delta: up 0.72, down 0.96]
delta over 84 refits: calm->stress median 1.62 (q10-q90 0.20..1.84), stress->calm median 1.02 (q10-q90 0.26..1.20)
DM log score (TVTP minus ref; + = TVTP better): vs MSGJ2 E -0.13, vs MSGJ2 F -0.62, vs MSGJ2 USD -0.13, vs GJR-t E -0.52, vs GJR-t F -0.09, vs GJR-t USD 0.15
Mean CDS by regime (smoothed, validation): calm 250 bp, stress 252 bp; share of days P(stress) > 0.5: 29.3%

Pass/fail: LR at 1%: FAIL, signs: FAIL, DM > -1.96: PASS, CDS stress > calm: PASS