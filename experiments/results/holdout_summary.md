# Holdout (2020-01-02 .. 2026-09-25), DESIGN §8.4, single run

Pass/fail: VaR coverage (<= 2 of 12 reject): PASS, ES Z2 at 2.5% (p >= 1%): FAIL, Log score DM vs GJR-t > -1.96: FAIL, Flag >= 4/7 events, <= 1 FA episode/yr: PASS

## VaR coverage (hits = share of days below VaR)
model target  alpha   hits  Kupiec_p   CC_p  hits 2021-24  hits rest
MSGJ1      E  0.010 0.0155    0.0368 0.0751        0.0160     0.0146
MSGJ1      E  0.025 0.0298    0.2246 0.2526        0.0331     0.0249
MSGJ1      F  0.010 0.0071    0.2148 0.4250        0.0060     0.0088
MSGJ1      F  0.025 0.0143    0.0022 0.0066        0.0130     0.0161
MSGJ1    USD  0.010 0.0131    0.2235 0.3560        0.0130     0.0132
MSGJ1    USD  0.025 0.0208    0.2603 0.0633        0.0221     0.0190
GJR-t      E  0.010 0.0179    0.0036 0.0083        0.0160     0.0205
GJR-t      E  0.025 0.0357    0.0082 0.0256        0.0361     0.0351
GJR-t      F  0.010 0.0071    0.2148 0.4250        0.0060     0.0088
GJR-t      F  0.025 0.0137    0.0012 0.0038        0.0140     0.0132
GJR-t    USD  0.010 0.0155    0.0368 0.0751        0.0160     0.0146
GJR-t    USD  0.025 0.0268    0.6430 0.3254        0.0301     0.0220
MSGJ1 rejections at 5%: 2 of 12
Z2 p at 2.5%: E 0.008, F 0.867, USD 0.564
DM log score MSGJ1 vs GJR-t (+ = MSGJ1 better): E -1.13, F -2.47, USD -3.03

## Stress flag
                    tau  m    AUC hits  FA_eps/yr  alarm_share
signal                                                        
GJR-J usd (v1)    0.925  3  0.935  7/7       0.30        0.067
RV20 (benchmark)  0.850  3  0.972  7/7       0.75        0.137
                                                    GJR-J usd (v1)  RV20 (benchmark)
2020-03-11 Covid                                                 3                 0
2021-03-20 CBRT governor dismissal                               2                 2
2021-11-23 Lira crisis then 20 Dec KKM                           2                 3
2022-02-24 Russia invades Ukraine                                2                 2
2023-02-06 Kahramanmaras earthquakes (BIST closed)               3                 0
2023-05-14 Elections then June policy U-turn                    18                 2
2025-03-19 Istanbul mayor detained                               2                 2

Full tables: holdout_tables.md