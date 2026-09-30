# §7.3 stress detection, validation 2013-01-02 .. 2019-12-31 (4 events, real-time signals)
Cost = 4.0 x missed events + 1 x false-alarm episodes (§12 Q3b); ties -> fewer alarm days.

                tau  m    AUC hits  mean_lag  FA_eps/yr  alarm_share  cost
signal                                                                    
GJR-J usd     0.925  3  0.899  3/4     6.000      0.144        0.023   5.0
GJR-J max     0.975  3  0.845  2/4     5.500      0.144        0.019   9.0
MSGJ2 P       0.900  3  0.567  2/4     1.500      0.431        0.007  11.0
MS-JD3 P      0.950  3  0.862  3/4     8.667      0.144        0.010   5.0
TVTP-MSGJ2 P  0.950  3  0.535  1/4     1.000      0.144        0.003  13.0
RV20          0.850  3  0.887  4/4     6.000      0.431        0.080   3.0
CDS lvl       0.990  3  0.553  1/4    13.000      0.000        0.003  12.0
CDS d20       0.975  3  0.821  2/4     8.000      0.431        0.023  11.0

## Detection lag per event (sessions; '-' = missed)
                                                              GJR-J usd GJR-J max MSGJ2 P MS-JD3 P TVTP-MSGJ2 P  RV20 CDS lvl CDS d20
2013-05-22 Taper tantrum then 31 May Gezi [EF]                       10        10       -       13            -    10       -      13
2013-12-17 Corruption probe then Jan 2014 emergency hike [EF]         -         -       2       11            -    11       -       -
2016-07-15 Coup attempt [EF]                                          6         -       -        -            -     3       -       -
2018-08-10 US sanctions / lira crisis [F]                             2         1       1        2            1     0      13       3

## Sensitivity of the chosen (tau, m) to the cost ratio
GJR-J usd: ratio 2 -> tau 0.925, m 3, hits 3, FA eps 1, ratio 8 -> tau 0.85, m 3, hits 4, FA eps 5
GJR-J max: ratio 2 -> tau 0.975, m 3, hits 2, FA eps 1, ratio 8 -> tau 0.925, m 3, hits 3, FA eps 6
MSGJ2 P: ratio 2 -> tau 0.95, m 3, hits 1, FA eps 1, ratio 8 -> tau 0.8, m 3, hits 3, FA eps 10
MS-JD3 P: ratio 2 -> tau 0.95, m 3, hits 3, FA eps 1, ratio 8 -> tau 0.8, m 3, hits 4, FA eps 8
TVTP-MSGJ2 P: ratio 2 -> tau 0.95, m 3, hits 1, FA eps 1, ratio 8 -> tau 0.95, m 3, hits 1, FA eps 1
RV20: ratio 2 -> tau 0.925, m 2, hits 3, FA eps 0, ratio 8 -> tau 0.85, m 3, hits 4, FA eps 3
CDS lvl: ratio 2 -> tau 0.99, m 3, hits 1, FA eps 0, ratio 8 -> tau 0.99, m 3, hits 1, FA eps 0
CDS d20: ratio 2 -> tau 0.99, m 3, hits 1, FA eps 0, ratio 8 -> tau 0.975, m 3, hits 2, FA eps 3

## OR flags (§7.8): GJR-J usd OR x, each part at its own (tau, m) for that ratio
ratio 4: GJR-J usd | RV20: hits 4/4, FA eps 4, cost 4 (alone: GJR-J usd 5); lags [10, 11, 3, 0]
ratio 4: GJR-J usd | CDS lvl: hits 3/4, FA eps 1, cost 5 (alone: GJR-J usd 5); lags [10, None, 6, 2]
ratio 4: GJR-J usd | CDS d20: hits 3/4, FA eps 3, cost 7 (alone: GJR-J usd 5); lags [10, None, 6, 2]
ratio 2: GJR-J usd | RV20: hits 3/4, FA eps 1, cost 3 (alone: GJR-J usd 3); lags [10, None, 6, 0]
ratio 2: GJR-J usd | CDS lvl: hits 3/4, FA eps 1, cost 3 (alone: GJR-J usd 3); lags [10, None, 6, 2]
ratio 2: GJR-J usd | CDS d20: hits 3/4, FA eps 1, cost 3 (alone: GJR-J usd 3); lags [10, None, 6, 2]
ratio 8: GJR-J usd | RV20: hits 4/4, FA eps 6, cost 6 (alone: GJR-J usd 5); lags [10, 8, 3, 0]
ratio 8: GJR-J usd | CDS lvl: hits 4/4, FA eps 5, cost 5 (alone: GJR-J usd 5); lags [10, 8, 3, 0]
ratio 8: GJR-J usd | CDS d20: hits 4/4, FA eps 5, cost 5 (alone: GJR-J usd 5); lags [10, 8, 3, 0]
§7.8 rule, GJR-J usd | RV20: does not replace GJR-J usd
§7.8 rule, GJR-J usd | CDS lvl: does not replace GJR-J usd
§7.8 rule, GJR-J usd | CDS d20: does not replace GJR-J usd
§7.8 rule, RV20 alone: does not replace GJR-J usd

Attribution (GJR-J max, listed type -> detected source): EF->E, EF->-, EF->-, F->F