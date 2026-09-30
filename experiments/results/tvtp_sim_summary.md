# TVTP simulation study (DESIGN §6.1b)

## null (delta* = 0.0, R = 20)
LR rejection rate at 5%: 0.10   mean LR 3.36
delta_hat calm->stress mean +0.10 (sd 0.35), stress->calm mean +0.08 (sd 0.39)
filtered accuracy: static 0.721  TVTP 0.720  (gain -0.001)

## weak (delta* = 0.75, R = 20)
LR rejection rate at 5%: 1.00   mean LR 40.31
delta_hat calm->stress mean +0.92 (sd 0.45), stress->calm mean -0.69 (sd 0.38)
filtered accuracy: static 0.736  TVTP 0.793  (gain +0.057)

## strong (delta* = 1.5, R = 20)
LR rejection rate at 5%: 1.00   mean LR 65.15
delta_hat calm->stress mean +2.12 (sd 1.49), stress->calm mean -1.74 (sd 1.27)
filtered accuracy: static 0.800  TVTP 0.871  (gain +0.071)

## Criteria (DESIGN §6.1b)
PASS  size: rejection <= 10% under null: 10%
PASS  power: rejection >= 80% under strong: 100%
FAIL  delta |rel. bias| < 30% under strong: [0.41, -0.16]
PASS  accuracy gain >= +5 pp under strong: +0.071