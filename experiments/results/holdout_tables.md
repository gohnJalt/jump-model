# Holdout out-of-sample (2020-01-01 ..), refit every 21 sessions


## Target E
       logscore  DM_ls_vs_ref  PIT_Berk_p  PIT_LB_p  PIT2_LB_p  hit0.01  Kupiec0.01  CC0.01  Z2_p0.01  FZ0_0.01  DM_FZ0_0.01  hit0.025  Kupiec0.025  CC0.025  Z2_p0.025  FZ0_0.025  DM_FZ0_0.025
model                                                                                                                                                                                           
MSGJ1    2.7204       -1.1295      0.0003    0.0010     0.5444   0.0155      0.0368  0.0751     0.003   -2.5803       0.1071    0.0298       0.2246   0.2526      0.008    -2.9287        0.3185
GJR-t    2.7240           NaN      0.0022    0.0006     0.5859   0.0179      0.0036  0.0083     0.000   -2.5738          NaN    0.0357       0.0082   0.0256      0.000    -2.9206           NaN

## Target F
       logscore  DM_ls_vs_ref  PIT_Berk_p  PIT_LB_p  PIT2_LB_p  hit0.01  Kupiec0.01  CC0.01  Z2_p0.01  FZ0_0.01  DM_FZ0_0.01  hit0.025  Kupiec0.025  CC0.025  Z2_p0.025  FZ0_0.025  DM_FZ0_0.025
model                                                                                                                                                                                           
MSGJ1    4.2983        -2.467         0.0       0.0        0.0   0.0071      0.2148   0.425     0.430   -3.5467      -1.2534    0.0143       0.0022   0.0066      0.867    -4.1943       -2.3298
GJR-t    4.3436           NaN         0.0       0.0        0.0   0.0071      0.2148   0.425     0.099   -3.9592          NaN    0.0137       0.0012   0.0038      0.764    -4.5322           NaN

## Target USD
       logscore  DM_ls_vs_ref  PIT_Berk_p  PIT_LB_p  PIT2_LB_p  hit0.01  Kupiec0.01  CC0.01  Z2_p0.01  FZ0_0.01  DM_FZ0_0.01  hit0.025  Kupiec0.025  CC0.025  Z2_p0.025  FZ0_0.025  DM_FZ0_0.025
model                                                                                                                                                                                           
MSGJ1    2.5905       -3.0255      0.1352    0.0273     0.1459   0.0131      0.2235  0.3560     0.057   -2.4140       1.0574    0.0208       0.2603   0.0633      0.564    -2.7678        0.8488
GJR-t    2.6063           NaN      0.2528    0.0143     0.1702   0.0155      0.0368  0.0751     0.002   -2.3323          NaN    0.0268       0.6430   0.3254      0.057    -2.7365           NaN
DCC-t    2.5931       -2.8884      0.3576    0.0331     0.0729   0.0167      0.0122  0.0100     0.000   -2.3558       0.3097    0.0310       0.1317   0.0818      0.002    -2.7450        0.2237

## Bivariate log score (E, F jointly)
       biv_logscore  DM_vs_DCC
model                         
MSGJ1        6.9918    -2.8253
DCC-t        7.0830        NaN

DM > 0: model beats the reference (GJR-t per target, DCC-t bivariate); |DM| > 1.96 ≈ 5% two-sided.