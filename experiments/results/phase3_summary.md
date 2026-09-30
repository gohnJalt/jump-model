
## rec2  (true K=2, R=40)
sig  truth [[0.169, 0.083], [0.298, 0.187]]
     rel.bias [[-0.0, -0.0], [0.01, 0.01]]
     rel.RMSE [[0.02, 0.02], [0.05, 0.03]]
dur  truth [61.092, 17.328]
     rel.bias [0.02, -0.02]
     rel.RMSE [0.19, 0.16]
lam  truth [[1.0, 1.0, 20.0], [20.0, 12.162, 20.0]]
     rel.bias [[0.15, 0.26, 0.01], [0.02, 0.06, -0.22]]
     rel.RMSE [[1.25, 0.61, 0.15], [0.33, 0.32, 0.51]]
rho  truth [-0.48, -0.58] mean est [-0.48, -0.58]
rhoC truth -0.73 mean est -0.75
filtered accuracy 0.912  | jump precision {'E': np.float64(0.71), 'F': np.float64(0.88), 'C': np.float64(0.7)}  recall {'E': np.float64(0.16), 'F': np.float64(0.48), 'C': np.float64(0.26)}

## rec3  (true K=3, R=40)
sig  truth [[0.151, 0.07], [0.227, 0.12], [0.336, 0.299]]
     rel.bias [[-0.0, 0.01], [0.01, 0.01], [0.01, 0.0]]
     rel.RMSE [[0.02, 0.02], [0.02, 0.03], [0.06, 0.06]]
dur  truth [35.5, 35.844, 18.364]
     rel.bias [-0.06, -0.02, 0.05]
     rel.RMSE [0.16, 0.17, 0.38]
lam  truth [[1.0, 1.0, 9.748], [3.888, 1.0, 20.0], [20.0, 18.782, 20.0]]
     rel.bias [[0.37, 0.12, -0.1], [0.29, 0.22, -0.13], [-0.25, -0.04, -0.48]]
     rel.RMSE [[1.42, 0.52, 0.36], [1.12, 0.65, 0.36], [0.56, 0.37, 0.7]]
rho  truth [-0.44, -0.56, -0.55] mean est [-0.45, -0.56, -0.55]
rhoC truth -0.73 mean est -0.78
filtered accuracy 0.834  | jump precision {'E': np.float64(0.7), 'F': np.float64(0.88), 'C': np.float64(0.71)}  recall {'E': np.float64(0.27), 'F': np.float64(0.51), 'C': np.float64(0.15)}

## disc  (true K=2, R=24)
sig  truth [[0.169, 0.083], [0.298, 0.187]]
     rel.bias [[-0.0, -0.0], [-0.01, -0.01]]
     rel.RMSE [[0.01, 0.01], [0.04, 0.04]]
dur  truth [61.092, 17.328]
     rel.bias [0.01, -0.03]
     rel.RMSE [0.2, 0.16]
lam  truth [[1.0, 1.0, 20.0], [20.0, 12.162, 20.0]]
     rel.bias [[0.13, -0.06, -0.02], [-0.08, -0.05, 0.01]]
     rel.RMSE [[1.22, 0.42, 0.18], [0.33, 0.27, 0.35]]
rho  truth [-0.48, -0.58] mean est [-0.48, -0.56]
rhoC truth -0.73 mean est -0.74
filtered accuracy 0.910  | jump precision {'E': np.float64(0.72), 'F': np.float64(0.9), 'C': np.float64(0.71)}  recall {'E': np.float64(0.17), 'F': np.float64(0.46), 'C': np.float64(0.27)}

## async  (true K=2, R=24)
sig  truth [[0.169, 0.083], [0.298, 0.187]]
     rel.bias [[0.0, 0.01], [0.0, -0.0]]
     rel.RMSE [[0.02, 0.02], [0.04, 0.04]]
dur  truth [61.092, 17.328]
     rel.bias [-0.17, -0.08]
     rel.RMSE [0.2, 0.16]
lam  truth [[1.0, 1.0, 20.0], [20.0, 12.162, 20.0]]
     rel.bias [[1.55, 0.3, -0.14], [0.09, 0.05, -0.21]]
     rel.RMSE [[2.18, 0.87, 0.23], [0.25, 0.33, 0.49]]
rho  truth [-0.48, -0.58] mean est [-0.35, -0.42]
rhoC truth -0.73 mean est -0.67
filtered accuracy 0.898  | jump precision {'E': np.float64(0.5), 'F': np.float64(0.63), 'C': np.float64(0.7)}  recall {'E': np.float64(0.16), 'F': np.float64(0.34), 'C': np.float64(0.18)}
rec2   K picked by bic: {2: 40}
rec2   K picked by oos: {2: 31, 3: 9}
rec3   K picked by bic: {2: 1, 3: 39}
rec3   K picked by oos: {3: 32, 4: 8}
merton K picked by bic: {1: 24}
merton K picked by oos: {1: 22, 2: 2}
garch  K picked by bic: {3: 9, 4: 15}
garch  K picked by oos: {3: 4, 4: 20}
garch fit K=1: mean λ/yr E,F,C by regime [[23.7, 25.0, 20.0]]  switches/yr 0.0
garch fit K=2: mean λ/yr E,F,C by regime [[6.9, 15.4, 6.7], [22.3, 22.2, 20.1]]  switches/yr 13.7
garch fit K=3: mean λ/yr E,F,C by regime [[4.1, 10.7, 4.3], [7.6, 17.2, 11.4], [22.1, 16.3, 20.7]]  switches/yr 20.8

## GATE (DESIGN §6.1a)
PASS  rec2 sig: max|bias| 0.01
PASS  rec2 dur: max|bias| 0.02
FAIL  rec2 lam: max|bias| 0.26, max RMSE 1.25
PASS  rec2 filtered accuracy: 0.912
PASS  rec2 jump precision: {'E': np.float64(0.71), 'F': np.float64(0.88), 'C': np.float64(0.7)}
PASS  rec3 sig: max|bias| 0.01
PASS  rec3 dur: max|bias| 0.06
FAIL  rec3 lam: max|bias| 0.48, max RMSE 1.42
FAIL  rec3 filtered accuracy: 0.834
FAIL  rec3 jump precision: {'E': np.float64(0.7), 'F': np.float64(0.88), 'C': np.float64(0.71)}
PASS  rec2 K-selection (bic): 100% correct
FAIL  rec2 K-selection (oos): 78% correct
PASS  rec3 K-selection (bic): 98% correct
PASS  rec3 K-selection (oos): 80% correct
PASS  merton K-selection (bic): 100% correct
PASS  merton K-selection (oos): 92% correct
FAIL  garch K-selection (bic): picked {3: 9, 4: 15}
FAIL  garch K-selection (oos): picked {3: 4, 4: 20}