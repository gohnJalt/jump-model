
## rec2  (true K=2, R=30)
sig  truth [[0.125, 0.041], [0.219, 0.146]]
     rel.bias [[0.21, 0.88], [0.16, 0.06]]
     rel.RMSE [[0.24, 0.91], [0.47, 0.26]]
dur  truth [22.611, 13.931]
     rel.bias [0.43, 0.09]
     rel.RMSE [0.78, 0.43]
lam  truth [[1.0, 1.0, 14.988], [20.0, 1.0, 20.0]]
     rel.bias [[0.78, -0.03, -0.31], [-0.07, 0.17, -0.33]]
     rel.RMSE [[2.78, 0.33, 0.39], [0.37, 0.64, 0.54]]
rho  truth [-0.47, -0.56] mean est [-0.48, -0.56]
alpha truth [[0.01, 0.01], [0.01, 0.015]] rel.bias [[-0.13, -0.05], [0.31, -0.12]] rel.RMSE [[0.57, 0.52], [1.43, 0.85]]
gamma truth [[0.01, 0.012], [0.104, 0.142]] rel.bias [[0.45, 0.14], [0.0, 0.15]] rel.RMSE [[0.8, 0.53], [0.29, 0.27]]
beta  truth [[0.975, 0.974], [0.918, 0.889]] rel.bias [[-0.0, -0.0], [-0.0, -0.01]] rel.RMSE [[0.01, 0.0], [0.02, 0.02]]
rhoC truth -0.77 mean est -0.85
filtered accuracy 0.717  | jump precision {'E': np.float64(0.75), 'F': np.float64(0.99), 'C': np.float64(0.78)}  recall {'E': np.float64(0.16), 'F': np.float64(0.98), 'C': np.float64(0.08)}
rec2   K picked by bic: {1: 18, 2: 12}
rec2   K picked by oos: {1: 3, 2: 22, 3: 5}
merton K picked by bic: {1: 16}
merton K picked by oos: {1: 15, 2: 1}
garch  K picked by bic: {1: 24}
garch  K picked by oos: {1: 11, 2: 9, 3: 4}
garch fit K=1: mean λ/yr E,F,C by regime [[7.5, 24.6, 15.6]]  switches/yr 0.0
garch fit K=2: mean λ/yr E,F,C by regime [[4.8, 22.6, 10.7], [14.1, 16.0, 15.7]]  switches/yr 30.8
garch fit K=3: mean λ/yr E,F,C by regime [[4.0, 15.3, 6.8], [9.2, 15.8, 12.2], [12.2, 16.7, 15.7]]  switches/yr 46.5

## GATE (DESIGN §6.1a)
FAIL  rec2 sig: max|bias| 0.88
FAIL  rec2 dur: max|bias| 0.43
FAIL  rec2 lam: max|bias| 0.78, max RMSE 2.78
FAIL  rec2 filtered accuracy: 0.717
PASS  rec2 jump precision: {'E': np.float64(0.75), 'F': np.float64(0.99), 'C': np.float64(0.78)}
FAIL  rec2 K-selection (bic): 40% correct
FAIL  rec2 K-selection (oos): 73% correct
PASS  merton K-selection (bic): 100% correct
PASS  merton K-selection (oos): 94% correct
PASS  garch K-selection (bic): picked {1: 24}
FAIL  garch K-selection (oos): picked {1: 11, 2: 9, 3: 4}