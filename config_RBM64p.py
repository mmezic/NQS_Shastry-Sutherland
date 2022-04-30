import numpy as np
"""lattice"""	
SITES    = 64            # 4, 8, 16, 20 ... number of vertices in a tile determines the tile shape 	
JEXCH1   = .2            # nn interaction	
JEXCH2   = 1            # nnn interaction	
"""machine learning"""	
TOTAL_SZ = None            # 0, None ... restriction of Hilbert space	
SAMPLER = 'local'       # 'local' = MetropolisLocal, 'exchange' = MetropolisExchange, 'exact' = ExactSampler
MACHINE = 'RBM'         # 'RBM', 'RBMSymm'
DTYPE = np.complex128   # type of weights in neural network
ALPHA = 8              # N_hidden / N_visible	
ETA   = .01            # learning rate (0.01 usually works)	
SAMPLES = 3000*2 # original 3000	
NUM_ITER = 1000*2
N_PRE_ITER = 100        # number of iteration before checking for convergence to speed up the process if the model is already pre-trained
VERBOSE = False         # should we print more detailed results
# pro tento array jsem to jeste nespoustel --> prave bezi
STEPS = np.array([1.00,1.05]) #np.array([.60,.90,.50,1.00]) #np.array([.67,.65,.60,.81,.85]) #np.array([0.730,0.720,0.710,0.700,0.690])#0.770,0.780,0.760,0.750,0.790,0.740])
RUNS = [1,0] # normal & MSR
NAME = "pp"
num_layers = 2
feature_dims = (8,4)
STEPS_A = [int(SITES/2), SITES, SITES*2, SITES*4]   # grid search across ALPHAs (used only in main-gridSearch.py)
STEPS_E = [1,0.1,0.01,0.001,0.0001]                 # grid search across ETAs   (used only in main-gridSearch.py)
