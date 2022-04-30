import numpy as np
"""lattice"""	
SITES    = 64            # 4, 8, 16, 20 ... number of vertices in a tile determines the tile shape 	
JEXCH1   = .2            # nn interaction	
JEXCH2   = 1            # nnn interaction	
"""machine learning"""	
TOTAL_SZ = 0            # 0, None ... restriction of Hilbert space	
SAMPLER = 'exchange'       # 'local' = MetropolisLocal, 'exchange' = MetropolisExchange, 'exact' = ExactSampler
MACHINE = 'RBM'         # 'RBM', 'sRBM'
DTYPE = np.complex128   # type of weights in neural network
ALPHA = 8              # N_hidden / N_visible	
ETA   = .01            # learning rate (0.01 usually works)	
SIGMA = .01             # variance of parameters during initialization
SAMPLES = 3000 # original 3000	
NUM_ITER = 1000
N_PRE_ITER = 100        # number of iteration before checking for convergence to speed up the process if the model is already pre-trained
VERBOSE = False         # should we print more detailed results
# pro tento array jsem to jeste nespoustel --> prave bezi
STEPS = np.array([1.05,0.95,0.85]) #np.array([1.2,1.0,0.9,0.95,1.15]) #np.flip(np.array([0,.1,.2,.3,.4,.5,.6,.9,1.0,1.1,1.2])) #np.array([0.725,0.718,0.825,0.84,0.85,0.87]) #np.flip(np.array([0.79,0.81,0.82,0.705,0.83])) #np.arange(0.715,0.731,step=0.01)
RUNS = [1,0]
NAME = "okraje"
num_layers = 2
feature_dims = (8,4)
STEPS_A = [int(SITES/2), SITES, SITES*2, SITES*4]   # grid search across ALPHAs (used only in main-gridSearch.py)
STEPS_E = [1,0.1,0.01,0.001,0.0001]                 # grid search across ETAs   (used only in main-gridSearch.py)
