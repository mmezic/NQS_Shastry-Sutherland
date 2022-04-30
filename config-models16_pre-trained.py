import numpy as np
"""lattice"""	
SITES    = 16            # 4, 8, 16, 20 ... number of vertices in a tile determines the tile shape 	
JEXCH1   = 0.2            # nn interaction	
JEXCH2   = 1            # nnn interaction	
H_Z      = 0
"""machine learning"""	
TOTAL_SZ = None            # 0, None ... restriction of Hilbert space	
SAMPLER = 'exact'       # 'local' = MetropolisLocal, 'exchange' = MetropolisExchange, 'exact' = ExactSampler
MACHINE = 'RBM'         # 'RBM', 'sRBM'
DTYPE = np.complex128   # type of weights in neural network
ALPHA = 2              # N_hidden / N_visible	
ETA   = .01            # learning rate (0.01 usually works)	
SIGMA = .01             # variance of parameters during initialization
SAMPLES = 3000
NUM_ITER = 3900
N_PRE_ITER = 100        # number of iteration before checking for convergence to speed up the process if the model is already pre-trained
VERBOSE = False         # should we print more detailed results
STEPS = np.concatenate((np.asarray([0.2,0.4,0.8]), np.arange(1.2,2.01,step=0.05)))
INDICES = range(42,45) #range(38)	
STEPS_A = [int(SITES/2), SITES, SITES*2, SITES*4]   # grid search across ALPHAs (used only in main-gridSearch.py)
STEPS_E = [1,0.1,0.01,0.001,0.0001]                 # grid search across ETAs   (used only in main-gridSearch.py)
num_layers = 2
feature_dims = (8,4)
characters = None
characters_2 = None