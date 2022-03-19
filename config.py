import numpy as np
"""lattice"""	
SITES    = 100            # 4, 8, 16, 20 ... number of vertices in a tile determines the tile shape 	
JEXCH1   = 0.2            # nn interaction	
JEXCH2   = 1            # nnn interaction	
H_Z      = 0
"""machine learning"""	
TOTAL_SZ = None            # 0, None ... restriction of Hilbert space	
SAMPLER = 'local'       # 'local' = MetropolisLocal, 'exchange' = MetropolisExchange, 'exact' = ExactSampler
MACHINE = 'RBM'         # 'RBM', 'RBMSymm'
DTYPE = np.complex128   # type of weights in neural network
ALPHA = 2              # N_hidden / N_visible	
ETA   = .01            # learning rate (0.01 usually works)	
SAMPLES = 320	
NUM_ITER = 20
N_PRE_ITER = 50        # number of iteration before checking for convergence to speed up the process if the model is already pre-trained
VERBOSE = True         # should we print more detailed results
STEPS = np.arange(0,1.95,step=0.1)	
STEPS_A = [int(SITES/2), SITES, SITES*2, SITES*4]   # grid search across ALPHAs (used only in main-gridSearch.py)
STEPS_E = [1,0.1,0.01,0.001,0.0001]                 # grid search across ETAs   (used only in main-gridSearch.py)
num_layers = 2
feature_dims = (8,4)
characters = None
characters_2 = None