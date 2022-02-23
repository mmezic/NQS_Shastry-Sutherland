import numpy as np
"""lattice"""	
SITES    = 20            # 4, 8, 16, 20 ... number of vertices in a tile determines the tile shape 	
JEXCH1   = .4            # nn interaction	
JEXCH2   = 1            # nnn interaction	
"""machine learning"""	
TOTAL_SZ = None            # 0, None ... restriction of Hilbert space	
SAMPLER = 'local'       # 'local' = MetropolisLocal, 'exchange' = MetropolisExchange, 'exact' = ExactSampler
MACHINE = 'GCNN'         # 'RBM', 'RBMSymm'
DTYPE = np.complex128   # type of weights in neural network
ALPHA = 16              # N_hidden / N_visible	
ETA   = .005            # learning rate (0.01 usually works)	
SAMPLES = 3000	
NUM_ITER = 600	
N_PRE_ITER = 30        # number of iteration before checking for convergence to speed up the process if the model is already pre-trained
VERBOSE = False         # should we print more detailed results
STEPS = np.arange(.4,1.21,step=1.1)	
num_layers = 3
feature_dims = (8,8,4)
characters = None
characters_2 = None