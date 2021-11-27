import numpy as np
"""lattice"""	
SITES    = 8            # 4, 8, 16, 20 ... number of vertices in a tile determines the tile shape 	
JEXCH1   = .8            # nn interaction	
JEXCH2   = 1            # nnn interaction	
"""machine learning"""	
TOTAL_SZ = None            # 0, None ... restriction of Hilbert space	
SAMPLER = 'exact'       # 'local' = MetropolisLocal, 'exchange' = MetropolisExchange, 'exact' = ExactSampler
MACHINE = 'RBM'         # 'RBM', 'RBMSymm'
DTYPE = np.complex128   # type of weights in neural network
ALPHA = 16              # N_hidden / N_visible	
ETA   = .005            # learning rate (0.01 usually works)	
SAMPLES = 3000*2	
NUM_ITER = 2500*2	
VERBOSE = False         # should we print more detailed results
STEPS = np.arange(0,1.21,step=0.2)