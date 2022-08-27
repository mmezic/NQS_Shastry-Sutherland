import numpy as np
"""lattice"""
SITES    = 8             # 4, 8, 16, 20, 36, 64, 100 ... number of particles
JEXCH1D  = 0.2            # nn interaction (denoted J) in the DS phase
JEXCH1A  = 0.9            # nn interaction (denoted J) in the AF phase
JEXCH2   = 1              # nnn interaction (denoted J')
H_Z      = 0              # external magnetic field (denoted h)

"""neural network architecture"""
MACHINE  = "RBM"          # RBM, RBM-b, sRBM, pRBM, GCNN, pmRBM, Jastrow, Jastrow+b
DTYPE    = np.complex128  # data-type of weights in neural network (pmRBM uses always just floats)
ALPHA    = 2              # size of the RBM, alpha = N_hidden / N_visible

"""machine learning"""
TOTAL_SZ = None           # 0, None ... restriction of Hilbert space's magnetization
ETA      = .01            # learning rate (0.01 usually works)
SIGMA    = .01            # initial variance of parameters (distributed according to a normal distribution)
SAMPLER  = 'exact'        # 'local' = MetropolisLocal, 'exchange' = MetropolisExchange, 'exact' = ExactSampler
SAMPLES  = 200           # number of Monte Carlo samples
NUM_ITER = 100           # number of convergence iterations

"""simulation parameters"""
VERBOSE = False                          # If True, prints the interdemidiate results to the screen. Otherwise only prints the final results after each convergence process.
INDICES = range(0,4*4)                    # A list of indices. Each index corresponds to one row in the non-simplyfied benchmarking table from the appendix B.

"""in case of GCNN machine, additional shape specification is needed"""
num_layers = 2            # number of layers
feature_dims = (8,4)      # dimensions of layers