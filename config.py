import numpy as np
"""lattice"""
SITES    = 8              # 4, 8, 16, 20, 36, 64, 100 ... number of particles
JEXCH1   = .2             # nn interaction (denoted J)
JEXCH2   = 1              # nnn interaction (denoted J')
H_Z      = 0              # external magnetic field (denoted h)

"""neural network"""
MACHINE  = "RBM"          # RBM, RBM-b, sRBM, pRBM, GCNN, pmRBM, Jastrow, Jastrow+b
DTYPE    = np.complex128  # data-type of weights in neural network (pmRBM uses always just floats)
ALPHA    = 2              # size of the RBM, alpha = N_hidden / N_visible

"""machine learning"""
TOTAL_SZ = None           # 0, None ... restriction of Hilbert space's magnetization
ETA      = .01            # learning rate (0.01 usually works)
SIGMA    = .01            # initial variance of parameters (distributed according to a normal distribution)
SAMPLER  = 'exact'        # 'local' = MetropolisLocal, 'exchange' = MetropolisExchange, 'exact' = ExactSampler
SAMPLES  = 1000           # number of Monte Carlo samples
NUM_ITER = 200            # number of convergence iterations
N_PRE_ITER = 50           # Used only in the case of transfer learning as a number of iteration before checking for convergence. It is used to speed up the process if the model is already pre-learned.

"""simulation parameters"""
VERBOSE = True                          # If True, prints the interdemidiate results to the screen. Otherwise only prints the final results after each convergence process.
STEPS = np.arange(0.0,1.21,step=0.2)	# A simulation is executed for each of these points specifying either the ratio J/J' (in the case of main.py) or the magnetic filed h (in the case of main-mag.py) 
RUNS = [1,1]                            # Specifying whether to exectute simulation with normal or MSR basis or both. RUNS = [run_normal, run_MSR]
NAME = "test_run"                       # A string added to the names of output runs.

"""in case of GCNN machine, additional shape specification is needed"""
num_layers = 2            # number of layers
feature_dims = (8,4)      # dimensions of layers