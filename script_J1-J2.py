# %% [markdown]
# # Exploring Shastry-Sutterland model with RBM variational wave function and other models

# %%
import sys, getopt	
sys.path.append('/storage/praha1/home/mezic/.local/lib/python3.7/site-packages')	
import netket as nk
import numpy as np
import time
import json
import plotly.graph_objects as go
import jax
import flax
import optax
print("NetKet version: {}".format(nk.__version__))
print("NumPy version: {}".format(np.__version__))

# %% [markdown]
# ### Setup relevant parameters and settings of the simulation 

# %%
"""lattice"""
SITES    = 20            # 4, 8, 16, 20 ... number of vertices in a tile determines the tile shape 
JEXCH1   = 1            # nn interaction
JEXCH2   = 1             # nnn interaction
TOTAL_SZ = None          # 0, None ... restriction of Hilbert space
#USE_MSR = True          # Should we use a Marshall sign rule? In this notebook, we use both.

"""machine learning"""
MACHINE  = "myRBM_trans"        # RBM, RBMSymm, RBMSymm_transl, RBMModPhase, GCNN, Jastrow, myRBM
USE_VISIBLE_BIAS = False        # in case of myRBM or myRBM_trans
DTYPE    = np.complex128 # data-type of weights in neural network
ALPHA    = 1            # N_hidden / N_visible
ETA      = .005          # learning rate (0.01 usually works)
SAMPLER  = 'exact'       # 'local' = MetropolisLocal, 'exchange' = MetropolisExchange
SAMPLES  = 2000          # number of Monte Carlo samples
NUM_ITER = 1000           # number of convergence iterations
N_LAYERS = 2             # number of layers (in case of G-CNN)
FEATURES = (8,4)         # dimensions of layers (in case of G-CNN)

OUT_NAME = "SS_"+str(SITES)+"j1="+str(JEXCH1) # output file name

# %% [markdown]
# ### Lattice definition
# Basic structure of tiled lattices is implemented in `lattice_and_ops.py` file. Class `Lattice` implements relative positional functions, 
# - e.g. *`rt(node)` returns the index of the right neighbour of a site with index `node`*
# 
# The `for` loop constructs full Shastry-Sutherland lattice with PBC using these auxiliary positional functions.

# %%
# Construction of custom graph according to tiled lattice structure defined in the Lattice class.
edge_colors = []
for node in range(SITES):
    edge_colors.append([node,(node+1)%SITES, 1]) # J1 bond
    edge_colors.append([node,(node+2)%SITES, 2]) # J2 bond

g = nk.graph.Graph(edges=edge_colors) #,n_nodes=3)
N = g.n_nodes

hilbert = nk.hilbert.Spin(s=.5, N=g.n_nodes, total_sz=TOTAL_SZ)

# %% [markdown]
# ### Characters of the symmetries
# In case of G-CNN, we need to specify the characters of the symmetry transformations.
# - DS phase anti-symmetric wrt permutations with negative sign and symmetric wer permutaions with postive sign
# - AF phase is always symmetric for all permutations  

# %%

print("There are", len(g.automorphisms()), "full symmetries.")
# deciding point between DS and AF phase is set to 0.5
if JEXCH1 < 0.5 or True:
    # DS phase is partly anti-symmetric
    characters = []
    from lattice_and_ops import permutation_sign
    for perm in g.automorphisms():
        # print(perm, "with sign", permutation_sign(np.asarray(perm)))
        characters.append(permutation_sign(np.asarray(perm)))
    characters_dimer_1 = np.asarray(characters,dtype=complex)
    characters_dimer_2 = characters_dimer_1
else:
    # AF phase if fully symmetric
    characters_dimer_1 = np.ones((len(g.automorphisms()),), dtype=complex)
    characters_dimer_2 = characters_dimer_1

# %% [markdown]
# ### Translations
# 
# If we want to include only translations, we have to exclude some symmetries from `g.automorphisms()`.
# 
# 
# ⚠️ TODO ⚠️ <span style="color:red"> This part is not fully automated yet. Translations are currently picked by hand from the group of all automorphisms. </span>

# %%
if MACHINE == "RBMSymm_transl" and N != 4:
    raise NotImplementedError("Extraction of translations from the group of automorphisms is not implemented yet.")
translations = []
for perm in g.automorphisms():
    aperm = np.asarray(perm)
    if aperm[1] == (aperm[0]+1)%SITES:
        translations.append(nk.utils.group._permutation_group.Permutation(aperm))
translation_group = nk.utils.group._permutation_group.PermutationGroup(translations,degree=SITES)
print("Out of", len(g.automorphisms()), "permutations,",len(translation_group), "translations were picked.")

# %% [markdown]
# ## Hamoltonian definition
# $$ H = J_{1} \sum\limits_{\langle i,j \rangle}^{L} \vec{\sigma}_{i} \cdot \vec{\sigma}_{j} + J_{2} \sum\limits_{\langle\langle i,j \rangle\rangle_{SS}}^{L}  \vec{\sigma}_{i} \cdot \vec{\sigma}_{j}\,. $$
# 
# Axiliary constant operators used to define hamiltonian are loaded from the external file, they are pre-defined in the `HamOps` class.

# %%
from lattice_and_ops import HamOps
ho = HamOps()
ha_1 = nk.operator.GraphOperator(hilbert, graph=g, bond_ops=ho.bond_operator(JEXCH1,JEXCH2, use_MSR=False), bond_ops_colors=ho.bond_color)
ha_2 = nk.operator.GraphOperator(hilbert, graph=g, bond_ops=ho.bond_operator(JEXCH1,JEXCH2, use_MSR=True), bond_ops_colors=ho.bond_color)


# %% [markdown]
# ## Exact diagonalization

# %%
if g.n_nodes < 20 or g.n_nodes < 30:
    start = time.time()
    if g.n_nodes < 15:
        evals, eigvects = nk.exact.full_ed(ha_1, compute_eigenvectors=True)
    else:
        evals, eigvects = nk.exact.lanczos_ed(ha_1, k=3, compute_eigenvectors=True)
    end = time.time()
    diag_time = end - start
    print("Ground state energy:",evals[0], "\nIt took ", round(diag_time,2), "s =", round((diag_time)/60,2),"min")
else:
    print("System is too large for exact diagonalization. Setting exact_ground_energy = 0 (which is wrong)")
    evals = [0,0,0]
    eigvects = None 
exact_ground_energy = evals[0]

# %% [markdown]
# ## Definition of the machine and other auxiliary `netket` objects
# We define two sets of these objects, usually: 
# - variables ending with ...`_1` belongs to the choice of standard basis,
# - variables ending with ...`_2` belongs to the choice of MSR basis.
# 
# But they can be used in a different way when we need to compare two different models.

# %%
optimizer_1 = nk.optimizer.Sgd(learning_rate=ETA)
optimizer_2 = nk.optimizer.Sgd(learning_rate=ETA)

# Selection of machine type
if MACHINE == "RBM":
    machine_1 = nk.models.RBM(dtype=DTYPE, alpha=ALPHA)#, use_visible_bias=False) 
    machine_2 = nk.models.RBM(dtype=DTYPE, alpha=ALPHA)#, use_visible_bias=False)
elif MACHINE == "RBMSymm":
    machine_1 = nk.models.RBMSymm(g.automorphisms(), dtype=DTYPE, alpha=ALPHA)#, use_visible_bias=False) 
    machine_2 = nk.models.RBMSymm(g.automorphisms(), dtype=DTYPE, alpha=ALPHA)#, use_visible_bias=False)
elif MACHINE == "RBMSymm_transl":
    machine_1 = nk.models.RBMSymm(translation_group, dtype=DTYPE, alpha=ALPHA)#, use_visible_bias=False) 
    machine_2 = nk.models.RBMSymm(translation_group, dtype=DTYPE, alpha=ALPHA)#, use_visible_bias=False)
elif MACHINE == "GCNN":
    machine_1 = nk.models.GCNN(symmetries=g.automorphisms(), dtype=DTYPE, layers=N_LAYERS, features=FEATURES, characters=characters_dimer_1)
    machine_2 = nk.models.GCNN(symmetries=g.automorphisms(), dtype=DTYPE, layers=N_LAYERS, features=FEATURES, characters=characters_dimer_2)
elif MACHINE == "Jastrow":
    from lattice_and_ops import Jastrow
    machine_1 = Jastrow()
    machine_2 = Jastrow()
elif MACHINE == "myRBM":
    from GCNN_Nomura import GCNN_my
    machine_1 = GCNN_my(symmetries=g.automorphisms(), dtype=DTYPE, layers=1, features=ALPHA*SITES, characters=characters_dimer_1, output_activation=nk.nn.log_cosh, use_bias=True, use_visible_bias=USE_VISIBLE_BIAS)
    machine_2 = GCNN_my(symmetries=g.automorphisms(), dtype=DTYPE, layers=1, features=ALPHA*SITES, characters=characters_dimer_2, output_activation=nk.nn.log_cosh, use_bias=True, use_visible_bias=USE_VISIBLE_BIAS)
elif MACHINE == "myRBM_trans":
    from GCNN_Nomura import GCNN_my
    machine_1 = GCNN_my(symmetries=translation_group, dtype=DTYPE, layers=1, features=ALPHA*SITES, characters=np.ones_like(translation_group[0],dtype=complex), output_activation=nk.nn.log_cosh, use_bias=True, use_visible_bias=USE_VISIBLE_BIAS)
    machine_2 = GCNN_my(symmetries=translation_group, dtype=DTYPE, layers=1, features=ALPHA*SITES, characters=np.ones_like(translation_group[0],dtype=complex), output_activation=nk.nn.log_cosh, use_bias=True, use_visible_bias=USE_VISIBLE_BIAS)
elif MACHINE == "RBMModPhase":
    machine_1 = nk.models.RBMModPhase(alpha=ALPHA, use_hidden_bias=True, dtype=DTYPE)
    machine_2 = nk.models.RBMModPhase(alpha=ALPHA, use_hidden_bias=True, dtype=DTYPE)

    # A linear schedule varies the learning rate from 0 to 0.01 across 600 steps.
    modulus_schedule_1=optax.linear_schedule(0,0.01,NUM_ITER)
    modulus_schedule_2=optax.linear_schedule(0,0.01,NUM_ITER)
    # The phase starts with a larger learning rate and then is decreased.
    phase_schedule_1=optax.linear_schedule(0.05,0.01,NUM_ITER)
    phase_schedule_2=optax.linear_schedule(0.05,0.01,NUM_ITER)
    # Combine the linear schedule with SGD
    optm_1=optax.sgd(modulus_schedule_1)
    optp_1=optax.sgd(phase_schedule_1)
    optm_2=optax.sgd(modulus_schedule_2)
    optp_2=optax.sgd(phase_schedule_2)
    # The multi-transform optimizer uses different optimisers for different parts of the parameters.
    optimizer_1 = optax.multi_transform({'o1': optm_1, 'o2': optp_1}, flax.core.freeze({"Dense_0":"o1", "Dense_1":"o2"}))
    optimizer_2 = optax.multi_transform({'o1': optm_2, 'o2': optp_2}, flax.core.freeze({"Dense_0":"o1", "Dense_1":"o2"}))
else:
    raise Exception(str("undefined MACHINE: ")+str(MACHINE))

# Selection of sampler type
if SAMPLER == 'local':
    sampler_1 = nk.sampler.MetropolisLocal(hilbert=hilbert)
    sampler_2 = nk.sampler.MetropolisLocal(hilbert=hilbert)
elif SAMPLER == 'exact':
    sampler_1 = nk.sampler.ExactSampler(hilbert=hilbert)
    sampler_2 = nk.sampler.ExactSampler(hilbert=hilbert)
else:
    sampler_1 = nk.sampler.MetropolisExchange(hilbert=hilbert, graph=g)
    sampler_2 = nk.sampler.MetropolisExchange(hilbert=hilbert, graph=g)
    if SAMPLER != 'exchange':
        print("Warning! Undefined fq.SAMPLER:", SAMPLER, ", dafaulting to MetropolisExchange fq.SAMPLER")


# Stochastic Reconfiguration as a preconditioner
sr_1  = nk.optimizer.SR(diag_shift=0.01)
sr_2  = nk.optimizer.SR(diag_shift=0.01)

# The variational state (former name: nk.variational.MCState)
vs_1 = nk.vqs.MCState(sampler_1 , machine_1 , n_samples=SAMPLES)
vs_2  = nk.vqs.MCState(sampler_2 , machine_2 , n_samples=SAMPLES)
vs_1.init_parameters(jax.nn.initializers.normal(stddev=0.001))
vs_2.init_parameters(jax.nn.initializers.normal(stddev=0.001))


gs_1 = nk.VMC(hamiltonian=ha_1 ,optimizer=optimizer_1 ,preconditioner=sr_1 ,variational_state=vs_1)
gs_2 = nk.VMC(hamiltonian=ha_2 ,optimizer=optimizer_2 ,preconditioner=sr_2 ,variational_state=vs_2) 

# %%
vs_1.n_parameters

# %%
if False:
    ETA2 = 0.002
    optimizer_1 = nk.optimizer.Sgd(learning_rate=ETA2)
    optimizer_2 = nk.optimizer.Sgd(learning_rate=ETA2)
    gs_1 = nk.VMC(hamiltonian=ha_1 ,optimizer=optimizer_1,preconditioner=sr_1,variational_state=vs_1)   # 0 ... symmetric
    gs_2 = nk.VMC(hamiltonian=ha_2 ,optimizer=optimizer_2,preconditioner=sr_2,variational_state=vs_2)   # 1 ... symmetric+MSR

# %% [markdown]
# # Calculation
# We let the calculation run for `NUM_ITERS` iterations for both cases _1 and _2 (without MSR and with MSR). If only one case is desired, set the variable `no_of_runs` to 1.

# %%
runs = [0,1]
no_of_runs = np.sum(runs) # 1 - one run for variables with ..._1;  2 - both runs for variables ..._1 and ..._2
run_only_2 = (runs[1]==1 and runs[0]==0) # in case of no_of_runs=1
NUM_ITER = 50
print("J_1 =", JEXCH1, end="; ")
if exact_ground_energy != 0:
    print("Expected exact energy:", exact_ground_energy)
for i,gs in enumerate([gs_1,gs_2][run_only_2:run_only_2+no_of_runs]):
    start = time.time()
    gs.run(out=OUT_NAME+str(i), n_iter=int(NUM_ITER))#, obs={'symmetry':P(0,1)})
    end = time.time()
    print(gs.energy,flush=True,end='')
    print("ε_",i," = ",error:=abs((gs.energy.mean.real.tolist() - exact_ground_energy)/exact_ground_energy)," = 10^", np.log10(error),sep='')
    print("The calculation for {} of type {} took {} min".format(MACHINE, i+1, (end-start)/60))

