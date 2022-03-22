from re import M
import sys, getopt	
sys.path.append('/storage/praha1/home/mezic/.local/lib/python3.7/site-packages')	
import netket as nk	
import numpy as np
import jax
import time
import json	
import copy
print("Python version: {}".format(sys.version))
print("NetKet version: {}".format(nk.__version__))	
print("NumPy version: {}".format(np.__version__))

file = sys.argv[-1]
if len(sys.argv) == 1:
    file = "config-models"
print(file)
fq = __import__(file)
from lattice_and_ops import Lattice
from lattice_and_ops import Operators
from lattice_and_ops import HamOps
from lattice_and_ops import permutation_sign
from lattice_and_ops import log_results
from GCNN_Nomura import GCNN_my
ho = HamOps()

OUT_NAME = fq.MACHINE+str(fq.SITES) # output file name
OUT_LOG_NAME = "out.txt"            # filename for output logging
print("N = ",fq.SITES, ", samples = ",fq.SAMPLES,", iters = ",fq.NUM_ITER, ", sampler = ",fq.SAMPLER, ", TOTAL_SZ = ", fq.TOTAL_SZ, ", machine = ", fq.MACHINE, ", dtype = ", fq.DTYPE, ", alpha = ", fq.ALPHA, ", eta = ", fq.ETA, sep="")
with open(OUT_LOG_NAME,"a") as out_log_file:
    out_log_file.write("N = {}, samples = {}, iters = {}, sampler = {}, TOTAL_SZ = {}, machine = {}, dtype = {}, alpha = {}, eta = {}\n".format(fq.SITES,fq.SAMPLES,fq.NUM_ITER,fq.SAMPLER, fq.TOTAL_SZ,fq.MACHINE, fq.DTYPE, fq.ALPHA, fq.ETA))

lattice = Lattice(fq.SITES)

# Define custom graph
edge_colors = []
for node in range(fq.SITES):
    edge_colors.append([node,lattice.rt(node), 1]) #horizontal connections
    edge_colors.append([node,lattice.bot(node), 1]) #vertical connections
    row, column = lattice.position(node)
    if column%2 == 0:
        if row%2 == 0:
            edge_colors.append([node,lattice.lrt(node),2])
        else:
            edge_colors.append([node,lattice.llft(node),2])

# Define the netket graph object
g = nk.graph.Graph(edges=edge_colors)

hilbert = nk.hilbert.Spin(s=.5, N=g.n_nodes, total_sz=fq.TOTAL_SZ)

# This part is only relevant for GCNN or myRBM machine
print("There are", len(g.automorphisms()), "full symmetries.")
# deciding point between DS and AF phase is set to 0.5
#####################""" assume only the fully symmetric models """##########################
characters_dimer_1 = np.ones((len(g.automorphisms()),), dtype=complex)
characters_dimer_2 = characters_dimer_1
    
# Hamiltonian definition
hd_1 = nk.operator.GraphOperator(hilbert, graph=g, bond_ops=ho.bond_operator(fq.JEXCH1D,fq.JEXCH2, use_MSR=False), bond_ops_colors=ho.bond_color)
hd_2 = nk.operator.GraphOperator(hilbert, graph=g, bond_ops=ho.bond_operator(fq.JEXCH1D,fq.JEXCH2, use_MSR=True), bond_ops_colors=ho.bond_color)
ha_1 = nk.operator.GraphOperator(hilbert, graph=g, bond_ops=ho.bond_operator(fq.JEXCH1A,fq.JEXCH2, use_MSR=False), bond_ops_colors=ho.bond_color)
ha_2 = nk.operator.GraphOperator(hilbert, graph=g, bond_ops=ho.bond_operator(fq.JEXCH1A,fq.JEXCH2, use_MSR=True), bond_ops_colors=ho.bond_color)

# Exact diagonalization
if g.n_nodes < 20:
    evals, eigvects = nk.exact.lanczos_ed(ha_1, k=3, compute_eigenvectors=True)
    exact_ground_energy = evals[0]
else:
    exact_ground_energy = 0
    eigvects = None


# extract translations from the full symmetry group
if not (fq.SITES in [4,16]):
    raise NotImplementedError("Extraction of translations from the group of automorphisms is not implemented yet.")
translations = []
for perm in g.automorphisms():
    aperm = np.asarray(perm)
    if fq.SITES == 4:
        if (aperm[0],aperm[1]) in ((0,1),(1,0),(2,3),(3,2)):
            translations.append(nk.utils.group._permutation_group.Permutation(aperm))
    elif fq.SITES == 16:
        if (aperm[0],aperm[1],aperm[3]) in ((0,1,3),(2,3,1),(8,9,11),(10,11,9)):
            translations.append(nk.utils.group._permutation_group.Permutation(aperm))
translation_group = nk.utils.group._permutation_group.PermutationGroup(translations,degree=fq.SITES)

# some parameters of the benchmarking of the models
ETAS = [0.2,0.05,0.01]
no_etas = len(ETAS)
no_repeats = 4
name = "none"
conv = np.zeros((no_repeats,4))
for m in range(0,30):
    start = time.time()
    for j in range(no_repeats):
        # model definition
        if m//no_etas == 0:
            name = "RBM_2"
            machine_1 = nk.models.RBM(dtype=fq.DTYPE, alpha=2)
            machine_2 = nk.models.RBM(dtype=fq.DTYPE, alpha=2)
            machine_3 = nk.models.RBM(dtype=fq.DTYPE, alpha=2)
            machine_4 = nk.models.RBM(dtype=fq.DTYPE, alpha=2)
        elif m//no_etas == 1:
            name = "RBM_16"
            machine_1 = nk.models.RBM(dtype=fq.DTYPE, alpha=16)
            machine_2 = nk.models.RBM(dtype=fq.DTYPE, alpha=16)
            machine_3 = nk.models.RBM(dtype=fq.DTYPE, alpha=16)
            machine_4 = nk.models.RBM(dtype=fq.DTYPE, alpha=16)
        elif m//no_etas == 2:
            name = "RBM_2notVisible"
            machine_1 = nk.models.RBM(dtype=fq.DTYPE, alpha=2,use_visible_bias=False)
            machine_2 = nk.models.RBM(dtype=fq.DTYPE, alpha=2,use_visible_bias=False)
            machine_3 = nk.models.RBM(dtype=fq.DTYPE, alpha=2,use_visible_bias=False)
            machine_4 = nk.models.RBM(dtype=fq.DTYPE, alpha=2,use_visible_bias=False)
        elif m//no_etas == 3:
            name = "RBM_16notVisible"
            machine_1 = nk.models.RBM(dtype=fq.DTYPE, alpha=16,use_visible_bias=False)
            machine_2 = nk.models.RBM(dtype=fq.DTYPE, alpha=16,use_visible_bias=False)
            machine_3 = nk.models.RBM(dtype=fq.DTYPE, alpha=16,use_visible_bias=False)
            machine_4 = nk.models.RBM(dtype=fq.DTYPE, alpha=16,use_visible_bias=False)
        elif m//no_etas == 4:
            name = "RBMSymm_16aut"
            alpha= 16
            machine_1 = nk.models.RBMSymm(g.automorphisms(), dtype=fq.DTYPE, alpha=alpha) 
            machine_2 = nk.models.RBMSymm(g.automorphisms(), dtype=fq.DTYPE, alpha=alpha)
            machine_3 = nk.models.RBMSymm(g.automorphisms(), dtype=fq.DTYPE, alpha=alpha) 
            machine_4 = nk.models.RBMSymm(g.automorphisms(), dtype=fq.DTYPE, alpha=alpha)
        elif m//no_etas == 5:
            name = "RBMSymm_128aut"
            alpha = 128
            machine_1 = nk.models.RBMSymm(g.automorphisms(), dtype=fq.DTYPE, alpha=alpha) 
            machine_2 = nk.models.RBMSymm(g.automorphisms(), dtype=fq.DTYPE, alpha=alpha)
            machine_3 = nk.models.RBMSymm(g.automorphisms(), dtype=fq.DTYPE, alpha=alpha) 
            machine_4 = nk.models.RBMSymm(g.automorphisms(), dtype=fq.DTYPE, alpha=alpha)
        elif m//no_etas == 6:
            name = "RBMSymm_16trans"
            alpha = 16
            machine_1 = nk.models.RBMSymm(translation_group, dtype=fq.DTYPE, alpha=alpha) 
            machine_2 = nk.models.RBMSymm(translation_group, dtype=fq.DTYPE, alpha=alpha)
            machine_3 = nk.models.RBMSymm(translation_group, dtype=fq.DTYPE, alpha=alpha) 
            machine_4 = nk.models.RBMSymm(translation_group, dtype=fq.DTYPE, alpha=alpha)
        elif m//no_etas == 7:
            name = "RBMSymm_128trans"
            alpha = 128
            machine_1 = nk.models.RBMSymm(translation_group, dtype=fq.DTYPE, alpha=alpha) 
            machine_2 = nk.models.RBMSymm(translation_group, dtype=fq.DTYPE, alpha=alpha)
            machine_3 = nk.models.RBMSymm(translation_group, dtype=fq.DTYPE, alpha=alpha) 
            machine_4 = nk.models.RBMSymm(translation_group, dtype=fq.DTYPE, alpha=alpha)
        elif m//no_etas == 8:
            name = "GCNN_my_32"
            alpha=32
            machine_1 = GCNN_my(symmetries=g.automorphisms(), dtype=fq.DTYPE, layers=1, features=alpha, characters=characters_dimer_1, output_activation=nk.nn.log_cosh, use_bias=True, use_visible_bias=True)
            machine_2 = GCNN_my(symmetries=g.automorphisms(), dtype=fq.DTYPE, layers=1, features=alpha, characters=characters_dimer_2, output_activation=nk.nn.log_cosh, use_bias=True, use_visible_bias=True)
            machine_3 = GCNN_my(symmetries=g.automorphisms(), dtype=fq.DTYPE, layers=1, features=alpha, characters=characters_dimer_1, output_activation=nk.nn.log_cosh, use_bias=True, use_visible_bias=True)
            machine_4 = GCNN_my(symmetries=g.automorphisms(), dtype=fq.DTYPE, layers=1, features=alpha, characters=characters_dimer_2, output_activation=nk.nn.log_cosh, use_bias=True, use_visible_bias=True)
        elif m//no_etas == 9:
            name = "GCNN_my_8"
            alpha=8
            machine_1 = GCNN_my(symmetries=g.automorphisms(), dtype=fq.DTYPE, layers=1, features=alpha, characters=characters_dimer_1, output_activation=nk.nn.log_cosh, use_bias=True, use_visible_bias=True)
            machine_2 = GCNN_my(symmetries=g.automorphisms(), dtype=fq.DTYPE, layers=1, features=alpha, characters=characters_dimer_2, output_activation=nk.nn.log_cosh, use_bias=True, use_visible_bias=True)
            machine_3 = GCNN_my(symmetries=g.automorphisms(), dtype=fq.DTYPE, layers=1, features=alpha, characters=characters_dimer_1, output_activation=nk.nn.log_cosh, use_bias=True, use_visible_bias=True)
            machine_4 = GCNN_my(symmetries=g.automorphisms(), dtype=fq.DTYPE, layers=1, features=alpha, characters=characters_dimer_2, output_activation=nk.nn.log_cosh, use_bias=True, use_visible_bias=True)
        elif m//no_etas == 10:
            name = "GCNN_my_32notVisible"
            alpha=32
            machine_1 = GCNN_my(symmetries=g.automorphisms(), dtype=fq.DTYPE, layers=1, features=alpha, characters=characters_dimer_1, output_activation=nk.nn.log_cosh, use_bias=True, use_visible_bias=True)
            machine_2 = GCNN_my(symmetries=g.automorphisms(), dtype=fq.DTYPE, layers=1, features=alpha, characters=characters_dimer_2, output_activation=nk.nn.log_cosh, use_bias=True, use_visible_bias=True)
            machine_3 = GCNN_my(symmetries=g.automorphisms(), dtype=fq.DTYPE, layers=1, features=alpha, characters=characters_dimer_1, output_activation=nk.nn.log_cosh, use_bias=True, use_visible_bias=True)
            machine_4 = GCNN_my(symmetries=g.automorphisms(), dtype=fq.DTYPE, layers=1, features=alpha, characters=characters_dimer_2, output_activation=nk.nn.log_cosh, use_bias=True, use_visible_bias=True)
        elif m//no_etas == 11:
            name = "GCNN_aut[8,4]"
            num_layers = 2
            feature_dims = [8,4]
            machine_1 = nk.models.GCNN(symmetries=g.automorphisms(), dtype=fq.DTYPE, layers=num_layers, features=feature_dims, characters=characters_dimer_1)
            machine_2 = nk.models.GCNN(symmetries=g.automorphisms(), dtype=fq.DTYPE, layers=num_layers, features=feature_dims, characters=characters_dimer_2)
            machine_3 = nk.models.GCNN(symmetries=g.automorphisms(), dtype=fq.DTYPE, layers=num_layers, features=feature_dims, characters=characters_dimer_1)
            machine_4 = nk.models.GCNN(symmetries=g.automorphisms(), dtype=fq.DTYPE, layers=num_layers, features=feature_dims, characters=characters_dimer_2)
        elif m//no_etas == 12:
            name = "GCNN_aut[16,16]"
            num_layers = 2
            feature_dims = [16,16]
            machine_1 = nk.models.GCNN(symmetries=g.automorphisms(), dtype=fq.DTYPE, layers=num_layers, features=feature_dims, characters=characters_dimer_1)
            machine_2 = nk.models.GCNN(symmetries=g.automorphisms(), dtype=fq.DTYPE, layers=num_layers, features=feature_dims, characters=characters_dimer_2)
            machine_3 = nk.models.GCNN(symmetries=g.automorphisms(), dtype=fq.DTYPE, layers=num_layers, features=feature_dims, characters=characters_dimer_1)
            machine_4 = nk.models.GCNN(symmetries=g.automorphisms(), dtype=fq.DTYPE, layers=num_layers, features=feature_dims, characters=characters_dimer_2)
        else:
            raise Exception(str("undefined MACHINE: ")+str(fq.MACHINE))

        # Selection of sampler type
        if fq.SAMPLER == 'local':
            sampler_1 = nk.sampler.MetropolisLocal(hilbert=hilbert)
            sampler_2 = nk.sampler.MetropolisLocal(hilbert=hilbert)
            sampler_3 = nk.sampler.MetropolisLocal(hilbert=hilbert)
            sampler_4 = nk.sampler.MetropolisLocal(hilbert=hilbert)
        elif fq.SAMPLER == 'exact':
            sampler_1 = nk.sampler.ExactSampler(hilbert=hilbert)
            sampler_2 = nk.sampler.ExactSampler(hilbert=hilbert)
            sampler_3 = nk.sampler.ExactSampler(hilbert=hilbert)
            sampler_4 = nk.sampler.ExactSampler(hilbert=hilbert)
        else:
            sampler_1 = nk.sampler.MetropolisExchange(hilbert=hilbert, graph=g)
            sampler_2 = nk.sampler.MetropolisExchange(hilbert=hilbert, graph=g)
            sampler_3 = nk.sampler.MetropolisExchange(hilbert=hilbert, graph=g)
            sampler_4 = nk.sampler.MetropolisExchange(hilbert=hilbert, graph=g)
            if fq.SAMPLER != 'exchange':
                print("Warning! Undefined fq.SAMPLER:", fq.SAMPLER, ", dafaulting to MetropolisExchange fq.SAMPLER")

        # Optimzer
        optimizer_1 = nk.optimizer.Sgd(learning_rate=ETAS[m%no_etas])
        optimizer_2 = nk.optimizer.Sgd(learning_rate=ETAS[m%no_etas])
        optimizer_3 = nk.optimizer.Sgd(learning_rate=ETAS[m%no_etas])
        optimizer_4 = nk.optimizer.Sgd(learning_rate=ETAS[m%no_etas])

        # Stochastic Reconfiguration
        sr_1 = nk.optimizer.SR(diag_shift=0.01)
        sr_2 = nk.optimizer.SR(diag_shift=0.01)
        sr_3 = nk.optimizer.SR(diag_shift=0.01)
        sr_4 = nk.optimizer.SR(diag_shift=0.01)

        # The variational state (drive to byla nk.variational.MCState)
        vs_1 = nk.vqs.MCState(sampler_1, machine_1, n_samples=fq.SAMPLES)
        vs_2 = nk.vqs.MCState(sampler_2, machine_2, n_samples=fq.SAMPLES)
        vs_1.init_parameters(jax.nn.initializers.normal(stddev=0.01))
        vs_2.init_parameters(jax.nn.initializers.normal(stddev=0.01))
        vs_3 = nk.vqs.MCState(sampler_3, machine_3, n_samples=fq.SAMPLES)
        vs_4 = nk.vqs.MCState(sampler_4, machine_4, n_samples=fq.SAMPLES)
        vs_3.init_parameters(jax.nn.initializers.normal(stddev=0.01))
        vs_4.init_parameters(jax.nn.initializers.normal(stddev=0.01))

        gs_1d = nk.VMC(hamiltonian=hd_1 ,optimizer=optimizer_1,preconditioner=sr_1,variational_state=vs_1)   # DS
        gs_2d = nk.VMC(hamiltonian=hd_2 ,optimizer=optimizer_2,preconditioner=sr_2,variational_state=vs_2)   # DS+MSR
        gs_1a = nk.VMC(hamiltonian=ha_1 ,optimizer=optimizer_3,preconditioner=sr_3,variational_state=vs_3)   # AF
        gs_2a = nk.VMC(hamiltonian=ha_2 ,optimizer=optimizer_4,preconditioner=sr_4,variational_state=vs_4)   # AF+MSR
        
        ops = Operators(lattice,hilbert,ho.mszsz,ho.exchange)

        no_of_runs = 4
        for i,gs in enumerate([gs_1d,gs_2d,gs_1a,gs_2a]):
            start = time.time()
            gs.run(out=OUT_NAME+"_"+str(m)+"_"+str(i), n_iter=int(fq.NUM_ITER),show_progress=fq.VERBOSE)
            end = time.time()
            if m == 0:
                print("The type {} of {} calculation took {} min".format(i,fq.MACHINE ,(end-start)/60))

        # finding the number of steps needed to converge
        threshold_energy = 0.995*exact_ground_energy
        data = []
        for i in range(no_of_runs):
            data.append(json.load(open(OUT_NAME+"_"+str(m)+"_"+str(i)+".log")))
        if type(data[0]["Energy"]["Mean"]) == dict: #DTYPE in (np.complex128, np.complex64):#, np.float64):# and False:
            energy_convergence = [data[i]["Energy"]["Mean"]["real"] for i in range(no_of_runs)]
        else:
            energy_convergence = [data[i]["Energy"]["Mean"] for i in range(no_of_runs)]
        steps_until_convergence = [next((i for i,v in enumerate(energy_convergence[j]) if v < threshold_energy), np.inf) for j in range(no_of_runs)]
        conv[j] = np.asarray(steps_until_convergence)
        
        if no_of_runs==2:
            log_results(m,gs_1d,gs_2d,ops,fq.SAMPLES,fq.NUM_ITER,exact_energy = m,steps_until_convergence=steps_until_convergence,filename=OUT_LOG_NAME)
        elif no_of_runs==4:
            log_results(m,gs_1d,gs_2d,ops,fq.SAMPLES,fq.NUM_ITER,exact_energy = m,steps_until_convergence=steps_until_convergence,filename=OUT_LOG_NAME)
            log_results(m,gs_1a,gs_2a,ops,fq.SAMPLES,fq.NUM_ITER,exact_energy = m,steps_until_convergence=steps_until_convergence,filename=OUT_LOG_NAME)
        else:
            log_results(m,gs_1d,gs_1d,ops,fq.SAMPLES,fq.NUM_ITER,exact_energy = m,steps_until_convergence=steps_until_convergence,filename=OUT_LOG_NAME)
    end = time.time()
    
    min_steps = np.min(conv,axis=0)
    average_steps = np.average(conv,axis=0)
    pm_steps = np.std(conv,axis=0) 
    with open('out-models_table.txt','a') as table_file:
                        # i       name     params   time      eta     min     avg     pm    MSR: min    avg     pm        min      avg     pm   MSR: min   avg    pm
        table_file.write("{:2.0f}  {:<15}  {:5.0f}  {:5.1f}  {:6.5f}  {:6.0f} {:6.1f} {:6.3}   {:6.0f} {:6.1f} {:6.3f}   {:6.0f} {:6.1f} {:6.3}  {:6.0f} {:6.1f} {:6.3f}\n".format(m,name,vs_1.n_parameters,(end-start)/60,ETAS[m%no_etas],min_steps[0],average_steps[0],pm_steps[0],min_steps[1],average_steps[1],pm_steps[1],min_steps[2],average_steps[2],pm_steps[2],min_steps[3],average_steps[3],pm_steps[3]))
    conv = np.zeros((no_repeats,4))