import sys, getopt, os
sys.path.append('/storage/praha1/home/mezic/.local/lib/python3.7/site-packages')
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "0") # Report only TF errors by default
import netket as nk	
import numpy as np
import jax
import time
import json	
print("Python version: {}".format(sys.version))
print("NetKet version: {}".format(nk.__version__))	
print("NumPy version: {}".format(np.__version__))
print("MPI:", nk.utils.mpi.available)
print("Devices:", jax.devices())

file = sys.argv[-1]
if len(sys.argv) == 1: # if no config file is specifies, use config.py by default
    file = "config"
print(file)
cf = __import__(file) # import configuration under cf alias
from lattice_and_ops import Lattice
from lattice_and_ops import Operators
from lattice_and_ops import HamOps
from lattice_and_ops import permutation_sign
from lattice_and_ops import log_results
ho = HamOps()

OUT_NAME = cf.MACHINE+str(cf.SITES) # output file name for logging the machine (.mpack file) and the convergence log (.log file)
OUT_LOG_NAME = "out.txt"            # filename for final logging of energies and order parameters
print("N = ",cf.SITES, ", samples = ",cf.SAMPLES,", iters = ",cf.NUM_ITER, ", sampler = ",cf.SAMPLER, ", TOTAL_SZ = ", cf.TOTAL_SZ, ", machine = ", cf.MACHINE, ", dtype = ", cf.DTYPE, ", alpha = ", cf.ALPHA, ", eta = ", cf.ETA, sep="")
with open(OUT_LOG_NAME,"a") as out_log_file: # heager of the log file
    out_log_file.write("N = {}, samples = {}, iters = {}, sampler = {}, TOTAL_SZ = {}, machine = {}, dtype = {}, alpha = {}, eta = {}\n".format(cf.SITES,cf.SAMPLES,cf.NUM_ITER,cf.SAMPLER, cf.TOTAL_SZ,cf.MACHINE, cf.DTYPE, cf.ALPHA, cf.ETA))

lattice = Lattice(cf.SITES)

# Define custom graph
edge_colors = []
for node in range(cf.SITES):
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

hilbert = nk.hilbert.Spin(s=.5, N=g.n_nodes, total_sz=cf.TOTAL_SZ)

# This part is only relevant for GCNN or pRBM used with the full symmetry group.
# We calculate the characters of the ground state irrep for the appropriate phase based on the value of JEXCH1.
# Only DS and AF phases are currently supported. 
print("There are", len(g.automorphisms()), "full symmetries.")
if cf.JEXCH1 < 0.6: # deciding point between DS and AF phase is set to 0.6
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

# This part is only relevant for GCNN or pRBM used only with the translation group enforced. 
if False and not (cf.SITES in [4,16]):
    raise NotImplementedError("Extraction of translations from the group of automorphisms is not implemented yet.")
# We extract the translations from the full symmetry group.
# This part in only implemented for the lattices of size 4 or 16.
translations = []
for perm in g.automorphisms():
    aperm = np.asarray(perm)
    if cf.SITES == 4:
        if (aperm[0],aperm[1]) in ((0,1),(1,0),(2,3),(3,2)):
            translations.append(nk.utils.group._permutation_group.Permutation(aperm))
    elif cf.SITES == 16:
        if (aperm[0],aperm[1],aperm[3]) in ((0,1,3),(2,3,1),(8,9,11),(10,11,9)):
            translations.append(nk.utils.group._permutation_group.Permutation(aperm))
translation_group = nk.utils.group._permutation_group.PermutationGroup(translations,degree=cf.SITES)

for JEXCH1 in cf.STEPS:
    # Hamiltonian definition
    ha_1 = nk.operator.GraphOperator(hilbert, graph=g, bond_ops=ho.bond_operator(JEXCH1,cf.JEXCH2, use_MSR=False), bond_ops_colors=ho.bond_color)
    ha_2 = nk.operator.GraphOperator(hilbert, graph=g, bond_ops=ho.bond_operator(JEXCH1,cf.JEXCH2, use_MSR=True), bond_ops_colors=ho.bond_color)

    # Exact diagonalization
    if g.n_nodes < 20:
        evals, eigvects = nk.exact.lanczos_ed(ha_1, k=3, compute_eigenvectors=True)
        exact_ground_energy = evals[0]
    else:
        exact_ground_energy = 0
        eigvects = None

    # Choice of the architecture
    if cf.MACHINE == "RBM":
        machine_1 = nk.models.RBM(dtype=cf.DTYPE, alpha=cf.ALPHA)
        machine_2 = nk.models.RBM(dtype=cf.DTYPE, alpha=cf.ALPHA)
    elif cf.MACHINE == "RBMSymm":
        machine_1 = nk.models.RBMSymm(g.automorphisms(), dtype=cf.DTYPE, alpha=cf.ALPHA) 
        machine_2 = nk.models.RBMSymm(g.automorphisms(), dtype=cf.DTYPE, alpha=cf.ALPHA)
    elif cf.MACHINE == "GCNN":
        machine_1 = nk.models.GCNN(symmetries=g.automorphisms(), dtype=cf.DTYPE, layers=cf.num_layers, features=cf.feature_dims, characters=characters_dimer_1)
        machine_2 = nk.models.GCNN(symmetries=g.automorphisms(), dtype=cf.DTYPE, layers=cf.num_layers, features=cf.feature_dims, characters=characters_dimer_2)
    elif cf.MACHINE == "pRBM":
        from pRBM import pRBM
        machine_1 = pRBM(symmetries=g.automorphisms(), dtype=cf.DTYPE, layers=1, features=cf.ALPHA, characters=characters_dimer_1, output_activation=nk.nn.log_cosh, use_bias=True, use_visible_bias=True)
        machine_2 = pRBM(symmetries=g.automorphisms(), dtype=cf.DTYPE, layers=1, features=cf.ALPHA, characters=characters_dimer_2, output_activation=nk.nn.log_cosh, use_bias=True, use_visible_bias=True)
    else:
        raise Exception(str("undefined MACHINE: ")+str(cf.MACHINE))

    # Selection of sampler type
    if cf.SAMPLER == 'local':
        sampler_1 = nk.sampler.MetropolisLocal(hilbert=hilbert)
        sampler_2 = nk.sampler.MetropolisLocal(hilbert=hilbert)
    elif cf.SAMPLER == 'exact':
        sampler_1 = nk.sampler.ExactSampler(hilbert=hilbert)
        sampler_2 = nk.sampler.ExactSampler(hilbert=hilbert)
    else:
        sampler_1 = nk.sampler.MetropolisExchange(hilbert=hilbert, graph=g)
        sampler_2 = nk.sampler.MetropolisExchange(hilbert=hilbert, graph=g)
        if cf.SAMPLER != 'exchange':
            print("Warning! Undefined cf.SAMPLER:", cf.SAMPLER, ", dafaulting to MetropolisExchange cf.SAMPLER")

    # Optimzer
    optimizer_1 = nk.optimizer.Sgd(learning_rate=cf.ETA)
    optimizer_2 = nk.optimizer.Sgd(learning_rate=cf.ETA)

    # Stochastic Reconfiguration
    sr_1 = nk.optimizer.SR(diag_shift=0.01)
    sr_2 = nk.optimizer.SR(diag_shift=0.01)

    # The variational state (drive to byla nk.variational.MCState)
    vs_1 = nk.vqs.MCState(sampler_1, machine_1, n_samples=cf.SAMPLES)
    vs_2 = nk.vqs.MCState(sampler_2, machine_2, n_samples=cf.SAMPLES)
    vs_1.init_parameters(jax.nn.initializers.normal(stddev=0.01))
    vs_2.init_parameters(jax.nn.initializers.normal(stddev=0.01))

    gs_1 = nk.VMC(hamiltonian=ha_1 ,optimizer=optimizer_1,preconditioner=sr_1,variational_state=vs_1)   # 0 ... normal basis
    gs_2 = nk.VMC(hamiltonian=ha_2 ,optimizer=optimizer_2,preconditioner=sr_2,variational_state=vs_2)   # 1 ...    MSR basis

    ops = Operators(lattice,hilbert,ho.mszsz,ho.exchange)

    runs = [1,1] # Here we can set for which basis to perform the simulation. E.g. [0,1] means don't run normal basis, run MSR basis. We run the simulation for both basis by default [1,1]
    no_of_runs = np.sum(runs) # 1 - one run for variables with ..._1;  2 - both runs for variables ..._1 and ..._2
    run_only_2 = (runs[1]==1 and runs[0]==0) # in case of no_of_runs=1
    if exact_ground_energy != 0 and cf.VERBOSE == True:
        print("J1 =",JEXCH1,"; Expected exact energy:", exact_ground_energy)
    for i,gs in enumerate([gs_1,gs_2][run_only_2:run_only_2+no_of_runs]):
        start = time.time()
        gs.run(out=OUT_NAME+"_"+str(round(JEXCH1,2))+"_"+str(i), n_iter=int(cf.NUM_ITER),show_progress=cf.VERBOSE)
        end = time.time()
        if JEXCH1 == cf.STEPS[0]:
            print("The type {} of {} calculation took {} min".format(i,cf.MACHINE ,(end-start)/60))


    # finding the number of steps needed to converge
    threshold_energy = 0.995*exact_ground_energy
    data = []
    for i in range(no_of_runs):
        data.append(json.load(open(OUT_NAME+"_"+str(round(JEXCH1,2))+"_"+str(i)+".log")))
    if type(data[0]["Energy"]["Mean"]) == dict:
        energy_convergence = [data[i]["Energy"]["Mean"]["real"] for i in range(no_of_runs)]
    else:
        energy_convergence = [data[i]["Energy"]["Mean"] for i in range(no_of_runs)]
    steps_until_convergence = [next((i for i,v in enumerate(energy_convergence[j]) if v < threshold_energy), -1) for j in range(no_of_runs)]

    if cf.VERBOSE == True:
        # print useful info about order parameters to the screen
        for i,gs in enumerate([gs_1,gs_2][run_only_2:run_only_2+no_of_runs]):
            print("Trained RBM with MSR:" if i else "Trained RBM without MSR:")
            print("m_d^2 =", gs.estimate(ops.m_dimer_op))
            print("m_p =", gs.estimate(ops.m_plaquette_op_MSR))
            print("m_s^2 =", float(ops.m_sSquared_slow(gs)[0].real))
            print("m_s^2 =", float(ops.m_sSquared_slow_MSR(gs)[0].real), "<--- MSR")
    
    # estimating the errorbars
    data = []
    for i in range(no_of_runs):
        data.append(json.load(open(OUT_NAME+"_"+str(round(JEXCH1,2))+"_"+str(i)+".log")))
    if type(data[0]["Energy"]["Mean"]) == dict:
        energy_convergence = [data[i]["Energy"]["Mean"]["real"] for i in range(no_of_runs)]
    else:
        energy_convergence = [data[i]["Energy"]["Mean"] for i in range(no_of_runs)]

    # logging the final energies
    with open("out_err.txt", "a") as file_mag:
        if JEXCH1 == cf.STEPS[0]:
            print("J1  exactE     E  err_of_mean  E_50avg  E_err_of_50avg    MSR: E  err_of_mean  E_50avg  E_err_of_50avg", file=file_mag)   
        print("{:5.2f}  {:10.5f}     {:10.5f} {:10.5f} {:10.5f} {:10.5f}     {:10.5f} {:10.5f} {:10.5f} {:10.5f}".format(JEXCH1, exact_ground_energy, gs_1.energy.mean.real, gs_1.energy.error_of_mean, np.mean(energy_convergence[0][-50:]), np.std(energy_convergence[0][-50:]), gs_2.energy.mean.real, gs_2.energy.error_of_mean, np.mean(energy_convergence[1][-50:]), np.std(energy_convergence[1][-50:])), file=file_mag)

    # our standardised logging of all the energies, order parameters and other properties
    if no_of_runs==2:
        log_results(JEXCH1,gs_1,gs_2,ops,cf.SAMPLES,cf.NUM_ITER,exact_ground_energy,steps_until_convergence,filename=OUT_LOG_NAME)
    else:
        log_results(JEXCH1,gs_1,gs_1,ops,cf.SAMPLES,cf.NUM_ITER,exact_ground_energy,steps_until_convergence,filename=OUT_LOG_NAME)
