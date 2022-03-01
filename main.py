import sys, getopt	
sys.path.append('/storage/praha1/home/mezic/.local/lib/python3.7/site-packages')	
import netket as nk	
import numpy as np
import jax
import time
import json	
print("Python version: {}".format(sys.version))
print("NetKet version: {}".format(nk.__version__))	
print("NumPy version: {}".format(np.__version__))

file = sys.argv[-1]
if len(sys.argv) == 1:
    file = "config"
print(file)
fq = __import__(file)
from lattice_and_ops import Lattice
from lattice_and_ops import Operators
from lattice_and_ops import HamOps
from lattice_and_ops import permutation_sign
from lattice_and_ops import log_results
ho = HamOps()

OUT_NAME = fq.MACHINE+str(fq.SITES) # output file name	
print("N = ",fq.SITES, ", samples = ",fq.SAMPLES,", iters = ",fq.NUM_ITER, ", sampler = ",fq.SAMPLER, ", TOTAL_SZ = ", fq.TOTAL_SZ, ", machine = ", fq.MACHINE, ", dtype = ", fq.DTYPE, ", alpha = ", fq.ALPHA, ", eta = ", fq.ETA, sep="")

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

# This pars is only relevant for GCNN machine
print("There are", len(g.automorphisms()), "full symmetries.")
# deciding point between DS and AF phase is set to 0.5
if fq.JEXCH1 < 0.5:
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
    
for JEXCH1 in fq.STEPS:
    # Hamiltonian definition
    ha_1 = nk.operator.GraphOperator(hilbert, graph=g, bond_ops=ho.bond_operator(JEXCH1,fq.JEXCH2, use_MSR=False), bond_ops_colors=ho.bond_color)
    ha_2 = nk.operator.GraphOperator(hilbert, graph=g, bond_ops=ho.bond_operator(JEXCH1,fq.JEXCH2, use_MSR=True), bond_ops_colors=ho.bond_color)

    # Exact diagonalization
    if g.n_nodes < 20: # and fq.VERBOSE == True:
        start = time.time()
        evals, eigvects = nk.exact.lanczos_ed(ha_1, k=3, compute_eigenvectors=True)
        end = time.time()
        diag_time = end - start
        exact_ground_energy = evals[0]
    else:
        exact_ground_energy = 0
        eigvects = None

    # Symmetric RBM Spin fq.MACHINE
    # and Symmetric RBM Spin fq.MACHINE with MSR
    if fq.MACHINE == "RBM":
        machine_1 = nk.models.RBM(dtype=fq.DTYPE, alpha=fq.ALPHA)
        machine_2 = nk.models.RBM(dtype=fq.DTYPE, alpha=fq.ALPHA)
    elif fq.MACHINE == "RBMSymm":
        machine_1 = nk.models.RBMSymm(g.automorphisms(), dtype=fq.DTYPE, alpha=fq.ALPHA) 
        machine_2 = nk.models.RBMSymm(g.automorphisms(), dtype=fq.DTYPE, alpha=fq.ALPHA)
    elif fq.MACHINE == "GCNN":
        machine_1 = nk.models.GCNN(symmetries=g.automorphisms(), dtype=fq.DTYPE, layers=fq.num_layers, features=fq.feature_dims, characters=characters_dimer_1)
        machine_2 = nk.models.GCNN(symmetries=g.automorphisms(), dtype=fq.DTYPE, layers=fq.num_layers, features=fq.feature_dims, characters=characters_dimer_2)
    elif fq.MACHINE == "myRBM":
        from GCNN_Nomura import GCNN_my
        machine_1 = GCNN_my(symmetries=g.automorphisms(), dtype=fq.DTYPE, layers=1, features=fq.ALPHA, characters=characters_dimer_1, output_activation=nk.nn.log_cosh, use_bias=True, use_visible_bias=True)
        machine_2 = GCNN_my(symmetries=g.automorphisms(), dtype=fq.DTYPE, layers=1, features=fq.ALPHA, characters=characters_dimer_2, output_activation=nk.nn.log_cosh, use_bias=True, use_visible_bias=True)
    else:
        raise Exception(str("undefined MACHINE: ")+str(fq.MACHINE))

    # Selection of sampler type
    if fq.SAMPLER == 'local':
        sampler_1 = nk.sampler.MetropolisLocal(hilbert=hilbert)
        sampler_2 = nk.sampler.MetropolisLocal(hilbert=hilbert)
    elif fq.SAMPLER == 'exact':
        sampler_1 = nk.sampler.ExactSampler(hilbert=hilbert)
        sampler_2 = nk.sampler.ExactSampler(hilbert=hilbert)
    else:
        sampler_1 = nk.sampler.MetropolisExchange(hilbert=hilbert, graph=g)
        sampler_2 = nk.sampler.MetropolisExchange(hilbert=hilbert, graph=g)
        if fq.SAMPLER != 'exchange':
            print("Warning! Undefined fq.SAMPLER:", fq.SAMPLER, ", dafaulting to MetropolisExchange fq.SAMPLER")

    # Optimzer
    optimizer_1 = nk.optimizer.Sgd(learning_rate=fq.ETA)
    optimizer_2 = nk.optimizer.Sgd(learning_rate=fq.ETA)

    # Stochastic Reconfiguration
    sr_1 = nk.optimizer.SR(diag_shift=0.1)
    sr_2 = nk.optimizer.SR(diag_shift=0.1)

    # The variational state (drive to byla nk.variational.MCState)
    vs_1 = nk.vqs.MCState(sampler_1, machine_1, n_samples=fq.SAMPLES)
    vs_2 = nk.vqs.MCState(sampler_2, machine_2, n_samples=fq.SAMPLES)
    vs_1.init_parameters(jax.nn.initializers.normal(stddev=0.001))
    vs_2.init_parameters(jax.nn.initializers.normal(stddev=0.001))

    gs_1 = nk.VMC(hamiltonian=ha_1 ,optimizer=optimizer_1,preconditioner=sr_1,variational_state=vs_1)   # 0 ... symmetric
    gs_2 = nk.VMC(hamiltonian=ha_2 ,optimizer=optimizer_2,preconditioner=sr_2,variational_state=vs_2)   # 1 ... symmetric+MSR

    ops = Operators(lattice,hilbert,ho.mszsz,ho.exchange)

    no_of_runs = 2 #2 ... bude se pocitat i druhý způsob (za použití MSR)
    use_2 = 0 # in case of one run
    if exact_ground_energy != 0 and fq.VERBOSE == True:
        print("Expected exact energy:", exact_ground_energy)
    for i,gs in enumerate([gs_1,gs_2][use_2:use_2+no_of_runs]):
        start = time.time()
        gs.run(out=OUT_NAME+"_"+str(round(JEXCH1,2))+"_"+str(i), n_iter=int(fq.NUM_ITER),show_progress=fq.VERBOSE)
        end = time.time()
        if JEXCH1 == fq.STEPS[0]:
            print("The type {} of {} calculation took {} min".format(i,fq.MACHINE ,(end-start)/60))

    threshold_energy = 0.995*exact_ground_energy
    data = []
    for i in range(no_of_runs):
        data.append(json.load(open(OUT_NAME+"_"+str(round(JEXCH1,2))+"_"+str(i)+".log")))
    if type(data[0]["Energy"]["Mean"]) == dict: #DTYPE in (np.complex128, np.complex64):#, np.float64):# and False:
        energy_convergence = [data[i]["Energy"]["Mean"]["real"] for i in range(no_of_runs)]
    else:
        energy_convergence = [data[i]["Energy"]["Mean"] for i in range(no_of_runs)]
    steps_until_convergence = [next((i for i,v in enumerate(energy_convergence[j]) if v < threshold_energy), -1) for j in range(no_of_runs)]

    if fq.VERBOSE == True:
        for i,gs in enumerate([gs_1,gs_2][use_2:use_2+no_of_runs]):
            print("Trained RBM with MSR:" if i else "Trained RBM without MSR:")
            print("m_d^2 =", gs.estimate(ops.m_dimer_op))
            print("m_p =", gs.estimate(ops.m_plaquette_op_MSR))
            print("m_s^2 =", gs.estimate(ops.m_s2_op_MSR))
            print("m_s^2 =", gs.estimate(ops.m_s2_op), "<--- no MSR!!")
    
    if no_of_runs==2:
        log_results(JEXCH1,gs_1,gs_2,ops,fq.SAMPLES,fq.NUM_ITER,exact_ground_energy,steps_until_convergence,filename="out.txt")
    else:
        log_results(JEXCH1,gs_1,gs_1,ops,fq.SAMPLES,fq.NUM_ITER,exact_ground_energy,steps_until_convergence,filename="out.txt")