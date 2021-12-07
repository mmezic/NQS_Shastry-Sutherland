import sys, getopt	
sys.path.append('/storage/praha1/home/mezic/.local/lib/python3.7/site-packages')	
import netket as nk	
import numpy as np
import jax
import time	
print("NetKet version: {}".format(nk.__version__))	
print("NumPy version: {}".format(np.__version__))
TF_CPP_MIN_LOG_LEVEL=0

file = sys.argv[-1]
if len(sys.argv) == 1:
    file = "config"
print(file)
fq = __import__(file)
from lattice_and_ops import Lattice
from lattice_and_ops import Operators

OUT_NAME = "SS-RBM_ops"+str(fq.SITES) # output file name	
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
print(g.edges(return_color=True))

hilbert = nk.hilbert.Spin(s=.5, N=g.n_nodes, total_sz=fq.TOTAL_SZ)


#Sigma^z*Sigma^z interactions
sigmaz = [[1, 0], [0, -1]]
mszsz = (np.kron(sigmaz, sigmaz)) #=sz*sz
#Exchange interactions
exchange = np.asarray([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]]) #=sx*sx+sy*sy = 1/2*(sp*sm+sm*sp)
full_spin = mszsz+exchange # = S*S = sx*sx + sy*sy + sz*sz
bond_color = [1, 2, 1, 2]

for JEXCH1 in fq.STEPS:
    bond_operator = [
        (JEXCH1 * mszsz).tolist(),
        (fq.JEXCH2 * mszsz).tolist(),
        (JEXCH1 * exchange).tolist(), # minus in case of MSR
        (fq.JEXCH2 * exchange).tolist(),
    ]
    bond_operatorMSR = [
        (JEXCH1 * mszsz).tolist(),
        (fq.JEXCH2 * mszsz).tolist(),
        (-JEXCH1 * exchange).tolist(), # minus in case of MSR
        (fq.JEXCH2 * exchange).tolist(),
    ]
    ha = nk.operator.GraphOperator(hilbert, graph=g, bond_ops=bond_operator, bond_ops_colors=bond_color)
    ha_MSR = nk.operator.GraphOperator(hilbert, graph=g, bond_ops=bond_operatorMSR, bond_ops_colors=bond_color)

    if g.n_nodes < 20 and fq.VERBOSE == True:
        start = time.time()
        evals, eigvects = nk.exact.lanczos_ed(ha, k=3, compute_eigenvectors=True)
        #evals = nk.exact.lanczos_ed(ha, compute_eigenvectors=False) #.lanczos_ed
        end = time.time()
        diag_time = end - start
        exact_ground_energy = evals[0]
    else:
        exact_ground_energy = 0
        eigvects = None

    # Symmetric RBM Spin fq.MACHINE
    # and Symmetric RBM Spin fq.MACHINE with MSR
    if fq.MACHINE == "RBMSymm":
        machine = nk.models.RBMSymm(g.automorphisms(), dtype=fq.DTYPE, alpha=fq.ALPHA)  #<--- zde je použita celá grupa symetrii (ne jen translace)
        machine_MSR = nk.models.RBMSymm(g.automorphisms(),dtype=fq.DTYPE, alpha=fq.ALPHA)
    else:
        machine = nk.models.RBM(dtype=fq.DTYPE, alpha=fq.ALPHA)  #<--- zde je použita celá grupa symetrii (ne jen translace)
        machine_MSR = nk.models.RBM(dtype=fq.DTYPE, alpha=fq.ALPHA)
        if fq.MACHINE != "RBM":
            print("Warning! Undefined machine:", fq.MACHINE_NAME, ", dafaulting to (npn-symmetric) RBM machine")

    # Meropolis Exchange Sampling
    if fq.SAMPLER == 'local':
        sampler = nk.sampler.MetropolisLocal(hilbert=hilbert)
        sampler_MSR = nk.sampler.MetropolisLocal(hilbert=hilbert)
    elif fq.SAMPLER == 'exact':
        sampler = nk.sampler.ExactSampler(hilbert=hilbert)
        sampler_MSR = nk.sampler.ExactSampler(hilbert=hilbert)
    else:
        sampler = nk.sampler.MetropolisExchange(hilbert=hilbert, graph=g)
        sampler_MSR = nk.sampler.MetropolisExchange(hilbert=hilbert, graph=g)
        if fq.SAMPLER != 'exchange':
            print("Warning! Undefined fq.SAMPLER:", fq.SAMPLER, ", dafaulting to MetropolisExchange fq.SAMPLER")

    # Optimzer
    optimizer = nk.optimizer.Sgd(learning_rate=fq.ETA)
    optimizer_MSR = nk.optimizer.Sgd(learning_rate=fq.ETA)

    # Stochastic Reconfiguration
    sr  = nk.optimizer.SR(diag_shift=0.1)
    sr_MSR  = nk.optimizer.SR(diag_shift=0.1)

    # The variational state (drive to byla nk.variational.MCState)
    vss = nk.vqs.MCState(sampler, machine, n_samples=fq.SAMPLES)
    vs_MSR  = nk.vqs.MCState(sampler_MSR, machine_MSR, n_samples=fq.SAMPLES)
    vss.init_parameters(jax.nn.initializers.normal(stddev=0.001))
    vs_MSR.init_parameters(jax.nn.initializers.normal(stddev=0.001))

    gs_normal = nk.VMC(hamiltonian=ha ,optimizer=optimizer,preconditioner=sr,variational_state=vss)                       # 0 ... symmetric
    gs_MSR = nk.VMC(hamiltonian=ha_MSR ,optimizer=optimizer_MSR,preconditioner=sr_MSR,variational_state=vs_MSR)   # 1 ... symmetric+MSR

    ops = Operators(lattice,hilbert,mszsz,exchange)

    no_of_runs = 2 #2 ... bude se pocitat i druhý způsob (za použití MSR)
    use_MSR = 0 # in case of one run
    #fq.NUM_ITER = 100
    if exact_ground_energy != 0 and fq.VERBOSE == True:
        print("Expected exact energy:", exact_ground_energy)
    for i,gs in enumerate([gs_normal,gs_MSR][use_MSR:use_MSR+no_of_runs]):
        start = time.time()
        gs.run(out=OUT_NAME+"_"+str(i), n_iter=int(fq.NUM_ITER),show_progress=fq.VERBOSE)#,obs={'DS_factor': m_dimer_op})#,'PS_factor':m_plaquette_op,'AF_factor':m_s2_op})
        end = time.time()
        print("The type {} of RBM calculation took {} min".format(i, (end-start)/60))


    if fq.VERBOSE == True:
        for i,gs in enumerate([gs_normal,gs_MSR][use_MSR:use_MSR+no_of_runs]):
            print("Trained RBM with MSR:" if i else "Trained RBM without MSR:")
            print("m_d^2 =", gs.estimate(ops.m_dimer_op))
            print("m_p =", gs.estimate(ops.m_plaquette_op))
            print("m_s^2 =", gs.estimate(ops.m_s2_op_MSR))
            print("m_s^2 =", gs.estimate(ops.m_s2_op), "<--- no MSR!!")
    print("{:9.5f}     {:9.5f}    {:9.5f}    {:9.5f}    {:9.5f}    {:9.5f}    {:9.5f}    {:9.5f}    {:9.5f}    {:9.5f}  {:9.5f}".format(JEXCH1, gs_normal.energy.mean.real, gs_MSR.energy.mean.real, gs_normal.estimate(ops.m_dimer_op).mean.real, gs_normal.estimate(ops.m_plaquette_op).mean.real, gs_normal.estimate(ops.m_s2_op).mean.real, gs_MSR.estimate(ops.m_dimer_op).mean.real, gs_MSR.estimate(ops.m_plaquette_op_MSR).mean.real, gs_MSR.estimate(ops.m_s2_op_MSR).mean.real, fq.SAMPLES, fq.NUM_ITER, sep='    '))
    file = open("out.txt", "a")
    file.write("{:9.5f}     {:9.5f}    {:9.5f}    {:9.5f}    {:9.5f}    {:9.5f}    {:9.5f}    {:9.5f}    {:9.5f}    {:9.5f}  {:9.5f}\n".format(JEXCH1, gs_normal.energy.mean.real, gs_MSR.energy.mean.real, gs_normal.estimate(ops.m_dimer_op).mean.real, gs_normal.estimate(ops.m_plaquette_op).mean.real, gs_normal.estimate(ops.m_s2_op).mean.real, gs_MSR.estimate(ops.m_dimer_op).mean.real, gs_MSR.estimate(ops.m_plaquette_op_MSR).mean.real, gs_MSR.estimate(ops.m_s2_op_MSR).mean.real, fq.SAMPLES, fq.NUM_ITER, sep='    '))
    file.close()