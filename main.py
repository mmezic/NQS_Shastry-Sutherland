import sys, getopt	
sys.path.append('/storage/praha1/home/mezic/.local/lib/python3.7/site-packages')	
import netket as nk	
import numpy as np
import jax
import time	
print("NetKet version: {}".format(nk.__version__))	
print("NumPy version: {}".format(np.__version__))	

file = sys.argv[-1]
if len(sys.argv) == 1:
    file = "config"
print(file)
fq = __import__(file)

OUT_NAME = "SS-RBM_ops"+str(fq.SITES) # output file name	
print("N = ",fq.SITES, ", samples = ",fq.SAMPLES,", iters = ",fq.NUM_ITER, ", sampler = ",fq.SAMPLER, ", TOTAL_SZ = ", fq.TOTAL_SZ, ", machine = ", fq.MACHINE, ", dtype = ", fq.DTYPE, ", alpha = ", fq.ALPHA, ", eta = ", fq.ETA, sep="")

if fq.SITES == 64:	
    indent = [0,0,0,0,0,0,0,0]	
    width = [8,8,8,8,8,8,8,8]	
    right_shift = 0	
    bottom_shift = 0	
elif fq.SITES == 36:	
    indent = [0,0,0,0,0,0]	
    width = [6,6,6,6,6,6]	
    right_shift = 0	
    bottom_shift = 0	
elif fq.SITES == 20: #tile shape definition	
    indent = [3,1,0,1,1,2]	
    width = [1,4,5,5,4,1]	
    right_shift = 2 #vertical shift of the center of right cell (in the upward direction)	
    bottom_shift = 4 #horizontal shift of the center of bottom cell (in the left direction)
elif fq.SITES == 16:
    indent = [0,0,0,0]
    width = [4,4,4,4]
    right_shift = 0
    bottom_shift = 0
elif fq.SITES == 8:
    indent = [1,0,1]
    width = [2,4,2]
    right_shift = 2
    bottom_shift = 2
elif fq.SITES == 4:
    indent = [0,0]
    width = [2,2]
    right_shift = 0
    bottom_shift = 0
else:
    raise Exception("Invalid number of fq.SITES given.")
N = sum(width) #number of nodes
left_shift = len(width) - right_shift #vertical shift of the center of left cell (in the upward direction)

# i j-->
# | .   .   .   0  
# V .   1   2   3   4
#   5   6   7   8   9
#   .   10  11  12  13  14
#   .   15  16  17  18
#   .   .   19

def getRandomNumber(): #returns a random integer
    return 4
def position(node): #returns positional indices i,j of the node
    row, n = 0, 0
    while n+width[row] <= node:
        n += width[row]
        row += 1
    column = indent[row] + node - n 
    return row, column
def index_n(row, column): #returns index n given positional indices
    return sum(width[0:row]) + column - indent[row]
def is_last(node):
    row, column = position(node)
    return (column == width[row] + indent[row] - 1)
def is_first(node):
    row, column = position(node)
    return (column == indent[row])
def is_lowest(node):
    row, column = position(node)
    if row == len(width) - 1:
        return True
    else:
        row += 1
        if column >= indent[row] and column < indent[row] + width[row]:
            return False
        else:
            return True
def rt(node): #returns index n of right neighbour
    if is_last(node):
        new_row = (position(node)[0] + right_shift)%len(width)
        return sum(width[0:new_row])
    else:
        return (node+1)%N
def lft(node): #returns index n of left neighbour
    if is_first(node):
        new_row = (position(node)[0] + left_shift)%len(width)
        return sum(width[0:new_row])+width[new_row]-1
    else:
        return (node-1)%N
def bot(node): #returns index n of bottom neighbour
    row, column = position(node)
    if is_lowest(node):
        no_of_columns = np.array(width)+np.array(indent)
        new_column = (column + bottom_shift)%max(no_of_columns)
        new_row = 0
        while new_column >= no_of_columns[new_row] or new_column < indent[new_row]:
            new_row += 1
        return index_n(new_row,new_column)
    else:
        return (node + width[row] + indent[row] - indent[row+1])
def lrt(node):
    return rt(bot(node))
def llft(node):
    return lft(bot(node))

# Define custom graph
edge_colors = []
for node in range(N):
    edge_colors.append([node,rt(node), 1]) #horizontal connections
    edge_colors.append([node,bot(node), 1]) #vertical connections
    row, column = position(node)
    if column%2 == 0:
        if row%2 == 0:
            edge_colors.append([node,lrt(node),2])
        else:
            edge_colors.append([node,llft(node),2])


# Define the netket graph object
g = nk.graph.Graph(edges=edge_colors)

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

    if g.n_nodes < 20:
        start = time.time()
        evals, eigvects = nk.exact.lanczos_ed(ha, k=3, compute_eigenvectors=True)
        #evals = nk.exact.lanczos_ed(ha, compute_eigenvectors=False) #.lanczos_ed
        end = time.time()
        diag_time = end - start
        exact_ground_energy = evals[0]
    else:
        exact_ground_energy = [0,0,0]
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


    
    def SS_old(i,j): # S_i * S_j
        return 2*nk.operator.spin.sigmap(hilbert,i,DTYPE="float64")@nk.operator.spin.sigmam(hilbert,j,DTYPE="float64")+2*nk.operator.spin.sigmam(hilbert,i,DTYPE="float64")@nk.operator.spin.sigmap(hilbert,j,DTYPE="float64")+nk.operator.spin.sigmaz(hilbert,i,DTYPE="float64")@nk.operator.spin.sigmaz(hilbert,j,DTYPE="float64")
        #return nk.operator.spin.sigmax(hilbert, i)@nk.operator.spin.sigmax(hilbert, j) + nk.operator.spin.sigmay(hilbert, i)@nk.operator.spin.sigmay(hilbert, j) + nk.operator.spin.sigmaz(hilbert, i)@nk.operator.spin.sigmaz(hilbert, j)

    def SS(i,j): #different method of definition
        if i==j:
            return nk.operator.LocalOperator(hilbert,operators=[[3,0],[0,3]],acting_on=[i])
        else:
            return nk.operator.LocalOperator(hilbert,operators=(mszsz+exchange),acting_on=[i,j])

    def SS_MSR(i,j): #different method of definition
        if i==j:
            return nk.operator.LocalOperator(hilbert,operators=[[3,0],[0,3]],acting_on=[i])
        elif (np.sum(position(i))+np.sum(position(j)))%2 == 0:                                      # same sublattice 
            return nk.operator.LocalOperator(hilbert,operators=(mszsz+exchange),acting_on=[i,j])
        else:                                                                                       # different sublattice 
            return nk.operator.LocalOperator(hilbert,operators=(mszsz-exchange),acting_on=[i,j])


    ss_operator = 0
    M = hilbert.size
    for node in range(M):
        row, column = position(node)
        if column%2 == 0:
            if row%2 == 0:
                ss_operator += SS(node,lrt(node))
            else:
                ss_operator += SS(node,llft(node))
    m_dimer_op = -1/3 * ss_operator/M*2


    def P(i,j,msr): # two particle permutation operator
        if msr == False:
            return .5*(SS(i,j)+nk.operator.LocalOperator(hilbert,operators=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],acting_on=[i,j]))
        else:
            return .5*(SS_MSR(i,j)+nk.operator.LocalOperator(hilbert,operators=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],acting_on=[i,j]))


    def P_cykl(i,j,k,l,msr): # 4 particle cyclic permutation operator
        return P(i,k,msr)@P(k,l,msr)@P(i,j,msr)
    def P_cykl_inv(i,j,k,l,msr): # inverse of 4 particle cyclic permutation operator
        return P(j,l,msr)@P(k,l,msr)@P(i,j,msr)

    def P_r(i,msr): # cyclic permutation of 4 fq.SITES located at position i
        return P_cykl(i,rt(i),lrt(i),bot(i),msr)
    # i --> i j  .... we assigne a lrt cell to each index i
    #       l k
    def P_r_inv(i,msr): # inverse of cyclic permutation of 4 fq.SITES located at position i
        return P_cykl_inv(i,rt(i),lrt(i),bot(i),msr)

    def Q_r(i,msr):
        return .5*(P_r(i,msr) + P_r_inv(i,msr))

    i = 0
    while sum(position(i))%2 == 0: # ensure that the plaquette is 'empty' (without J_2 bond inside)
        i += 1
    m_plaquette_op = Q_r(i,False)-Q_r(lrt(i),False)
    m_plaquette_op_MSR = Q_r(i,True)-Q_r(lrt(i),True)


    def m_sSquared_slow(state):
        ss_operator = 0
        M = hilbert.size
        m_s2 = 0
        for i in range(M):
            for j in range(M):
                ss_operator += SS(i,j) * (-1)**np.sum(position(i)+position(j))
            if i%3==2 or i==(M-1):
                m_s2 += (state.transpose()@(ss_operator@state))[0,0]
                ss_operator = 0
        m_s2 = m_s2/M**2
        return m_s2

    m_s2_op_MSR = 0
    M = hilbert.size
    for i in range(M):
        for j in range(M):
            m_s2_op_MSR += SS_MSR(i,j) * (-1)**np.sum(position(i)+position(j)) #(i+j) #chyba?
    m_s2_op_MSR = m_s2_op_MSR/(M**2)

    m_s2_op = 0
    for i in range(M):
        for j in range(M):
            m_s2_op += SS(i,j) * (-1)**np.sum(position(i)+position(j)) #(i+j) #chyba?
    m_s2_op = m_s2_op/(M**2)


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
            print("m_d^2 =", gs.estimate(m_dimer_op))
            print("m_p =", gs.estimate(m_plaquette_op))
            print("m_s^2 =", gs.estimate(m_s2_op_MSR))
            print("m_s^2 =", gs.estimate(m_s2_op), "<--- no MSR!!")
    print("{:9.5f}     {:9.5f}    {:9.5f}    {:9.5f}    {:9.5f}    {:9.5f}    {:9.5f}    {:9.5f}    {:9.5f}    {:9.5f}  {:9.5f}".format(JEXCH1, gs_normal.energy.mean.real, gs_MSR.energy.mean.real, gs_normal.estimate(m_dimer_op).mean.real, gs_normal.estimate(m_plaquette_op).mean.real, gs_normal.estimate(m_s2_op).mean.real, gs_MSR.estimate(m_dimer_op).mean.real, gs_MSR.estimate(m_plaquette_op_MSR).mean.real, gs_MSR.estimate(m_s2_op_MSR).mean.real, fq.SAMPLES, fq.NUM_ITER, sep='    '))