import numpy as np
import netket as nk

"""
Builds S-S lattice given the number of sites. And implements positional functions on the lattice sites.
"""
class Lattice:
    def __init__(self,SITES):
        self.SITES = SITES
        if self.SITES == 100:	
            self.indent = [0,0,0,0,0,0,0,0,0,0]	
            self.width = [10,10,10,10,10,10,10,10,10,10]	
            self.right_shift = 0	
            self.bottom_shift = 0
        elif self.SITES == 64:	
            self.indent = [0,0,0,0,0,0,0,0]	
            self.width = [8,8,8,8,8,8,8,8]	
            self.right_shift = 0	
            self.bottom_shift = 0	
        elif self.SITES == 36:	
            self.indent = [0,0,0,0,0,0]	
            self.width = [6,6,6,6,6,6]	
            self.right_shift = 0	
            self.bottom_shift = 0	
        elif self.SITES == 20:
            self.indent = [3,1,0,1,1,2]	
            self.width = [1,4,5,5,4,1]	
            self.right_shift = 2  # vertical shift of the center of right cell (in the upward direction)	
            self.bottom_shift = 4 # horizontal shift of the center of bottom cell (in the left direction)
        elif self.SITES == 16:
            self.indent = [0,0,0,0]
            self.width = [4,4,4,4]
            self.right_shift = 0
            self.bottom_shift = 0
        elif self.SITES == 8:
            self.indent = [1,0,1]
            self.width = [2,4,2]
            self.right_shift = 2
            self.bottom_shift = 2
        elif self.SITES == 4:
            self.indent = [0,0]
            self.width = [2,2]
            self.right_shift = 0
            self.bottom_shift = 0
        else:
            raise Exception("Invalid number of SITES given. Supported numbers of sites: 4,8,16,20,36,64,100.")
        self.N = sum(self.width) #number of nodes

        assert self.N == self.SITES, "Error during the Lattice initialization. The checksum of sites failed."

        deg45 = True # special case when angle of tiling is 45 deg
        i = 0
        while deg45 and i < len(self.width)-1:
            deg45 = (self.width[i] == self.width[i+1] + 2 or self.width[i] == self.width[i+1] - 2)
            i += 1
        self.vertical_gap = False # is there a 1-site-sized gap between bottom (left) tile and left (top) tile ? 
        self.horizontal_gap = False # is there a 1-site-sized gap between bottom (left) tile and bottom right tile ?
        if deg45:
            if self.width[i]%2 == 0:
                self.vertical_gap = True
            else:
                self.horizontal_gap = True
                Exception("Not implemented ERROR during lattice definition. Please rotate given lattice by 90 degrees. This special case happens only when dealing with 45 deg tilings.")

        self.left_shift = len(self.width) - self.right_shift + self.vertical_gap #vertical shift of the center of (top) left tile (in the upward direction)

        # Exymple of the tile with 20 sites. Here, i and j denote cortinates and the numbers denote cardinal indices of each site. 
        # i j-->
        # | .   .   .   0  
        # V .   1   2   3   4
        #   5   6   7   8   9
        #   .   10  11  12  13  14
        #   .   15  16  17  18
        #   .   .   19

    def position(self,node): # returns positional indices i,j of the node given by the cardinal index
        row, n = 0, 0
        while n+self.width[row] <= node:
            n += self.width[row]
            row += 1
        column = self.indent[row] + node - n 
        return row, column
    def index_n(self,row, column): # returns cardinal index n given positional indices
        return sum(self.width[0:row]) + column - self.indent[row]
    def is_last(self,node): # returns true if the node is on the rightmost edge of the tile
        row, column = self.position(node)
        return (column == self.width[row] + self.indent[row] - 1)
    def is_first(self,node): # returns true if the node is on the leftmost edge of the tile
        row, column = self.position(node)
        return (column == self.indent[row])
    def is_lowest(self,node): # returns true if the node is on the bottom edge of the tile
        row, column = self.position(node)
        if row == len(self.width) - 1:
            return True
        else:
            row += 1
            if column >= self.indent[row] and column < self.indent[row] + self.width[row]:
                return False
            else:
                return True
    def rt(self, node): # returns ordinal index n of right neighbour
        if self.is_last(node):
            new_row = (self.position(node)[0] + self.right_shift)%(len(self.width)+self.vertical_gap)
            if new_row == len(self.width): # special case of gap
                new_row -= self.right_shift
            return sum(self.width[0:new_row])
        else:
            return (node+1)%self.N
    def lft(self, node): # returns ordinal index n of left neighbour
        if self.is_first(node):
            new_row = (self.position(node)[0] + self.left_shift)%(len(self.width)+self.vertical_gap)
            if new_row == len(self.width):
                new_row -= self.left_shift
            return sum(self.width[0:new_row])+self.width[new_row]-1
        else:
            return (node-1)%self.N
    def bot(self, node): # returns ordinal index n of bottom neighbour
        row, column = self.position(node)
        if self.is_lowest(node):
            no_of_columns = np.array(self.width)+np.array(self.indent)
            new_column = (column + self.bottom_shift)%max(no_of_columns)
            new_row = 0
            while new_column >= no_of_columns[new_row] or new_column < self.indent[new_row]:
                new_row += 1
            return self.index_n(new_row,new_column)
        else:
            return (node + self.width[row] + self.indent[row] - self.indent[row+1])
    def lrt(self, node):
        return self.rt(self.bot(node))
    def llft(self, node):
        return self.lft(self.bot(node))


"""
This clas contains definitions of order parameter operators for a given lattice. Both MSR and non-MSR operators are defined here.
"""
class Operators:
    def __init__(self,lattice,hilbert, mszsz_interaction, exchange_interaction):
        self.lattice = lattice
        self.hilbert = hilbert
        self.mszsz = mszsz_interaction
        self.exchange = exchange_interaction

        ## Total magnetization
        self.m_z = sum(nk.operator.spin.sigmaz(hilbert, i) for i in range(hilbert.size))

        ## DS operator
        ss_operator = 0
        M = self.hilbert.size
        for node in range(M):
            row, column = self.lattice.position(node)
            if column%2 == 0:
                if row%2 == 0:
                    ss_operator += self.SS(node,self.lattice.lrt(node))
                else:
                    ss_operator += self.SS(node,self.lattice.llft(node))
        self.m_dimer_op = -1/3 * ss_operator/M*2

        ## PS operator
        i = 0
        while sum(self.lattice.position(i))%2 == 0: # ensure that the plaquette is 'empty' (without J_2 bond inside)
            i += 1
        self.m_plaquette_op = self.Q_r(i,False)-self.Q_r(self.lattice.lrt(i),False)
        self.m_plaquette_op_MSR = self.Q_r(i,True)-self.Q_r(self.lattice.lrt(i),True)
        if lattice.SITES < 5: # plaquette operators cannot be defined for small lattices -> replacing by identity
            self.m_plaquette_op = nk.operator.LocalOperator(self.hilbert,operators=[[1,0],[0,1]],acting_on=[0])
            self.m_plaquette_op_MSR = self.m_plaquette_op

        ## AF operator
        self.m_s2_op_MSR = 0
        M = self.hilbert.size
        for i in range(M):
            for j in range(M):
                self.m_s2_op_MSR += self.SS_MSR(i,j) * (-1)**np.sum(self.lattice.position(i)+self.lattice.position(j))
        self.m_s2_op_MSR = self.m_s2_op_MSR/(M**2)

        self.m_s2_op = 0
        for i in range(M):
            for j in range(M):
                self.m_s2_op += self.SS(i,j) * (-1)**np.sum(self.lattice.position(i)+self.lattice.position(j))
        self.m_s2_op = self.m_s2_op/(M**2)

        

    """ Returns S_i * S_j coupling operator for normal basis. The ordinal integers i and j denote two sites in the lattice (i and j are not coordinates). """
    def SS(self,i,j):
        if i==j:
            return nk.operator.LocalOperator(self.hilbert,operators=[[3,0],[0,3]],acting_on=[i])
        else:
            return nk.operator.LocalOperator(self.hilbert,operators=(self.mszsz+self.exchange),acting_on=[i,j])

    """ Returns S_i * S_j coupling operator for MSR basis. The ordinal integers i and j denote two sites in the lattice (i and j are not coordinates). """
    def SS_MSR(self,i,j):
        if i==j:
            return nk.operator.LocalOperator(self.hilbert,operators=[[3,0],[0,3]],acting_on=[i])
        elif (np.sum(self.lattice.position(i))+np.sum(self.lattice.position(j)))%2 == 0:            # same sublattice 
            return nk.operator.LocalOperator(self.hilbert,operators=(self.mszsz+self.exchange),acting_on=[i,j])
        else:                                                                                       # different sublattice 
            return nk.operator.LocalOperator(self.hilbert,operators=(self.mszsz-self.exchange),acting_on=[i,j])

    """ Alternative equivalent definition of the S_i * S_j coupling operator using sigma matrices. """
    def SS_old(self,i,j): # S_i * S_j
        return 2*nk.operator.spin.sigmap(self.hilbert,i,DTYPE="float64")@nk.operator.spin.sigmam(self.hilbert,j,DTYPE="float64")+2*nk.operator.spin.sigmam(self.hilbert,i,DTYPE="float64")@nk.operator.spin.sigmap(self.self.hilbert,j,DTYPE="float64")+nk.operator.spin.sigmaz(self.hilbert,i,DTYPE="float64")@nk.operator.spin.sigmaz(self.hilbert,j,DTYPE="float64")

    """ Two-particle permutation operator. $$\hat{P}_{ij} = \frac{1}{2} (\hat{S}_i\cdot \hat{S}_j + \hat{1})$$. The ordinal integers i and j denote two sites in the lattice (i and j are not coordinates). """
    def P(self,i,j,msr=False): 
        if msr == False:
            return .5*(self.SS(i,j)+nk.operator.LocalOperator(self.hilbert,operators=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],acting_on=[i,j]))
        else:
            return .5*(self.SS_MSR(i,j)+nk.operator.LocalOperator(self.hilbert,operators=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],acting_on=[i,j]))

    """ Four-particle cyclic permutation operator defined as a product of three two-particle permutations. The ordinal integers i, j, k and l denote two sites in the lattice (these are not coordinates). """
    def P_cykl(self,i,j,k,l,msr):
        return self.P(i,k,msr)@self.P(k,l,msr)@self.P(i,j,msr)

    """ Inverse four-particle cyclic permutation operator defined as a product of three two-particle permutations. The ordinal integers i, j, k and l denote two sites in the lattice (these are not coordinates). """
    def P_cykl_inv(self,i,j,k,l,msr):
        return self.P(j,l,msr)@self.P(k,l,msr)@self.P(i,j,msr)

    """ Cyclic permutation of four sites. The top left site is located at the index i. """
    def P_r(self,i,msr):
        return self.P_cykl(i,self.lattice.rt(i),self.lattice.lrt(i),self.lattice.bot(i),msr)
        # i --> i j  .... we assigne a lower right cell to the input index i
        #       l k
    
    """ Inverse cyclic permutation of four sites. The top left site is located at the index i. """
    def P_r_inv(self,i,msr):
        return self.P_cykl_inv(i,self.lattice.rt(i),self.lattice.lrt(i),self.lattice.bot(i),msr)

    """ Returns the Q operator where index i indicates the top left corner:
    i---X
    |   |
    X---X  
    It is implicitly expected that there is no diagonal bond inside. """
    def Q_r(self,i,msr):
        return .5*(self.P_r(i,msr) + self.P_r_inv(i,msr))

    """ Returns the PS operator at lattice position i."""
    def m_plaquette_op_at(self,i,msr=False,secondary_diagonal=False):
        if self.lattice.SITES < 5: # plaquette operators cannot be defined for small lattices!
            raise Exception("PS operator cannot be defined for small lattices!")
        if secondary_diagonal:
            return self.Q_r(i,msr)-self.Q_r(self.lattice.llft(i),msr)
        else:
            return self.Q_r(i,msr)-self.Q_r(self.lattice.lrt(i),msr)

    """ Calculates the PS order parameter in a proper way (as a difference of two sub-checkerboards). This implementation works only for N=64."""
    def PS_order_full(self, state, msr=False):
        if self.lattice.SITES != 64:
            print("WARNING: Current implementation of improved PS order parameter works only for N=64 lattice. Returning NaN instead...")
            return np.nan
        sum_of_plaquettes = 0
        for i in [1,3,5,7,17,19,21,23,33,35,37,39,49,51,53,55]:
            sum_of_plaquettes += state.estimate(self.Q_r(i,msr)).mean.real
        for i in [8,10,12,14,24,26,28,30,40,42,44,46,56,58,60,62]:
            sum_of_plaquettes -= state.estimate(self.Q_r(i,msr)).mean.real
        return sum_of_plaquettes/16
    

    """ Calculates the value of AF order parameter from either a vector (ndarray) representation of state or NN representation. This method is slower but uses less memory.
    The value of partial sum is saved in batches of MEMORY_SIZE to avoid "out of memory" errors. """
    def m_sSquared_slow(self,state):
        MEMORY_SIZE = 8
        m_s2_partial_operator = 0
        M = self.hilbert.size
        m_s2 = 0
        variance = 0
        for i in range(M):
            for j in range(M):
                m_s2_partial_operator += self.SS(i,j) * (-1)**np.sum(self.lattice.position(i)+self.lattice.position(j))
                if (j+1)%MEMORY_SIZE == 0 or j == M-1:
                    if type(state) == np.ndarray:
                        m_s2 += (state.transpose()@(m_s2_partial_operator@state))#[0]
                    else:
                        m_s2 += state.estimate(m_s2_partial_operator).mean
                        variance += state.estimate(m_s2_partial_operator).error_of_mean
                    m_s2_partial_operator = 0
        m_s2 = m_s2/(M**2)
        variance = variance/(M**2)
        return m_s2, variance
    
    def m_sSquared_slow_MSR(self,state):
        MEMORY_SIZE = 8
        m_s2_partial_operator = 0
        M = self.hilbert.size
        m_s2 = 0
        variance = 0
        for i in range(M):
            for j in range(M):
                m_s2_partial_operator += self.SS_MSR(i,j) * (-1)**np.sum(self.lattice.position(i)+self.lattice.position(j))
                if (j+1)%MEMORY_SIZE == 0 or j == M-1:
                    if type(state) == np.ndarray:
                        m_s2 += (state.transpose()@(m_s2_partial_operator@state))#[0]
                    elif type(state) == nk.vqs.MCState:
                        m_s2 += state.expect(m_s2_partial_operator).mean
                        variance += state.expect(m_s2_partial_operator).error_of_mean
                    else:
                        m_s2 += state.estimate(m_s2_partial_operator).mean
                        variance += state.estimate(m_s2_partial_operator).error_of_mean
                    m_s2_partial_operator = 0
        m_s2 = m_s2/(M**2)
        variance = variance/(M**2)
        return m_s2, variance

"""
Class containing auxiliary operators which helps with defining a hamiltonian.
"""
class HamOps:
    # Sigma^z*Sigma^z interactions:
    sigmaz = [[1, 0], [0, -1]]
    mszsz = (np.kron(sigmaz, sigmaz)) # = sz*sz = [[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]
    # Exchange interactions:
    exchange = np.asarray([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]]) # = sx*sx+sy*sy = 1/2*(sp*sm+sm*sp)
    full_spin = mszsz+exchange # = S*S = sx*sx + sy*sy + sz*sz
    bond_color = [1, 2, 1, 2, 2, 3] # J mszsz; J' mszsz; J exchange; J' exchange; h_z applied to all vertices <=> adding h_z to all SS-bonds (J-bonds have color 1; J'-bonds have color 2; number 3 is an auxiliary color in case of no PBC)
    def __init__(self) -> None:
        pass
    
    scale = 2 # = 2/hbar; this is normalization factor due to S = hbar/2 * σ
    mag_field_z = scale * np.asarray([[2,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,-2]]) # = sz*1 + 1*sz
    
    """ Returns a list of operators compatible with the bond_color: J mszsz; J' mszsz; J exchange; J' exchange; h_z; h_z. """
    def bond_operator(self, jexch1=1, jexch2=1, use_MSR=False, h_z = 0):
        sign = -1 if use_MSR else 1
        return [(jexch1 * self.mszsz).tolist(),(jexch2 * self.mszsz).tolist(),(sign*jexch1 * self.exchange).tolist(),(jexch2 * self.exchange).tolist(), (h_z * self.mag_field_z).tolist(), (h_z * self.mag_field_z).tolist(),]

""" Calculates the parity of the given permutation (number of swaps needed). Returns 1 or -1. """
def permutation_sign(permutation):
    length = len(permutation)
    count = 0
    for i in range(length):
        for j in range(i,length):
            if permutation[i]>permutation[j]:
                count +=1
    return -1 if count%2 else 1

""" Writes the log including values of energies and order parameters to both screen and a file. 
NOTE: gs_2 is expedted here to have MSR basis. """
def log_results(JEXCH1,gs_1,gs_2,ops,samples,iters,exact_energy,steps_until_convergence,filename=None):
    if ops.hilbert.size > 30: # If the system is too large, AF order parameter tends to fail due to memory overflow. The method m_sSquared_slow addresses this issue.
        m_s2_1, m_s2v_1 = ops.m_sSquared_slow(gs_1)
        m_s2_1, m_s2v_1 = float(m_s2_1.real), float(m_s2v_1)
        m_s2_2, m_s2v_2 = ops.m_sSquared_slow_MSR(gs_2)
        m_s2_2, m_s2v_2 = float(m_s2_2.real), float(m_s2v_2)
    else:
        m_s2_1, m_s2v_1 = gs_1.estimate(ops.m_s2_op).mean.real, gs_1.estimate(ops.m_s2_op).error_of_mean
        m_s2_2, m_s2v_2 = gs_2.estimate(ops.m_s2_op_MSR).mean.real, gs_2.estimate(ops.m_s2_op_MSR).error_of_mean
    m_PS = ops.PS_order_full(gs_1)
    print("{:6.3f} {:10.5f} {:8.5f}  {:10.5f} {:8.5f}  {:8.4f} {:7.4f}  {:7.4f} {:7.4f}  {:7.4f} {:7.4f}  {:8.4f} {:7.4f}  {:7.4f} {:7.4f}  {:7.4f} {:7.4f}  {:10.5f} {:5.0f} {:5.0f} {}".format(
        JEXCH1, 
        gs_1.energy.mean.real,                          gs_1.energy.error_of_mean, 
        gs_2.energy.mean.real,                          gs_2.energy.error_of_mean, 
        # gs_1.estimate(ops.m_z).mean.real,               gs_1.estimate(ops.m_z).error_of_mean, 
        m_PS,    gs_1.estimate(ops.m_plaquette_op).mean.real, # ZMENA 
        gs_1.estimate(ops.m_dimer_op).mean.real,        gs_1.estimate(ops.m_dimer_op).error_of_mean, 
        m_s2_1,                                         m_s2v_1, 
        # gs_2.estimate(ops.m_z).mean.real,               gs_2.estimate(ops.m_z).error_of_mean, 
        0 , 0,  #gs_2.estimate(ops.m_plaquette_op_MSR).mean.real,gs_2.estimate(ops.m_plaquette_op_MSR).error_of_mean, # ZMENA
        gs_2.estimate(ops.m_dimer_op).mean.real,        gs_2.estimate(ops.m_dimer_op).error_of_mean, 
        m_s2_2,                                         m_s2v_2, 
        exact_energy, samples, iters, str(steps_until_convergence)[1:-1]))
    if filename is not None:
        file = open(filename, "a")
        print("{:6.3f} {:10.5f} {:8.5f}  {:10.5f} {:8.5f}  {:8.4f} {:6.4f}  {:7.4f} {:7.4f}  {:7.4f} {:7.4f}  {:7.4f} {:7.4f}  {:7.4f} {:7.4f}  {:7.4f} {:7.4f}  {:10.5f} {:5.0f} {:5.0f} {}".format(
            JEXCH1, 
            gs_1.energy.mean.real,                          gs_1.energy.error_of_mean, 
            gs_2.energy.mean.real,                          gs_2.energy.error_of_mean, 
            # gs_1.estimate(ops.m_z).mean.real,               gs_1.estimate(ops.m_z).error_of_mean, 
            m_PS,    gs_1.estimate(ops.m_plaquette_op).mean.real, # ZMENA
            gs_1.estimate(ops.m_dimer_op).mean.real,        gs_1.estimate(ops.m_dimer_op).error_of_mean, 
            m_s2_1,                                         m_s2v_1, 
            # gs_2.estimate(ops.m_z).mean.real,               gs_2.estimate(ops.m_z).error_of_mean, 
            0 , 0, #gs_2.estimate(ops.m_plaquette_op_MSR).mean.real,gs_2.estimate(ops.m_plaquette_op_MSR).error_of_mean, # ZMENA
            gs_2.estimate(ops.m_dimer_op).mean.real,        gs_2.estimate(ops.m_dimer_op).error_of_mean, 
            m_s2_2,                                         m_s2v_2, 
            exact_energy, samples, iters, str(steps_until_convergence)[1:-1]),file=file)
        file.close()


import netket.nn as nknn
import flax.linen as nn
import jax
import jax.numpy as jnp
""" Implementation of Jastrow ansatz with visible biases (Jastrow+b). """
class Jastrow_b(nknn.Module):
    @nknn.compact
    def __call__(self, x):
        x = jnp.atleast_2d(x)
        return jax.vmap(self.single_evaluate, in_axes=(0))(x)

    def single_evaluate(self, x):
        v_bias = self.param(
            "visible_bias", nn.initializers.normal(), (x.shape[-1],), complex
        )

        J = self.param(
            "kernel", nn.initializers.normal(), (x.shape[-1],x.shape[-1]), complex
        )

        return x.T@J@x + jnp.dot(x, v_bias)