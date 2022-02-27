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
        elif self.SITES == 20: #tile shape definition	
            self.indent = [3,1,0,1,1,2]	
            self.width = [1,4,5,5,4,1]	
            self.right_shift = 2 #vertical shift of the center of right cell (in the upward direction)	
            self.bottom_shift = 4 #horizontal shift of the center of bottom cell (in the left direction)
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
            raise Exception("Invalid number of self.SITES given.")
        self.N = sum(self.width) #number of nodes

        assert self.N == self.SITES, "Error! Lattice is given wrongly. The checksum of sites failed."

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


        # i j-->
        # | .   .   .   0  
        # V .   1   2   3   4
        #   5   6   7   8   9
        #   .   10  11  12  13  14
        #   .   15  16  17  18
        #   .   .   19

    def position(self,node): #returns positional indices i,j of the node
        row, n = 0, 0
        while n+self.width[row] <= node:
            n += self.width[row]
            row += 1
        column = self.indent[row] + node - n 
        return row, column
    def index_n(self,row, column): #returns index n given positional indices
        return sum(self.width[0:row]) + column - self.indent[row]
    def is_last(self,node):
        row, column = self.position(node)
        return (column == self.width[row] + self.indent[row] - 1)
    def is_first(self,node):
        row, column = self.position(node)
        return (column == self.indent[row])
    def is_lowest(self,node):
        row, column = self.position(node)
        if row == len(self.width) - 1:
            return True
        else:
            row += 1
            if column >= self.indent[row] and column < self.indent[row] + self.width[row]:
                return False
            else:
                return True
    def rt(self, node): #returns index n of right neighbour
        if self.is_last(node):
            new_row = (self.position(node)[0] + self.right_shift)%(len(self.width)+self.vertical_gap)
            if new_row == len(self.width): # special case of gap
                new_row -= self.right_shift
            return sum(self.width[0:new_row])
        else:
            return (node+1)%self.N
    def lft(self, node): #returns index n of left neighbour
        if self.is_first(node):
            new_row = (self.position(node)[0] + self.left_shift)%(len(self.width)+self.vertical_gap)
            if new_row == len(self.width):
                new_row -= self.left_shift
            return sum(self.width[0:new_row])+self.width[new_row]-1
        else:
            return (node-1)%self.N
    def bot(self, node): #returns index n of bottom neighbour
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
Defines order parameter operators for a given lattice. Both MSR and non-MSR operators are defined.
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
                self.m_s2_op_MSR += self.SS_MSR(i,j) * (-1)**np.sum(self.lattice.position(i)+self.lattice.position(j)) #(i+j) #chyba?
        self.m_s2_op_MSR = self.m_s2_op_MSR/(M**2)

        self.m_s2_op = 0
        for i in range(M):
            for j in range(M):
                self.m_s2_op += self.SS(i,j) * (-1)**np.sum(self.lattice.position(i)+self.lattice.position(j)) #(i+j) #chyba?
        self.m_s2_op = self.m_s2_op/(M**2)

        

    def SS_old(self,i,j): # S_i * S_j
        return 2*nk.operator.spin.sigmap(self.hilbert,i,DTYPE="float64")@nk.operator.spin.sigmam(self.hilbert,j,DTYPE="float64")+2*nk.operator.spin.sigmam(self.hilbert,i,DTYPE="float64")@nk.operator.spin.sigmap(self.self.hilbert,j,DTYPE="float64")+nk.operator.spin.sigmaz(self.hilbert,i,DTYPE="float64")@nk.operator.spin.sigmaz(self.hilbert,j,DTYPE="float64")
        #return nk.operator.spin.sigmax(self.hilbert, i)@nk.operator.spin.sigmax(self.hilbert, j) + nk.operator.spin.sigmay(self.hilbert, i)@nk.operator.spin.sigmay(self.hilbert, j) + nk.operator.spin.sigmaz(self.hilbert, i)@nk.operator.spin.sigmaz(self.hilbert, j)

    def SS(self,i,j): #different method of definition
        if i==j:
            return nk.operator.LocalOperator(self.hilbert,operators=[[3,0],[0,3]],acting_on=[i])
        else:
            return nk.operator.LocalOperator(self.hilbert,operators=(self.mszsz+self.exchange),acting_on=[i,j])

    def SS_MSR(self,i,j): #different method of definition
        if i==j:
            return nk.operator.LocalOperator(self.hilbert,operators=[[3,0],[0,3]],acting_on=[i])
        elif (np.sum(self.lattice.position(i))+np.sum(self.lattice.position(j)))%2 == 0:            # same sublattice 
            return nk.operator.LocalOperator(self.hilbert,operators=(self.mszsz+self.exchange),acting_on=[i,j])
        else:                                                                                       # different sublattice 
            return nk.operator.LocalOperator(self.hilbert,operators=(self.mszsz-self.exchange),acting_on=[i,j])
    

    def P(self,i,j,msr=False): # two particle permutation operator
        if msr == False:
            return .5*(self.SS(i,j)+nk.operator.LocalOperator(self.hilbert,operators=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],acting_on=[i,j]))
        else:
            return .5*(self.SS_MSR(i,j)+nk.operator.LocalOperator(self.hilbert,operators=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],acting_on=[i,j]))


    def P_cykl(self,i,j,k,l,msr): # 4 particle cyclic permutation operator
        return self.P(i,k,msr)@self.P(k,l,msr)@self.P(i,j,msr)
    def P_cykl_inv(self,i,j,k,l,msr): # inverse of 4 particle cyclic permutation operator
        return self.P(j,l,msr)@self.P(k,l,msr)@self.P(i,j,msr)

    def P_r(self,i,msr): # cyclic permutation of 4 fq.SITES located at position i
        return self.P_cykl(i,self.lattice.rt(i),self.lattice.lrt(i),self.lattice.bot(i),msr)
        # i --> i j  .... we assigne a lrt cell to each index i
        #       l k
    def P_r_inv(self,i,msr): # inverse of cyclic permutation of 4 fq.SITES located at position i
        return self.P_cykl_inv(i,self.lattice.rt(i),self.lattice.lrt(i),self.lattice.bot(i),msr)

    def Q_r(self,i,msr):
        return .5*(self.P_r(i,msr) + self.P_r_inv(i,msr))


    def m_sSquared_slow(self,state):
        ss_operator = 0
        M = self.hilbert.size
        m_s2 = 0
        for i in range(M):
            for j in range(M):
                ss_operator += self.SS(i,j) * (-1)**np.sum(self.lattice.position(i)+self.lattice.position(j))
            if i%3==2 or i==(M-1):
                m_s2 += (state.transpose()@(ss_operator@state))[0,0]
                ss_operator = 0
        m_s2 = m_s2/M**2
        return m_s2

"""
Class containing auxiliary operators which helps with defining a hamiltonian.
"""
class HamOps:
    #Sigma^z*Sigma^z interactions
    sigmaz = [[1, 0], [0, -1]]
    mszsz = (np.kron(sigmaz, sigmaz)) # = sz*sz = [[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]
    #Exchange interactions
    exchange = np.asarray([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]]) # = sx*sx+sy*sy = 1/2*(sp*sm+sm*sp)
    full_spin = mszsz+exchange # = S*S = sx*sx + sy*sy + sz*sz
    bond_color = [1, 2, 1, 2, 2] # J1 mszsz; J2 mszsz; J1 exchange; J2 exchange; h_z applied to all vertices <=> adding h_z to all SS-bonds
    def __init__(self) -> None:
        pass
    
    scale = 2 # = 2/hbar normalization factor thans to S = hbar/2 * σ
    mag_field_z = scale * np.asarray([[2,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,-2]]) # = sz*1 + 1*sz

    def bond_operator(self, jexch1=1, jexch2=1, use_MSR=False, h_z = 0):
        sign = -1 if use_MSR else 1
        return [(jexch1 * self.mszsz).tolist(),(jexch2 * self.mszsz).tolist(),(sign*jexch1 * self.exchange).tolist(),(jexch2 * self.exchange).tolist(), (h_z * self.mag_field_z).tolist(),]

"""
Calculates the number of swaps forming the given permutation. Returns 1 or -1.
"""
def permutation_sign(permutation):
    length = len(permutation)
    count = 0
    for i in range(length):
        for j in range(i,length):
            if permutation[i]>permutation[j]:
                count +=1
    return -1 if count%2 else 1


def log_results(JEXCH1,gs_1,gs_2,ops,samples,iters,steps_until_convergence,filename=None):
    print("{:6.3f} {:10.5f} {:8.5f}  {:10.5f} {:8.5f}  {:7.4f} {:7.4f}  {:7.4f} {:7.4f}  {:7.4f} {:7.4f}  {:7.4f} {:7.4f}  {:7.4f} {:7.4f}  {:7.4f} {:7.4f}  {:5.0f} {:5.0f} {}".format(
        JEXCH1, 
        gs_1.energy.mean.real,                          gs_1.energy.variance, 
        gs_2.energy.mean.real,                          gs_2.energy.variance, 
        gs_1.estimate(ops.m_z).mean.real,               gs_1.estimate(ops.m_z).variance, 
        gs_1.estimate(ops.m_dimer_op).mean.real,        gs_1.estimate(ops.m_dimer_op).variance, 
        gs_1.estimate(ops.m_s2_op).mean.real,           gs_1.estimate(ops.m_s2_op).variance, 
        # gs_1.estimate(ops.m_plaquette_op).mean.real,    gs_1.estimate(ops.m_plaquette_op).variance, 
        gs_2.estimate(ops.m_z).mean.real,               gs_2.estimate(ops.m_z).variance, 
        gs_2.estimate(ops.m_dimer_op).mean.real,        gs_2.estimate(ops.m_dimer_op).variance, 
        gs_2.estimate(ops.m_s2_op_MSR).mean.real,       gs_2.estimate(ops.m_s2_op_MSR).variance, 
        # gs_2.estimate(ops.m_plaquette_op_MSR).mean.real,gs_2.estimate(ops.m_plaquette_op_MSR).variance, 
        samples, iters, str(steps_until_convergence)[1:-1]))
    if filename is not None:
        file = open(filename, "a")
        print("{:6.3f} {:10.5f} {:8.5f}  {:10.5f} {:8.5f}  {:7.4f} {:7.4f}  {:7.4f} {:7.4f}  {:7.4f} {:7.4f}  {:7.4f} {:7.4f}  {:7.4f} {:7.4f}  {:7.4f} {:7.4f}  {:5.0f} {:5.0f} {}".format(
            JEXCH1, 
            gs_1.energy.mean.real,                          gs_1.energy.variance, 
            gs_2.energy.mean.real,                          gs_2.energy.variance, 
            gs_1.estimate(ops.m_z).mean.real,               gs_1.estimate(ops.m_z).variance, 
            gs_1.estimate(ops.m_dimer_op).mean.real,        gs_1.estimate(ops.m_dimer_op).variance, 
            gs_1.estimate(ops.m_s2_op).mean.real,           gs_1.estimate(ops.m_s2_op).variance, 
            # gs_1.estimate(ops.m_plaquette_op).mean.real,    gs_1.estimate(ops.m_plaquette_op).variance, 
            gs_2.estimate(ops.m_z).mean.real,               gs_2.estimate(ops.m_z).variance, 
            gs_2.estimate(ops.m_dimer_op).mean.real,        gs_2.estimate(ops.m_dimer_op).variance, 
            gs_2.estimate(ops.m_s2_op_MSR).mean.real,       gs_2.estimate(ops.m_s2_op_MSR).variance, 
            # gs_2.estimate(ops.m_plaquette_op_MSR).mean.real,gs_2.estimate(ops.m_plaquette_op_MSR).variance, 
            samples, iters, str(steps_until_convergence)[1:-1]),file=file)
        file.close()


import netket.nn as nknn
import flax.linen as nn
import jax
import jax.numpy as jnp
class Jastrow(nknn.Module):
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