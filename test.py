import os, sys
# detect MPI rank
from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
# set only one visible device
os.environ["CUDA_VISIBLE_DEVICES"] = f"{rank}"
# force to use gpu
os.environ["JAX_PLATFORM_NAME"] = "gpu"
import jax
import netket as nk
import mpi4jax
# print(jax.lib.xla_bridge.get_backend().platform)
print("NetKet version: {}".format(nk.__version__))
print("MPI utils available: {}".format(nk.utils.mpi.available))
print("Jax version: {}".format(jax.__version__))
print("Jax devices: {}".format(jax.devices()))
