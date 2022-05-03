# Study of Shastry-Sutherland Model using Machine Learning

This repository contains codes for my master thesis. I study the properties of various stochastic methods on different lattices. The primary functioning code can be found in the [main_notebook.ipynb](main_notebook.ipynb) file. All of my code is based on the [NetKet](https://www.netket.org) library.

## Submitting to a cluster
The file [main.py](main.py) contains an essential code that can be submitted to a computational cluster. It loads the parameters of the simulation from one of `config_[something].py` files so that multiple different simulations can be submitted without conflicts. The name of the config file may be given as an argument during submission, for example:
```
python3 main.py -f config_RBM16
```
I use three different clusters with different scripts for submission:
- For [MetaCentrum](https://metavo.metacentrum.cz/) I use [run.qsub](run.qsub).
- For cluster Barbora from [IT4I](https://www.it4i.cz/) I use [Barbora-CPU.sh](Barbora-CPU.sh) or [Barbora-GPU.sh](Barbora-GPU.sh).
    - the GPU is worth only for larger systems (at least 64 spins)
- For our local server at KFES, I run the jobs directly from the terminal since there is no scheduler.

## File structure
I have two auxiliary files, which I load from almost all of my scripts and notebooks. The file [lattice_and_ops.py](lattice_and_ops.py) contains auxiliary classes and functions. For example, it contains the implementation of the structure of lattice, a method for a quick definition of Shastry-Shutterland Hamiltonian, possibly with a magnetic field or Marshall sign rule, and implementation of the order parameters.

The file [pRBM.py](pRBM.py) contains a slight modification of the source code from the `NetKet` implementation of the G-CNN to also include a visible bias. This file is only needed when using this model (I abbreviate it as *pRBM* or *pRBM* in the code).

There are a few modified versions of [main](main.py) file depending on the use case:
 - [main.py](main.py) generates a dependence on the **coupling constant J**.
 - [main_pre-trained.py](main_pre-trained.py) uses **transfer learning** - the weights at the end of the last run are saved and used in the next run while changing the values of J'.
 - [main-mag.py](main-mag.py) generates a dependence on the **magnetic field h**.
 - [main-mag_pre-trained.py](main-mag_pre-trained.py) uses **transfer learning** while changing the values of $h$.
 - [main_benchmark_table.py](main_benchmark_table.py) is used to generate a **benchmarking table** of convergence steps and accuracies of many different models for two given values of J' and possibly for given values of learning rate.

