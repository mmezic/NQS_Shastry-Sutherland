# Study of Shastry-Sutherland Model using Machine Learning

This repository contains codes for my master thesis. I study the properties of various stochastic methods on different lattices. The primary functioning code can be found in the [main_notebook.ipynb](main_notebook.ipynb) file. All of my code is based on the [NetKet](https://www.netket.org) library.

## Quick start
### Installation
The scripts require Python 3.7 or higher installed on Linux or macOS (and experimentally on Windows). The basic installation can be done by running:
```
pip install --upgrade pip
pip install --upgrade 'netket' 
```
For a more detailed description or a guide to installation on GPU, see the official [NetKet website](https://www.netket.org/get_started/).

### Usage
The most straightforward way is to start with the notebook file [main_notebook.ipynb](main_notebook.ipynb). It contains the definitions of the Shastry-Sutherland Hamiltonian and all types of neural network architectures presented in my thesis. The parameters of the simulation may be specified at the beginning of the notebook. After the simulation is over, it also produces an interactive plot of the energy dependence on the number of iterations. You can see there how well the selected architecture performs. By default, it executes the simulation in both the normal and the MSR basis.

More technical usage of my other scripts is described below.

## File structure
For more advanced usage, i.e., to generate the data that are presented in my thesis, I prepared the following files. These scripts execute multiple (independent or transferred) runs, and after each run, they write the results (final energy, order parameters, ...) to the output file. The output files may be used to generate the same plots as were presented in the thesis.

The basic script is in the [main](main.py) file, and slight modifications of this file to specific use-cases are as follows:
 - [main.py](main.py) generates a dependence on the **coupling constant J**.
 - [main_pre-trained.py](main_pre-trained.py) uses **transfer learning** - the weights at the end of the last run are saved and used in the next run while changing the values of J'.
 - [main-mag.py](main-mag.py) generates a dependence on the **magnetic field h**.
 - [main-mag_pre-trained.py](main-mag_pre-trained.py) uses **transfer learning** while changing the values of $h$.
 - [main_benchmark_table.py](main_benchmark_table.py) is used to generate a **benchmarking table** of convergence steps and accuracies of many different models for two given values of J' and possibly for given values of learning rate.

 I employ two auxiliary files, which I load from almost all of my scripts and notebooks. 

- The file [lattice_and_ops.py](lattice_and_ops.py) contains auxiliary classes and functions. For example, it contains the implementation of the structure of lattice, a method for a quick definition of Shastry-Shutterland Hamiltonian, possibly with a magnetic field or Marshall sign rule, and implementation of the order parameters.
- The file [pRBM.py](pRBM.py) contains a slight modification of the source code from the `NetKet` implementation of the G-CNN to also include a visible bias. This file is only needed when using this model (I abbreviate it as *pRBM*).


## Submitting to a cluster
The file [main.py](main.py) and its modifications contain code that can be submitted to a computational cluster. It loads the parameters of the simulation from one of the configuration files so that multiple different simulations can be submitted without conflicts. The name of the config file may be given as an argument during submission, for example:
```
python3 main.py -f config
```
will load the simulation parameters from the [config.py](config.py) file. 

I use three different clusters with different scripts for submission:
- For [MetaCentrum](https://metavo.metacentrum.cz/), I use [run_MetaCentrum.qsub](run_MetaCentrum.qsub).
- For cluster Barbora from [IT4I](https://www.it4i.cz/), I use [run_Barbora_CPU.sh](run_Barbora_CPU.sh) or [run_Barbora_GPU.sh](run_Barbora_GPU.sh).
    - the GPU is worth only for larger systems (at least 40 spins)
- For our local server at KFES, I run the jobs directly from the terminal since there is no scheduler.
