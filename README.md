# Study of Shastry-Sutherland Model using Machine Learning

This reposiroty contains codes for my masters' thesis. I study the properties of various stochastic methods primarly in the [main_notebook.ipynb](main_notebook.ipynb) file.

The file [main.py](main.py) contains the essential code which can be submited to MetaCentrum using [run.qsub](run.qsub). It loads the parameters of the simulation from one of `config_[something].py` files so that multiple different simulations can be submited without conflicts.

The file [lattice_and_ops.py](lattice_and_ops.py) contains auxiliary classes and functions which are called from both [main.py](main.py) and [main_notebook.ipynb](main_notebook.ipynb).

There are a few modified versions of [main](main.py) file:
 - [main-mag.py](main-mag.py) is used to go throught a list of values of magnetic field $h$ instead of coupling constant $J'$
 - [main-models.py](main-models.py) is used to generate a table of convergence steps and accuracies of many different models for two given values of $J'$ and possibly for given values of learning rate
 - suffix `_pre-trained` denotes that the weights from the end of last run is used in the next run while changing the values of $J'$ resp. $h$

 ## submission
 - [run.sh](run.sh) is a scirpt to submit jobs to *metacentrum*
 - [Barbora-{C/G}PU.sh]() submits jobs to Barbora CPU/GPU queue
