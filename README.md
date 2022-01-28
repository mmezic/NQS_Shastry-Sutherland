# Study of Shastry-Sutherland Model using Machine Learning

This reposiroty contains codes for my masters' thesis. I study the properties of various stochastic methods primarly in the [main_notebook.ipynb](main_notebook.ipynb) file.

The file [main.py](main.py) contains the essential code which can be submited to MetaCentrum using [run.qsub](run.qsub). It loads the parameters of the simulation from one of `config___.py` files so that multiple different simulations can be submited without conflicts.

The file [lattice_and_ops.py](lattice_and_ops.py) contains auxiliary classes and functions which are called from both [main.py](main.py) and [main_notebook.ipynb](main_notebook.ipynb).
