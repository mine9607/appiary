# Appiary General Notes:

## Environment Setup

Note: This project uses a pre-defined conda env to enable GPU utilization (tensorflow)

1.  Install conda (miniconda) if not yet installed: https://docs.anaconda.com/miniconda/install/
2.  Create a conda environment: `conda create --name <env_name>` (installs latest version of python)
3.  List all conda environments: `conda env list`
4.  Activate the conda environment: `conda activate <env_name>`
5.  Install the necessary dependencies: `conda install conda-forge <package_name>`
6.  List all dependencies in an activated conda environment: `conda list` or `conda list -n <env_name>`

