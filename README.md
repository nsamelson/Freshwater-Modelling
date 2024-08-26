# Freshwater Hybrid Modelling
#### Variational Graph Auto-Encoder to embed Latex equations

This repository contains the code to preprocess, train and test Latex equations with a VGAE.

## Environment Setup

1. Clone the project
2. Create a `dataset` and `trained_models` within the root folder
3. Setup a virtual environment in the root folder called venv (more on that [here](https://docs.python.org/3/library/venv.html)):
    - Download python version 3.10.10
    - Create a new environment with `python -m venv venv`.
    - Activate the virtual environment with the command: `source venv/bin/activate`.
    - Make sure the Python version is 3.10.10 with `python -V`.
4. Install the librairies with `pip install -r requirements.txt`.

## Code Architecture

The code is split into 4 different folders:
1. **Node** contains the API to transform Latex equations into MathML 
2. **Preprocessing** takes care of processing the dataset of equations and build a big XML file, a vocabulary and the Graph Dataset
3. **Models** involves the files to train, do hyperparameter search, and also the VGAE model
4. **Utils** contains the code to plot, save, and extract experiment data

All these files can be called through `main.py` in the root folder.
The training parameters are held within the `config.py` file.

## Authors

- Nicolas SAMELSON - https://github.com/nsamelson
