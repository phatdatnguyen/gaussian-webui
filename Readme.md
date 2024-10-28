## Introduction
This web UI is for simple computational chemistry calculations with [Gaussian 16](https://gaussian.com/):

* Single-Point Calculation


* Geometry Optimization


* Frequency Analysis


## Installation  (Linux only)
You will need [Anaconda](https://www.anaconda.com/download) for this app.
- Clone this repo: Open terminal

```
git clone https://github.com/phatdatnguyen/gaussian-webui
```

- Create and activate Anaconda environment:

```
cd gaussian-webui
conda create -p ./gaussian-env
conda activate ./gaussian-env
```

- Install packages:

```
conda install conda-forge::rdkit
conda install autode -c conda-forge
pip install cclib
pip install plotly
pip install gradio
pip install gradio_molecule2d
pip install gradio_molecule3d
```

## Start web UI
To start the web UI:

```
conda activate ./gaussian-env
set PYTHONUTF8=1
python webui.py
```