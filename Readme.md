## Introduction
This web UI is for simple computational chemistry calculations with [Gaussian 16](https://gaussian.com/):

* Single-Point Calculation

* Geometry Optimization

* Frequency Analysis

* Absorption/Emission Spectrum Prediction

* NMR Prediction


## Installation  (Linux only)
- Clone this repo: Open terminal

```
git clone https://github.com/phatdatnguyen/gaussian-webui
```

- Create and activate virtual environment:

```
cd gaussian-webui
python3 -m venv gaussian-env
source gaussian-env/bin/activate
```

- Install packages:

```
pip install psutil
pip install rdkit
pip install cclib
pip install plotly
pip install gradio
pip install gradio_molecule2d
pip install nglview
```

## Start web UI
To start the web UI:

```
source gaussian-env/bin/activate
python3 webui.py
```