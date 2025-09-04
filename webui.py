import os
import glob
import gradio as gr
from molecular_structure import molecular_structure_tab_content
from single_point_calculation import single_point_calculation_tab_content
from geometry_optimization import geometry_optimization_tab_content
from frequency_analysis import frequency_analysis_tab_content
from uv_vis_prediction import uv_vis_prediction_tab_content
from fluorescence_prediction import fluorescence_prediction_tab_content
from nmr_prediction import nmr_prediction_tab_content
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
import socket

# clean up the directory
for filepath in glob.iglob('./*.log'):
    os.remove(filepath)
for filepath in glob.iglob('./static/**/*.cube', recursive=True):
    os.remove(filepath)
for filepath in glob.iglob('./static/**/*.xyz', recursive=True):
    os.remove(filepath)
for filepath in glob.iglob('./static/*.html'):
    os.remove(filepath)
for filepath in glob.iglob('./static/**/*.html', recursive=True):
    os.remove(filepath)

# create a FastAPI app
app = FastAPI()

# create a static directory to store the static files
static_dir = Path('./static')
static_dir.mkdir(parents=True, exist_ok=True)

# mount FastAPI StaticFiles server
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# function to find an available port
def find_available_port(start_port=7860):
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port  # Available port found
            except OSError:
                port += 1  # Try next port

available_port = find_available_port()

with gr.Blocks(css='styles.css') as blocks:
    with gr.Tabs() as tabs:
        molecular_structure_tab_content()
        single_point_calculation_tab_content()
        geometry_optimization_tab_content()
        frequency_analysis_tab_content()
        uv_vis_prediction_tab_content()
        fluorescence_prediction_tab_content()
        nmr_prediction_tab_content()

# mount Gradio app to FastAPI app
app = gr.mount_gradio_app(app, blocks, path="/")

# serve the app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=available_port)
