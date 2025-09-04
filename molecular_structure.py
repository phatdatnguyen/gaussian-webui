import gradio as gr
from gradio_molecule2d import molecule2d

def molecular_structure_tab_content():
    with gr.Tab("Molecular Structure") as molecular_structure_tab:
        molecule2d(label="Molecule")    
        
    return molecular_structure_tab