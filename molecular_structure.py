import os
from rdkit import Chem
from rdkit.Chem import AllChem
import gradio as gr
from gradio_molecule2d import molecule2d
from utils import conformer_to_xyz_file

def on_draw_molecule(molecule_editor: molecule2d):
    mol = Chem.MolFromSmiles(molecule_editor)
    if mol is None:
        return ""
    
    return Chem.MolToSmiles(mol, canonical=True)

def on_generate_conformers(input_smiles: gr.Textbox, charge_slider: gr.Slider, multiplicity_dropdown: gr.Dropdown, num_confs: gr.Slider, file_name: gr.Textbox, file_type_dropdown: gr.Dropdown, progress=gr.Progress()):
    try:
        # Generate conformers
        mol = Chem.MolFromSmiles(input_smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=num_confs)
        t = progress.tqdm(range(num_confs), total=num_confs, desc="Generating")
        
        os.makedirs('./conformers', exist_ok=True)
        for conf_id in t:
            # Create a unique file name for each conformer
            conf_file_name = f'{file_name}_{conf_id + 1}'
            conf_file_path = f'./conformers/{conf_file_name}'
            # Write conformers geometry to file
            if file_type_dropdown == 'xyz':
                conf_file_path += '.xyz'
                conformer_to_xyz_file(mol, conf_id, conf_file_path, charge_slider, multiplicity_dropdown)
            elif file_type_dropdown == 'pdb':
                conf_file_path += '.pdb'
                Chem.MolToPDBFile(mol, conf_file_path, confId=conf_id)
            elif file_type_dropdown == 'mol':
                conf_file_path += '.mol'
                Chem.MolToMolFile(mol, conf_file_path, confId=conf_id)
    except Exception as exc:
        gr.Warning(f'Error: {exc}')
        return None
    
    return f'{num_confs} conformers generated and saved.'

def molecular_structure_tab_content():
    with gr.Tab("Molecular Structure") as molecular_structure_tab:
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Accordion("Molecular Structure"):
                    molecule_editor = molecule2d(label="Molecule")
            with gr.Column(scale=1):
                with gr.Accordion("Generate Conformers"):
                    input_smiles_texbox = gr.Textbox(label="SMILES")
                    charge_slider = gr.Slider(label="Charge", value=0, minimum=-2, maximum=2, step=1)
                    multiplicity_dropdown = gr.Dropdown(label="Multiplicity", value=1, choices=[("Singlet", 1), ("Doublet", 2), ("Triplet", 3), ("Quartet", 4), ("Quintet", 5), ("Sextet", 6)])
                    num_confs_slider = gr.Slider(label="Number of conformers", value=1, minimum=1, maximum=100, step=1)
                    file_name_textbox = gr.Textbox(label="File name", value="conformer")
                    file_type_dropdown = gr.Dropdown(label="File type", value="xyz", choices=["xyz", "pdb", "mol"])
                    generate_button = gr.Button(value="Generate")
                    status_markdown = gr.Markdown()

    molecule_editor.change(on_draw_molecule, molecule_editor, input_smiles_texbox)
    generate_button.click(on_generate_conformers, [input_smiles_texbox, charge_slider, multiplicity_dropdown, num_confs_slider, file_name_textbox, file_type_dropdown], status_markdown)
        
    return molecular_structure_tab