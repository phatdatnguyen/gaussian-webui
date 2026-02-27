import os
from rdkit import Chem
from rdkit.Chem import AllChem
import gradio as gr
from gradio_molecule2d import molecule2d
from utils import get_files_in_working_directory, conformer_to_xyz_file

def on_draw_molecule(molecule_editor):
    mol = Chem.MolFromSmiles(molecule_editor)
    if mol is None:
        return ""
    
    return Chem.MolToSmiles(mol, canonical=True)

def on_generate_conformers(working_directory_path, input_smiles, charge, multiplicity, num_confs, file_name, file_type, progress=gr.Progress()):
    try:
        # Generate conformers
        mol = Chem.MolFromSmiles(input_smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=num_confs)
        
        for conf_id in progress.tqdm(range(num_confs), total=num_confs, desc="Generating"):
            # Create a unique file name for each conformer
            conf_file_path = os.path.join(working_directory_path, f'{file_name}_{conf_id + 1}')
            # Write conformers geometry to file
            if file_type == 'xyz':
                conf_file_path += '.xyz'
                conformer_to_xyz_file(mol, conf_id, conf_file_path, charge, multiplicity)
            elif file_type == 'pdb':
                conf_file_path += '.pdb'
                Chem.MolToPDBFile(mol, conf_file_path, confId=conf_id)
            else: # file_type_dropdown == 'mol'
                conf_file_path += '.mol'
                Chem.MolToMolFile(mol, conf_file_path, confId=conf_id)

        status = 'Conformers generated.'
        return f"<span style='color:green;'>{status}</span>", get_files_in_working_directory(working_directory_path)
    except Exception as exc:
        status = f'Error generating conformers: {exc}'
        return f"<span style='color:red;'>{status}</span>", get_files_in_working_directory(working_directory_path)

def show_selected_file(selected_file):
    gr.Warning(selected_file)
    return selected_file

def conformer_generation_tab_content(working_directory_path_state, working_directory_file_list_state, status_markdown):
    with gr.Tab("Conformer generation") as conformer_generation_tab:
        with gr.Row():
            with gr.Column(scale=2):
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
    generate_button.click(on_generate_conformers, [working_directory_path_state, input_smiles_texbox, charge_slider, multiplicity_dropdown, num_confs_slider, file_name_textbox, file_type_dropdown], [status_markdown, working_directory_file_list_state])
        
    return conformer_generation_tab