from rdkit import Chem
from rdkit.Chem import AllChem
import gradio as gr
from gradio_molecule2d import molecule2d
from utils import conformer_to_xyz_file

def on_generate_conformers(molecule_editor: molecule2d, charge_slider: gr.Slider, multiplicity_dropdown: gr.Dropdown, file_name: gr.Textbox, num_confs: gr.Slider, progress=gr.Progress()):
    # Generate conformers
    mol = Chem.MolFromSmiles(molecule_editor)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=num_confs)
    t = progress.tqdm(range(num_confs), total=num_confs, desc="Generating")

    for conf_id in t:
        # Create a unique file name for each conformer
        conf_file_name = f'{file_name}_{conf_id + 1}.xyz'
        # Write conformers geometry to xyz file
        conformer_to_xyz_file(mol, conf_id, conf_file_name, charge_slider, multiplicity_dropdown)

    return f'{num_confs} conformers generated and saved.'

with gr.Blocks(css='styles.css') as app:
    with gr.Row():
        with gr.Column(scale=1):
            molecule_editor = molecule2d(label="Molecule")
            charge_slider = gr.Slider(label="Charge", value=0, minimum=-2, maximum=2, step=1)
            multiplicity_dropdown = gr.Dropdown(label="Multiplicity", value=1, choices=[("Singlet", 1), ("Doublet", 2),
                                                                                                 ("Triplet", 3), ("Quartet", 4),
                                                                                                 ("Quintet", 5), ("Sextet ", 6)])
        with gr.Column(scale=1):
            file_name_textbox = gr.Textbox(label="File name", value="conformer")
            num_confs_slider = gr.Slider(label="Number of conformers", value=30, minimum=1, maximum=50, step=1)
            generate_button = gr.Button(value="Generate")
            status_markdown = gr.Markdown()

    generate_button.click(on_generate_conformers, [molecule_editor, charge_slider, multiplicity_dropdown, file_name_textbox, num_confs_slider], status_markdown)


app.launch()
