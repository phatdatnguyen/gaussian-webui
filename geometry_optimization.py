import os
import multiprocessing
import psutil
import subprocess
import time
import math
import gradio as gr
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from rdkit import Chem
from rdkit.Chem import AllChem
import cclib
import nglview
from utils import *

max_n_procs = multiprocessing.cpu_count()
max_memory = math.floor(psutil.virtual_memory().total / (1024 ** 3))

def on_create_molecule(input_smiles_textbox: gr.Textbox):
    os.makedirs("structures", exist_ok=True)
    file_path = "./structures/molecule_opt.pdb"
    try:
        global mol_opt
        mol_opt = Chem.MolFromSmiles(input_smiles_textbox)
        mol_opt = Chem.AddHs(mol_opt)
        smiles = Chem.CanonSmiles(input_smiles_textbox)
        AllChem.EmbedMolecule(mol_opt)
        Chem.MolToPDBFile(mol_opt, file_path)

        # Create the NGL view widget
        view = nglview.show_rdkit(mol_opt)
        
        # Write the widget to HTML
        if os.path.exists('./static/molecule_opt.html'):
            os.remove('./static/molecule_opt.html')
        nglview.write_html('./static/molecule_opt.html', [view])

        # Read the HTML file
        timestamp = int(time.time())
        html = f'<iframe src="/static/molecule_opt.html?ts={timestamp}" height="300" width="400" title="NGL View"></iframe>'
    except Exception as exc:
        gr.Warning("Error!\n" + str(exc)) 
        return None, None, gr.update(interactive=False)
    
    return smiles, html, gr.update(interactive=True)

def on_upload_molecule(load_molecule_uploadbutton: gr.UploadButton):
    uploaded_file_path = load_molecule_uploadbutton
    _, file_extension = os.path.splitext(uploaded_file_path)

    try:
        global mol_opt

        if file_extension.lower() == ".pdb":
            mol_opt = Chem.MolFromPDBFile(uploaded_file_path, sanitize=False, removeHs=False)
        elif file_extension.lower() == ".xyz":    
            mol_opt = mol_from_xyz_file(uploaded_file_path)
        elif file_extension.lower() == ".mol":    
            mol_opt = Chem.MolFromMolFile(uploaded_file_path, sanitize=False, removeHs=False)
        elif file_extension.lower() == ".mol2":    
            mol_opt = Chem.MolFromMol2File(uploaded_file_path, sanitize=False, removeHs=False)
        elif file_extension.lower() == ".log":    
            mol_opt = mol_from_gaussian_file(uploaded_file_path)
        else:
            raise Exception("File must be in supported formats (pdb, xyz, mol, mol2, log).")
        Chem.SanitizeMol(mol_opt)

        if file_extension.lower() == ".xyz" or file_extension.lower() == ".log":
            smiles = "(No bonding information)"
        else:
            smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol_opt))    
        if mol_opt.GetNumConformers()==0:
            AllChem.EmbedMolecule(mol_opt)

        # Create the NGL view widget
        view = nglview.show_rdkit(mol_opt)
        
        # Write the widget to HTML
        if os.path.exists('./static/molecule_opt.html'):
            os.remove('./static/molecule_opt.html')
        nglview.write_html('./static/molecule_opt.html', [view])

        # Read the HTML file
        timestamp = int(time.time())
        html = f'<iframe src="/static/molecule_opt.html?ts={timestamp}" height="300" width="400" title="NGL View"></iframe>'
    except Exception as exc:
        gr.Warning("Error!\n" + str(exc))
        return None, None, gr.update(interactive=False), None, None, None, None, None, gr.update(interactive=False), None, None
    
    # Read calculation results if the uploaded file is a Gaussian log file
    if file_extension.lower() == ".log":
        try:
            # Get results
            gaussian_mol = mol_from_gaussian_file(uploaded_file_path)
            
            # Create the NGL view widget
            view = nglview.show_rdkit(gaussian_mol)
            
            # Write the widget to HTML
            if os.path.exists('./static/molecule_opt_gaussian_output.html'):
                os.remove('./static/molecule_opt_gaussian_output.html')
            nglview.write_html('./static/molecule_opt_gaussian_output.html', [view])

            # Read the HTML file
            timestamp = int(time.time())
            html = f'<iframe src="/static/molecule_opt_gaussian_output.html?ts={timestamp}" height="300" width="400" title="NGL View"></iframe>'

            parser = cclib.io.ccopen(uploaded_file_path)
            data = parser.parse()
            energy = data.scfenergies[-1] / 27.2114
            energy_textbox = '{:.4f} (hartree)'.format(energy)
            dipole_moment = data.moments[1]
            dipole_magnitude = (dipole_moment[0]**2 + dipole_moment[1]**2 + dipole_moment[2]**2) ** 0.5
            dipole_moment_textbox = "{:.4f} (Debye)".format(dipole_magnitude)

            scf_energies = [x / 27.2114 for x in data.scfenergies]
            energy_plot = plt.figure()
            plt.plot(range(1, len(scf_energies) + 1), scf_energies, 'k')
            plt.xlabel("Conformer")
            plt.ylabel("Energy (hartree)")
            plt.title("SCF Energy")
            ax = plt.gca()
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            plt.tight_layout()

            MO_energies = data.moenergies
            visualization_dropdown_choices=["Electron density", "Electrostatic potential"]
            MO_df = pd.DataFrame(columns=["Molecular orbital", "Energy (hartree)"])
            for i, MO_energy in enumerate(MO_energies[0]):
                if i in data.homos:
                    MO_df = MO_df._append({"Molecular orbital": f"MO {i+1} (HOMO)", "Energy (hartree)": "{:.4f}".format(MO_energy)}, ignore_index=True)
                else:
                    MO_df = MO_df._append({"Molecular orbital": f"MO {i+1}", "Energy (hartree)": "{:.4f}".format(MO_energy)}, ignore_index=True)
                MO_name = f"MO {i+1}"
                visualization_dropdown_choices.append(MO_name)
            visualization_dropdown = gr.Dropdown(label="Visualization", value="Electron density", choices=visualization_dropdown_choices)
            
            return smiles, html, gr.update(interactive=True), energy_textbox, dipole_moment_textbox, energy_plot, html, visualization_dropdown, gr.update(interactive=True), "", MO_df
        except:
            return smiles, html, gr.update(interactive=True), None, None, None, None, None, gr.update(interactive=False), None, None
    else:
        return smiles, html, gr.update(interactive=True), None, None, None, None, None, gr.update(interactive=False), None, None

def on_mm_checkbox_change(mm_checkbox: gr.Checkbox):
    return gr.update(visible=mm_checkbox), gr.update(visible=mm_checkbox)

def on_solvation_checkbox_change(solvation_checkbox: gr.Checkbox):
    return gr.update(visible=solvation_checkbox), gr.update(visible=solvation_checkbox)

def write_opt_gaussian_input(mol, file_name, method='B3LYP', basis='6-31G(d,p)', charge=0, multiplicity=1, solvation=None, solvent=None, n_proc=4, memory=2):
    # Open the file for writing
    with open(file_name + '.gjf', 'w') as f:
        # Link0 commands
        f.write(f'%NProcShared={n_proc}\n')
        f.write(f'%Mem={memory}GB\n')
        f.write(f'%Chk={file_name}.chk\n')
        # Route section
        route_section = f'#P {method}/{basis} Opt'
        if solvation:
            route_section += f' scrf=({solvation},solvent={solvent})'
        route_section += '\n\n'
        f.write(route_section)
        # Title section
        f.write('Geometry Optimization\n\n')
        # Charge and multiplicity
        f.write(f'{charge} {multiplicity}\n')
        # Atomic coordinates
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()
            pos = conf.GetAtomPosition(idx)
            f.write(f'{symbol:<2} {pos.x:>12.6f} {pos.y:>12.6f} {pos.z:>12.6f}\n')
        f.write('\n')  # Blank line to end the molecule specification

    print(f"Gaussian input file '{file_name}' has been written.")

def on_optimize_geometry(mm_checkbox: gr.Checkbox, force_field_dropdown: gr.Dropdown, max_iters_slider: gr.Slider, solvation_checkbox: gr.Checkbox, solvation_dropdown: gr.Dropdown, solvent_dropdown: gr.Dropdown,
                         functional_textbox: gr.Dropdown, basis_set_textbox: gr.Dropdown, charge_slider: gr.Slider, multiplicity_dropdown: gr.Dropdown,
                         n_cores_slider: gr.Slider, memory_slider: gr.Slider, file_name_textbox: gr.Textbox):
    try:
        if mm_checkbox:
            if force_field_dropdown=="MMFF":
                AllChem.MMFFOptimizeMolecule(mol_opt, maxIters=max_iters_slider)
            else:
                AllChem.UFFOptimizeMolecule(mol_opt, maxIters=max_iters_slider)

        # Write Gaussian input file
        if solvation_checkbox:
            solvation = solvation_dropdown
        else:
            solvation = None
        write_opt_gaussian_input(mol_opt, file_name=file_name_textbox, method=functional_textbox, basis=basis_set_textbox, charge=charge_slider, multiplicity=multiplicity_dropdown, solvation=solvation, solvent=solvent_dropdown, n_proc=n_cores_slider, memory=memory_slider)

        # Run calculation
        start = time.time()
        subprocess.run(['g16', file_name_textbox + '.gjf'], check=True)
        print(f"Gaussian job '{file_name_textbox + '.gjf'}' has been submitted.")
        end = time.time()
        duration = end - start

        # Get results
        gaussian_mol = mol_from_gaussian_file(f'{file_name_textbox}.log')
        # Create the NGL view widget
        view = nglview.show_rdkit(gaussian_mol)
        
        # Write the widget to HTML
        if os.path.exists('./static/molecule_opt_gaussian_output.html'):
            os.remove('./static/molecule_opt_gaussian_output.html')
        nglview.write_html('./static/molecule_opt_gaussian_output.html', [view])

        # Read the HTML file
        timestamp = int(time.time())
        html = f'<iframe src="/static/molecule_opt_gaussian_output.html?ts={timestamp}" height="300" width="400" title="NGL View"></iframe>'

        parser = cclib.io.ccopen(file_name_textbox + '.log')
        data = parser.parse()
        energy = data.scfenergies[-1] / 27.2114
        energy_textbox = '{:.4f} (hartree)'.format(energy)
        dipole_moment = data.moments[1]
        dipole_magnitude = (dipole_moment[0]**2 + dipole_moment[1]**2 + dipole_moment[2]**2) ** 0.5
        dipole_moment_textbox = "{:.4f} (Debye)".format(dipole_magnitude)

        scf_energies = [x / 27.2114 for x in data.scfenergies]
        energy_plot = plt.figure()
        plt.plot(range(1, len(scf_energies) + 1), scf_energies, 'k')
        plt.xlabel("Conformer")
        plt.ylabel("Energy (hartree)")
        plt.title("SCF Energy")
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        plt.tight_layout()

        MO_energies = data.moenergies
        visualization_dropdown_choices=["Electron density", "Electrostatic potential"]
        MO_df = pd.DataFrame(columns=["Molecular orbital", "Energy (hartree)"])
        for i, MO_energy in enumerate(MO_energies[0]):
            if i in data.homos:
                MO_df = MO_df._append({"Molecular orbital": f"MO {i+1} (HOMO)", "Energy (hartree)": "{:.4f}".format(MO_energy)}, ignore_index=True)
            else:
                MO_df = MO_df._append({"Molecular orbital": f"MO {i+1}", "Energy (hartree)": "{:.4f}".format(MO_energy)}, ignore_index=True)
            MO_name = f"MO {i+1}"
            visualization_dropdown_choices.append(MO_name)
        visualization_dropdown = gr.Dropdown(label="Visualization", value="Electron density", choices=visualization_dropdown_choices)

    except Exception as exc:
        gr.Warning("Calculation error!\n" + str(exc))
        return None, gr.update(visible=False), None, None, None, None, None, gr.update(interactive=False), None, None

    calculation_status = "Calculation finished. ({0:.3f} s)".format(duration)
    output_file_path = f'{file_name_textbox}.log'
    return calculation_status, gr.update(value=output_file_path, visible=True), energy_textbox, dipole_moment_textbox, energy_plot, html, visualization_dropdown, gr.update(interactive=True), "", MO_df

def on_visualization_change(visualization_dropdown: gr.Dropdown):
    if visualization_dropdown == "Electron density":
        return gr.Slider(label="Isolevel", value=0.5, minimum=0, maximum=1, step=0.01, visible=True)
    return gr.Slider(label="Isolevel", value=0.5, minimum=0, maximum=1, step=0.01, visible=False)

def on_visualization(file_name_textbox: gr.Textbox, visualization_dropdown: gr.Dropdown, visualization_color1: gr.ColorPicker, visualization_color2: gr.ColorPicker, visualization_opacity: gr.Slider, visualization_isolevel: gr.Slider):
    # Set options for cube file generation
    try:
        gaussian_mol = mol_from_gaussian_file(f'{file_name_textbox}.log')

        # Convert to formatted checkpoint file
        subprocess.run(['formchk', f'{file_name_textbox}.chk', f'{file_name_textbox}.fchk'], check=True)
        print(f"Checkpoint file has been converted to '{file_name_textbox + '.fchk'}' .")

        # Generate cube file
        os.makedirs("./static/opt_visualization", exist_ok=True)
        if visualization_dropdown == "Electron density":
            subprocess.run(['cubegen', '0', 'Density=SCF', f'{file_name_textbox}' + '.fchk', f'{file_name_textbox}' + '.cube'], check=True)
            print(f"Cube file generation job '{file_name_textbox + '.fchk'}' has been submitted.")

            # Get the cube file
            view = nglview.show_rdkit(gaussian_mol)
            view.add_component(f"./{file_name_textbox + '.cube'}")

            # Adjust visualization settings
            view.component_1.update_surface(opacity=visualization_opacity, color=visualization_color1, isolevel=visualization_isolevel)
            view.camera = 'orthographic'
        elif visualization_dropdown == "Electrostatic potential":
            subprocess.run(['cubegen', '0', 'Potential=SCF', f'{file_name_textbox}' + '.fchk', f'{file_name_textbox}' + '.cube'], check=True)
            print(f"Cube file generation job '{file_name_textbox + '.fchk'}' has been submitted.")

            # Get the cube file
            view = nglview.show_rdkit(gaussian_mol)
            view.add_component(f"./{file_name_textbox + '.cube'}")

            # Adjust visualization settings
            view.component_1.update_surface(opacity=visualization_opacity, color=visualization_color1, isolevel=visualization_isolevel)
            view.camera = 'orthographic'
        else:
            MO_index = int(visualization_dropdown.split(" ")[1])
            subprocess.run(['cubegen', '0', f'MO={MO_index}', f'{file_name_textbox}' + '.fchk', f'{file_name_textbox}' + '.cube'], check=True)
            print(f"Cube file generation job '{file_name_textbox + '.fchk'}' has been submitted.")

            # Get the cube file
            view = nglview.show_rdkit(gaussian_mol)
            view.add_component(f"./{file_name_textbox + '.cube'}")
            view.add_component(f"./{file_name_textbox + '.cube'}")

            # Adjust visualization settings
            view.component_1.update_surface(opacity=visualization_opacity, color=visualization_color1, isolevel=2)
            view.component_2.update_surface(opacity=visualization_opacity, color=visualization_color2, isolevel=-2)
            view.camera = 'orthographic'
        
        # Write the widget to HTML
        if os.path.exists('./static/opt_visualization/cube_file.html'):
            os.remove('./static/opt_visualization/cube_file.html')
        nglview.write_html('./static/opt_visualization/cube_file.html', [view])

        # Read the HTML file
        timestamp = int(time.time())
        html = f'<iframe src="/static/opt_visualization/cube_file.html?ts={timestamp}" height="400" width="600" title="NGL View"></iframe>'
    except Exception as exc:
        gr.Warning("Visualization error!\n" + str(exc))
        return None
    
    return html

def geometry_optimization_tab_content():
    with gr.Tab("Geometry Optimization") as geometry_optimization_tab:
        with gr.Accordion("Molecule"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    input_smiles_texbox = gr.Textbox(label="SMILES")
                    create_molecule_button = gr.Button(value="Create molecule")
                with gr.Column(scale=1):
                    load_molecule_uploadbutton = gr.UploadButton(label="Load molecule", file_types=['.pdb', '.xyz', '.mol', '.mol2', '.log'])
                with gr.Column(scale=1):
                    smiles_texbox = gr.Textbox(label="SMILES")
                    molecule_viewer = gr.HTML(label="Molecule")
        with gr.Accordion("Geometry Optimization"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    mm_checkbox = gr.Checkbox(label="Optimize geometry with molecular mechanics", value=True)
                    with gr.Row():
                        with gr.Column(scale=1):
                            force_field_dropdown = gr.Dropdown(label="Force field", value="MMFF", choices=["MMFF", "UFF"])
                        with gr.Column(scale=1):
                            max_iters_slider = gr.Slider(label="Max iterations", value=200, minimum=0, maximum=1000, step=1)
                    solvation_checkbox = gr.Checkbox(label="Solvation", value=False)
                    with gr.Row():
                        with gr.Column(scale=1):
                            solvation_dropdown = gr.Dropdown(label="Solvation model", value="iefpcm", choices=[("IEFPCM", "iefpcm"), ("SMD", "smd"),  ("I-PCM", "ipcm"), ("SCI-PCM", "scipcm"), ("CPCM", "cpcm")], visible=False)
                        with gr.Column(scale=1):
                            solvent_dropdown = gr.Dropdown(label="Solvent", value="water", choices=["water", ("DMSO", "dmso"),  "nitromethane", "acetonitrile", "methanol", "ethanol", "acetone", "dichloromethane",
                                                                                                    "dichloroethane", ("THF", "thf"), "aniline", "chlorobenzene", "chloroform", ("diethyl ether", "diethylether"),
                                                                                                    "toluene", "benzene", ("CCl4", "ccl4"), "cyclohexane", "heptane"], allow_custom_value=True, visible=False)
                with gr.Column(scale=1):
                    functional_textbox = gr.Dropdown(label="Functional", value="B3LYP", choices=["LSDA", "BVP86", "B3LYP", "CAM-B3LYP", "B3PW91", "B97D", "MPW1PW91", "PBEPBE", "HSEH1PBE", "HCTH", "TPSSTPSS", "WB97XD",
                                                                                                 "M06-2X"], allow_custom_value=True)
                    basis_set_textbox = gr.Dropdown(label="Basis set", value="6-31G(d,p)", choices=["STO-3G", "3-21G", "6-31G", "6-31G'", "6-31G(d,p)", "6-31G(3d,p)", "6-31G(d,3p)", "6-31G(3d,3p)", "6-31+G(d,p)", "6-31++G(d,p)",
                                                                                               "6-311G", "6-311G(d,p)", "cc-pVDZ", "cc-pVTZ", "cc-pVQZ", "aug-cc-pVDZ", "aug-cc-pVTZ", "aug-cc-pVQZ",
                                                                                               "LanL2DZ", "LanL2MB", "SDD", "DGDZVP", "DGDZVP2", "DGTZVP", "GEN", "GENECP"], allow_custom_value=True)
                    charge_slider = gr.Slider(label="Charge", value=0, minimum=-2, maximum=2, step=1)
                    multiplicity_dropdown = gr.Dropdown(label="Multiplicity", value=1, choices=[("Singlet", 1), ("Doublet", 2),
                                                                                                 ("Triplet", 3), ("Quartet", 4),
                                                                                                 ("Quintet", 5), ("Sextet ", 6)])
                with gr.Column(scale=1):
                    n_cores_slider = gr.Slider(label="Number of cores", value=max_n_procs, minimum=1, maximum=max_n_procs, step=1)
                    memory_slider = gr.Slider(label="Memory (GB)", value=max_memory, minimum=1, maximum=max_memory, step=1)
                    file_name_textbox = gr.Textbox(label="File name", value="molecule_opt")
                    optimize_button = gr.Button(value="Optimize", interactive=False)
        with gr.Accordion("Calculation Results"):
            with gr.Row():
                status_markdown = gr.Markdown()
                download_button = gr.DownloadButton(label="Download calculation output file", visible=False)
            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    energy_plot = gr.Plot(label="Energy plot")
                with gr.Column(scale=1):
                    conformers_viewer = gr.HTML(label="Conformer")
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    with gr.Row():
                        energy_texbox = gr.Textbox(label="Energy", value="Not calculated")
                        dipole_moment_texbox = gr.Textbox(label="Dipole moment", value="Not calculated")
                    visualization_dropdown = gr.Dropdown(label="Visualization", value="Electron density", choices=["Electron density", "Electrostatic potential"])
                    with gr.Row():
                        visualization_color1 = gr.ColorPicker(label="Color 1", value="#0000ff")
                        visualization_color2 = gr.ColorPicker(label="Color 2", value="#ff0000")
                        visualization_opacity = gr.Slider(label="Opacity", value=0.8, minimum=0, maximum=1, step=0.01)
                        visualization_isolevel = gr.Slider(label="Isolevel", value=0.05, minimum=0, maximum=1, step=0.01)
                    visualize_button = gr.Button(value="Visualize", interactive=False)
                    visualization_html = gr.HTML(label="Visualization")
                with gr.Column(scale=1):
                    MO_dataframe = gr.DataFrame(label="Molecular orbitals")
            
        create_molecule_button.click(on_create_molecule, input_smiles_texbox, [smiles_texbox, molecule_viewer, optimize_button])
        load_molecule_uploadbutton.upload(on_upload_molecule, load_molecule_uploadbutton, [smiles_texbox, molecule_viewer, optimize_button, energy_texbox, dipole_moment_texbox, energy_plot, conformers_viewer, visualization_dropdown, visualize_button, visualization_html, MO_dataframe])
        mm_checkbox.change(on_mm_checkbox_change, mm_checkbox, [force_field_dropdown, max_iters_slider])
        solvation_checkbox.change(on_solvation_checkbox_change, solvation_checkbox, [solvation_dropdown, solvent_dropdown])
        optimize_button.click(on_optimize_geometry, [mm_checkbox, force_field_dropdown, max_iters_slider, solvation_checkbox, solvation_dropdown, solvent_dropdown,
                                                     functional_textbox, basis_set_textbox, charge_slider, multiplicity_dropdown,
                                                     n_cores_slider, memory_slider, file_name_textbox],
                                                    [status_markdown, download_button, energy_texbox, dipole_moment_texbox, energy_plot, conformers_viewer, visualization_dropdown, visualize_button, visualization_html, MO_dataframe])
        visualization_dropdown.change(on_visualization_change, visualization_dropdown, visualization_isolevel)
        visualize_button.click(on_visualization, [file_name_textbox, visualization_dropdown, visualization_color1, visualization_color2, visualization_opacity, visualization_isolevel], [visualization_html])
        
    return geometry_optimization_tab