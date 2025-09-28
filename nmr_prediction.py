import os
import multiprocessing
import psutil
import subprocess
import time
import math
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import AllChem
import re
import nglview
from utils import *

max_n_procs = multiprocessing.cpu_count()
max_memory = math.floor(psutil.virtual_memory().total / (1024 ** 3))

def on_create_molecule(input_smiles_textbox: gr.Textbox):
    os.makedirs("structures", exist_ok=True)
    file_path = "./structures/molecule_nmr.pdb"
    try:
        global mol_nmr
        mol_nmr = Chem.MolFromSmiles(input_smiles_textbox)
        mol_nmr = Chem.AddHs(mol_nmr)
        smiles = Chem.CanonSmiles(input_smiles_textbox)
        AllChem.EmbedMolecule(mol_nmr)
        Chem.MolToPDBFile(mol_nmr, file_path)

        # Create the NGL view widget
        view = nglview.show_rdkit(mol_nmr)
        
        # Write the widget to HTML
        if os.path.exists('./static/molecule_nmr.html'):
            os.remove('./static/molecule_nmr.html')
        nglview.write_html('./static/molecule_nmr.html', [view])

        # Read the HTML file
        timestamp = int(time.time())
        html = f'<iframe src="/static/molecule_nmr.html?ts={timestamp}" height="300" width="400" title="NGL View"></iframe>'
    except Exception as exc:
        gr.Warning("Error!\n" + str(exc))

        return None, None, None
    
    return smiles, html, gr.update(interactive=True)

def on_upload_molecule(load_molecule_uploadbutton: gr.UploadButton):
    uploaded_file_path = load_molecule_uploadbutton
    _, file_extension = os.path.splitext(uploaded_file_path)

    try:
        global mol_nmr

        if file_extension.lower() == ".pdb":
            mol_nmr = Chem.MolFromPDBFile(uploaded_file_path, sanitize=False, removeHs=False)
        elif file_extension.lower() == ".xyz":    
            mol_nmr = mol_from_xyz_file(uploaded_file_path)
        elif file_extension.lower() == ".mol":    
            mol_nmr = Chem.MolFromMolFile(uploaded_file_path, sanitize=False, removeHs=False)
        elif file_extension.lower() == ".mol2":    
            mol_nmr = Chem.MolFromMol2File(uploaded_file_path, sanitize=False, removeHs=False)
        elif file_extension.lower() == ".log":    
            mol_nmr = mol_from_gaussian_file(uploaded_file_path)
        else:
            raise Exception("File must be in supported formats (pdb, xyz, mol, mol2, log).")
        Chem.SanitizeMol(mol_nmr)
        
        if file_extension.lower() == ".xyz" or file_extension.lower() == ".log":
            smiles = "(No bonding information)"
        else:
            smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol_nmr))
        if mol_nmr.GetNumConformers()==0:
            AllChem.EmbedMolecule(mol_nmr)

        # Create the NGL view widget
        view = nglview.show_rdkit(mol_nmr)
        
        # Write the widget to HTML
        if os.path.exists('./static/molecule_nmr.html'):
            os.remove('./static/molecule_nmr.html')
        nglview.write_html('./static/molecule_nmr.html', [view])

        # Read the HTML file
        timestamp = int(time.time())
        html = f'<iframe src="/static/molecule_nmr.html?ts={timestamp}" height="300" width="400" title="NGL View"></iframe>'
    except Exception as exc:
        gr.Warning("Error!\n" + str(exc))
        return None, None, gr.update(interactive=False), None, None, None, gr.update(interactive=False)
    
    # Read calculation results if the uploaded file is a Gaussian log file
    if file_extension.lower() == ".log":
        try:
            # Get results
            shielding_data = parse_nmr_shielding_constants(uploaded_file_path)
            reference_shieldings = {
                'H': 31.5,
                'C': 186.0
            }       

            shifts_data = calculate_chemical_shifts(shielding_data, reference_shieldings)

            # Create the DataFrame
            atom_indices = []
            element_symbols = []
            shielding_constants = []
            chemical_shifts = []

            shielding_dict = {(atom_idx, element): shielding for atom_idx, element, shielding in shielding_data}

            for atom_idx, element, shift in shifts_data:
                key = (atom_idx, element)
                shielding = shielding_dict.get(key, None)
                if shielding is not None:
                    atom_indices.append(atom_idx)
                    element_symbols.append(element)
                    shielding_constants.append('{:.4f}'.format(shielding))
                    chemical_shifts.append('{:.4f}'.format(shift))
                else:
                    atom_indices.append(atom_idx)
                    element_symbols.append(element)
                    shielding_constants.append("")
                    chemical_shifts.append("")

            nmr_df = pd.DataFrame({
                'Index': atom_indices,
                'Symbol': element_symbols,
                'Shielding (ppm)': shielding_constants,
                'Chemical Shift (ppm, ref: TMS)': chemical_shifts
            })

            # Generate 1H NMR spectrum
            nmr_spectrum_1H = generate_nmr_spectrum_interactive(
                shifts_data,
                element_symbol='H',
                linewidth=1,           # Line width in Hz
                frequency=500,         # 500 MHz spectrometer
                plot_range=(-2, 13)    # Plot from 0 to 10 ppm
            )

            # Generate 13C NMR spectrum
            nmr_spectrum_13C = generate_nmr_spectrum_interactive(
                shifts_data,
                element_symbol='C',
                linewidth=1,           # Line width in Hz
                frequency=500,         # 500 MHz spectrometer
                plot_range=(-25, 225)  # Plot from 0 to 200 ppm
            )

            return smiles, html, gr.update(interactive=True), nmr_df, nmr_spectrum_1H, nmr_spectrum_13C, gr.update(interactive=True)
        except:
            return smiles, html, gr.update(interactive=True), None, None, None, gr.update(interactive=False)
    else:
        return smiles, html, gr.update(interactive=True), None, None, None, gr.update(interactive=False)

def on_mm_checkbox_change(mm_checkbox: gr.Checkbox):
    return gr.update(visible=mm_checkbox), gr.update(visible=mm_checkbox)

def on_solvation_checkbox_change(solvation_checkbox: gr.Checkbox):
    return gr.update(visible=solvation_checkbox), gr.update(visible=solvation_checkbox)

def write_nmr_gaussian_input(mol, file_name, method='B3LYP', basis='6-31G(d,p)', charge=0, multiplicity=1, solvation=None, solvent=None, n_proc=4, memory=2):
    # Open the file for writing
    with open(file_name + '.gjf', 'w') as f:
        # Link0 commands
        f.write(f'%NProcShared={n_proc}\n')
        f.write(f'%Mem={memory}GB\n')
        f.write(f'%Chk={file_name}.chk\n')
        # Route section
        route_section = f'#P {method}/{basis} NMR=GIAO'
        if solvation:
            route_section += f' scrf=({solvation},solvent={solvent})'
        route_section += '\n\n'
        f.write(route_section)
        # Title section
        f.write('NMR Chemical Shielding Calculation using GIAO\n\n')
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

def parse_nmr_shielding_constants(logfile):
    shielding_data = []

    # Regular expression pattern to match the shielding tensor lines
    pattern = re.compile(
        r'\s*(\d+)\s+([A-Za-z]+)\s+Isotropic\s*=\s*([-\d.]+)\s+Anisotropy\s*=\s*([-\d.]+)'
    )

    with open(logfile, 'r') as f:
        lines = f.readlines()

    read_shielding = False

    # Now, parse the shielding constants
    for i, line in enumerate(lines):
        if 'SCF GIAO Magnetic shielding tensor' in line:
            read_shielding = True
            j = i + 1  # Start parsing from the line after the header
            while j < len(lines):
                line = lines[j]
                match = pattern.match(line)
                if match:
                    atom_idx = int(match.group(1))
                    element_symbol = match.group(2)
                    isotropic_value = float(match.group(3))
                    # Optionally, you can retrieve anisotropy_value if needed
                    # anisotropy_value = float(match.group(4))
                    shielding_data.append((atom_idx, element_symbol, isotropic_value))
                elif 'Eigenvalues:' in line or 'XX=' in line:
                    # Skip lines containing tensor components and eigenvalues
                    pass
                elif line.strip() == '':
                    # Stop parsing if we reach an empty line (end of the shielding section)
                    break
                j += 1
            break  # Exit after parsing shielding constants

    if not read_shielding:
        print("No NMR shielding data found in the log file.")

    return shielding_data

def calculate_chemical_shifts(shielding_data, reference_shieldings):
    shifts_data = []
    for atom_idx, element, shielding in shielding_data:
        if element in reference_shieldings:
            delta = reference_shieldings[element] - shielding
            shifts_data.append((atom_idx, element, delta))
        else:
            print(f"No reference shielding constant for element {element}. Skipping atom {atom_idx}.")
    return shifts_data

def lorentzian(x, x0, gamma):
    return (gamma**2) / ((x - x0)**2 + gamma**2)

def generate_nmr_spectrum_interactive(shifts_data, element_symbol, linewidth=0.5, points=10000, frequency=400, plot_range=None):
    # Filter shifts for the desired element
    shifts = [(atom_idx, element, shift) for atom_idx, element, shift in shifts_data if element == element_symbol]

    if not shifts:
        print(f"No chemical shifts found for element {element_symbol}.")
        return

    # Create the chemical shift axis
    if plot_range:
        min_shift, max_shift = plot_range
    else:
        min_shift = min(shift for _, _, shift in shifts) - 1
        max_shift = max(shift for _, _, shift in shifts) + 1

    x = np.linspace(min_shift, max_shift, points)

    # Initialize the spectrum
    spectrum = np.zeros_like(x)

    # Convert linewidth from Hz to ppm
    lw_ppm = linewidth / frequency

    # Count occurrences of each chemical shift for intensity scaling
    shift_counts = {}
    for atom_idx, element, shift in shifts:
        shift_counts[shift] = shift_counts.get(shift, 0) + 1

    # Generate Lorentzian peaks with intensities and collect peak positions for annotations
    annotations = []
    for atom_idx, element, shift in shifts:
        intensity = shift_counts[shift]
        spectrum += intensity * lorentzian(x, shift, lw_ppm)
        annotations.append({
            'shift': shift,
            'atom_label': f"{element}-{atom_idx}",
            'intensity': intensity
        })

    # Normalize the spectrum
    spectrum /= np.max(spectrum)

    # Create the Plotly figure
    fig = go.Figure()

    # Add the spectrum trace
    fig.add_trace(go.Scatter(
        x=x,
        y=spectrum,
        mode='lines',
        line=dict(color='black'),
        name=f'{element_symbol} NMR Spectrum'
    ))

    # Add peak annotations
    for ann in annotations:
        fig.add_annotation(
            x=ann['shift'],
            y=lorentzian(ann['shift'], ann['shift'], lw_ppm) / np.max(spectrum),
            text=ann['atom_label'],
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            yshift=10,
            font=dict(color='red'),
        )

    # Invert x-axis for NMR spectrum convention
    fig.update_layout(
        xaxis=dict(
            autorange='reversed',
            title='Chemical Shift (ppm)',
        ),
        yaxis=dict(
            title='Intensity (a.u.)',
        ),
        title=f'{element_symbol} NMR Spectrum',
        showlegend=False,
    )

    # Adjust plot range if specified
    if plot_range:
        fig.update_xaxes(range=plot_range)

    return fig

def on_nmr_predict(solvation_checkbox: gr.Checkbox, solvation_dropdown: gr.Dropdown, solvent_dropdown: gr.Dropdown,
                   functional_textbox: gr.Dropdown, basis_set_textbox: gr.Dropdown, charge_slider: gr.Slider, multiplicity_dropdown: gr.Dropdown,
                   n_cores_slider: gr.Slider, memory_slider: gr.Slider, file_name_textbox: gr.Textbox):
    try:
        # Write Gaussian input file
        if solvation_checkbox:
            solvation = solvation_dropdown
        else:
            solvation = None
        write_nmr_gaussian_input(mol_nmr, file_name=file_name_textbox, method=functional_textbox, basis=basis_set_textbox, charge=charge_slider, multiplicity=multiplicity_dropdown, solvation=solvation, solvent=solvent_dropdown, n_proc=n_cores_slider, memory=memory_slider)

        # Run calculation
        start = time.time()
        subprocess.run(['g16', file_name_textbox + '.gjf'], check=True)
        print(f"Gaussian job '{file_name_textbox + '.gjf'}' has been submitted.")
        end = time.time()
        duration = end - start

        # Get results
        shielding_data = parse_nmr_shielding_constants(file_name_textbox + '.log')
        reference_shieldings = {
            'H': 31.5,
            'C': 186.0
        }       

        shifts_data = calculate_chemical_shifts(shielding_data, reference_shieldings)

        # Create the DataFrame
        atom_indices = []
        element_symbols = []
        shielding_constants = []
        chemical_shifts = []

        shielding_dict = {(atom_idx, element): shielding for atom_idx, element, shielding in shielding_data}

        for atom_idx, element, shift in shifts_data:
            key = (atom_idx, element)
            shielding = shielding_dict.get(key, None)
            if shielding is not None:
                atom_indices.append(atom_idx)
                element_symbols.append(element)
                shielding_constants.append('{:.4f}'.format(shielding))
                chemical_shifts.append('{:.4f}'.format(shift))
            else:
                atom_indices.append(atom_idx)
                element_symbols.append(element)
                shielding_constants.append("")
                chemical_shifts.append("")

        nmr_df = pd.DataFrame({
            'Index': atom_indices,
            'Symbol': element_symbols,
            'Shielding (ppm)': shielding_constants,
            'Chemical Shift (ppm, ref: TMS)': chemical_shifts
        })

        # Generate 1H NMR spectrum
        nmr_spectrum_1H = generate_nmr_spectrum_interactive(
            shifts_data,
            element_symbol='H',
            linewidth=1,           # Line width in Hz
            frequency=500,         # 500 MHz spectrometer
            plot_range=(-2, 13)    # Plot from 0 to 10 ppm
        )

        # Generate 13C NMR spectrum
        nmr_spectrum_13C = generate_nmr_spectrum_interactive(
            shifts_data,
            element_symbol='C',
            linewidth=1,           # Line width in Hz
            frequency=500,         # 500 MHz spectrometer
            plot_range=(-25, 225)  # Plot from 0 to 200 ppm
        )

    except Exception as exc:
        gr.Warning("Calculation error!\n" + str(exc))
        return None, gr.update(visible=False), None, None, None, gr.update(interactive=False)

    calculation_status = "Calculation finished. ({0:.3f} s)".format(duration)
    output_file_path = f'{file_name_textbox}.log'
    return calculation_status, gr.update(value=output_file_path, visible=True), nmr_df, nmr_spectrum_1H, nmr_spectrum_13C, gr.update(interactive=True)

def on_export_nmr(nmr_dataframe: gr.Dataframe, nmr_filename_textbox: gr.Textbox):
    file_path = nmr_filename_textbox + '.csv'
    nmr_dataframe.to_csv(file_path)

    return "NMR data exported: " + file_path

def nmr_prediction_tab_content():
    with gr.Tab("NMR Prediction") as nmr_prediction_tab:
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
        with gr.Accordion("NMR Prediction"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    with gr.Row():
                        mm_checkbox = gr.Checkbox(label="Optimize geometry with molecular mechanics", value=False)
                    with gr.Row():
                        with gr.Column(scale=1):
                            force_field_dropdown = gr.Dropdown(label="Force field", value="MMFF", choices=["MMFF", "UFF"], visible=False)
                        with gr.Column(scale=1):
                            max_iters_slider = gr.Slider(label="Max iterations", value=200, minimum=0, maximum=1000, step=1, visible=False)
                    with gr.Row():
                        solvation_checkbox = gr.Checkbox(label="Solvation (IEFPCM)", value=True)
                    with gr.Row():
                        with gr.Column(scale=1):
                            solvation_dropdown = gr.Dropdown(label="Solvation model", value="iefpcm", choices=[("IEFPCM", "iefpcm"), ("SMD", "smd"),  ("I-PCM", "ipcm"), ("SCI-PCM", "scipcm"), ("CPCM", "cpcm")])
                        with gr.Column(scale=1):
                            solvent_dropdown = gr.Dropdown(label="Solvent", value="chloroform", choices=["water", ("DMSO", "dmso"),  "nitromethane", "acetonitrile", "methanol", "ethanol", "acetone", "dichloromethane",
                                                                                                         "dichloroethane", ("THF", "thf"), "aniline", "chlorobenzene", "chloroform", ("diethyl ether", "diethylether"),
                                                                                                         "toluene", "benzene", ("CCl4", "ccl4"), "cyclohexane", "heptane"], allow_custom_value=True)
                with gr.Column(scale=1):
                    functional_textbox = gr.Dropdown(label="Functional", value="B3LYP", choices=["LSDA", ("B–VP86", "BVP86"), "B3LYP", "CAM-B3LYP", "B3PW91", "B97D", "MPW1PW91", "PBEPBE", "HSEH1PBE", "HCTH", "TPSSTPSS", ("ωB97XD", "WB97XD"), "M06-2X"], allow_custom_value=True)
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
                    file_name_textbox = gr.Textbox(label="File name", value="molecule_nmr")
                    predict_button = gr.Button(value="Predict", interactive=False)
        with gr.Accordion("Prediction Results"):
            with gr.Row():
                status_markdown = gr.Markdown()
                download_button = gr.DownloadButton(label="Download calculation output file", visible=False)
            with gr.Row():
                with gr.Column(scale=1):
                    nmr_dataframe = gr.DataFrame(label="NMR signals")
                    nmr_filename_textbox = gr.Textbox(label="File name", value="nmr_data")
                    export_nmr_button = gr.Button(value="Export", interactive=False)
                    export_nmr_status_markdown = gr.Markdown(value="")
                with gr.Column(scale=2):
                    with gr.Row():    
                        nmr_spectrum_1H = gr.Plot(label="1H NMR spectrum")
                    with gr.Row():
                        nmr_spectrum_13C = gr.Plot(label="13C NMR spectrum")
                
        create_molecule_button.click(on_create_molecule, input_smiles_texbox, [smiles_texbox, molecule_viewer, predict_button])
        load_molecule_uploadbutton.upload(on_upload_molecule, load_molecule_uploadbutton, [smiles_texbox, molecule_viewer, predict_button, nmr_dataframe, nmr_spectrum_1H, nmr_spectrum_13C, export_nmr_button])
        mm_checkbox.change(on_mm_checkbox_change, mm_checkbox, [force_field_dropdown, max_iters_slider])
        solvation_checkbox.change(on_solvation_checkbox_change, solvation_checkbox, [solvation_dropdown, solvent_dropdown])
        predict_button.click(on_nmr_predict, [solvation_checkbox, solvation_dropdown, solvent_dropdown,
                                              functional_textbox, basis_set_textbox, charge_slider, multiplicity_dropdown,
                                              n_cores_slider, memory_slider, file_name_textbox],
                                             [status_markdown, download_button, nmr_dataframe, nmr_spectrum_1H, nmr_spectrum_13C, export_nmr_button])
        export_nmr_button.click(on_export_nmr, [nmr_dataframe, nmr_filename_textbox], export_nmr_status_markdown)
        
    return nmr_prediction_tab