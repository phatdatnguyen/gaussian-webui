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
import cclib
import nglview
from utils import *

max_n_procs = multiprocessing.cpu_count()
max_memory = math.floor(psutil.virtual_memory().total / (1024 ** 3))

def on_create_molecule(input_smiles_textbox: gr.Textbox):
    os.makedirs("structures", exist_ok=True)
    file_path = "./structures/molecule_fluorescence.pdb"
    try:
        global mol_fluorescence
        mol_fluorescence = Chem.MolFromSmiles(input_smiles_textbox)
        mol_fluorescence = Chem.AddHs(mol_fluorescence)
        smiles = Chem.CanonSmiles(input_smiles_textbox)
        AllChem.EmbedMolecule(mol_fluorescence)
        Chem.MolToPDBFile(mol_fluorescence, file_path)

        # Create the NGL view widget
        view = nglview.show_rdkit(mol_fluorescence)
        
        # Write the widget to HTML
        if os.path.exists('./static/molecule_fluorescence.html'):
            os.remove('./static/molecule_fluorescence.html')
        nglview.write_html('./static/molecule_fluorescence.html', [view])

        # Read the HTML file
        timestamp = int(time.time())
        html = f'<iframe src="/static/molecule_fluorescence.html?ts={timestamp}" height="300" width="400" title="NGL View"></iframe>'
    except Exception as exc:
        gr.Warning("Error!\n" + str(exc))

        return None, None, gr.update(interactive=False)
    
    return smiles, html, gr.update(interactive=True)

def on_upload_molecule(load_molecule_uploadbutton: gr.UploadButton):
    uploaded_file_path = load_molecule_uploadbutton
    _, file_extension = os.path.splitext(uploaded_file_path)

    try:
        global mol_fluorescence

        if file_extension.lower() == ".pdb":
            mol_fluorescence = Chem.MolFromPDBFile(uploaded_file_path, sanitize=False, removeHs=False)
        elif file_extension.lower() == ".xyz":    
            mol_fluorescence = mol_from_xyz_file(uploaded_file_path)
        elif file_extension.lower() == ".mol":    
            mol_fluorescence = Chem.MolFromMolFile(uploaded_file_path, sanitize=False, removeHs=False)
        elif file_extension.lower() == ".mol2":    
            mol_fluorescence = Chem.MolFromMol2File(uploaded_file_path, sanitize=False, removeHs=False)
        elif file_extension.lower() == ".log":    
            mol_fluorescence = mol_from_gaussian_file(uploaded_file_path)
        else:
            raise Exception("File must be in supported formats (pdb, xyz, mol, mol2, log).")
        Chem.SanitizeMol(mol_fluorescence)
        
        if file_extension.lower() == ".xyz" or file_extension.lower() == ".log":
            smiles = "(No bonding information)"
        else:
            smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol_fluorescence))    
        if mol_fluorescence.GetNumConformers()==0:
            AllChem.EmbedMolecule(mol_fluorescence)

        # Create the NGL view widget
        view = nglview.show_rdkit(mol_fluorescence)
        
        # Write the widget to HTML
        if os.path.exists('./static/molecule_fluorescence.html'):
            os.remove('./static/molecule_fluorescence.html')
        nglview.write_html('./static/molecule_fluorescence.html', [view])

        # Read the HTML file
        timestamp = int(time.time())
        html = f'<iframe src="/static/molecule_fluorescence.html?ts={timestamp}" height="300" width="400" title="NGL View"></iframe>'
    except Exception as exc:
        gr.Warning("Error!\n" + str(exc))
        return None, None, gr.update(interactive=False), None, None
    
    # Read calculation results if the uploaded file is a Gaussian log file
    if file_extension.lower() == ".log":
        try:
            # Read the fluorescence spectrum from the Gaussian output using cclib
            parser = cclib.io.ccopen(uploaded_file_path)
            data = parser.parse()
            # Extract excitation energies and oscillator strengths
            if hasattr(data, 'etenergies') and hasattr(data, 'etoscs'):
                # etenergies in cm^(-1), etoscs are dimensionless
                energies_wavenumber = np.array(data.etenergies)

                # Keep only the lowest energy transition only (S1 -> S0)
                energies_wavenumber = np.array([energies_wavenumber[0]])
                oscs = np.array([1]) # set osc to 1 for the emission peak

                # Convert energies to wavelength (eV to nm)
                wavelengths = 1e7 / energies_wavenumber

                # Build spectrum
                fluorescence_spectrum = generate_fluorescence_spectrum_interactive(wavelengths=wavelengths, oscs=oscs, points=10000, plot_range=(0, 800))

                # Build peak dataframe
                peak_df = pd.DataFrame({
                    "Emission Wavelength (nm)": wavelengths,
                    "Oscillator Strength": oscs
                })

                # Sort by wavelength (ascending order)
                peak_df = peak_df.sort_values("Emission Wavelength (nm)").reset_index(drop=True)

            return smiles, html, gr.update(interactive=True), fluorescence_spectrum, peak_df
        except:
            return smiles, html, gr.update(interactive=True), None, None
    else:
        return smiles, html, gr.update(interactive=True), None, None

def on_mm_checkbox_change(mm_checkbox: gr.Checkbox):
    return gr.update(visible=mm_checkbox), gr.update(visible=mm_checkbox)

def on_solvation_checkbox_change(solvation_checkbox: gr.Checkbox):
    return gr.update(visible=solvation_checkbox), gr.update(visible=solvation_checkbox)

def write_fluorescence_gaussian_input(mol, file_name, n_states=10, method='B3LYP', basis='6-31G(d,p)', charge=0, multiplicity=1, solvation=None, solvent=None, n_proc=4, memory=2):
    # Open the file for writing
    with open(file_name + '.gjf', 'w') as f:
        # First job: excited-state optimization (S1)
        f.write(f'%NProcShared={n_proc}\n')
        f.write(f'%Mem={memory}GB\n')
        f.write(f'%Chk={file_name}.chk\n')
        route1 = f'#P {method}/{basis} TD(Singlets,NStates={n_states},Root=1) Opt'
        if solvation:
            route1 += f' scrf=({solvation},solvent={solvent})'
        route1 += '\n\n'
        f.write(route1)
        f.write('S1 optimization (TD-DFT, Root=1)\n\n')
        f.write(f'{charge} {multiplicity}\n')
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            sym = atom.GetSymbol()
            pos = conf.GetAtomPosition(idx)
            f.write(f'{sym:<2} {pos.x:>12.6f} {pos.y:>12.6f} {pos.z:>12.6f}\n')
        f.write('\n')

        # Second job: TD single point at optimized S1 geometry
        f.write('--Link1--\n')
        f.write(f'%NProcShared={n_proc}\n')
        f.write(f'%Mem={memory}GB\n')
        f.write(f'%Chk={file_name}.chk\n')
        route2 = f'#P {method}/{basis} TD(Singlets,NStates={n_states}) Geom=AllCheck Guess=Read'
        if solvation:
            route2 += f' scrf=({solvation},solvent={solvent})'
        route2 += '\n\n'
        f.write(route2)
        f.write('TD single-point at optimized S1 geometry (emission energies)\n\n')
        f.write(f'{charge} {multiplicity}\n\n')

    print(f"Wrote chained Gaussian input '{file_name}.gjf' (S1 Opt -> TD SP for emission).")

# Gaussian broadening
def gaussian(x, x0, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-(x - x0)**2 / (2 * sigma**2))

def generate_fluorescence_spectrum_interactive(wavelengths, oscs, points=10000, plot_range=None):
    # Check if wavelengths and oscs are valid
    if wavelengths is None or len(wavelengths) == 0:
        print(f"No absorption peak detected.")
        return
    
    # Create the chemical shift axis
    if plot_range:
        min_wavelength, max_wavelength = plot_range
    else:
        min_wavelength = np.min(wavelengths) - 10
        max_wavelength = np.max(wavelengths) + 10
    
    x = np.linspace(min_wavelength, max_wavelength, points)
    
    # Initialize the spectrum
    spectrum = np.zeros_like(x)
    
    # Generate annotations for each peak with Gaussian broadening
    annotations = []
    base_fwhm = 10.0
    for wavelength, osc in zip(wavelengths, oscs):
        fwhm = base_fwhm * (1 + osc)   # stronger osc → broader peak
        spectrum += osc * gaussian(x, wavelength, fwhm)
        annotations.append({
            'wavelength': wavelength,
            'oscilation strength': oscs,
            'text': f"{wavelength:.2f}",
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
        name=f'fluorescence Spectrum'
    ))
    
    # Add peak annotations
    for ann in annotations:
        fig.add_annotation(
            x=ann['wavelength'],
            y=1.05,
            text=ann['text'],
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            yshift=10,
            textangle=-90,
            font=dict(color='red'),
        )

    fig.update_layout(
        xaxis=dict(
            title='Wavelength (nm)',
        ),
        yaxis=dict(
            title='Relative Intensity (a.u.)',
        ),
        title=f'fluorescence Spectrum',
        showlegend=False,
    )

    # Adjust plot range if specified
    if plot_range:
        fig.update_xaxes(range=plot_range)

    return fig

def on_fluorescence_predict(mm_checkbox: gr.Checkbox, force_field_dropdown: gr.Dropdown, max_iters_slider: gr.Slider, n_state_slider: gr.Slider, solvation_checkbox: gr.Checkbox, solvation_dropdown: gr.Dropdown, solvent_dropdown: gr.Dropdown,
                      functional_textbox: gr.Dropdown, basis_set_textbox: gr.Dropdown, charge_slider: gr.Slider, multiplicity_dropdown: gr.Dropdown,
                      n_cores_slider: gr.Slider, memory_slider: gr.Slider, file_name_textbox: gr.Textbox):
    try:
        if mm_checkbox:
            if force_field_dropdown=="MMFF":
                AllChem.MMFFOptimizeMolecule(mol_fluorescence, maxIters=max_iters_slider)
            else:
                AllChem.UFFOptimizeMolecule(mol_fluorescence, maxIters=max_iters_slider)

        # Write Gaussian input file
        if solvation_checkbox:
            solvation = solvation_dropdown
        else:
            solvation = None
        write_fluorescence_gaussian_input(mol_fluorescence, file_name=file_name_textbox, n_states=n_state_slider, method=functional_textbox, basis=basis_set_textbox, charge=charge_slider, multiplicity=multiplicity_dropdown, solvation=solvation, solvent=solvent_dropdown, n_proc=n_cores_slider, memory=memory_slider)

        # Run calculation
        start = time.time()
        subprocess.run(['g16', file_name_textbox + '.gjf'], check=True)
        print(f"Gaussian job '{file_name_textbox + '.gjf'}' has been submitted.")
        end = time.time()
        duration = end - start

        # Read the fluorescence spectrum from the Gaussian output using cclib
        parser = cclib.io.ccopen(f'{file_name_textbox}.log')
        data = parser.parse()
        # Extract excitation energies and oscillator strengths
        if hasattr(data, 'etenergies') and hasattr(data, 'etoscs'):
            # etenergies in cm^(-1), etoscs are dimensionless
            energies_wavenumber = np.array(data.etenergies)

            # Keep only the lowest energy transition only (S1 -> S0)
            energies_wavenumber = np.array([energies_wavenumber[0]])
            oscs = np.array([1]) # set osc to 1 for the emission peak

            # Convert energies to wavelength (eV to nm)
            wavelengths = 1e7 / energies_wavenumber

            # Build spectrum
            fluorescence_spectrum = generate_fluorescence_spectrum_interactive(wavelengths=wavelengths, oscs=oscs, points=10000, plot_range=(0, 800))

            # Build peak dataframe
            peak_df = pd.DataFrame({
                "Emission Wavelength (nm)": wavelengths,
                "Oscillator Strength": oscs
            })

            # Sort by wavelength (ascending order)
            peak_df = peak_df.sort_values("Emission Wavelength (nm)").reset_index(drop=True)

    except Exception as exc:
        gr.Warning("Calculation error!\n" + str(exc))
        return None, gr.update(visible=False), None, None

    calculation_status = "Calculation finished. ({0:.3f} s)".format(duration)
    output_file_path = f'{file_name_textbox}.log'
    return calculation_status, gr.update(value=output_file_path, visible=True), fluorescence_spectrum, peak_df

def fluorescence_prediction_tab_content():
    with gr.Tab("Fluorescence Spectrum Prediction") as fluorescence_prediction_tab:
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
        with gr.Accordion("Fluorescence Spectrum Prediction"):
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
                        n_states_slider = gr.Slider(label="Number of excited states", value=10, minimum=5, maximum=100, step=1)
                    with gr.Row():
                        solvation_checkbox = gr.Checkbox(label="Solvation", value=True)
                    with gr.Row():
                        with gr.Column(scale=1):
                            solvation_dropdown = gr.Dropdown(label="Solvation model", value="smd", choices=[("IEFPCM", "iefpcm"), ("SMD", "smd"),  ("I-PCM", "ipcm"), ("SCI-PCM", "scipcm"), ("CPCM", "cpcm")])
                        with gr.Column(scale=1):
                            solvent_dropdown = gr.Dropdown(label="Solvent", value="water", choices=["water", ("DMSO", "dmso"),  "nitromethane", "acetonitrile", "methanol", "ethanol", "acetone", "dichloromethane",
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
                    file_name_textbox = gr.Textbox(label="File name", value="molecule_fluorescence")
                    predict_button = gr.Button(value="Predict", interactive=False)
        with gr.Accordion("Fluorescence Spectrum"):
            with gr.Row():
                status_markdown = gr.Markdown()
                download_button = gr.DownloadButton(label="Download calculation output file", visible=False)
            with gr.Row(equal_height=True):
                with gr.Column():
                    fluorescence_spectrum_plot = gr.Plot(label="Fluorescence Spectrum")
                with gr.Column():
                    peak_table = gr.Dataframe(headers=["Wavelength (nm)", "Oscillator Strength"], datatype=["number", "number"], label="Absorption Peaks")
                
        create_molecule_button.click(on_create_molecule, input_smiles_texbox, [smiles_texbox, molecule_viewer, predict_button])
        load_molecule_uploadbutton.upload(on_upload_molecule, load_molecule_uploadbutton, [smiles_texbox, molecule_viewer, predict_button, fluorescence_spectrum_plot, peak_table])
        mm_checkbox.change(on_mm_checkbox_change, mm_checkbox, [force_field_dropdown, max_iters_slider])
        solvation_checkbox.change(on_solvation_checkbox_change, solvation_checkbox, [solvation_dropdown, solvent_dropdown])
        predict_button.click(on_fluorescence_predict, [mm_checkbox, force_field_dropdown, max_iters_slider, n_states_slider, solvation_checkbox, solvation_dropdown, solvent_dropdown,
                                                       functional_textbox, basis_set_textbox, charge_slider, multiplicity_dropdown,
                                                       n_cores_slider, memory_slider, file_name_textbox],
                                                      [status_markdown, download_button, fluorescence_spectrum_plot, peak_table])
        
    return fluorescence_prediction_tab