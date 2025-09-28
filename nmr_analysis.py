import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import plotly.graph_objects as go
import gradio as gr
from gradio_molecule2d import molecule2d
import cclib
import re
from utils import conformer_to_xyz_file
import nmrglue as ng

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

def on_analyze_conformers_energy(conformers_dir: gr.File):
    file_names = []
    energies = []
    for file in conformers_dir:
        _, file_extension = os.path.splitext(file.name)
        if file_extension.lower() != ".log":
            continue

        # Get results
        file_name = os.path.basename(file.name)
        parser = cclib.io.ccopen(file.name)
        data = parser.parse()
        energy = data.scfenergies[-1] / 27.2114 # hartree

        # Add to lists
        file_names.append(file_name)
        energies.append(energy)
        
    # Create dataframe
    df = pd.DataFrame(list(zip(file_names, energies)), columns =["Conformer", "Energy (hartree)"])
    return df

def on_export_conformers_energy(export_conformers_energy_filename_textbox: gr.Textbox, conformers_energy_dataframe: gr.Dataframe):
    file_name = export_conformers_energy_filename_textbox + '.csv'
    conformers_energy_dataframe.to_csv(file_name)
    return "Energies exported: " + file_name

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

def on_upload_gaussian_output(output_file_uploadbutton: gr.UploadButton):
    _, file_extension = os.path.splitext(output_file_uploadbutton)

    if file_extension.lower() != ".log":
        return None, None
    
    # Get results
    shielding_data = parse_nmr_shielding_constants(output_file_uploadbutton)
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

    return nmr_spectrum_1H, nmr_spectrum_13C

def read_and_plot_nmr_spectrum(file_path, file_format, p0=-50.0, p1=0.0, plot_title=None):
    # Read the NMR data using nmrglue
    if file_format.lower() == 'bruker':
        # For Bruker data, file_path should be the directory containing the fid file
        dic, data = ng.bruker.read(file_path)
        # Process the data to obtain the spectrum
        udic = ng.bruker.guess_udic(dic, data)
        uc = ng.fileiobase.uc_from_udic(udic)
        # remove the digital filter
        data = ng.bruker.remove_digital_filter(dic, data)
        # process the spectrum
        data = ng.proc_base.zf_size(data, 32768)            # zero fill to 32768 points
        data = ng.proc_base.fft(data)                       # Fourier transform
        data = ng.proc_base.ps(data, p0=p0, p1=p1)          # phase correction
        data = ng.proc_base.di(data)                        # discard the imaginaries
        data = ng.proc_base.rev(data)                       # reverse the data
        # Generate the frequency axis
        x_axis = uc.ppm_scale()
        # Reverse the axis if needed
        x_axis = x_axis[::-1]
        data = data[::-1]
    else:
        return None
    
    # Create the Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=np.real(data),
        mode='lines',
        line=dict(color='blue'),
        name='Experimental NMR Spectrum'
    ))
    
    # Invert x-axis for NMR spectrum convention
    fig.update_layout(
        xaxis=dict(
            autorange='reversed',
            title=f'Chemical Shift (ppm)',
        ),
        yaxis=dict(
            title='Intensity (a.u.)',
        ),
        title=plot_title or 'Experimental NMR Spectrum',
        showlegend=False,
    )
    
    return fig

def on_load_experimental_1H_nmr_spectrum(experimental_1H_nmr_spectrum_dir_textbox: gr.Textbox):
    return read_and_plot_nmr_spectrum(experimental_1H_nmr_spectrum_dir_textbox, file_format='bruker', p0=-78.0, p1=0, plot_title="1H NMR spectrum")

def on_load_experimental_13C_nmr_spectrum(experimental_1H_nmr_spectrum_dir_textbox: gr.Textbox):
    return read_and_plot_nmr_spectrum(experimental_1H_nmr_spectrum_dir_textbox, file_format='bruker', p0=180.00, p1=90, plot_title="13C NMR spectrum")

with gr.Blocks(css='styles.css') as app:
    with gr.Tabs() as tabs:
        with gr.Tab("Generate Conformers"):
            with gr.Row():
                with gr.Column(scale=1):
                    molecule_editor = molecule2d(label="Molecule")
                    charge_slider = gr.Slider(label="Charge", value=0, minimum=-2, maximum=2, step=1)
                    multiplicity_dropdown = gr.Dropdown(label="Multiplicity", value=1, choices=[("Singlet", 1), ("Doublet", 2),
                                                                                                        ("Triplet", 3), ("Quartet", 4),
                                                                                                        ("Quintet", 5), ("Sextet", 6)])
                with gr.Column(scale=1):
                    file_name_textbox = gr.Textbox(label="File name", value="conformer")
                    num_confs_slider = gr.Slider(label="Number of conformers", value=30, minimum=1, maximum=50, step=1)
                    generate_button = gr.Button(value="Generate")
                    status_markdown = gr.Markdown()
        with gr.Tab("Energy Analysis for Conformers"):
            with gr.Row():
                with gr.Column(scale=1):
                    conformers_dir = gr.File(label="Folder", file_count="directory")
                with gr.Column(scale=1):
                    analyze_energy_button = gr.Button(value="Analyze")
            with gr.Row():
                with gr.Column(scale=2):
                    conformers_energy_dataframe = gr.Dataframe(label="Energy")
                with gr.Column(scale=1):
                    export_conformers_energy_filename_textbox = gr.Textbox(label="File name", value="conformers_energies")
                    export_conformers_energy_button = gr.Button(value="Export")
                    export_conformers_energy_button_status_markdown = gr.Markdown(value="")
        with gr.Tab("NMR Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    output_file_uploadbutton = gr.UploadButton(label="Gaussian output file")
                with gr.Column(scale=1):
                    experimental_1H_nmr_spectrum_dir_textbox = gr.Textbox(label="Experimental 1H NMR spectrum folder", value=os.getcwd())
                    experimental_1H_nmr_spectrum_load_button = gr.Button(value="Load")
                with gr.Column(scale=1):
                    experimental_13C_nmr_spectrum_dir_textbox = gr.Textbox(label="Experimental 13C NMR spectrum folder", value=os.getcwd())
                    experimental_13C_nmr_spectrum_load_button = gr.Button(value="Load")
            with gr.Row():
                with gr.Column(scale=1):
                    predicted_1H_nmr_spectrum = gr.Plot(label="Predicted 1H NMR spectrum")
                with gr.Column(scale=1):
                    predicted_13C_nmr_spectrum = gr.Plot(label="Predicted 13C NMR spectrum")
            with gr.Row():
                with gr.Column(scale=1):
                    experimental_1H_nmr_spectrum = gr.Plot(label="Experimental 1H NMR spectrum")
                with gr.Column(scale=1):
                    experimental_13C_nmr_spectrum = gr.Plot(label="Experimental 13C NMR spectrum")

    generate_button.click(on_generate_conformers, [molecule_editor, charge_slider, multiplicity_dropdown, file_name_textbox, num_confs_slider], status_markdown)
    analyze_energy_button.click(on_analyze_conformers_energy, conformers_dir, conformers_energy_dataframe)
    export_conformers_energy_button.click(on_export_conformers_energy, [export_conformers_energy_filename_textbox, conformers_energy_dataframe], export_conformers_energy_button_status_markdown)
    output_file_uploadbutton.upload(on_upload_gaussian_output, output_file_uploadbutton, [predicted_1H_nmr_spectrum, predicted_13C_nmr_spectrum])
    experimental_1H_nmr_spectrum_load_button.click(on_load_experimental_1H_nmr_spectrum, experimental_1H_nmr_spectrum_dir_textbox, experimental_1H_nmr_spectrum)
    experimental_13C_nmr_spectrum_load_button.click(on_load_experimental_13C_nmr_spectrum, experimental_13C_nmr_spectrum_dir_textbox, experimental_13C_nmr_spectrum)

app.launch()
