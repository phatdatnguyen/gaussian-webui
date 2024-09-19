import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from gradio_molecule2d import molecule2d
from gradio_molecule3d import Molecule3D
from rdkit import Chem
from rdkit.Chem import AllChem
import autode as ade
import cclib

ade.Config.lcode = 'g16'
ade.Config.hcode = 'g16'

def on_create_molecule(molecule_editor: molecule2d):
    os.makedirs("structures", exist_ok=True)
    file_path = ".\\structures\\molecule.pdb"
    try:
        global mol
        mol = Chem.MolFromSmiles(molecule_editor)
        mol = Chem.AddHs(mol)
        smiles = Chem.CanonSmiles(molecule_editor)
        AllChem.EmbedMolecule(mol)
        Chem.MolToPDBFile(mol, file_path)
    except Exception as exc:
        gr.Warning("Error!\n" + str(exc))
        return [None, None]
    
    return smiles, file_path

def on_upload_molecule(load_molecule_uploadbutton: gr.UploadButton):
    file_path = load_molecule_uploadbutton
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() != ".pdb":
        gr.Warning("Invalid file!\nFile must be in .pdb format.")
        return [None, None]

    try:
        global mol
        mol = Chem.MolFromPDBFile(file_path, sanitize=False, removeHs=False)
        smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol))    
        AllChem.EmbedMolecule(mol)
        Chem.MolToPDBFile(mol, file_path)
    except Exception as exc:
        gr.Warning("Error!\n" + str(exc))
        return [None, None]  
    
    return smiles, file_path

def lorentzian(wavenumber, position, intensity, width=10):
    return intensity * (width ** 2) / ((wavenumber - position) ** 2 + width ** 2)

def on_frequency_analyze(smiles_texbox: gr.Textbox, functional_textbox: gr.Textbox, basis_set_textbox: gr.Textbox,
                              charge_slider: gr.Slider, multiplicity_dropdown: gr.Dropdown, n_cores_slider: gr.Slider):
    energy_textbox = "Not calculated"

    try:
        # Get the AutoDE species
        species = ade.Molecule(smiles=smiles_texbox, charge=charge_slider, mult=multiplicity_dropdown)

        # Get the functional and basis set
        ade.Config.n_cores = n_cores_slider
        kwds = ade.Config.G16.keywords
        kwds.opt.functional = functional_textbox
        kwds.opt = ['Opt', 'Freq', functional_textbox, basis_set_textbox]
        
        # Run geometry optimization and frequency analysis
        start = time.time()
        species.optimise(method=ade.methods.G16())
        species.energy.to('kcal mol-1')      
        end = time.time()
        duration = end - start

        # Get results
        energy = species.energy.real
        energy_textbox = "{:.4f} (kcal/mol)".format(energy)

        log_files = list(Path('.').glob('*.log'))
        newest_log_file_path = max(log_files, key=lambda f: f.stat().st_mtime)
        parser = cclib.io.ccopen(newest_log_file_path.name)
        data = parser.parse()
        dipole_moment = data.moments[1]
        dipole_magnitude = (dipole_moment[0]**2 + dipole_moment[1]**2 + dipole_moment[2]**2) ** 0.5
        dipole_moment_textbox = "{:.4f} (Debye)".format(dipole_magnitude)

        MO_energies = data.moenergies
        MO_df = pd.DataFrame(columns=["Molecular orbital", "Energy (hartree)"])
        for i, MO_energy in enumerate(MO_energies[0]):
            if i in data.homos:
                MO_df = MO_df._append({"Molecular orbital": f"MO {i+1} (HOMO)", "Energy (hartree)": "{:.4f}".format(MO_energy)}, ignore_index=True)
            else:
                MO_df = MO_df._append({"Molecular orbital": f"MO {i+1}", "Energy (hartree)": "{:.4f}".format(MO_energy)}, ignore_index=True)

        frequencies = data.vibfreqs
        ir_intensities = data.vibirs
        wavenumbers = np.linspace(min(frequencies) - 100, max(frequencies) + 100, 2000)
        spectrum = np.zeros_like(wavenumbers)
        for freq, intensity in zip(frequencies, ir_intensities):
            spectrum += lorentzian(wavenumbers, freq, intensity)
        ir_spectrum = plt.figure()
        plt.plot(wavenumbers, spectrum, 'k')
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.xlabel("Wavenumber ($cm^{-1}$)")
        plt.ylabel("Transmittance")
        plt.title("IR Spectrum")

        zpve = data.zpve
        h = data.enthalpy
        g = data.freeenergy
        s = data.entropy
        thermo_html = """
        <div>
            <table>
                <thead>
                    <tr><td><b>Thermodynamic quantity</b></td><td><b>Value</b></td></tr>
                </thead>
                <tbody>
                    <tr><td>Zero-point potential energy</td><td>{:.4f} hartree</td></tr>
                    <tr><td>Enthalpy</td><td>{:.4f} hartree</td></tr>
                    <tr><td>Entropy</td><td>{:.4f} hartree</td></tr>
                    <tr><td>Gibbs free energy</td><td>{:.4f} hartree</td></tr>
                </tbody>
            </table>
        </div>
        """.format(zpve, h, s, g)

    except Exception as exc:
        gr.Warning("Optimization error!\n" + str(exc))
        return [None, None, None, None, None, None]

    calculation_status = "Optimization finished. ({0:.3f} s)".format(duration)
    return calculation_status, energy_textbox, dipole_moment_textbox, MO_df, ir_spectrum, thermo_html

reps = [
    {
      "model": 0,
      "chain": "",
      "resname": "",
      "style": "stick",
      "color": "whiteCarbon",
      "residue_range": "",
      "around": 0,
      "byres": False,
      "visible": False
    }
]

def frequency_analysis_tab_content():
    with gr.Tab("Frequency Analysis") as single_point_calculation_tab:
        with gr.Accordion("Molecule"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    molecule_editor = molecule2d(label="Molecule")
                with gr.Column(scale=1):
                    create_molecule_button = gr.Button(value="Create molecule")
                    smiles_texbox = gr.Textbox(label="SMILES")
                    molecule_viewer = Molecule3D(label="Molecule", reps=reps)
                    load_molecule_uploadbutton = gr.UploadButton(label="Load molecule")
        with gr.Accordion("Frequency Analysis"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    functional_textbox = gr.Textbox(label="Functional", value="B3LYP", visible=True)
                    basis_set_textbox = gr.Textbox(label="Basis set", value="STO-3G", visible=True)
                    charge_slider = gr.Slider(label="Charge", value=0, minimum=-2, maximum=2, step=1)
                    multiplicity_dropdown = gr.Dropdown(label="Multiplicity", value=1, choices=[("Singlet", 1), ("Doublet", 2),
                                                                                                 ("Triplet", 3), ("Quartet", 4),
                                                                                                 ("Quintet", 5), ("Sextet ", 6)])
                with gr.Column(scale=1):
                    n_cores_slider = gr.Slider(label="Number of cores", value=1, minimum=1, maximum=16, step=1)
                    analyze_button = gr.Button(value="Analyze")
        with gr.Accordion("Frequency Results"):
            with gr.Row():
                status_markdown = gr.Markdown()
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    energy_texbox = gr.Textbox(label="Energy", value="Not calculated")
                    dipole_moment_texbox = gr.Textbox(label="Dipole moment", value="Not calculated")
                with gr.Column(scale=1):
                    MO_dataframe = gr.DataFrame(label="Molecular orbitals")
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    ir_spectrum_plot = gr.Plot(label="IR spectrum")
                with gr.Column(scale=1):
                    thermo_html = gr.HTML() 
                
        create_molecule_button.click(on_create_molecule, molecule_editor, [smiles_texbox, molecule_viewer])
        load_molecule_uploadbutton.upload(on_upload_molecule, load_molecule_uploadbutton, [smiles_texbox, molecule_viewer])
        analyze_button.click(on_frequency_analyze, [smiles_texbox, functional_textbox, basis_set_textbox,
                                                           charge_slider, multiplicity_dropdown, n_cores_slider],
                                                           [status_markdown, energy_texbox, dipole_moment_texbox, MO_dataframe, ir_spectrum_plot, thermo_html])
        
    return single_point_calculation_tab