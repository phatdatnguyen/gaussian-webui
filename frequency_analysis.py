import os
import multiprocessing
import psutil
import subprocess
import time
import math
import gradio as gr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from gradio_molecule2d import molecule2d
from gradio_molecule3d import Molecule3D
from rdkit import Chem
from rdkit.Chem import AllChem
import cclib
from utils import *

max_n_procs = multiprocessing.cpu_count()
max_memory = math.floor(psutil.virtual_memory().total / (1024 ** 3))

def on_create_molecule(molecule_editor: molecule2d):
    os.makedirs("structures", exist_ok=True)
    file_path = ".\\structures\\molecule_freq.pdb"
    try:
        global mol_freq
        mol_freq = Chem.MolFromSmiles(molecule_editor)
        mol_freq = Chem.AddHs(mol_freq)
        smiles = Chem.CanonSmiles(molecule_editor)
        AllChem.EmbedMolecule(mol_freq)
        Chem.MolToPDBFile(mol_freq, file_path)

        analyze_button = gr.Button(value="Analyze", interactive=True)
    except Exception as exc:
        gr.Warning("Error!\n" + str(exc))

        analyze_button = gr.Button(value="Analyze", interactive=False)

        return [None, None, analyze_button]
    
    return smiles, file_path, analyze_button

def on_upload_molecule(load_molecule_uploadbutton: gr.UploadButton):
    os.makedirs("structures", exist_ok=True)
    file_path = ".\\structures\\molecule_freq.pdb"
    uploaded_file_path = load_molecule_uploadbutton
    _, file_extension = os.path.splitext(uploaded_file_path)

    try:
        global mol_freq

        if file_extension.lower() == ".pdb":
            mol_freq = Chem.MolFromPDBFile(uploaded_file_path, sanitize=False, removeHs=False)
        elif file_extension.lower() == ".xyz":    
            mol_freq = mol_from_xyz_file(uploaded_file_path)
        elif file_extension.lower() == ".mol":    
            mol_freq = Chem.MolFromMolFile(uploaded_file_path, sanitize=False, removeHs=False)
        elif file_extension.lower() == ".mol2":    
            mol_freq = Chem.MolFromMol2File(uploaded_file_path, sanitize=False, removeHs=False)
        elif file_extension.lower() == ".log":    
            mol_freq = mol_from_gaussian_file(uploaded_file_path)
        else:
            raise Exception("File must be in supported formats (pdb, xyz, mol, mol2, log).")
        
        smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol_freq))    
        if mol_freq.GetNumConformers()==0:
            AllChem.EmbedMolecule(mol_freq)
        Chem.MolToPDBFile(mol_freq, file_path)

        analyze_button = gr.Button(value="Analyze", interactive=True)
    except Exception as exc:
        gr.Warning("Error!\n" + str(exc))

        analyze_button = gr.Button(value="Analyze", interactive=False)

        return [None, None, analyze_button]  
    
    return smiles, file_path, analyze_button

def on_mm_checkbox_change(mm_checkbox: gr.Checkbox):
    if mm_checkbox:
        force_field_dropdown = gr.Dropdown(label="Force field", value="MMFF", choices=["MMFF", "UFF"], visible=True)
        max_iters_slider = gr.Slider(label="Max iterations", value=200, minimum=0, maximum=1000, step=1, visible=True)
    else:
        force_field_dropdown = gr.Dropdown(label="Force field", value="MMFF", choices=["MMFF", "UFF"], visible=False)
        max_iters_slider = gr.Slider(label="Max iterations", value=200, minimum=0, maximum=1000, step=1, visible=False)

    return force_field_dropdown, max_iters_slider

def on_solvation_checkbox_change(solvation_checkbox: gr.Checkbox):
    if solvation_checkbox:
        solvent_dropdown = gr.Dropdown(label="Solvent", value="water", choices=["water", ("DMSO", "dmso"),  "nitromethane", "acetonitrile", "methanol", "ethanol", "acetone", "dichloromethane",
                                                                                "dichloroethane", ("THF", "thf"), "aniline", "chlorobenzene", "chloroform", ("diethyl ether", "diethylether"),
                                                                                "toluene", "benzene", ("CCl4", "ccl4"), "cyclohexane", "heptane"], allow_custom_value=True, visible=True)
    else:
        solvent_dropdown = gr.Dropdown(label="Solvent", value="water", choices=["water", ("DMSO", "dmso"),  "nitromethane", "acetonitrile", "methanol", "ethanol", "acetone", "dichloromethane",
                                                                                "dichloroethane", ("THF", "thf"), "aniline", "chlorobenzene", "chloroform", ("diethyl ether", "diethylether"),
                                                                                "toluene", "benzene", ("CCl4", "ccl4"), "cyclohexane", "heptane"], allow_custom_value=True, visible=False)

    return solvent_dropdown

def write_opt_freq_gaussian_input(mol, file_name, method='B3LYP', basis='6-31G(d,p)', charge=0, multiplicity=1, solvent=None, n_proc=4, memory=2):
    # Ensure the molecule has 3D coordinates
    if not mol.GetNumConformers():
        raise ValueError("Molecule does not have 3D coordinates. Please generate conformers first.")

    # Open the file for writing
    with open(file_name + '.gjf', 'w') as f:
        # Link0 commands
        f.write(f'%NProcShared={n_proc}\n')
        f.write(f'%Mem={memory}GB\n')
        f.write(f'%Chk={file_name}.chk\n')
        # Route section
        route_section = f'#P {method}/{basis} Opt Freq'
        if solvent:
            route_section += f' scrf=(iefpcm,solvent={solvent})'
        route_section += '\n\n'
        f.write(route_section)
        # Title section
        f.write('Geometry Optimization and Frequency Analysis\n\n')
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

def lorentzian(wavenumber, position, intensity, width=10):
    return intensity * (width ** 2) / ((wavenumber - position) ** 2 + width ** 2)

def on_frequency_analyze(mm_checkbox: gr.Checkbox, force_field_dropdown: gr.Dropdown, max_iters_slider: gr.Slider, solvation_checkbox: gr.Checkbox, solvent_dropdown: gr.Dropdown,
                         functional_textbox: gr.Dropdown, basis_set_textbox: gr.Dropdown, charge_slider: gr.Slider, multiplicity_dropdown: gr.Dropdown,
                         n_cores_slider: gr.Slider, memory_slider: gr.Slider, file_name_textbox: gr.Textbox):
    try:
        if mm_checkbox:
            if force_field_dropdown=="MMFF":
                AllChem.MMFFOptimizeMolecule(mol_freq, maxIters=max_iters_slider)
            else:
                AllChem.UFFOptimizeMolecule(mol_freq, maxIters=max_iters_slider)

        # Write Gaussian input file
        if solvation_checkbox:
            solvent = solvent_dropdown
        else:
            solvent = None
        write_opt_freq_gaussian_input(mol_freq, file_name=file_name_textbox, method=functional_textbox, basis=basis_set_textbox, charge=charge_slider, multiplicity=multiplicity_dropdown, solvent=solvent,
                                      n_proc=n_cores_slider, memory=memory_slider)

        # Run calculation
        start = time.time()
        subprocess.run(['g16', file_name_textbox + '.gjf'], check=True)
        print(f"Gaussian job '{file_name_textbox + '.gjf'}' has been submitted.")
        end = time.time()
        duration = end - start

        # Get results
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
        plt.tight_layout()

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
        return [None, None, None, None, None, None, None]

    calculation_status = "Optimization finished. ({0:.3f} s)".format(duration)
    return calculation_status, energy_textbox, dipole_moment_textbox, energy_plot, MO_df, ir_spectrum, thermo_html

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
    with gr.Tab("Frequency Analysis") as frequency_analysis_tab:
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
                    mm_checkbox = gr.Checkbox(label="Optimize geometry with molecular mechanics", value=True)
                    force_field_dropdown = gr.Dropdown(label="Force field", value="MMFF", choices=["MMFF", "UFF"])
                    max_iters_slider = gr.Slider(label="Max iterations", value=200, minimum=0, maximum=1000, step=1)
                    solvation_checkbox = gr.Checkbox(label="Solvation (IEFPCM)", value=False)
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
                    file_name_textbox = gr.Textbox(label="File name", value="molecule_freq")
                    analyze_button = gr.Button(value="Analyze", interactive=False)
        with gr.Accordion("Frequency Results"):
            with gr.Row():
                status_markdown = gr.Markdown()
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    energy_texbox = gr.Textbox(label="Energy", value="Not calculated")
                    dipole_moment_texbox = gr.Textbox(label="Dipole moment", value="Not calculated")
                    energy_plot = gr.Plot(label="SCF energy")
                with gr.Column(scale=1):
                    MO_dataframe = gr.DataFrame(label="Molecular orbitals")
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    ir_spectrum_plot = gr.Plot(label="IR spectrum")
                with gr.Column(scale=1):
                    thermo_html = gr.HTML() 
                
        create_molecule_button.click(on_create_molecule, molecule_editor, [smiles_texbox, molecule_viewer, analyze_button])
        load_molecule_uploadbutton.upload(on_upload_molecule, load_molecule_uploadbutton, [smiles_texbox, molecule_viewer, analyze_button])
        mm_checkbox.change(on_mm_checkbox_change, mm_checkbox, [force_field_dropdown, max_iters_slider])
        solvation_checkbox.change(on_solvation_checkbox_change, solvation_checkbox, solvent_dropdown)
        analyze_button.click(on_frequency_analyze, [mm_checkbox, force_field_dropdown, max_iters_slider, solvation_checkbox, solvent_dropdown,
                                                    functional_textbox, basis_set_textbox, charge_slider, multiplicity_dropdown,
                                                    n_cores_slider, memory_slider, file_name_textbox],
                                                   [status_markdown, energy_texbox, dipole_moment_texbox, energy_plot, MO_dataframe, ir_spectrum_plot, thermo_html])
        
    return frequency_analysis_tab