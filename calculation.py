import os
import time
import shutil
import signal
import threading
import multiprocessing
import psutil
import subprocess
import math
import gradio as gr
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import cclib
from utils import *

max_n_procs = multiprocessing.cpu_count()
max_memory = math.floor(psutil.virtual_memory().total / (1024 ** 3))

# Basis set dropdown entry that stands for a basis set file supplied by the user, for instance one
# downloaded from the Basis Set Exchange in Gaussian format
CUSTOM_BASIS_SET = "Custom"

# Handle on the g16 process of the calculation in progress, so the stop button can reach it. The UI
# runs one Gaussian job at a time, so a single module level handle is enough.
running_calculation_lock = threading.Lock()
running_calculation_process = None
calculation_stopped_by_user = False
# how often the run handler refreshes the elapsed time in the status bar
CALCULATION_POLL_SECONDS = 1
# how long a stopped job may take to wind down before it is killed outright
CALCULATION_TERMINATE_TIMEOUT_SECONDS = 10

def on_working_directory_file_list_change(working_directory_file_list, input_file_name):
    structure_file_names = sorted([f for f in working_directory_file_list if f.endswith('.xyz') or f.endswith('.pdb') or f.endswith('.mol') or f.endswith('.mol2') or f.endswith('.log')], key=natural_sort_key)
    input_file_names = sorted([f for f in working_directory_file_list if f.endswith('.gjf')], key=natural_sort_key)
    if input_file_name + ".gjf" in input_file_names:
        input_file_name_value = input_file_name + ".gjf"
    else:
        input_file_name_value = input_file_names[0] if len(input_file_names) > 0 else None
    
    return gr.update(choices=structure_file_names, value=structure_file_names[0] if len(structure_file_names) > 0 else None, interactive=True), \
           gr.update(choices=input_file_names, value=input_file_name_value, interactive=True)
   
def on_change_calculation_type(calculation_type):
    if calculation_type == "Single-Point":
        file_name = "single_point"
    elif calculation_type == "Geometry Optimization":
        file_name = "geometry_optimization"
    elif calculation_type == "Frequency":
        file_name = "frequency"
    elif calculation_type == "Absorption Spectrum":
        file_name = "absorption_spectrum"
    elif calculation_type == "Emission Spectrum":
        file_name = "emission_spectrum"
    else: # calculation_type == "NMR Spectrum"
        file_name = "nmr_spectrum"

    if calculation_type == "NMR Spectrum":
        method_types = ["HF", "DFT", "MP2"]
    else:
        method_types = ["HF", "DFT", "Semi-empirical", "MP2", "MP4", "CCSD", "BD", "Compound"]

    show_n_state = calculation_type in ["Absorption Spectrum", "Emission Spectrum"]
    show_spin_spin_couplings = (calculation_type == "NMR Spectrum")

    return file_name, gr.update(choices=method_types, value="DFT"), gr.update(visible=show_n_state), gr.update(visible=show_spin_spin_couplings)

def on_mm_checkbox_change(mm_checkbox: gr.Checkbox):
    return gr.update(visible=mm_checkbox), gr.update(visible=mm_checkbox)

def on_solvation_checkbox_change(solvation_checkbox: gr.Checkbox):
    return gr.update(visible=solvation_checkbox), gr.update(visible=solvation_checkbox)

def on_method_type_change(method_type, basis_set):
    show_method_name = (method_type=="Semi-empirical" or method_type=="Compound")
    show_functional = method_type=="DFT"
    show_basis_set = not (method_type=="Semi-empirical" or method_type=="Compound")

    if method_type == "Semi-empirical":
        method_names = ["PM6", "PDDG", "AM1", "PM3", "PM3MM", "INDO", "CNDO"]
        default_method_name = "CNDO"
    elif method_type == "Compound":
        method_names = ["CBS-4M", "CBS-Q", "CBS-QB3", "CBS-APNO", "W1BD", "W1U", "W1RO", "G1", "G2", "G3", "G3S", "G4"]
        default_method_name = "G3"
    else:
        method_names = [""]
        default_method_name = ""

    return gr.update(choices=method_names, value=default_method_name, visible=show_method_name), gr.update(visible=show_functional), gr.update(visible=show_basis_set), \
           gr.update(visible=show_basis_set and basis_set == CUSTOM_BASIS_SET)

def on_basis_set_change(basis_set):
    return gr.update(visible=basis_set == CUSTOM_BASIS_SET)

def on_upload_custom_basis_set(working_directory_path, file_path):
    if working_directory_path is None:
        gr.Warning("Please open a working directory first.")
        return gr.update(), gr.update()

    # Keep the basis set next to the calculation it belongs to, like any other working directory file
    file_name = os.path.basename(file_path)
    shutil.copy2(file_path, os.path.join(working_directory_path, file_name))

    return file_name, get_files_in_working_directory(working_directory_path)

def read_custom_basis_set(working_directory_path, custom_basis_set_file_name):
    if custom_basis_set_file_name is None or custom_basis_set_file_name.strip() == "":
        raise ValueError("Please choose a custom basis set file.")

    custom_basis_set_file_path = os.path.join(working_directory_path, custom_basis_set_file_name)
    if not os.path.exists(custom_basis_set_file_path):
        raise ValueError(f'Custom basis set file "{custom_basis_set_file_name}" is not in the working directory.')

    with open(custom_basis_set_file_path, 'r') as file:
        custom_basis_set = file.read()
    if custom_basis_set.strip() == "":
        raise ValueError(f'Custom basis set file "{custom_basis_set_file_name}" is empty.')

    return custom_basis_set

def on_generate_input_file(working_directory_path, input_structure_file_dropdown, calculation_type,
                           use_mm, force_field, max_iters,
                           use_solvation, solvation_model, solvent,
                           method_type, method_name, functional, basis_set_textbox, custom_basis_set_file_name, n_states, spin_spin_coupling, charge, multiplicity,
                           n_cores, memory, input_file_name):
    if input_structure_file_dropdown is None or input_structure_file_dropdown == "":
        gr.Warning("Please select an input structure")
        return "", get_files_in_working_directory(working_directory_path)

    try:
        # Read the user supplied basis set, it is written into the input file instead of being named
        # in the route section
        if basis_set_textbox == CUSTOM_BASIS_SET and uses_basis_set(method_type):
            custom_basis_set = read_custom_basis_set(working_directory_path, custom_basis_set_file_name)
        else:
            custom_basis_set = None

        # Get the molecule object 
        file_path = os.path.join(working_directory_path, input_structure_file_dropdown)
        if input_structure_file_dropdown.endswith('.pdb'):
            mol = Chem.MolFromPDBFile(file_path, sanitize=False, removeHs=False)
        elif input_structure_file_dropdown.endswith('.mol'):    
            mol = Chem.MolFromMolFile(file_path, sanitize=False, removeHs=False)
        elif input_structure_file_dropdown.endswith('.mol2'): 
            mol = Chem.MolFromMol2File(file_path, sanitize=False, removeHs=False)
        elif input_structure_file_dropdown.endswith('.xyz'):    
            mol = add_bonds(mol_from_xyz_file(file_path))
        else: # file_name.endswith('.log')
            mol = add_bonds(mol_from_gaussian_file(file_path))

        Chem.SanitizeMol(mol)
        if mol.GetNumConformers()==0:
            AllChem.EmbedMolecule(mol)

        # Optimize geometry with molecular mechanics
        if use_mm:
            if force_field=="MMFF":
                AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iters)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)

        # Generate input file
        input_file_path = os.path.join(working_directory_path, input_file_name)
        if calculation_type == "Single-Point":
            write_sp_gaussian_input(mol, input_file_path, method_type, method_name, functional=functional, basis=basis_set_textbox,
                                    charge=charge, multiplicity=multiplicity,
                                    solvation=use_solvation, solvation_model=solvation_model, solvent=solvent, n_proc=n_cores, memory=memory, custom_basis_set=custom_basis_set)
        elif calculation_type == "Geometry Optimization":
            write_opt_gaussian_input(mol, input_file_path, method_type, method_name, functional=functional, basis=basis_set_textbox,
                                     charge=charge, multiplicity=multiplicity,
                                     solvation=use_solvation, solvation_model=solvation_model, solvent=solvent, n_proc=n_cores, memory=memory, custom_basis_set=custom_basis_set)
        elif calculation_type == "Frequency":
            write_opt_freq_gaussian_input(mol, input_file_path, method_type, method_name, functional=functional, basis=basis_set_textbox,
                                          charge=charge, multiplicity=multiplicity,
                                          solvation=use_solvation, solvation_model=solvation_model, solvent=solvent, n_proc=n_cores, memory=memory, custom_basis_set=custom_basis_set)
        elif calculation_type == "Absorption Spectrum":
            write_uv_vis_gaussian_input(mol, input_file_path, method_type, method_name, functional=functional, basis=basis_set_textbox,
                                        n_states=n_states, charge=charge, multiplicity=multiplicity,
                                        solvation=use_solvation, solvation_model=solvation_model, solvent=solvent, n_proc=n_cores, memory=memory, custom_basis_set=custom_basis_set)
        elif calculation_type == "Emission Spectrum":
            write_fluorescence_gaussian_input(mol, input_file_path, method_type, method_name, functional=functional, basis=basis_set_textbox,
                                              n_states=n_states, charge=charge, multiplicity=multiplicity,
                                              solvation=use_solvation, solvation_model=solvation_model, solvent=solvent, n_proc=n_cores, memory=memory, custom_basis_set=custom_basis_set)
        else: # calculation_type == "NMR Spectrum"
            write_nmr_gaussian_input(mol, input_file_path, method_type, functional=functional, basis=basis_set_textbox,
                                     spin_spin_coupling=spin_spin_coupling, charge=charge, multiplicity=multiplicity,
                                     solvation=use_solvation, solvation_model=solvation_model, solvent=solvent, n_proc=n_cores, memory=memory, custom_basis_set=custom_basis_set)

        status = "Input file generated."
        return f"<span style='color:green;'>{status}</span>", get_files_in_working_directory(working_directory_path)
    except Exception as exc:
        status = exc
        return f"<span style='color:red;'>{status}</span>", get_files_in_working_directory(working_directory_path)

def terminate_calculation_process(process):
    # g16 runs the Gaussian link executables as child processes, so terminating g16 on its own would
    # leave them behind. The job was started in its own process group, signal all of it at once.
    try:
        process_group_id = os.getpgid(process.pid)
    except (ProcessLookupError, PermissionError):
        return

    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return

    # give the links a chance to shut down cleanly before insisting
    deadline = time.time() + CALCULATION_TERMINATE_TIMEOUT_SECONDS
    while time.time() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.5)

    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass

def calculation_running_status(input_file_name, elapsed):
    # blue rather than the orange used for a stopped run, the two must not look alike
    return f"<span style='color:#1e90ff;'>Running {input_file_name}... ({round(elapsed)} s elapsed).</span>"

def on_run_calculation(working_directory_path, input_file_name):
    global running_calculation_process, calculation_stopped_by_user

    idle_buttons = (gr.update(interactive=True), gr.update(interactive=False))

    if input_file_name is None or input_file_name=="":
        gr.Warning("Please choose an input file.")
        yield "", gr.skip(), *idle_buttons
        return

    with running_calculation_lock:
        if running_calculation_process is not None:
            gr.Warning("A calculation is already running.")
            yield gr.skip(), gr.skip(), gr.skip(), gr.skip()
            return

    input_file_path = os.path.join(working_directory_path, input_file_name)
    output_file_path = os.path.join(working_directory_path, os.path.splitext(input_file_name)[0] + ".log")

    # Run calculation
    cmd = ["g16", f"{input_file_path}", f"{output_file_path}"]
    start = time.time()
    print(f"Running command: {' '.join(cmd)}")
    try:
        # start_new_session puts g16 in its own process group, which is what makes stopping work
        process = subprocess.Popen(cmd, start_new_session=True)
    except Exception as exc:
        status = f"Error running calculation: {exc}"
        yield f"<span style='color:red;'>{status}</span>", get_files_in_working_directory(working_directory_path), *idle_buttons
        return

    with running_calculation_lock:
        running_calculation_process = process
        calculation_stopped_by_user = False

    try:
        while process.poll() is None:
            yield calculation_running_status(input_file_name, time.time() - start), gr.skip(), gr.update(interactive=False), gr.update(interactive=True)
            time.sleep(CALCULATION_POLL_SECONDS)
    finally:
        with running_calculation_lock:
            was_stopped = calculation_stopped_by_user
            running_calculation_process = None
            calculation_stopped_by_user = False

    duration = round(time.time() - start, 3)
    files = get_files_in_working_directory(working_directory_path)

    if was_stopped:
        status = f"Calculation stopped after {duration} s."
        yield f"<span style='color:orange;'>{status}</span>", files, *idle_buttons
    elif process.returncode != 0:
        status = f"Error running calculation: g16 exited with code {process.returncode}, see {os.path.basename(output_file_path)}."
        yield f"<span style='color:red;'>{status}</span>", files, *idle_buttons
    else:
        status = f"Calculation finished ({duration} s)."
        yield f"<span style='color:green;'>{status}</span>", files, *idle_buttons

def on_stop_calculation():
    global calculation_stopped_by_user

    with running_calculation_lock:
        process = running_calculation_process
        if process is None:
            gr.Warning("No calculation is running.")
            return
        calculation_stopped_by_user = True

    # terminate in the background, the escalation to SIGKILL must not hold up the click handler
    threading.Thread(target=terminate_calculation_process, args=(process,), daemon=True).start()
    gr.Warning("Stopping the calculation...")

def calculation_tab_content(working_directory_path_state, working_directory_file_list_state, status_markdown):
    with gr.Tab("Calculation") as calculation_tab:
        with gr.Accordion("Settings", open=False):
            with gr.Row():
                n_cores_slider = gr.Slider(label="Number of cores", value=max_n_procs, minimum=1, maximum=max_n_procs, step=1)
                memory_slider = gr.Slider(label="Memory (GB)", value=max_memory, minimum=1, maximum=max_memory, step=1)
        with gr.Accordion("Generate Input File"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_structure_file_dropdown = gr.Dropdown(label="Input structure", choices=[""], value="", interactive=False)
                with gr.Column(scale=4):
                    calculation_type_radio = gr.Radio(label="Type of calculation", value="Single-Point", choices=["Single-Point", "Geometry Optimization", "Frequency", "Absorption Spectrum", "Emission Spectrum", "NMR Spectrum"])
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        mm_checkbox = gr.Checkbox(label="Optimize geometry with molecular mechanics", value=False)
                    with gr.Row():
                        with gr.Column(scale=1):
                            force_field_dropdown = gr.Dropdown(label="Force field", value="MMFF", choices=["MMFF", "UFF"], visible=False)
                        with gr.Column(scale=1):
                            max_iters_slider = gr.Slider(label="Max iterations", value=200, minimum=0, maximum=1000, step=1, visible=False)
                    with gr.Row():
                        solvation_checkbox = gr.Checkbox(label="Solvation", value=False)
                    with gr.Row():
                        with gr.Column(scale=1):
                            solvation_dropdown = gr.Dropdown(label="Solvation model", value="iefpcm", choices=[("IEFPCM", "iefpcm"), ("SMD", "smd"),  ("I-PCM", "ipcm"), ("SCI-PCM", "scipcm"), ("CPCM", "cpcm")], visible=False)
                        with gr.Column(scale=1):
                            solvent_dropdown = gr.Dropdown(label="Solvent", value="water", choices=["water", ("DMSO", "dmso"),  "nitromethane", "acetonitrile", "methanol", "ethanol", "acetone", "dichloromethane",
                                                                                                    "dichloroethane", ("THF", "thf"), "aniline", "chlorobenzene", "chloroform", ("diethyl ether", "diethylether"),
                                                                                                    "toluene", "benzene", ("CCl4", "ccl4"), "cyclohexane", "heptane"], allow_custom_value=True, visible=False)
                with gr.Column(scale=1):
                    with gr.Row():
                        method_type_dropdown = gr.Dropdown(label="Type of method", value="DFT", choices=["HF", "DFT", "Semi-empirical", "MP2", "MP4", "CCSD", "BD", "Compound"])
                        method_name_dropdown = gr.Dropdown(label="Method", choices=[], visible=False)
                    with gr.Row():
                        functional_dropdown = gr.Dropdown(label="Functional", value="B3LYP", choices=["LSDA", "BVP86", "B3LYP", "CAM-B3LYP", "B3PW91", "B97D", "MPW1PW91", "PBEPBE", "HSEH1PBE", "HCTH", "TPSSTPSS", "WB97XD", "M06-2X"], allow_custom_value=True)
                        basis_set_dropdown = gr.Dropdown(label="Basis set", value="3-21G", choices=["STO-3G", "3-21G", "6-31G", "6-31G'", "6-31G(d,p)", "6-31G(3d,p)", "6-31G(d,3p)", "6-31G(3d,3p)", "6-31+G(d,p)", "6-31++G(d,p)",
                                                                                                    "6-311G", "6-311G(d,p)", "cc-pVDZ", "cc-pVTZ", "cc-pVQZ", "aug-cc-pVDZ", "aug-cc-pVTZ", "aug-cc-pVQZ",
                                                                                                    "LanL2DZ", "LanL2MB", "SDD", "DGDZVP", "DGDZVP2", "DGTZVP", CUSTOM_BASIS_SET], allow_custom_value=True)
                    with gr.Row(visible=False) as custom_basis_set_row:
                        custom_basis_set_upload_button = gr.UploadButton(label="Choose custom basis set", file_types=[".gbs", ".bas", ".basis", ".txt", ".dat"])
                        custom_basis_set_textbox = gr.Textbox(label="Custom basis set file", info="Gaussian format, for example downloaded from the Basis Set Exchange", interactive=False)
                    with gr.Row():
                        n_states_slider = gr.Slider(label="Number of excited states", value=10, minimum=5, maximum=100, step=1, visible=False)
                        spin_spin_coupling_checkbox = gr.Checkbox(label="Compute spin-spin couplings", value=False, visible=False)
                    with gr.Row():
                        charge_slider = gr.Slider(label="Charge", value=0, minimum=-2, maximum=2, step=1)
                        multiplicity_dropdown = gr.Dropdown(label="Multiplicity", value=1, choices=[("Singlet", 1), ("Doublet", 2),
                                                                                                    ("Triplet", 3), ("Quartet", 4),
                                                                                                    ("Quintet", 5), ("Sextet ", 6)])
                with gr.Column(scale=1):
                    input_file_name_textbox = gr.Textbox(label="File name", value="single_point")
                    generate_input_file_button = gr.Button(value="Generate input file")
        with gr.Accordion("Run calculation"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_file_name_dropdown = gr.Dropdown(label="Input file", choices=[""], value="", interactive=False)
                with gr.Column(scale=2):
                    with gr.Row():
                        run_button = gr.Button("Run")
                        stop_button = gr.Button("Stop", interactive=False)

        working_directory_file_list_state.change(on_working_directory_file_list_change, [working_directory_file_list_state, input_file_name_textbox], [input_structure_file_dropdown, input_file_name_dropdown])
        calculation_type_radio.change(on_change_calculation_type, calculation_type_radio, [input_file_name_textbox, method_type_dropdown, n_states_slider, spin_spin_coupling_checkbox])        
        mm_checkbox.change(on_mm_checkbox_change, mm_checkbox, [force_field_dropdown, max_iters_slider])
        solvation_checkbox.change(on_solvation_checkbox_change, solvation_checkbox, [solvation_dropdown, solvent_dropdown])
        method_type_dropdown.change(on_method_type_change, [method_type_dropdown, basis_set_dropdown], [method_name_dropdown, functional_dropdown, basis_set_dropdown, custom_basis_set_row])
        basis_set_dropdown.change(on_basis_set_change, basis_set_dropdown, custom_basis_set_row)
        custom_basis_set_upload_button.upload(on_upload_custom_basis_set, [working_directory_path_state, custom_basis_set_upload_button], [custom_basis_set_textbox, working_directory_file_list_state])
        generate_input_file_button.click(on_generate_input_file, [working_directory_path_state, input_structure_file_dropdown, calculation_type_radio,
                                                                  mm_checkbox, force_field_dropdown, max_iters_slider, solvation_checkbox, solvation_dropdown, solvent_dropdown,
                                                                  method_type_dropdown, method_name_dropdown, functional_dropdown, basis_set_dropdown, custom_basis_set_textbox, n_states_slider, spin_spin_coupling_checkbox, charge_slider, multiplicity_dropdown,
                                                                  n_cores_slider, memory_slider, input_file_name_textbox],
                                                                 [status_markdown, working_directory_file_list_state])
        run_button.click(on_run_calculation, [working_directory_path_state, input_file_name_dropdown], [status_markdown, working_directory_file_list_state, run_button, stop_button])
        stop_button.click(on_stop_calculation, None, None)
        
    return calculation_tab