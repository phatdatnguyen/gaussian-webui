import os
import time
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import gradio as gr
import cclib
import nglview
from utils import get_files_in_working_directory, mol_from_gaussian_file, generate_ir_spectrum_interactive, generate_absorption_emission_spectrum_interactive, parse_nmr_shielding_constants, calculate_chemical_shifts, generate_nmr_spectrum_interactive

def on_working_directory_file_list_change(working_directory_file_list):
    output_file_names = [f for f in working_directory_file_list if f.endswith('.log') ]
    return gr.update(choices=output_file_names, value=output_file_names[0] if len(output_file_names) > 0 else None, interactive=True)

def on_load_result_file(working_directory_path, calculation_result_file_name):
    data = None
    try:
        calculation_result_file_path = os.path.join(working_directory_path, calculation_result_file_name)
        parser = cclib.io.ccopen(calculation_result_file_path)
        data = parser.parse()

        # Display energy result
        if hasattr(data, "scfenergies") and hasattr(data, "moenergies"):
            show_enery_result = True
            energy = data.scfenergies[-1] / 27.2114 # in unit hartree
            energy_text = '{:.4f} (hartree)'.format(energy)
            dipole_moment = data.moments[1]
            dipole_magnitude = (dipole_moment[0]**2 + dipole_moment[1]**2 + dipole_moment[2]**2) ** 0.5
            dipole_moment_text = "{:.4f} (Debye)".format(dipole_magnitude)

            visualization_dropdown_choices=["Electron density", "Electrostatic potential"]
            MO_energies = data.moenergies
            if MO_energies is not None:
                rows = []
                for i, MO_energy in enumerate(MO_energies[0]):
                    label = f"MO {i+1} (HOMO)" if i in data.homos else f"MO {i+1}"
                    rows.append({"Molecular orbital": label, "Energy (hartree)": "{:.4f}".format(MO_energy)})
                    visualization_dropdown_choices.append(f"MO {i+1}")
                MO_df = pd.DataFrame(rows, columns=["Molecular orbital", "Energy (hartree)"])
            else:
                MO_df = None
        else:
            show_enery_result = False
            energy_text = ""
            dipole_moment_text = ""
            MO_df = None
            visualization_dropdown_choices = ["Electron density", "Electrostatic potential"]

        # Display geometry optimization result
        if hasattr(data, "scfenergies") and len(data.scfenergies) > 1:
            show_geometry_optimization = True
            scf_energies = [x / 27.2114 for x in data.scfenergies]
            energy_plot = plt.figure()
            plt.plot(range(1, len(scf_energies) + 1), scf_energies, 'k')
            plt.xlabel("Conformer")
            plt.ylabel("Energy (hartree)")
            plt.title("SCF Energy")
            ax = plt.gca()
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            plt.tight_layout()
        else:
            show_geometry_optimization = False
            energy_plot = None

        # Display frequency result
        if hasattr(data, "vibfreqs"):
            show_frequency = True
            frequencies = data.vibfreqs
            intensities = data.vibirs
            
            ir_df = pd.DataFrame({
                "Frequencies": frequencies,
                "Intensity": intensities
            })
            
            ir_spectrum = generate_ir_spectrum_interactive(frequencies, intensities, width=12.0, transmittance=True)

            thero_df = pd.DataFrame([["{:.4f}".format(data.zpve), "{:.4f}".format(data.enthalpy), "{:.4f}".format(data.entropy), "{:.4f}".format(data.freeenergy)]],
                                    columns=["Zero-point potential energy", "Enthalpy", "Entropy", "Gibbs free energy"])
        else:
            show_frequency = False
            ir_df = None
            ir_spectrum = None
            thero_df = None
        
        # Display absorption/emission result
        if hasattr(data, 'etenergies') and hasattr(data, 'etoscs'):
            show_absorption_emission = True
        else:
            show_absorption_emission = False

        # Display NMR result
        show_nmr = False
        with open(calculation_result_file_path) as output_file:
            if "NMR Chemical Shielding Calculation using GIAO" in output_file.read():
                show_nmr = True
        if show_nmr:
            # Get results
            shielding_data = parse_nmr_shielding_constants(calculation_result_file_path)
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
        else:
            nmr_df = None
            nmr_spectrum_1H = None
            nmr_spectrum_13C = None

        status = f"Result file loaded."
        return f"<span style='color:green;'>{status}</span>", data, \
               gr.update(visible=show_enery_result), energy_text, dipole_moment_text, MO_df, gr.update(choices=visualization_dropdown_choices), gr.update(interactive=True), \
               gr.update(visible=show_geometry_optimization), energy_plot, \
               gr.update(visible=show_frequency), ir_df, ir_spectrum, thero_df, gr.update(interactive=show_frequency), \
               gr.update(visible=show_absorption_emission), None, None, gr.update(interactive=show_absorption_emission), \
               gr.update(visible=show_nmr), nmr_df, nmr_spectrum_1H, nmr_spectrum_13C, gr.update(interactive=show_nmr)
    except Exception as exc:
        status = f"Error loading result file: {exc}"
        return f"<span style='color:red;'>{status}</span>", data, \
               gr.update(visible=False), "", "", None, gr.update(), gr.update(interactive=False), \
               gr.update(visible=False), None, \
               gr.update(visible=False), None, None, None, gr.update(interactive=False), \
               gr.update(visible=False), None, None, gr.update(interactive=False), \
               gr.update(visible=False), None, None, None, gr.update(interactive=False)

def on_visualization_change(visualization_dropdown):
    return gr.update(visible=(visualization_dropdown == "Electron density"))

def on_visualize_surface(working_directory_path, output_file_name, visualization, color1, color2, opacity, isolevel):
    # Set options for cube file generation
    try:
        output_file_path = os.path.join(working_directory_path, output_file_name)
        mol = mol_from_gaussian_file(output_file_path)

        file_stem = os.path.splitext(output_file_name)[0]
        check_file_path = os.path.join(working_directory_path, file_stem + ".chk")
        formated_check_file_path = os.path.join(working_directory_path, file_stem + ".fchk")
        cube_file_path = os.path.join(working_directory_path, file_stem + ".cube")
        
        # Convert to formatted checkpoint file
        subprocess.run(["formchk", check_file_path, formated_check_file_path], check=True)

        # Generate cube file
        os.makedirs("./static/surface_visualization", exist_ok=True)
        if visualization == "Electron density":
            subprocess.run(['cubegen', '0', 'Density=SCF', formated_check_file_path, cube_file_path], check=True)
            
            # Get the cube file
            view = nglview.show_rdkit(mol)
            view.add_component(cube_file_path)

            # Adjust visualization settings
            view.component_1.update_surface(opacity=opacity, color=color1, isolevel=isolevel)
            view.camera = 'orthographic'
        elif visualization == "Electrostatic potential":
            subprocess.run(['cubegen', '0', 'Potential=SCF', formated_check_file_path, cube_file_path], check=True)
            
            # Get the cube file
            view = nglview.show_rdkit(mol)
            view.add_component(cube_file_path)

            # Adjust visualization settings
            view.component_1.update_surface(opacity=opacity, color=color1, isolevel=isolevel)
            view.camera = 'orthographic'
        else: # MO visualization
            MO_index = int(visualization.split(" ")[1])
            subprocess.run(['cubegen', '0', f'MO={MO_index}', formated_check_file_path, cube_file_path], check=True)
            
            # Get the cube file
            view = nglview.show_rdkit(mol)
            view.add_component(cube_file_path)
            view.add_component(cube_file_path)

            # Adjust visualization settings
            view.component_1.update_surface(opacity=opacity, color=color1, isolevel=2)
            view.component_2.update_surface(opacity=opacity, color=color2, isolevel=-2)
            view.camera = 'orthographic'
        
        # Write the widget to HTML
        if os.path.exists('./static/surface_visualization/cube_file.html'):
            os.remove('./static/surface_visualization/cube_file.html')
        nglview.write_html('./static/surface_visualization/cube_file.html', [view])

        # Read the HTML file
        timestamp = int(time.time())
        html = f'<iframe src="/static/surface_visualization/cube_file.html?ts={timestamp}" height="600" width="600" title="NGL View"></iframe>'
    except Exception as exc:
        return "Visualization error!\n" + str(exc)
    
    return html

def on_show_absorption_spectrum(data):
    if hasattr(data, 'etenergies') and hasattr(data, 'etoscs'):
        # etenergies in cm^(-1), etoscs are dimensionless
        energies_wavenumber = np.array(data.etenergies)
        oscs = np.array(data.etoscs)
        
        # Convert energies to wavelength (eV to nm)
        wavelengths = 1e7 / energies_wavenumber
                    
        # Build spectrum
        absorption_spectrum = generate_absorption_emission_spectrum_interactive(wavelengths=wavelengths, oscs=oscs, points=10000, plot_range=(0, 800))

        # Build peak dataframe
        configurations_list = data.etsecs
        configurations_str_list = []
        for configurations in configurations_list:
            transitions_str_list = []
            for transition_begin, transition_end, coefficient in configurations:
                transitions_str_list.append(f"{transition_begin[0]} -> {transition_end[0]} (coefficient: {coefficient:.4f})")
            transitions_str = "\n".join(transitions_str_list)
            configurations_str_list.append(transitions_str)
        symmetry_list = [symmetry for symmetry in data.etsyms]
        peak_df = pd.DataFrame({
            "Absorption Wavelength (nm)": np.round(wavelengths, 4),
            "Oscillator Strength": oscs,
            "Transitions": configurations_str_list,
            "Symmetry": symmetry_list
        })

        # Sort by wavelength (ascending order)
        peak_df = peak_df.sort_values("Absorption Wavelength (nm)").reset_index(drop=True)

        return peak_df, absorption_spectrum
    else:
        return None, None

def on_show_emission_spectrum(data):
    if hasattr(data, 'etenergies') and hasattr(data, 'etoscs'):
        # etenergies in cm^(-1), etoscs are dimensionless
        energies_wavenumber = np.array(data.etenergies)

        # Keep only the lowest energy transition only (S1 -> S0)
        energies_wavenumber = np.array([energies_wavenumber[0]])
        oscs = np.array([1]) # set osc to 1 for the emission peak

        # Convert energies to wavelength (eV to nm)
        wavelengths = 1e7 / energies_wavenumber

        # Build spectrum
        emission_spectrum = generate_absorption_emission_spectrum_interactive(wavelengths=wavelengths, oscs=oscs, points=10000, plot_range=(0, 800))

        # Build peak dataframe
        configurations = data.etsecs[0]
        transitions_str_list = []
        for transition_begin, transition_end, coefficient in configurations:
            transitions_str_list.append(f"{transition_begin[0]} -> {transition_end[0]} (coefficient: {coefficient:.4f})")
        transitions_str = "\n".join(transitions_str_list)
        symmetry = data.etsyms[0]
        peak_df = pd.DataFrame({
            "Emission Wavelength (nm)": np.round(wavelengths, 4),
            "Oscillator Strength": oscs,
            "Transitions": transitions_str,
            "Symmetry": symmetry
        })

        # Sort by wavelength (ascending order)
        peak_df = peak_df.sort_values("Emission Wavelength (nm)").reset_index(drop=True)

        return peak_df, emission_spectrum
    else:
        return None, None

def on_export_data(working_directory_path, file_name, df):
    try:
        file_path = os.path.join(working_directory_path, file_name + ".csv")
        df.to_csv(file_path, encoding='utf-8', index=False)
        status = "Data exported successfully."
        return f"<span style='color:green;'>{status}</span>", get_files_in_working_directory(working_directory_path)
    except Exception as exc:
        status = f"Error exporting data: {exc}"
        return f"<span style='color:red;'>{status}</span>", get_files_in_working_directory(working_directory_path)

def result_tab_content(working_directory_path_state, working_directory_file_list_state, status_markdown):
    with gr.Tab("Result") as result_tab:
        with gr.Row():
            with gr.Column(scale=1):
                calculation_result_file_dropdown = gr.Dropdown(label="Calculation result", choices=[""], value="", interactive=False)
            with gr.Column(scale=1):
                load_button = gr.Button("Load")
                data_state = gr.State()
        with gr.Accordion(label="Energy", visible=False) as energy_result_accordion:
            with gr.Row():
                with gr.Column(scale=1):
                    energy_texbox = gr.Textbox(label="Energy", value="Not calculated")
                    dipole_moment_texbox = gr.Textbox(label="Dipole moment", value="Not calculated")
                    mo_dataframe = gr.DataFrame(label="Molecular orbitals")
                with gr.Column(scale=1):
                    visualization_dropdown = gr.Dropdown(label="Visualization", value="Electron density", choices=["Electron density", "Electrostatic potential"])
                    with gr.Row():
                        visualization_color1 = gr.ColorPicker(label="Color 1", value="#0000ff")
                        visualization_color2 = gr.ColorPicker(label="Color 2", value="#ff0000")
                        visualization_opacity = gr.Slider(label="Opacity", value=0.8, minimum=0, maximum=1, step=0.01)
                        visualization_isolevel = gr.Slider(label="Isolevel", value=0.05, minimum=0, maximum=1, step=0.01)
                    visualize_button = gr.Button(value="Visualize", interactive=False)
                    visualization_html = gr.HTML(label="Visualization")
        with gr.Accordion(label="Geometry Optimization", visible=False) as opt_result_accordion:
            with gr.Row():
                energy_plot = gr.Plot(label="Energy plot")
        with gr.Accordion(label="Frequency", visible=False) as frequency_result_accordion:
            with gr.Row():
                with gr.Column(scale=1):
                    ir_dataframe = gr.Dataframe(label="Vibrational frequencies")
                    ir_filename_textbox = gr.Textbox(label="File name", value="ir_data")
                    export_ir_button = gr.Button(value="Export", interactive=False)
                with gr.Column(scale=2):
                    ir_spectrum_plot = gr.Plot(label="IR spectrum")
            with gr.Row():
                thermo_dataframe = gr.Dataframe(label="Thermodynamics") 
        with gr.Accordion(label="Absorption/Emission Spectrum", visible=False) as absorption_emission_result_accordion:
            with gr.Row():
                with gr.Column(scale=1):
                    absorption_spectrum_button = gr.Button("Show absorption spectrum")
                    emission_spectrum_button = gr.Button("Show emission spectrum")
                with gr.Column(scale=2):
                    peak_dataframe = gr.Dataframe(label="Absorption Peaks", headers=["Wavelength (nm)", "Oscillator Strength"], datatype=["number", "number", "markdown", "markdown"], line_breaks=True, wrap=True)
                    peak_filename_textbox = gr.Textbox(label="File name", value="peak_data")
                    export_peak_button = gr.Button(value="Export", interactive=False)
            with gr.Row():
                absorption_emisson_spectrum_plot = gr.Plot(label="UV-Vis Spectrum")
        with gr.Accordion(label="NMR", visible=False) as nmr_result_accordion:
            with gr.Row():
                with gr.Column(scale=1):
                    nmr_dataframe = gr.DataFrame(label="NMR signals")
                    nmr_filename_textbox = gr.Textbox(label="File name", value="nmr_data")
                    export_nmr_button = gr.Button(value="Export", interactive=False)
                with gr.Column(scale=2):
                    with gr.Row():    
                        nmr_spectrum_1H = gr.Plot(label="1H NMR spectrum")
                    with gr.Row():
                        nmr_spectrum_13C = gr.Plot(label="13C NMR spectrum")
    
    working_directory_file_list_state.change(on_working_directory_file_list_change, working_directory_file_list_state, calculation_result_file_dropdown)
    load_button.click(on_load_result_file, [working_directory_path_state, calculation_result_file_dropdown],
                                           [status_markdown, data_state,
                                            energy_result_accordion, energy_texbox, dipole_moment_texbox, mo_dataframe, visualization_dropdown, visualize_button,
                                            opt_result_accordion, energy_plot,
                                            frequency_result_accordion, ir_dataframe, ir_spectrum_plot, thermo_dataframe, export_ir_button,
                                            absorption_emission_result_accordion, peak_dataframe, absorption_emisson_spectrum_plot, export_peak_button,
                                            nmr_result_accordion, nmr_dataframe, nmr_spectrum_1H, nmr_spectrum_13C, export_nmr_button])
    visualization_dropdown.change(on_visualization_change, visualization_dropdown, visualization_isolevel)
    visualize_button.click(on_visualize_surface, [working_directory_path_state, calculation_result_file_dropdown, visualization_dropdown, visualization_color1, visualization_color2, visualization_opacity, visualization_isolevel], visualization_html)
    export_ir_button.click(on_export_data, [working_directory_path_state, ir_filename_textbox, ir_dataframe], [status_markdown, working_directory_file_list_state])
    absorption_spectrum_button.click(on_show_absorption_spectrum, data_state, [peak_dataframe, absorption_emisson_spectrum_plot])
    export_peak_button.click(on_export_data, [working_directory_path_state, peak_filename_textbox, peak_dataframe], [status_markdown, working_directory_file_list_state])
    emission_spectrum_button.click(on_show_emission_spectrum, data_state, [peak_dataframe, absorption_emisson_spectrum_plot])
    export_nmr_button.click(on_export_data, [working_directory_path_state, nmr_filename_textbox, nmr_dataframe], [status_markdown, working_directory_file_list_state])

    return result_tab