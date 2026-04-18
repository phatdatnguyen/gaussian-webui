import os
import re
import cclib
from rdkit import Chem
import numpy as np
import plotly.graph_objects as go

def get_files_in_working_directory(working_directory_path):
    files = [f for f in os.listdir(working_directory_path) if not f.endswith('Zone.Identifier')]
    return files

def conformer_to_xyz_file(mol, conf_id, file_path, charge=0, multiplicity=1):
    # Get atom information
    atoms = mol.GetAtoms()
    xyz_lines = []
    for atom in atoms:
        pos = mol.GetConformer(conf_id).GetAtomPosition(atom.GetIdx())
        xyz_lines.append(f"{atom.GetSymbol()} {pos.x} {pos.y} {pos.z}")

    # Construct the XYZ string
    xyz_string = f"{charge} {multiplicity}\n" + "\n".join(xyz_lines)
    
    with open(file_path, 'w') as file:
        file.write(xyz_string)

def add_bonds(mol, bond_factor = 1.25):
    # Create a new empty molecule
    mol_new = Chem.RWMol()
    
    # Add atoms
    for atom in mol.GetAtoms():
        mol_new.AddAtom(atom)

    # Add conformer
    conf = mol.GetConformer()
    mol_new.AddConformer(conf)

    # Add bonds based on covalent radii
    for i in range(mol.GetNumAtoms()):
        for j in range(i + 1, mol.GetNumAtoms()):
            dist = np.linalg.norm(np.array(conf.GetAtomPosition(i)) - np.array(conf.GetAtomPosition(j)))
            pt = Chem.GetPeriodicTable()
            r_cov_i = pt.GetRcovalent(mol.GetAtomWithIdx(i).GetSymbol())
            r_cov_j = pt.GetRcovalent(mol.GetAtomWithIdx(j).GetSymbol())
            if dist < (r_cov_i + r_cov_j) * bond_factor:
                mol_new.AddBond(i, j, Chem.BondType.SINGLE)

    # Convert to a regular Mol object
    mol_new = mol_new.GetMol()
    
    return mol_new

def mol_from_xyz_file(file_path, return_charge_and_multiplicity=False):
    with open(file_path, 'r') as file:
        xyz_string = file.read()

    lines = xyz_string.split('\n')
    charge, multiplicity = map(int, lines[0].split())

    # Create a new empty molecule
    mol = Chem.RWMol()

    # Parse the atomic coordinates and add atoms
    elements = []
    for line in lines[1:]:
        if line.strip():
            parts = line.split()
            element = parts[0].capitalize()  # Correctly format the element symbol
            atom = Chem.Atom(element)
            mol.AddAtom(atom)
            elements.append(element)

    # Add 3D coordinates to the molecule
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, line in enumerate(lines[1:]):
        if line.strip():
            parts = line.split()
            x, y, z = map(float, parts[1:4])
            conf.SetAtomPosition(i, (x, y, z))
    mol.AddConformer(conf)
                
    # Convert to a regular Mol object
    mol = mol.GetMol()
    
    if return_charge_and_multiplicity:
        return mol, charge, multiplicity
    else:
        return mol

def mol_from_gaussian_file(filename):
    # Parse the Gaussian file using cclib
    parser = cclib.io.ccopen(filename)
    print(filename)
    data = parser.parse()

    # Extract atomic numbers and coordinates
    atomic_numbers = data.atomnos  # List of atomic numbers
    coordinates = data.atomcoords[-1]  # Final geometry (last set of coordinates)

    # Create an empty editable RDKit molecule
    mol = Chem.RWMol()

    # Add atoms to the molecule
    for atomic_num in atomic_numbers:
        atom = Chem.Atom(int(atomic_num))
        mol.AddAtom(atom)

    # Create a conformer to store atomic positions
    conf = Chem.Conformer(len(atomic_numbers))
    for i, (x, y, z) in enumerate(coordinates):
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf)

    # Convert RWMol to Mol before adding bonds
    mol = mol.GetMol()

    return mol

def write_sp_gaussian_input(mol, file_name, method_type, method_name, functional='B3LYP', basis='6-31G(d)', charge=0, multiplicity=1, solvation=False, solvation_model=None, solvent=None, n_proc=4, memory=2):
    # Open the file for writing
    with open(file_name + '.gjf', 'w') as f:
        # Link0 commands
        f.write(f'%NProcShared={n_proc}\n')
        f.write(f'%Mem={memory}GB\n')
        f.write(f'%Chk={file_name}.chk\n')
        
        # Route section
        if method_type == "HF":
            method = f'hf/{basis.lower()}'
        elif method_type == "DFT":
            method = f'{functional.lower()}/{basis.lower()}'
        elif method_type in ["MP2", "CCSD", "BD"]:
            method = f'{method_type.lower()}/{basis.lower()}'
        elif method_type == "MP4":
            method = f'mp4(sdtq)/{basis.lower()}'
        else: # method_type in ["Semi-empirical", "Compound"]:
            method = f'{method_name.lower()}'
       
        route_section = f'#P {method} sp'

        if solvation:
            route_section += f' scrf=({solvation_model},solvent={solvent})'
        route_section += '\n\n'
        f.write(route_section)

        # Title section
        f.write('Single Point Energy Calculation\n\n')
        
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

def write_opt_gaussian_input(mol, file_name, method_type, method_name, functional='B3LYP', basis='6-31G(d,p)', charge=0, multiplicity=1, solvation=False, solvation_model=None, solvent=None, n_proc=4, memory=2):
    # Open the file for writing
    with open(file_name + '.gjf', 'w') as f:
        # Link0 commands
        f.write(f'%NProcShared={n_proc}\n')
        f.write(f'%Mem={memory}GB\n')
        f.write(f'%Chk={file_name}.chk\n')
        
        # Route section
        if method_type == "HF":
            method = f'hf/{basis.lower()}'
        elif method_type == "DFT":
            method = f'{functional.lower()}/{basis.lower()}'
        elif method_type in ["MP2", "CCSD", "BD"]:
            method = f'{method_type.lower()}/{basis.lower()}'
        elif method_type == "MP4":
            method = f'mp4(sdtq)/{basis.lower()}'
        else: # method_type in ["Semi-empirical", "Compound"]:
            method = f'{method_name.lower()}'
       
        route_section = f'#P {method} opt'

        if solvation:
            route_section += f' scrf=({solvation_model},solvent={solvent})'
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

def write_opt_freq_gaussian_input(mol, file_name, method_type, method_name, functional='B3LYP', basis='6-31G(d,p)', charge=0, multiplicity=1, solvation=False, solvation_model=None, solvent=None, n_proc=4, memory=2):
    # Open the file for writing
    with open(file_name + '.gjf', 'w') as f:
        # Link0 commands
        f.write(f'%NProcShared={n_proc}\n')
        f.write(f'%Mem={memory}GB\n')
        f.write(f'%Chk={file_name}.chk\n')
        
        # Route section
        if method_type == "HF":
            method = f'hf/{basis.lower()}'
        elif method_type == "DFT":
            method = f'{functional.lower()}/{basis.lower()}'
        elif method_type in ["MP2", "CCSD", "BD"]:
            method = f'{method_type.lower()}/{basis.lower()}'
        elif method_type == "MP4":
            method = f'mp4(sdtq)/{basis.lower()}'
        else: # method_type in ["Semi-empirical", "Compound"]:
            method = f'{method_name.lower()}'
       
        route_section = f'#P {method} opt freq'
        
        if solvation:
            route_section += f' scrf=({solvation_model},solvent={solvent})'
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

def write_uv_vis_gaussian_input(mol, file_name, method_type, method_name, functional='B3LYP', basis='6-31G(d,p)', n_states=10, charge=0, multiplicity=1, solvation=False, solvation_model=None, solvent=None, n_proc=4, memory=2):
    # Open the file for writing
    with open(file_name + '.gjf', 'w') as f:
        # Get method for both jobs
        if method_type == "HF":
            method = f'hf/{basis.lower()}'
        elif method_type == "DFT":
            method = f'{functional.lower()}/{basis.lower()}'
        elif method_type in ["MP2", "CCSD", "BD"]:
            method = f'{method_type.lower()}/{basis.lower()}'
        elif method_type == "MP4":
            method = f'mp4(sdtq)/{basis.lower()}'
        else: # method_type in ["Semi-empirical", "Compound"]:
            method = f'{method_name.lower()}'
        
        # First job: ground-state optimization
        f.write(f'%NProcShared={n_proc}\n')
        f.write(f'%Mem={memory}GB\n')
        f.write(f'%Chk={file_name}.chk\n')
        route1 = f'#P {method} opt'
        if solvation:
            route1 += f' scrf=({solvation_model},solvent={solvent})'
        route1 += '\n\n'
        f.write(route1)
        f.write('S0 geometry optimization\n\n')
        f.write(f'{charge} {multiplicity}\n')
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            sym = atom.GetSymbol()
            pos = conf.GetAtomPosition(idx)
            f.write(f'{sym:<2} {pos.x:>12.6f} {pos.y:>12.6f} {pos.z:>12.6f}\n')
        f.write('\n')
        
        # Second job: TD single-point using geometry from checkpoint
        f.write('--Link1--\n')
        f.write(f'%NProcShared={n_proc}\n')
        f.write(f'%Mem={memory}GB\n')
        f.write(f'%Chk={file_name}.chk\n')
        route2 = f'#P {method} TD(NStates={n_states}) Geom=AllCheck Guess=Read'
        if solvation:
            route2 += f' scrf=({solvation_model},solvent={solvent})'
        route2 += '\n\n'
        f.write(route2)
        f.write('TD-DFT vertical excitations at optimized S0 geometry\n\n')
        # When using Geom=AllCheck, coordinates block can be a placeholder
        f.write(f'{charge} {multiplicity}\n\n')

def write_fluorescence_gaussian_input(mol, file_name, method_type, method_name, functional='B3LYP', basis='6-31G(d,p)', n_states=10, charge=0, multiplicity=1, solvation=False, solvation_model=None, solvent=None, n_proc=4, memory=2):
    # Get method for both jobs
    if method_type == "HF":
        method = f'hf/{basis.lower()}'
    elif method_type == "DFT":
        method = f'{functional.lower()}/{basis.lower()}'
    elif method_type in ["MP2", "CCSD", "BD"]:
        method = f'{method_type.lower()}/{basis.lower()}'
    elif method_type == "MP4":
        method = f'mp4(sdtq)/{basis.lower()}'
    else: # method_type in ["Semi-empirical", "Compound"]:
        method = f'{method_name.lower()}'
    
    # First job: excited-state optimization (S1)
    with open(file_name + '_S1_Opt.gjf', 'w') as f:
        f.write(f'%NProcShared={n_proc}\n')
        f.write(f'%Mem={memory}GB\n')
        f.write(f'%Chk={file_name}.chk\n')
        route1 = f'#P {method} TD(Singlets,NStates={n_states},Root=1) opt'
        if solvation:
            route1 += f' scrf=({solvation_model},solvent={solvent})'
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
    with open(file_name + '_S1_SP.gjf', 'w') as f:
        f.write(f'%NProcShared={n_proc}\n')
        f.write(f'%Mem={memory}GB\n')
        f.write(f'%Chk={file_name}.chk\n')
        route2 = f'#P {method} TD(Singlets,NStates={n_states},Root=1) Geom=AllCheck Guess=Read'
        if solvation:
            route2 += f' scrf=({solvation_model},solvent={solvent})'
        route2 += '\n\n'
        f.write(route2)
        f.write('TD single-point at optimized S1 geometry (emission energies)\n\n')

def write_nmr_gaussian_input(mol, file_name, method_type, functional='B3LYP', basis='6-31G(d,p)', spin_spin_coupling=False, charge=0, multiplicity=1, solvation=False, solvation_model=None, solvent=None, n_proc=4, memory=2):
    # Open the file for writing
    with open(file_name + '.gjf', 'w') as f:
        # Link0 commands
        f.write(f'%NProcShared={n_proc}\n')
        f.write(f'%Mem={memory}GB\n')
        f.write(f'%Chk={file_name}.chk\n')
        
        # Route section
        if method_type == "HF":
            method = f'hf/{basis.lower()}'
        elif method_type == "DFT":
            method = f'{functional.lower()}/{basis.lower()}'
        elif method_type == "MP2":
            method = f'mp2/{basis.lower()}'
        else: # other methods
            raise Exception(f"Cannot use {method_type} method to predict NMR spectrum.")

        if spin_spin_coupling:
            route_section = f'#P {method} NMR=(GIAO,spinspin)'
        else:
            route_section = f'#P {method} NMR=GIAO'
        if solvation:
            route_section += f' scrf=({solvation_model},solvent={solvent})'
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

def lorentzian_ir(wavenumber, position, intensity, width=10):
    return intensity * (width ** 2) / ((wavenumber - position) ** 2 + width ** 2)

def generate_ir_spectrum_interactive(
    frequencies,
    intensities,
    width=10.0,
    points=4000,
    plot_range=None,
    normalize=True,
    transmittance=True
):
    # Safety check
    if frequencies is None or len(frequencies) == 0:
        print("No IR frequencies detected.")
        return None

    # Define plotting range
    if plot_range:
        min_wn, max_wn = plot_range
    else:
        min_wn = np.min(frequencies) - 100
        max_wn = np.max(frequencies) + 100

    x = np.linspace(min_wn, max_wn, points)
    spectrum = np.zeros_like(x)

    # Generate Lorentzian broadened spectrum
    annotations = []
    for freq, inten in zip(frequencies, intensities):
        spectrum += lorentzian_ir(x, freq, inten, width)

        annotations.append({
            "freq": freq,
            "intensity": inten,
            "text": f"{freq:.1f}"
        })

    # Normalize if requested
    if normalize and np.max(spectrum) != 0:
        spectrum /= np.max(spectrum)

    # Convert to transmittance if requested
    if transmittance:
        spectrum = 1 - spectrum  # simple inversion

    # Create figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=spectrum,
        mode='lines',
        line=dict(color='black'),
        name="IR Spectrum"
    ))

    # Add peak annotations
    for ann in annotations:
        fig.add_annotation(
            x=ann["freq"],
            y=max(spectrum),
            text=ann["text"],
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-30,
            textangle=-90,
            font=dict(color='red', size=10)
        )

    # Layout
    fig.update_layout(
        title="IR Spectrum",
        xaxis=dict(
            title="Wavenumber (cm⁻¹)",
            autorange="reversed"  # Important for IR
        ),
        yaxis=dict(
            title="Transmittance" if transmittance else "Intensity (a.u.)"
        ),
        showlegend=False
    )

    # Apply manual range if provided
    if plot_range:
        fig.update_xaxes(range=plot_range[::-1])  # reverse manually

    return fig

def gaussian(x, x0, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-(x - x0)**2 / (2 * sigma**2))

def generate_absorption_emission_spectrum_interactive(wavelengths, oscs, points=10000, plot_range=None):
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
        name=f'Spectrum'
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
        title=f'Spectrum',
        showlegend=False,
    )

    # Adjust plot range if specified
    if plot_range:
        fig.update_xaxes(range=plot_range)

    return fig

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

def lorentzian_nmr(x, x0, gamma):
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
        spectrum += intensity * lorentzian_nmr(x, shift, lw_ppm)
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
            y=lorentzian_nmr(ann['shift'], ann['shift'], lw_ppm) / np.max(spectrum),
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