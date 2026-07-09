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

def _fortran_float(token):
    """Convert a Gaussian Fortran float token (e.g. '0.311173D+02') to float."""
    return float(token.replace('D', 'E').replace('d', 'e'))

def parse_nmr_jcouplings(logfile, atom_symbols=None):
    """Parse NMR spin-spin (J) coupling constants from a Gaussian output file.

    Gaussian prints the isotropic couplings as the symmetric lower-triangular
    matrix under 'Total nuclear spin-spin coupling J (Hz):', in column blocks of
    up to five nuclei, using Fortran D-notation and 1-based atom indices::

        Total nuclear spin-spin coupling J (Hz):
                        1             2             3   ...
              1  0.000000D+00
              2  0.311173D+02  0.000000D+00
              ...

    ``atom_symbols`` is an optional {atom_idx: element} map (1-based, e.g. built
    from parse_nmr_shielding_constants) used to tag each nucleus; when omitted the
    element fields are left blank. Returns a list of tuples
    ``(atom_a_idx, element_a, atom_b_idx, element_b, J_iso_Hz)`` for a != b, each
    pair reported once (a > b), matching the shape parse_nmr_jcouplings uses in the
    ORCA reader.
    """
    atom_symbols = atom_symbols or {}
    jcouplings = []

    with open(logfile, 'r') as f:
        lines = f.readlines()

    # Anchor on the final "Total ... J (Hz)" block (skip the K matrix and the
    # per-mechanism FC/SD/PSO/DSO contributions printed earlier).
    start = None
    for i, line in enumerate(lines):
        if 'Total nuclear spin-spin coupling J (Hz)' in line:
            start = i + 1
            break
    if start is None:
        print("No NMR spin-spin coupling data found in the log file.")
        return jcouplings

    col_indices = []
    for line in lines[start:]:
        tokens = line.split()
        if not tokens:
            break
        # A header line lists only column (nucleus) indices - all plain integers.
        if all(re.fullmatch(r'\d+', t) for t in tokens):
            col_indices = [int(t) for t in tokens]
            continue
        # A data row: leading integer row index followed by Fortran floats.
        if re.fullmatch(r'\d+', tokens[0]) and 'D' in line:
            row_idx = int(tokens[0])
            values = tokens[1:]
            for k, value in enumerate(values):
                if k >= len(col_indices):
                    break
                col_idx = col_indices[k]
                if col_idx == row_idx:
                    continue  # skip the zero diagonal
                j_hz = _fortran_float(value)
                jcouplings.append((
                    row_idx, atom_symbols.get(row_idx, ''),
                    col_idx, atom_symbols.get(col_idx, ''),
                    j_hz,
                ))
            continue
        # Anything else (e.g. "End of Minotr ...") marks the end of the matrix.
        break

    if not jcouplings:
        print("No NMR spin-spin coupling data found in the log file.")
    return jcouplings

# Symbols indexed by atomic number (Z=1..54), enough for typical organic NMR.
_ATOMIC_SYMBOLS = [
    '', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe',
]

def parse_gaussian_geometry(logfile):
    """Parse the last 'Input/Standard orientation' block from a Gaussian log.

    Returns (symbols, coords) in the SAME atom order as Gaussian's NMR nucleus
    indices (center numbering), which is what the equivalence helpers below expect.
    ``symbols`` is a 0-indexed list of element symbols (center i maps to index i-1);
    ``coords`` is an (N, 3) numpy array in Angstrom.
    """
    # atom line: center#  atomic#  atomic-type  X  Y  Z
    line_pattern = re.compile(
        r'^\s*\d+\s+(\d+)\s+\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$'
    )
    with open(logfile, 'r') as f:
        lines = f.readlines()

    symbols, coords = [], []
    for i, line in enumerate(lines):
        if 'orientation:' in line:
            block_symbols, block_coords = [], []
            # Skip the two column-header rows and the dashed separators.
            j = i + 1
            while j < len(lines) and not line_pattern.match(lines[j]):
                if 'Coordinates' not in lines[j] and 'Number' not in lines[j] \
                        and set(lines[j].strip()) not in (set('-'), set()):
                    break
                j += 1
            while j < len(lines):
                m = line_pattern.match(lines[j])
                if not m:
                    break
                atomic_number = int(m.group(1))
                symbol = _ATOMIC_SYMBOLS[atomic_number] if atomic_number < len(_ATOMIC_SYMBOLS) else 'X'
                block_symbols.append(symbol)
                block_coords.append([float(m.group(2)), float(m.group(3)), float(m.group(4))])
                j += 1
            if block_symbols:
                symbols, coords = block_symbols, block_coords  # keep the last block
    return symbols, np.asarray(coords, dtype=float)

# Cordero covalent radii (Angstrom) for the elements common in organic NMR.
# Used only for simple distance-based bond perception.
_COVALENT_RADII = {
    'H': 0.31, 'B': 0.84, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39,
}

def perceive_bonds(coords, symbols, tolerance=1.3):
    """Infer bonds from interatomic distances and covalent radii.

    Returns a list of sets, where adjacency[i] holds the indices bonded to atom i.
    Two atoms are bonded when their distance is below (r_i + r_j) * tolerance.
    """
    coords = np.asarray(coords, dtype=float)
    n = len(symbols)
    adjacency = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            r_i = _COVALENT_RADII.get(symbols[i], 0.77)
            r_j = _COVALENT_RADII.get(symbols[j], 0.77)
            dist = float(np.linalg.norm(coords[i] - coords[j]))
            if 0.0 < dist < (r_i + r_j) * tolerance:
                adjacency[i].add(j)
                adjacency[j].add(i)
    return adjacency

def morgan_ranks(coords, symbols):
    """Assign each atom a topological-symmetry rank (Morgan-style refinement).

    Atoms start with an invariant of (element, degree) on the perceived bond graph
    and are iteratively re-ranked by (own rank, sorted neighbour ranks) until the set
    of distinct ranks stops growing. Two atoms sharing a final rank are equivalent by
    the symmetry of the whole molecular graph. Ranks are opaque grouping keys only.
    """
    adjacency = perceive_bonds(coords, symbols)
    n = len(symbols)
    ranks = [hash((symbols[i], len(adjacency[i]))) for i in range(n)]
    for _ in range(n):  # converges in at most N refinement rounds
        new_ranks = [
            hash((ranks[i], tuple(sorted(ranks[j] for j in adjacency[i]))))
            for i in range(n)
        ]
        if len(set(new_ranks)) == len(set(ranks)):
            break  # partition stable: no new classes distinguished
        ranks = new_ranks
    return adjacency, ranks

def compute_hydrogen_equivalence(coords, symbols):
    """Group hydrogens that are bonded to symmetry-equivalent heavy atoms.

    Freely-rotating groups (CH3, CH2, NH2) average their protons to a single signal,
    and so do protons on heavy atoms that are equivalent by molecular symmetry (e.g.
    the two methyls of isopropanol or tert-butyl). Hydrogens are therefore keyed on the
    Morgan rank of their bonded heavy atom, which reduces to the bonded-atom identity
    when the molecule has no symmetry. ``coords``/``symbols`` must be in Gaussian NMR
    nucleus order (use parse_gaussian_geometry, NOT cclib, which reorders atoms).
    Returns a dict {hydrogen_atom_idx: group_key} keyed by Gaussian's 1-based index.

    NOTE: equivalence is topological only. Diastereotopic protons on the same carbon are
    treated as equivalent (a deliberate simplification, since a static geometry carries
    no stereochemistry in the graph).
    """
    adjacency, ranks = morgan_ranks(coords, symbols)
    equivalence = {}
    for idx, sym in enumerate(symbols):
        if sym != 'H':
            continue
        heavy_neighbors = [j for j in adjacency[idx] if symbols[j] != 'H']
        # Key on the bonded heavy atom's symmetry rank so protons on equivalent heavy
        # atoms merge. With no heavy neighbor (e.g. H2), the hydrogen forms its own group.
        # +1 converts the 0-based list index to Gaussian's 1-based nucleus index.
        equivalence[idx + 1] = ('heavy', ranks[heavy_neighbors[0]]) if heavy_neighbors else f"H{idx + 1}"
    return equivalence

def compute_carbon_equivalence(coords, symbols):
    """Group carbons that are topologically equivalent by molecular symmetry.

    Unlike hydrogens (which also average within a freely-rotating group), equivalent
    carbons are related by the symmetry of the whole molecule (e.g. the two methyls of
    isopropanol, the three methyls of tert-butyl, or the para carbons of a symmetric
    ring). Carbons sharing a Morgan rank (see morgan_ranks) merge into a single averaged
    signal, independent of small DFT shift differences.

    ``coords``/``symbols`` must be in Gaussian NMR nucleus order (use
    parse_gaussian_geometry, NOT cclib, which reorders atoms). Returns a dict
    {carbon_atom_idx: rank} keyed by Gaussian's 1-based index; the rank is only used as
    an opaque grouping key.
    """
    _, ranks = morgan_ranks(coords, symbols)
    return {i + 1: ranks[i] for i, sym in enumerate(symbols) if sym == 'C'}

# Multiplicity letters for first-order multiplets (number of equal coupling
# partners -> name). More than one coupling group concatenates letters (e.g. dd).
_MULTIPLICITY_LETTER = {1: 'd', 2: 't', 3: 'q', 4: 'p', 5: 'h', 6: 'hept'}

def build_nmr_peak_table(shifts_data, element_symbol, jcouplings=None, equivalence=None, j_threshold=0.5):
    """Group nuclei of ``element_symbol`` into averaged multiplet peaks.

    Equivalent nuclei merge into one peak whose chemical shift is the average of
    the members and whose intensity is the member count. ``equivalence`` is an
    optional {atom_idx: group_key} map (e.g. from compute_hydrogen_equivalence);
    when omitted, nuclei are grouped by rounded chemical shift (~0.01 ppm).

    Coupling between two equivalent groups A and B is treated to first order:
    every nucleus of A is split by B into (size(B) + 1) lines with a single
    coupling constant equal to the AVERAGE of all pairwise A-B couplings. The
    multiplicity therefore comes from the partner group SIZE (not from how many
    couplings the calculation happened to print for one atom), which is what
    makes a CH3-CH2 pair read as the expected triplet/quartet even though the
    individual per-atom couplings differ in a single static geometry. Couplings
    whose averaged magnitude is below ``j_threshold`` (Hz) are dropped as
    unresolvable.

    Returns a list of peak dicts sorted by ascending shift::

        {'atom_idxs': [...], 'element': str, 'shift': float, 'count': int,
         'couplings': [{'partner_idxs': (...), 'n': int, 'J': float}, ...]}
    """
    shifts = [(i, e, s) for i, e, s in shifts_data if e == element_symbol]
    if not shifts:
        return []

    def key_of(atom_idx, shift):
        if equivalence is not None and atom_idx in equivalence:
            return ('grp', equivalence[atom_idx])
        return ('shift', round(shift, 2))

    groups = {}
    for atom_idx, element, shift in shifts:
        k = key_of(atom_idx, shift)
        grp = groups.setdefault(k, {'atom_idxs': [], 'shifts': []})
        grp['atom_idxs'].append(atom_idx)
        grp['shifts'].append(shift)

    # Pairwise homonuclear coupling lookup: (atom_a, atom_b) -> J_Hz (both ways).
    pair_j = {}
    if jcouplings:
        for idx_a, elem_a, idx_b, elem_b, j_hz in jcouplings:
            if elem_a == element_symbol and elem_b == element_symbol:
                pair_j[(idx_a, idx_b)] = j_hz
                pair_j[(idx_b, idx_a)] = j_hz

    group_items = list(groups.items())
    peaks = []
    for k, grp in group_items:
        atom_idxs = sorted(grp['atom_idxs'])
        avg_shift = sum(grp['shifts']) / len(grp['shifts'])
        couplings = []
        for k2, grp2 in group_items:
            if k2 == k:
                continue  # mutually equivalent nuclei do not split each other
            partner_idxs = sorted(grp2['atom_idxs'])
            js = [pair_j[(a, b)] for a in atom_idxs for b in partner_idxs if (a, b) in pair_j]
            if not js:
                continue
            j_avg = sum(js) / len(js)
            if abs(j_avg) < j_threshold:
                continue  # negligible / unresolvable coupling
            couplings.append({
                'partner_idxs': tuple(partner_idxs),
                'n': len(partner_idxs),  # multiplicity from partner GROUP size
                'J': j_avg,
            })
        couplings.sort(key=lambda c: -abs(c['J']))
        peaks.append({
            'atom_idxs': atom_idxs,
            'element': element_symbol,
            'shift': avg_shift,
            'count': len(atom_idxs),
            'couplings': couplings,
        })

    peaks.sort(key=lambda p: p['shift'])
    return peaks

def multiplicity_label(couplings):
    """Human-readable multiplicity (s/d/t/q/...) for a peak's coupling list."""
    if not couplings:
        return 's'
    return ''.join(_MULTIPLICITY_LETTER.get(c['n'], 'm') for c in couplings)

def lorentzian_nmr(x, x0, gamma):
    return (gamma**2) / ((x - x0)**2 + gamma**2)

def generate_nmr_spectrum_interactive(shifts_data, element_symbol, linewidth=0.5, points=10000, frequency=400, plot_range=None, jcouplings=None, equivalence=None):
    # Filter shifts for the desired element
    shifts = [(atom_idx, element, shift) for atom_idx, element, shift in shifts_data if element == element_symbol]

    if not shifts:
        print(f"No chemical shifts found for element {element_symbol}.")
        return

    # Group nuclei into averaged multiplet peaks. Equivalent nuclei (per the
    # `equivalence` map, or by rounded shift when absent) collapse to one peak
    # whose shift is averaged and whose height scales with the member count.
    peaks = build_nmr_peak_table(shifts_data, element_symbol,
                                 jcouplings=jcouplings, equivalence=equivalence)

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

    # Generate Lorentzian multiplets per group and collect positions for annotations
    annotations = []
    for peak in peaks:
        shift = peak['shift']
        intensity = peak['count']
        # First-order multiplet: each coupled partner halves the weight and
        # splits the peak by J/freq (Hz -> ppm). Coupling to n equivalent
        # partners splits n times, giving an (n+1)-line Pascal multiplet.
        sub_peaks = [(shift, 1.0)]
        for c in peak['couplings']:
            half_j_ppm = (c['J'] / 2.0) / frequency
            for _ in range(c['n']):
                sub_peaks = [(p - half_j_ppm, w / 2.0) for p, w in sub_peaks] + \
                            [(p + half_j_ppm, w / 2.0) for p, w in sub_peaks]
        for pos, weight in sub_peaks:
            spectrum += intensity * weight * lorentzian_nmr(x, pos, lw_ppm)
        label = f"{element_symbol}-" + ",".join(str(i) for i in peak['atom_idxs'])
        annotations.append({
            'shift': shift,
            'atom_label': label,
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

    # Annotate at the centroid (chemical shift) using the actual spectrum value
    # so the arrow lands on the multiplet envelope rather than above it.
    for ann in annotations:
        idx = int(np.argmin(np.abs(x - ann['shift'])))
        fig.add_annotation(
            x=ann['shift'],
            y=spectrum[idx],
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