import cclib
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def AddBonds(mol, bond_factor = 1.25):
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

def mol_from_gaussian_file(filename, bond_factor=1.25):
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

    # Use the provided AddBonds function to add bonds
    mol_with_bonds = AddBonds(mol, bond_factor=bond_factor)

    # Sanitize the molecule (optional, but recommended)
    Chem.SanitizeMol(mol_with_bonds)

    return mol_with_bonds

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

def mol_from_xyz_file(file_path, bond_factor=1.25, return_charge_and_multiplicity=False):
    with open(file_path, 'r') as file:
        xyz_string = file.read()

    lines = xyz_string.split('\n')
    charge, multiplicity = map(int, lines[0].split())

    # Create a new empty molecule
    mol = Chem.RWMol()

    # Parse the atomic coordinates and add atoms
    coords = []
    elements = []
    for line in lines[1:]:
        if line.strip():
            parts = line.split()
            element = parts[0].capitalize()  # Correctly format the element symbol
            atom = Chem.Atom(element)
            mol.AddAtom(atom)
            elements.append(element)
            x, y, z = map(float, parts[1:4])
            coords.append([x, y, z])

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

    # Use the provided AddBonds function to add bonds
    mol_with_bonds = AddBonds(mol, bond_factor=bond_factor)

    # Sanitize the molecule (optional, but recommended)
    Chem.SanitizeMol(mol_with_bonds)
    
    if return_charge_and_multiplicity:
        return mol_with_bonds, charge, multiplicity
    else:
        return mol_with_bonds