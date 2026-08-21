"""Structure file reading and writing, and the distance based bond perception."""
import math
import unittest

import numpy as np
from rdkit import Chem

from helpers import TempWorkingDirTestCase, make_molecule
from utils import (add_bonds, conformer_to_xyz_file, get_files_in_working_directory,
                   mol_from_xyz_file, natural_sort_key)


class XyzRoundTripTests(TempWorkingDirTestCase):
    """These .xyz files are NOT standard: line 1 is "charge multiplicity", not an atom count."""

    def test_first_line_is_charge_and_multiplicity(self):
        mol = make_molecule('CCO')
        conformer_to_xyz_file(mol, 0, self.path('m.xyz'), charge=-1, multiplicity=3)
        lines = self.read_file('m.xyz').splitlines()
        self.assertEqual(lines[0], '-1 3')
        self.assertEqual(len(lines), mol.GetNumAtoms() + 1, 'no atom count or comment line')

    def test_atoms_and_coordinates_survive_a_round_trip(self):
        mol = make_molecule('CCO')
        conformer_to_xyz_file(mol, 0, self.path('m.xyz'))
        reloaded = mol_from_xyz_file(self.path('m.xyz'))

        self.assertEqual(reloaded.GetNumAtoms(), mol.GetNumAtoms())
        self.assertEqual([a.GetSymbol() for a in reloaded.GetAtoms()],
                         [a.GetSymbol() for a in mol.GetAtoms()])
        original = mol.GetConformer()
        restored = reloaded.GetConformer()
        for i in range(mol.GetNumAtoms()):
            a, b = original.GetAtomPosition(i), restored.GetAtomPosition(i)
            self.assertAlmostEqual(a.x, b.x, places=6)
            self.assertAlmostEqual(a.y, b.y, places=6)
            self.assertAlmostEqual(a.z, b.z, places=6)

    def test_charge_and_multiplicity_can_be_read_back(self):
        conformer_to_xyz_file(make_molecule('CCO'), 0, self.path('m.xyz'), 2, 5)
        _, charge, multiplicity = mol_from_xyz_file(self.path('m.xyz'),
                                                    return_charge_and_multiplicity=True)
        self.assertEqual((charge, multiplicity), (2, 5))

    def test_writing_a_chosen_conformer(self):
        from rdkit.Chem import AllChem
        mol = Chem.AddHs(Chem.MolFromSmiles('CCCCO'))
        AllChem.EmbedMultipleConfs(mol, numConfs=3, randomSeed=1)
        conformer_to_xyz_file(mol, 0, self.path('a.xyz'))
        conformer_to_xyz_file(mol, 2, self.path('b.xyz'))
        self.assertNotEqual(self.read_file('a.xyz'), self.read_file('b.xyz'))

    def test_a_standard_xyz_file_does_not_parse_as_one_of_ours(self):
        # First line would be an atom count, which is not "<charge> <multiplicity>"
        self.write_file('standard.xyz', '3\nwater\nO 0.0 0.0 0.0\nH 0.0 0.8 0.4\nH 0.0 -0.8 0.4\n')
        with self.assertRaises(ValueError):
            mol_from_xyz_file(self.path('standard.xyz'))


class AddBondsTests(unittest.TestCase):
    def test_bonds_are_inferred_for_a_bondless_molecule(self):
        mol = make_molecule('CCO')
        stripped = Chem.RWMol()
        for atom in mol.GetAtoms():
            stripped.AddAtom(Chem.Atom(atom.GetSymbol()))
        stripped.AddConformer(mol.GetConformer())
        rebuilt = add_bonds(stripped.GetMol())
        self.assertEqual(rebuilt.GetNumBonds(), mol.GetNumBonds())

    def test_perception_is_sane_for_water(self):
        water = Chem.RWMol()
        for symbol in ('O', 'H', 'H'):
            water.AddAtom(Chem.Atom(symbol))
        conf = Chem.Conformer(3)
        conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
        conf.SetAtomPosition(1, (0.0, 0.76, 0.59))
        conf.SetAtomPosition(2, (0.0, -0.76, 0.59))
        water.AddConformer(conf)
        rebuilt = add_bonds(water.GetMol())
        self.assertEqual(rebuilt.GetNumBonds(), 2)
        self.assertEqual({b.GetBeginAtomIdx() for b in rebuilt.GetBonds()}, {0})

    def test_a_tighter_factor_finds_fewer_bonds(self):
        mol = make_molecule('CCO')
        stripped = Chem.RWMol()
        for atom in mol.GetAtoms():
            stripped.AddAtom(Chem.Atom(atom.GetSymbol()))
        stripped.AddConformer(mol.GetConformer())
        loose = add_bonds(stripped.GetMol(), bond_factor=1.25).GetNumBonds()
        tight = add_bonds(stripped.GetMol(), bond_factor=0.5).GetNumBonds()
        self.assertLess(tight, loose)


class WorkingDirectoryListingTests(TempWorkingDirTestCase):
    def test_zone_identifier_files_are_hidden(self):
        self.write_file('m.xyz')
        self.write_file('m.xyz:Zone.Identifier')
        self.assertEqual(get_files_in_working_directory(self.working_directory), ['m.xyz'])


class NaturalSortTests(unittest.TestCase):
    def test_numbers_sort_numerically_not_lexicographically(self):
        names = ['conformer_10.xyz', 'conformer_2.xyz', 'conformer_1.xyz']
        self.assertEqual(sorted(names, key=natural_sort_key),
                         ['conformer_1.xyz', 'conformer_2.xyz', 'conformer_10.xyz'])

    def test_sorting_is_case_insensitive(self):
        names = ['Zebra.log', 'apple.log', 'Banana.log']
        self.assertEqual(sorted(names, key=natural_sort_key),
                         ['apple.log', 'Banana.log', 'Zebra.log'])

    def test_leading_numbers_are_handled(self):
        self.assertEqual(sorted(['10x', '2x', 'wd1'], key=natural_sort_key), ['2x', '10x', 'wd1'])

    def test_names_without_digits_still_sort(self):
        self.assertEqual(sorted(['b', 'a'], key=natural_sort_key), ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
