"""Conformer generation: force field minimisation, duplicate removal and file output."""
import os
import unittest

from rdkit import Chem
from rdkit.Chem import AllChem

from helpers import FakeProgress, TempWorkingDirTestCase, make_molecule, status_colour, status_text
from conformer_generation import (generate_unique_conformers, is_duplicate_conformer,
                                  on_draw_molecule, on_generate_conformers, optimize_conformers)


class DrawMoleculeTests(unittest.TestCase):
    def test_smiles_is_canonicalised(self):
        self.assertEqual(on_draw_molecule('OCC'), 'CCO')

    def test_invalid_smiles_gives_an_empty_string(self):
        self.assertEqual(on_draw_molecule('not a molecule'), '')


class OptimizeConformersTests(unittest.TestCase):
    def test_one_energy_per_conformer(self):
        mol = Chem.AddHs(Chem.MolFromSmiles('CCCCO'))
        AllChem.EmbedMultipleConfs(mol, numConfs=5, randomSeed=1)
        energies = optimize_conformers(mol)
        self.assertEqual(len(energies), 5)
        self.assertTrue(all(isinstance(e, float) for e in energies))

    def test_minimisation_lowers_the_energy(self):
        mol = Chem.AddHs(Chem.MolFromSmiles('CCCCO'))
        AllChem.EmbedMultipleConfs(mol, numConfs=3, randomSeed=1)
        properties = AllChem.MMFFGetMoleculeProperties(mol)
        before = [AllChem.MMFFGetMoleculeForceField(mol, properties, confId=c.GetId()).CalcEnergy()
                  for c in mol.GetConformers()]
        after = optimize_conformers(mol)
        self.assertLessEqual(min(after), min(before) + 1e-6)

    def test_uff_is_used_when_mmff_cannot_type_the_molecule(self):
        # a molecule MMFF has no parameters for still gets energies through UFF
        mol = Chem.AddHs(Chem.MolFromSmiles('[Cu]'))
        AllChem.EmbedMultipleConfs(mol, numConfs=1, randomSeed=1)
        if mol.GetNumConformers():
            self.assertEqual(len(optimize_conformers(mol)), mol.GetNumConformers())


class DuplicateDetectionTests(unittest.TestCase):
    def setUp(self):
        self.mol = Chem.AddHs(Chem.MolFromSmiles('CCCCO'))
        AllChem.EmbedMultipleConfs(self.mol, numConfs=6, randomSeed=1)
        optimize_conformers(self.mol)
        self.heavy = Chem.RemoveHs(Chem.Mol(self.mol))

    def test_a_conformer_is_a_duplicate_of_itself(self):
        kept = [(0.0, 0)]
        self.assertTrue(is_duplicate_conformer(
            0.0, 0, kept, self.heavy, self.heavy, energy_threshold=0.1, rms_threshold=0.5))

    def test_same_geometry_but_far_in_energy_is_not_a_duplicate(self):
        kept = [(100.0, 0)]
        self.assertFalse(is_duplicate_conformer(
            0.0, 0, kept, self.heavy, self.heavy, energy_threshold=0.1, rms_threshold=0.5))

    def test_same_energy_but_different_geometry_is_not_a_duplicate(self):
        # energy alone would discard this, the RMSD test is what keeps it
        kept = [(0.0, 1)]
        self.assertFalse(is_duplicate_conformer(
            0.0, 0, kept, self.heavy, self.heavy, energy_threshold=0.1, rms_threshold=0.01))

    def test_zero_thresholds_disable_duplicate_detection(self):
        kept = [(0.0, 0)]
        self.assertFalse(is_duplicate_conformer(
            0.0, 0, kept, self.heavy, self.heavy, energy_threshold=0.0, rms_threshold=0.0))


class GenerateUniqueConformersTests(unittest.TestCase):
    def test_a_flexible_molecule_reaches_the_requested_count(self):
        mol = make_molecule('CCCCCCO')
        _, conformers, _ = generate_unique_conformers(mol, 15, 0.1, 0.5)
        self.assertEqual(len(conformers), 15)

    def test_conformers_come_back_lowest_energy_first(self):
        mol = make_molecule('CCCCCCO')
        _, conformers, _ = generate_unique_conformers(mol, 10, 0.1, 0.5)
        energies = [energy for energy, _ in conformers]
        self.assertEqual(energies, sorted(energies))

    def test_kept_conformers_are_pairwise_distinct(self):
        mol = make_molecule('CCCCCCO')
        unique_mol, conformers, _ = generate_unique_conformers(mol, 8, 0.5, 0.5)
        heavy = Chem.RemoveHs(Chem.Mol(unique_mol))
        from rdkit.Chem import rdMolAlign
        for i, (energy_a, id_a) in enumerate(conformers):
            for energy_b, id_b in conformers[i + 1:]:
                if abs(energy_a - energy_b) < 0.5:
                    rms = rdMolAlign.GetBestRMS(heavy, heavy, prbId=id_a, refId=id_b)
                    self.assertGreaterEqual(rms, 0.5)

    def test_a_rigid_molecule_returns_what_exists_without_hanging(self):
        # benzene has a single conformer, asking for 20 must terminate quickly
        _, conformers, _ = generate_unique_conformers(make_molecule('c1ccccc1'), 20, 0.1, 0.5)
        self.assertEqual(len(conformers), 1)

    def test_duplicates_are_counted(self):
        _, conformers, discarded = generate_unique_conformers(make_molecule('c1ccccc1'), 5, 0.1, 0.5)
        self.assertGreater(discarded, 0)

    def test_zero_thresholds_keep_everything(self):
        _, conformers, discarded = generate_unique_conformers(make_molecule('c1ccccc1'), 6, 0.0, 0.0)
        self.assertEqual(len(conformers), 6)
        self.assertEqual(discarded, 0)

    def test_conformer_ids_address_the_returned_molecule(self):
        unique_mol, conformers, _ = generate_unique_conformers(make_molecule('CCCCO'), 4, 0.1, 0.5)
        for _, conf_id in conformers:
            self.assertIsNotNone(unique_mol.GetConformer(conf_id))


class GenerateConformersHandlerTests(TempWorkingDirTestCase):
    def generate(self, smiles='CCCCCCO', count=5, energy=0.1, rms=0.5, name='conformer',
                 file_type='xyz', charge=0, multiplicity=1):
        return on_generate_conformers(self.working_directory, smiles, charge, multiplicity, count,
                                      energy, rms, name, file_type, FakeProgress())

    def test_files_and_table_agree(self):
        status, files, table = self.generate(count=5)
        self.assertEqual(status_colour(status), 'green')
        self.assertEqual(len(table), 5)
        self.assertEqual(list(table.columns), ['ID', 'Energy (kcal/mol)'])
        for conformer_id in table['ID']:
            self.assertIn(f'conformer_{conformer_id}.xyz', files)
            self.assertTrue(os.path.exists(self.path(f'conformer_{conformer_id}.xyz')))

    def test_ids_are_sequential_and_energies_ascending(self):
        _, _, table = self.generate(count=6)
        self.assertEqual(list(table['ID']), list(range(1, 7)))
        energies = list(table['Energy (kcal/mol)'])
        self.assertEqual(energies, sorted(energies))

    def test_the_first_file_is_the_lowest_energy_conformer(self):
        _, _, table = self.generate(count=4)
        self.assertEqual(table['Energy (kcal/mol)'].iloc[0], min(table['Energy (kcal/mol)']))

    def test_charge_and_multiplicity_reach_the_file(self):
        self.generate(count=1, charge=-1, multiplicity=2)
        self.assertEqual(self.read_file('conformer_1.xyz').splitlines()[0], '-1 2')

    def test_every_output_format(self):
        for file_type in ('xyz', 'pdb', 'mol'):
            with self.subTest(file_type=file_type):
                _, _, table = self.generate(count=2, name=f'c_{file_type}', file_type=file_type)
                self.assertTrue(os.path.exists(self.path(f'c_{file_type}_1.{file_type}')))

    def test_an_unreachable_count_warns_instead_of_failing(self):
        status, _, table = self.generate(smiles='c1ccccc1', count=10)
        self.assertEqual(status_colour(status), 'orange')
        self.assertIn('lower the thresholds', status_text(status))
        self.assertEqual(len(table), 1)

    def test_invalid_smiles_reports_an_error_and_an_empty_table(self):
        status, _, table = self.generate(smiles='not a molecule')
        self.assertEqual(status_colour(status), 'red')
        self.assertIn('invalid SMILES', status_text(status))
        self.assertEqual(len(table), 0)
        self.assertEqual(list(table.columns), ['ID', 'Energy (kcal/mol)'])

    def test_the_file_list_is_returned_so_the_rest_of_the_ui_refreshes(self):
        _, files, _ = self.generate(count=2)
        self.assertEqual(sorted(files), ['conformer_1.xyz', 'conformer_2.xyz'])

    def test_a_tighter_threshold_never_yields_more_conformers(self):
        _, _, loose = self.generate(count=12, energy=0.1, name='loose')
        _, _, tight = self.generate(count=12, energy=3.0, name='tight')
        self.assertLessEqual(len(tight), len(loose))


if __name__ == '__main__':
    unittest.main()
