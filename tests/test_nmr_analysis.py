"""NMR log parsing, chemical equivalence and the multiplet table."""
import unittest

import numpy as np

from helpers import (GEOMETRY_LOG, NMR_JCOUPLING_LOG, NMR_SHIELDING_LOG, TempWorkingDirTestCase,
                     make_molecule)
from utils import (_fortran_float, build_nmr_peak_table, calculate_chemical_shifts,
                   compute_carbon_equivalence, compute_hydrogen_equivalence, morgan_ranks,
                   multiplicity_label, parse_gaussian_geometry, parse_nmr_jcouplings,
                   parse_nmr_shielding_constants, perceive_bonds)


def molecule_geometry(smiles):
    """Coordinates and symbols in Gaussian nucleus order, which is RDKit's atom order here."""
    mol = make_molecule(smiles, optimize=True)
    conf = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                        conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())])
    return coords, symbols


class ShieldingParsingTests(TempWorkingDirTestCase):
    def test_isotropic_values_are_read_for_every_nucleus(self):
        self.write_file('nmr.log', NMR_SHIELDING_LOG)
        data = parse_nmr_shielding_constants(self.path('nmr.log'))
        self.assertEqual(data, [
            (1, 'C', 225.1454),
            (2, 'C', 195.2693),
            (3, 'O', 360.8253),
            (4, 'H', 31.5000),
        ])

    def test_tensor_component_and_eigenvalue_lines_are_skipped(self):
        self.write_file('nmr.log', NMR_SHIELDING_LOG)
        data = parse_nmr_shielding_constants(self.path('nmr.log'))
        self.assertEqual(len(data), 4)

    def test_a_log_without_nmr_data_yields_nothing(self):
        self.write_file('plain.log', 'Normal termination of Gaussian 16\n')
        self.assertEqual(parse_nmr_shielding_constants(self.path('plain.log')), [])


class ChemicalShiftTests(unittest.TestCase):
    def test_shift_is_reference_minus_shielding(self):
        shifts = calculate_chemical_shifts([(1, 'C', 195.2693), (2, 'H', 31.5)],
                                           {'C': 203.4, 'H': 32.9})
        self.assertAlmostEqual(shifts[0][2], 203.4 - 195.2693, places=4)
        self.assertAlmostEqual(shifts[1][2], 32.9 - 31.5, places=4)

    def test_elements_without_a_reference_are_dropped(self):
        shifts = calculate_chemical_shifts([(1, 'C', 100.0), (2, 'O', 300.0)], {'C': 203.4})
        self.assertEqual([entry[1] for entry in shifts], ['C'])


class FortranFloatTests(unittest.TestCase):
    def test_d_notation_is_understood(self):
        self.assertAlmostEqual(_fortran_float('0.418515D+02'), 41.8515)
        self.assertAlmostEqual(_fortran_float('-0.420058D-01'), -0.0420058)
        self.assertAlmostEqual(_fortran_float('0.000000D+00'), 0.0)


class JCouplingParsingTests(TempWorkingDirTestCase):
    def setUp(self):
        super().setUp()
        self.write_file('nmr.log', NMR_JCOUPLING_LOG)
        self.couplings = parse_nmr_jcouplings(self.path('nmr.log'))

    def test_the_k_matrix_printed_earlier_is_ignored(self):
        # 99.9999 Hz only appears in the K matrix, which must not be picked up
        self.assertNotIn(99.9999, [round(entry[4], 4) for entry in self.couplings])

    def test_pairs_are_read_from_the_j_matrix(self):
        lookup = {(a, b): j for a, _, b, _, j in self.couplings}
        self.assertAlmostEqual(lookup[(2, 1)], 41.8515, places=4)
        self.assertAlmostEqual(lookup[(3, 1)], -0.0420058, places=6)
        self.assertAlmostEqual(lookup[(4, 2)], -10.8619, places=4)

    def test_the_zero_diagonal_is_skipped(self):
        self.assertFalse([entry for entry in self.couplings if entry[0] == entry[2]])

    def test_wrapped_column_blocks_are_read(self):
        # nuclei 6 and 7 only appear together in the second column block
        lookup = {(a, b): j for a, _, b, _, j in self.couplings}
        self.assertAlmostEqual(lookup[(7, 6)], 3.98266, places=4)

    def test_each_pair_is_reported_once(self):
        pairs = [(a, b) for a, _, b, _, _ in self.couplings]
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_element_symbols_are_attached_when_supplied(self):
        couplings = parse_nmr_jcouplings(self.path('nmr.log'), {1: 'C', 2: 'C', 3: 'O'})
        first = next(entry for entry in couplings if entry[0] == 2 and entry[2] == 1)
        self.assertEqual((first[1], first[3]), ('C', 'C'))

    def test_a_log_without_couplings_yields_nothing(self):
        self.write_file('plain.log', 'Normal termination\n')
        self.assertEqual(parse_nmr_jcouplings(self.path('plain.log')), [])


class GeometryParsingTests(TempWorkingDirTestCase):
    def test_atomic_numbers_become_symbols(self):
        self.write_file('geom.log', GEOMETRY_LOG)
        symbols, coords = parse_gaussian_geometry(self.path('geom.log'))
        self.assertEqual(symbols, ['O', 'H', 'H'])
        self.assertEqual(coords.shape, (3, 3))

    def test_the_last_orientation_block_wins(self):
        self.write_file('geom.log', GEOMETRY_LOG)
        _, coords = parse_gaussian_geometry(self.path('geom.log'))
        self.assertAlmostEqual(coords[0][2], -0.120000, places=6)
        self.assertAlmostEqual(coords[1][1], 0.760000, places=6)

    def test_a_log_without_geometry_yields_empty(self):
        self.write_file('plain.log', 'Normal termination\n')
        symbols, coords = parse_gaussian_geometry(self.path('plain.log'))
        self.assertEqual(symbols, [])
        self.assertEqual(coords.size, 0)


class BondPerceptionTests(unittest.TestCase):
    def test_ethanol_connectivity(self):
        coords, symbols = molecule_geometry('CCO')
        adjacency = perceive_bonds(coords, symbols)
        # C1-C2, C2-O3 plus five C-H and one O-H
        self.assertEqual(sum(len(neighbours) for neighbours in adjacency) // 2, 8)
        self.assertIn(1, adjacency[0])
        self.assertIn(2, adjacency[1])

    def test_symmetry_ranks_group_equivalent_atoms(self):
        coords, symbols = molecule_geometry('CC(C)O')  # isopropanol
        _, ranks = morgan_ranks(coords, symbols)
        methyl_carbons = [i for i, s in enumerate(symbols) if s == 'C']
        # the two methyl carbons share a rank, the CH is different
        rank_counts = {}
        for i in methyl_carbons:
            rank_counts[ranks[i]] = rank_counts.get(ranks[i], 0) + 1
        self.assertIn(2, rank_counts.values(), 'the two methyls must be equivalent')


class EquivalenceTests(unittest.TestCase):
    def test_methyl_protons_average_to_one_group(self):
        coords, symbols = molecule_geometry('CCO')
        equivalence = compute_hydrogen_equivalence(coords, symbols)
        groups = {}
        for atom_idx, key in equivalence.items():
            groups.setdefault(key, []).append(atom_idx)
        sizes = sorted(len(members) for members in groups.values())
        self.assertEqual(sizes, [1, 2, 3], 'ethanol gives OH, CH2 and CH3')

    def test_symmetric_methyls_merge(self):
        coords, symbols = molecule_geometry('CC(C)O')
        equivalence = compute_hydrogen_equivalence(coords, symbols)
        groups = {}
        for atom_idx, key in equivalence.items():
            groups.setdefault(key, []).append(atom_idx)
        sizes = sorted(len(members) for members in groups.values())
        self.assertEqual(sizes, [1, 1, 6], 'the two isopropyl methyls average together')

    def test_carbons_are_grouped_by_whole_molecule_symmetry(self):
        coords, symbols = molecule_geometry('CC(C)O')
        equivalence = compute_carbon_equivalence(coords, symbols)
        self.assertEqual(len(equivalence), 3, 'one entry per carbon')
        self.assertEqual(len(set(equivalence.values())), 2, 'two distinct carbon environments')

    def test_indices_are_one_based_like_gaussian(self):
        coords, symbols = molecule_geometry('CCO')
        equivalence = compute_hydrogen_equivalence(coords, symbols)
        self.assertNotIn(0, equivalence)
        self.assertEqual(min(equivalence), symbols.index('H') + 1)


class PeakTableTests(unittest.TestCase):
    def test_equivalent_nuclei_merge_into_one_peak(self):
        shifts = [(4, 'H', 1.20), (5, 'H', 1.22), (6, 'H', 1.19)]
        equivalence = {4: ('heavy', 1), 5: ('heavy', 1), 6: ('heavy', 1)}
        peaks = build_nmr_peak_table(shifts, 'H', equivalence=equivalence)
        self.assertEqual(len(peaks), 1)
        self.assertEqual(peaks[0]['count'], 3)
        self.assertAlmostEqual(peaks[0]['shift'], (1.20 + 1.22 + 1.19) / 3, places=6)

    def test_only_the_requested_element_is_tabulated(self):
        shifts = [(1, 'C', 20.0), (4, 'H', 1.2)]
        self.assertEqual(len(build_nmr_peak_table(shifts, 'H')), 1)
        self.assertEqual(len(build_nmr_peak_table(shifts, 'C')), 1)
        self.assertEqual(build_nmr_peak_table(shifts, 'N'), [])

    def test_peaks_come_out_in_ascending_shift_order(self):
        shifts = [(1, 'H', 7.2), (2, 'H', 1.2), (3, 'H', 3.6)]
        peaks = build_nmr_peak_table(shifts, 'H')
        self.assertEqual([round(p['shift'], 1) for p in peaks], [1.2, 3.6, 7.2])

    def test_multiplicity_comes_from_the_partner_group_size(self):
        # a CH3 next to a CH2: the CH3 is split into a triplet, the CH2 into a quartet
        shifts = [(1, 'H', 1.2), (2, 'H', 1.2), (3, 'H', 1.2), (4, 'H', 3.6), (5, 'H', 3.6)]
        equivalence = {1: 'a', 2: 'a', 3: 'a', 4: 'b', 5: 'b'}
        jcouplings = [(a, 'H', b, 'H', 7.0) for a in (1, 2, 3) for b in (4, 5)]
        peaks = build_nmr_peak_table(shifts, 'H', jcouplings=jcouplings, equivalence=equivalence)
        by_shift = {round(p['shift'], 1): p for p in peaks}
        self.assertEqual(multiplicity_label(by_shift[1.2]['couplings']), 't')
        self.assertEqual(multiplicity_label(by_shift[3.6]['couplings']), 'q')

    def test_equivalent_nuclei_do_not_split_each_other(self):
        shifts = [(1, 'H', 1.2), (2, 'H', 1.2)]
        equivalence = {1: 'a', 2: 'a'}
        jcouplings = [(1, 'H', 2, 'H', 12.0)]
        peaks = build_nmr_peak_table(shifts, 'H', jcouplings=jcouplings, equivalence=equivalence)
        self.assertEqual(peaks[0]['couplings'], [])

    def test_small_couplings_are_dropped(self):
        shifts = [(1, 'H', 1.2), (2, 'H', 3.6)]
        equivalence = {1: 'a', 2: 'b'}
        jcouplings = [(1, 'H', 2, 'H', 0.2)]
        peaks = build_nmr_peak_table(shifts, 'H', jcouplings=jcouplings, equivalence=equivalence,
                                     j_threshold=0.5)
        self.assertTrue(all(not p['couplings'] for p in peaks))

    def test_heteronuclear_couplings_are_ignored(self):
        shifts = [(1, 'H', 1.2), (2, 'H', 3.6)]
        equivalence = {1: 'a', 2: 'b'}
        jcouplings = [(1, 'H', 9, 'C', 120.0)]
        peaks = build_nmr_peak_table(shifts, 'H', jcouplings=jcouplings, equivalence=equivalence)
        self.assertTrue(all(not p['couplings'] for p in peaks))

    def test_without_equivalence_nuclei_group_by_rounded_shift(self):
        shifts = [(1, 'H', 1.200), (2, 'H', 1.204), (3, 'H', 5.0)]
        peaks = build_nmr_peak_table(shifts, 'H')
        self.assertEqual(sorted(p['count'] for p in peaks), [1, 2])


class MultiplicityLabelTests(unittest.TestCase):
    def test_no_coupling_is_a_singlet(self):
        self.assertEqual(multiplicity_label([]), 's')

    def test_letters_for_common_multiplicities(self):
        for n, letter in ((1, 'd'), (2, 't'), (3, 'q'), (4, 'p'), (5, 'h'), (6, 'hept')):
            self.assertEqual(multiplicity_label([{'n': n, 'J': 7.0}]), letter)

    def test_several_couplings_concatenate(self):
        self.assertEqual(multiplicity_label([{'n': 1, 'J': 7.0}, {'n': 1, 'J': 2.0}]), 'dd')

    def test_unusual_counts_fall_back_to_m(self):
        self.assertEqual(multiplicity_label([{'n': 9, 'J': 7.0}]), 'm')


if __name__ == '__main__':
    unittest.main()
