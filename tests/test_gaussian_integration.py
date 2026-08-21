"""End to end tests that actually run Gaussian.

Skipped unless g16 is on PATH. To enable them:

    export g16root=$HOME/gaussian
    . $g16root/g16/bsd/g16.profile
    export GAUSS_SCRDIR=$HOME/gaussian/scratch

Every job is a tiny HF/STO-3G calculation so the suite still finishes quickly.
"""
import os
import unittest

import cclib

from helpers import BSE_STYLE_BASIS, TempWorkingDirTestCase, requires_gaussian, status_colour, status_text
from calculation import CUSTOM_BASIS_SET, on_generate_input_file, on_run_calculation
from result import on_load_result_file
from utils import (add_bonds, calculate_chemical_shifts, compute_hydrogen_equivalence,
                   build_nmr_peak_table, mol_from_gaussian_file, parse_gaussian_geometry,
                   parse_nmr_jcouplings, parse_nmr_shielding_constants)


@requires_gaussian
class GaussianPipelineTests(TempWorkingDirTestCase):
    """Generate an input, run g16, then read the log back the way the Result tab does."""

    def setUp(self):
        super().setUp()
        self.write_structure('O', 'water.xyz')  # 3 atoms, fastest useful molecule

    def generate(self, calculation_type, output, structure='water.xyz', method_type='HF',
                 basis='STO-3G', custom_basis_file=None, functional='B3LYP'):
        status, _ = on_generate_input_file(
            self.working_directory, structure, calculation_type,
            False, 'MMFF', 200, False, 'iefpcm', 'water',
            method_type, '', functional, basis, custom_basis_file, 3, False, 0, 1, 1, 1, output)
        self.assertEqual(status_colour(status), 'green', status_text(status))
        return output + '.gjf'

    def run_job(self, input_file):
        steps = list(on_run_calculation(self.working_directory, input_file))
        status = steps[-1][0]
        self.assertEqual(status_colour(status), 'green', status_text(status))
        return os.path.splitext(input_file)[0] + '.log'

    def test_single_point_runs_and_parses(self):
        log = self.run_job(self.generate('Single-Point', 'sp'))
        data = cclib.io.ccopen(self.path(log)).parse()
        self.assertEqual(data.natom, 3)
        self.assertTrue(hasattr(data, 'scfenergies'))
        self.assertLess(data.scfenergies[-1], 0)

    def test_the_result_tab_shows_the_energy_accordion(self):
        log = self.run_job(self.generate('Single-Point', 'sp'))
        outputs = on_load_result_file(self.working_directory, log)
        status, data = outputs[0], outputs[1]
        self.assertEqual(status_colour(status), 'green', status_text(status))
        self.assertTrue(outputs[2]['visible'], 'the Energy accordion must open')
        self.assertIn('hartree', outputs[3])
        self.assertIn('Debye', outputs[4])

    def test_geometry_optimization_produces_a_trajectory(self):
        log = self.run_job(self.generate('Geometry Optimization', 'opt'))
        data = cclib.io.ccopen(self.path(log)).parse()
        self.assertGreater(len(data.scfenergies), 1, 'an optimisation has several steps')
        outputs = on_load_result_file(self.working_directory, log)
        self.assertTrue(outputs[8]['visible'], 'the optimisation accordion must open')

    def test_frequency_run_yields_vibrations(self):
        log = self.run_job(self.generate('Frequency', 'freq'))
        data = cclib.io.ccopen(self.path(log)).parse()
        self.assertTrue(hasattr(data, 'vibfreqs'))
        self.assertEqual(len(data.vibfreqs), 3, 'water has 3N-6 = 3 modes')
        outputs = on_load_result_file(self.working_directory, log)
        self.assertTrue(outputs[10]['visible'], 'the Frequency accordion must open')

    def test_a_structure_can_be_read_back_out_of_a_log(self):
        log = self.run_job(self.generate('Geometry Optimization', 'opt'))
        mol = add_bonds(mol_from_gaussian_file(self.path(log)))
        self.assertEqual(mol.GetNumAtoms(), 3)
        self.assertEqual(mol.GetNumBonds(), 2)

    def test_gaussian_accepts_and_actually_uses_the_custom_basis_set(self):
        # The gen section is only worth writing if Gaussian reads it. Running the same
        # molecule with the custom basis and with a named one must give different
        # energies; if the gen section were ignored the two would agree.
        self.write_file('basis.gbs', BSE_STYLE_BASIS)
        custom_input = self.generate('Single-Point', 'custom', basis=CUSTOM_BASIS_SET,
                                     custom_basis_file='basis.gbs')
        self.assertIn('/gen', self.read_file(custom_input))
        custom_log = self.run_job(custom_input)
        self.assertIn('Normal termination', self.read_file(custom_log))

        named_log = self.run_job(self.generate('Single-Point', 'named', basis='STO-3G'))

        custom_energy = cclib.io.ccopen(self.path(custom_log)).parse().scfenergies[-1]
        named_energy = cclib.io.ccopen(self.path(named_log)).parse().scfenergies[-1]
        self.assertNotAlmostEqual(custom_energy, named_energy, places=3,
                                  msg='Gaussian appears to have ignored the gen section')

    def test_a_deliberately_broken_basis_set_makes_gaussian_fail_loudly(self):
        # Guards the test above: our own validation must not be the only thing standing
        # between a bad file and a wrong answer. This one covers every element of the
        # molecule, so it gets past our checks and has to be rejected by Gaussian itself.
        self.write_file('broken.gbs',
                        'H     0\nS    1   1.00\n      not-a-number     nonsense\n****\n'
                        'O     0\nS    1   1.00\n      not-a-number     nonsense\n****\n')
        input_file = self.generate('Single-Point', 'broken', basis=CUSTOM_BASIS_SET,
                                   custom_basis_file='broken.gbs')
        steps = list(on_run_calculation(self.working_directory, input_file))
        self.assertEqual(status_colour(steps[-1][0]), 'red',
                         'Gaussian should have refused this basis set')


@requires_gaussian
class GaussianNmrTests(TempWorkingDirTestCase):
    """The NMR chain: shieldings, couplings, geometry, equivalence and the peak table."""

    @classmethod
    def setUpClass(cls):
        cls._log_cache = {}

    def setUp(self):
        super().setUp()
        self.write_structure('CCO', 'ethanol.xyz', seed=7)

    def nmr_log(self):
        status, _ = on_generate_input_file(
            self.working_directory, 'ethanol.xyz', 'NMR Spectrum',
            True, 'MMFF', 500, False, 'iefpcm', 'water',
            'HF', '', 'B3LYP', 'STO-3G', None, 3, True, 0, 1, 1, 1, 'nmr')
        self.assertEqual(status_colour(status), 'green', status_text(status))
        steps = list(on_run_calculation(self.working_directory, 'nmr.gjf'))
        self.assertEqual(status_colour(steps[-1][0]), 'green', status_text(steps[-1][0]))
        return self.path('nmr.log')

    def test_the_whole_nmr_chain(self):
        log = self.nmr_log()

        shielding = parse_nmr_shielding_constants(log)
        self.assertEqual(len(shielding), 9, 'ethanol has 9 nuclei')
        self.assertEqual(sorted(element for _, element, _ in shielding),
                         ['C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'O'])

        symbols = {idx: element for idx, element, _ in shielding}
        couplings = parse_nmr_jcouplings(log, symbols)
        self.assertTrue(couplings, 'spinspin was requested, so couplings must be present')
        self.assertTrue(all(a != b for a, _, b, _, _ in couplings))
        pairs = [(a, b) for a, _, b, _, _ in couplings]
        self.assertEqual(len(pairs), len(set(pairs)), 'each pair reported once')
        # the lower triangle of a 9x9 matrix
        self.assertEqual(len(pairs), 9 * 8 // 2)

        geometry_symbols, coords = parse_gaussian_geometry(log)
        self.assertEqual(len(geometry_symbols), 9)
        self.assertEqual([s for s in geometry_symbols if s != 'H'], ['C', 'C', 'O'])

        equivalence = compute_hydrogen_equivalence(coords, geometry_symbols)
        groups = {}
        for atom_idx, key in equivalence.items():
            groups.setdefault(key, []).append(atom_idx)
        self.assertEqual(sorted(len(m) for m in groups.values()), [1, 2, 3],
                         'ethanol gives OH, CH2 and CH3')

        shifts = calculate_chemical_shifts(shielding, {'H': 32.9, 'C': 203.4})
        peaks = build_nmr_peak_table(shifts, 'H', jcouplings=couplings, equivalence=equivalence)
        self.assertEqual(len(peaks), 3)
        self.assertEqual(sorted(p['count'] for p in peaks), [1, 2, 3])
        self.assertEqual([p['shift'] for p in peaks], sorted(p['shift'] for p in peaks))

    def test_the_result_tab_opens_the_nmr_accordion(self):
        log = self.nmr_log()
        outputs = on_load_result_file(self.working_directory, os.path.basename(log))
        self.assertEqual(status_colour(outputs[0]), 'green', status_text(outputs[0]))
        self.assertTrue(outputs[19]['visible'], 'the NMR accordion must open')


@requires_gaussian
class GaussianStopTests(TempWorkingDirTestCase):
    """Stopping a real g16 job, which is the case the process group kill exists for."""

    def test_a_real_job_can_be_stopped(self):
        import threading
        import time
        import psutil
        import calculation

        self.addCleanup(setattr, calculation, 'running_calculation_process', None)
        # something big enough to still be running a second from now
        self.write_structure('c1ccc2c(c1)ccc1ccccc12', 'anthracene.xyz')
        status, _ = on_generate_input_file(
            self.working_directory, 'anthracene.xyz', 'Frequency',
            False, 'MMFF', 200, False, 'iefpcm', 'water',
            'DFT', '', 'B3LYP', '6-31G(d,p)', None, 3, False, 0, 1, 1, 1, 'slow')
        self.assertEqual(status_colour(status), 'green', status_text(status))

        steps = []
        worker = threading.Thread(target=lambda: steps.extend(
            on_run_calculation(self.working_directory, 'slow.gjf')))
        worker.start()
        self.addCleanup(worker.join, 60)

        deadline = time.time() + 30
        while time.time() < deadline and calculation.running_calculation_process is None:
            time.sleep(0.1)
        self.assertIsNotNone(calculation.running_calculation_process, 'g16 never started')
        g16_process = psutil.Process(calculation.running_calculation_process.pid)
        time.sleep(3)
        # the Gaussian links g16 forks are exactly what a bare terminate would strand
        children = g16_process.children(recursive=True)

        from calculation import on_stop_calculation
        on_stop_calculation()
        worker.join(timeout=60)
        self.assertFalse(worker.is_alive())
        self.assertEqual(status_colour(steps[-1][0]), 'orange', status_text(steps[-1][0]))

        gone, alive = psutil.wait_procs([g16_process] + children, timeout=30)
        self.assertFalse(alive, f'these Gaussian processes survived the stop: {alive}')


if __name__ == '__main__':
    unittest.main()
