"""Calculation tab: dropdown wiring, input generation, and the run/stop lifecycle."""
import os
import threading
import time
import unittest

import psutil

from helpers import BSE_STYLE_BASIS, TempWorkingDirTestCase, status_colour, status_text
import calculation
from calculation import (CUSTOM_BASIS_SET, on_basis_set_change, on_change_calculation_type,
                         on_generate_input_file, on_method_type_change, on_mm_checkbox_change,
                         on_run_calculation, on_solvation_checkbox_change, on_stop_calculation,
                         on_upload_custom_basis_set, on_working_directory_file_list_change,
                         read_custom_basis_set)


class FileListWiringTests(unittest.TestCase):
    def test_structure_and_input_dropdowns_are_filled_and_sorted(self):
        files = ['b.xyz', 'a.pdb', 'run_10.gjf', 'run_2.gjf', 'x.log', 'notes.txt', 'j.chk']
        structures, inputs = on_working_directory_file_list_change(files, 'run_2')
        self.assertEqual(structures['choices'], ['a.pdb', 'b.xyz', 'x.log'])
        self.assertEqual(inputs['choices'], ['run_2.gjf', 'run_10.gjf'])

    def test_the_just_generated_input_file_is_preselected(self):
        _, inputs = on_working_directory_file_list_change(['a.gjf', 'run_2.gjf'], 'run_2')
        self.assertEqual(inputs['value'], 'run_2.gjf')

    def test_falls_back_to_the_first_input_file(self):
        _, inputs = on_working_directory_file_list_change(['a.gjf'], 'missing')
        self.assertEqual(inputs['value'], 'a.gjf')

    def test_empty_directory_selects_nothing(self):
        structures, inputs = on_working_directory_file_list_change([], 'x')
        self.assertIsNone(structures['value'])
        self.assertIsNone(inputs['value'])


class FormVisibilityTests(unittest.TestCase):
    def test_calculation_type_drives_the_default_file_name(self):
        expected = {
            'Single-Point': 'single_point',
            'Geometry Optimization': 'geometry_optimization',
            'Frequency': 'frequency',
            'Absorption Spectrum': 'absorption_spectrum',
            'Emission Spectrum': 'emission_spectrum',
            'NMR Spectrum': 'nmr_spectrum',
        }
        for calculation_type, file_name in expected.items():
            with self.subTest(calculation_type=calculation_type):
                self.assertEqual(on_change_calculation_type(calculation_type)[0], file_name)

    def test_nmr_offers_only_the_methods_it_supports(self):
        _, methods, _, _ = on_change_calculation_type('NMR Spectrum')
        self.assertEqual(methods['choices'], ['HF', 'DFT', 'MP2'])

    def test_excited_state_count_appears_only_for_spectra(self):
        for calculation_type, visible in (('Absorption Spectrum', True),
                                          ('Emission Spectrum', True),
                                          ('Single-Point', False)):
            with self.subTest(calculation_type=calculation_type):
                self.assertEqual(on_change_calculation_type(calculation_type)[2]['visible'], visible)

    def test_spin_spin_checkbox_appears_only_for_nmr(self):
        self.assertTrue(on_change_calculation_type('NMR Spectrum')[3]['visible'])
        self.assertFalse(on_change_calculation_type('Frequency')[3]['visible'])

    def test_force_field_controls_follow_the_checkbox(self):
        self.assertTrue(all(update['visible'] for update in on_mm_checkbox_change(True)))
        self.assertFalse(any(update['visible'] for update in on_mm_checkbox_change(False)))

    def test_solvation_controls_follow_the_checkbox(self):
        self.assertTrue(all(update['visible'] for update in on_solvation_checkbox_change(True)))
        self.assertFalse(any(update['visible'] for update in on_solvation_checkbox_change(False)))

    def test_dft_shows_the_functional_and_basis_set(self):
        method_name, functional, basis, custom = on_method_type_change('DFT', '3-21G')
        self.assertFalse(method_name['visible'])
        self.assertTrue(functional['visible'])
        self.assertTrue(basis['visible'])
        self.assertFalse(custom['visible'])

    def test_semi_empirical_hides_the_basis_set(self):
        method_name, functional, basis, custom = on_method_type_change('Semi-empirical', '3-21G')
        self.assertTrue(method_name['visible'])
        self.assertIn('PM6', method_name['choices'])
        self.assertFalse(functional['visible'])
        self.assertFalse(basis['visible'])
        self.assertFalse(custom['visible'])

    def test_custom_basis_row_appears_only_when_custom_is_selected(self):
        self.assertTrue(on_basis_set_change(CUSTOM_BASIS_SET)['visible'])
        self.assertFalse(on_basis_set_change('6-31G(d,p)')['visible'])

    def test_custom_basis_row_hides_when_the_method_takes_no_basis(self):
        self.assertTrue(on_method_type_change('DFT', CUSTOM_BASIS_SET)[3]['visible'])
        self.assertFalse(on_method_type_change('Compound', CUSTOM_BASIS_SET)[3]['visible'])


class CustomBasisUploadTests(TempWorkingDirTestCase):
    def test_upload_copies_into_the_working_directory_and_refreshes_the_list(self):
        source = os.path.join(self.temp_root, 'downloaded.gbs')
        with open(source, 'w') as handle:
            handle.write(BSE_STYLE_BASIS)
        file_name, files = on_upload_custom_basis_set(self.working_directory, source)
        self.assertEqual(file_name, 'downloaded.gbs')
        self.assertIn('downloaded.gbs', files)
        self.assertEqual(self.read_file('downloaded.gbs'), BSE_STYLE_BASIS)

    def test_upload_without_a_working_directory_warns(self):
        file_name, files = on_upload_custom_basis_set(None, 'whatever.gbs')
        self.assertEqual(file_name['__type__'], 'update')

    def test_reading_back_the_uploaded_file(self):
        self.write_file('b.gbs', BSE_STYLE_BASIS)
        self.assertEqual(read_custom_basis_set(self.working_directory, 'b.gbs'), BSE_STYLE_BASIS)

    def test_unhelpful_inputs_are_reported(self):
        self.write_file('empty.gbs', '')
        for file_name, message in ((None, 'choose a custom basis set'),
                                   ('', 'choose a custom basis set'),
                                   ('missing.gbs', 'not in the working directory'),
                                   ('empty.gbs', 'is empty')):
            with self.subTest(file_name=file_name):
                with self.assertRaises(ValueError) as caught:
                    read_custom_basis_set(self.working_directory, file_name)
                self.assertIn(message, str(caught.exception))


class GenerateInputFileTests(TempWorkingDirTestCase):
    def setUp(self):
        super().setUp()
        self.write_structure('CCO', 'ethanol.xyz')

    def generate(self, structure='ethanol.xyz', calculation_type='Single-Point', method_type='DFT',
                 method_name='', functional='B3LYP', basis='3-21G', custom_basis_file=None,
                 output='job', use_mm=False, solvation=False, charge=0, multiplicity=1):
        return on_generate_input_file(
            self.working_directory, structure, calculation_type,
            use_mm, 'MMFF', 200, solvation, 'iefpcm', 'water',
            method_type, method_name, functional, basis, custom_basis_file, 10, False,
            charge, multiplicity, 4, 2, output)

    def test_an_input_file_is_produced_and_announced(self):
        status, files = self.generate()
        self.assertEqual(status_colour(status), 'green')
        self.assertIn('job.gjf', files)

    def test_no_structure_selected_warns_without_writing(self):
        status, _ = self.generate(structure='')
        self.assertEqual(status, '')
        self.assertFalse(os.path.exists(self.path('job.gjf')))

    def test_every_calculation_type_writes_its_file(self):
        expected = {
            'Single-Point': 'job.gjf',
            'Geometry Optimization': 'job.gjf',
            'Frequency': 'job.gjf',
            'Absorption Spectrum': 'job.gjf',
            'NMR Spectrum': 'job.gjf',
            'Emission Spectrum': 'job_S1_Opt.gjf',
        }
        for calculation_type, produced in expected.items():
            with self.subTest(calculation_type=calculation_type):
                status, _ = self.generate(calculation_type=calculation_type)
                self.assertEqual(status_colour(status), 'green', status_text(status))
                self.assertTrue(os.path.exists(self.path(produced)))

    def test_structures_can_come_from_every_supported_format(self):
        from rdkit import Chem
        from helpers import make_molecule
        mol = make_molecule('CCO')
        Chem.MolToPDBFile(mol, self.path('e.pdb'))
        Chem.MolToMolFile(mol, self.path('e.mol'))
        for structure in ('ethanol.xyz', 'e.pdb', 'e.mol'):
            with self.subTest(structure=structure):
                status, _ = self.generate(structure=structure, output='from_' + structure[:-4])
                self.assertEqual(status_colour(status), 'green', status_text(status))

    def test_molecular_mechanics_preoptimisation_changes_the_geometry(self):
        self.generate(output='plain')
        self.generate(output='minimised', use_mm=True)
        self.assertNotEqual(self.read_file('plain.gjf'), self.read_file('minimised.gjf'))

    def test_solvation_reaches_the_route_section(self):
        self.generate(solvation=True)
        self.assertIn('scrf=(iefpcm,solvent=water)', self.read_file('job.gjf'))

    def test_charge_and_multiplicity_reach_the_input_file(self):
        self.generate(charge=1, multiplicity=2)
        self.assertIn('\n1 2\n', self.read_file('job.gjf'))

    def test_custom_basis_set_is_embedded(self):
        self.write_file('basis.gbs', BSE_STYLE_BASIS)
        status, _ = self.generate(basis=CUSTOM_BASIS_SET, custom_basis_file='basis.gbs')
        self.assertEqual(status_colour(status), 'green', status_text(status))
        text = self.read_file('job.gjf')
        self.assertIn('b3lyp/gen', text)
        self.assertIn('****', text)

    def test_custom_basis_set_without_a_file_is_reported(self):
        status, _ = self.generate(basis=CUSTOM_BASIS_SET, custom_basis_file=None)
        self.assertEqual(status_colour(status), 'red')
        self.assertIn('choose a custom basis set', status_text(status))

    def test_custom_basis_set_missing_an_element_is_reported(self):
        self.write_structure('CI', 'iodo.xyz')
        self.write_file('basis.gbs', BSE_STYLE_BASIS)
        status, _ = self.generate(structure='iodo.xyz', basis=CUSTOM_BASIS_SET,
                                  custom_basis_file='basis.gbs')
        self.assertEqual(status_colour(status), 'red')
        self.assertIn('no basis functions for I', status_text(status))

    def test_nmr_with_an_unsupported_method_is_reported(self):
        status, _ = self.generate(calculation_type='NMR Spectrum', method_type='CCSD')
        self.assertEqual(status_colour(status), 'red')


# A fake g16 that forks a child, so stopping can be checked to take the whole group down.
FAKE_G16_LONG = """#!/bin/bash
echo "fake gaussian" > "$2"
sleep 300 &
echo $! > "$CHILD_PID_FILE"
wait
"""
FAKE_G16_OK = """#!/bin/bash
echo "Normal termination of Gaussian 16" > "$2"
sleep 0.2
exit 0
"""
FAKE_G16_FAIL = """#!/bin/bash
echo "Error termination" > "$2"
exit 1
"""


class RunCalculationTests(TempWorkingDirTestCase):
    def setUp(self):
        super().setUp()
        self.write_file('job.gjf', '#P hf/sto-3g sp\n')
        self.addCleanup(self.reset_module_state)

    def reset_module_state(self):
        calculation.running_calculation_process = None
        calculation.calculation_stopped_by_user = False

    def run_to_completion(self, input_file='job.gjf'):
        return list(on_run_calculation(self.working_directory, input_file))

    def test_no_input_file_selected_warns(self):
        steps = self.run_to_completion(input_file='')
        self.assertEqual(steps[-1][0], '')
        self.assertTrue(steps[-1][2]['interactive'], 'run button must stay usable')
        self.assertFalse(steps[-1][3]['interactive'], 'stop button must stay disabled')

    def test_a_successful_run_reports_green_and_refreshes_the_file_list(self):
        self.install_fake_g16(FAKE_G16_OK)
        steps = self.run_to_completion()
        status, files, run_button, stop_button = steps[-1]
        self.assertEqual(status_colour(status), 'green')
        self.assertIn('Calculation finished', status_text(status))
        self.assertIn('job.log', files)
        self.assertTrue(run_button['interactive'])
        self.assertFalse(stop_button['interactive'])

    def test_progress_is_streamed_while_the_job_runs(self):
        self.install_fake_g16(FAKE_G16_OK)
        steps = self.run_to_completion()
        self.assertGreater(len(steps), 1, 'the handler must yield before it finishes')
        running_status = steps[0][0]
        self.assertIn('Running job.gjf', running_status)
        self.assertIn('elapsed', running_status)
        self.assertEqual(status_colour(running_status), '#1e90ff')
        self.assertFalse(steps[0][2]['interactive'], 'run button disabled while running')
        self.assertTrue(steps[0][3]['interactive'], 'stop button enabled while running')

    def test_a_failing_job_reports_the_exit_code(self):
        self.install_fake_g16(FAKE_G16_FAIL)
        status = self.run_to_completion()[-1][0]
        self.assertEqual(status_colour(status), 'red')
        self.assertIn('exited with code 1', status_text(status))

    def test_a_missing_g16_is_reported_rather_than_raised(self):
        # point PATH somewhere without g16
        previous = os.environ['PATH']
        os.environ['PATH'] = os.path.join(self.temp_root, 'nothing-here')
        self.addCleanup(os.environ.__setitem__, 'PATH', previous)
        status = self.run_to_completion()[-1][0]
        self.assertEqual(status_colour(status), 'red')
        self.assertIn('Error running calculation', status_text(status))

    def test_the_process_handle_is_cleared_afterwards(self):
        self.install_fake_g16(FAKE_G16_OK)
        self.run_to_completion()
        self.assertIsNone(calculation.running_calculation_process)

    def test_stopping_kills_the_whole_process_group(self):
        child_pid_file = os.path.join(self.temp_root, 'child.pid')
        os.environ['CHILD_PID_FILE'] = child_pid_file
        self.addCleanup(os.environ.pop, 'CHILD_PID_FILE', None)
        self.install_fake_g16(FAKE_G16_LONG)

        steps = []
        worker = threading.Thread(target=lambda: steps.extend(
            on_run_calculation(self.working_directory, 'job.gjf')))
        worker.start()
        self.addCleanup(worker.join, 30)

        deadline = time.time() + 20
        while time.time() < deadline:
            if os.path.exists(child_pid_file) and calculation.running_calculation_process:
                break
            time.sleep(0.1)
        self.assertTrue(os.path.exists(child_pid_file), 'fake g16 never started')

        g16_pid = calculation.running_calculation_process.pid
        with open(child_pid_file) as handle:
            child_pid = int(handle.read().strip())
        self.assertTrue(psutil.pid_exists(child_pid))

        on_stop_calculation()
        worker.join(timeout=30)
        self.assertFalse(worker.is_alive(), 'the run handler must return after a stop')

        status = steps[-1][0]
        self.assertEqual(status_colour(status), 'orange')
        self.assertIn('stopped', status_text(status).lower())
        # the forked child would survive if only g16 itself had been signalled
        for pid in (g16_pid, child_pid):
            self.assertTrue(self.process_is_gone(pid), f'pid {pid} survived the stop')
        self.assertIsNone(calculation.running_calculation_process)

    @staticmethod
    def process_is_gone(pid, timeout=15):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not psutil.pid_exists(pid):
                return True
            try:
                if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                    return True
            except psutil.NoSuchProcess:
                return True
            time.sleep(0.2)
        return False

    def test_stopping_when_nothing_runs_is_harmless(self):
        self.assertIsNone(on_stop_calculation())

    def test_a_second_run_is_refused_while_one_is_in_flight(self):
        self.install_fake_g16(FAKE_G16_OK)
        calculation.running_calculation_process = object()
        try:
            steps = self.run_to_completion()
        finally:
            calculation.running_calculation_process = None
        self.assertEqual(len(steps), 1, 'the handler must bail out after a single yield')
        self.assertFalse(os.path.exists(self.path('job.log')), 'no second g16 may be started')

    def test_a_run_still_works_after_a_previous_one_was_stopped(self):
        self.install_fake_g16(FAKE_G16_OK)
        self.run_to_completion()
        status = self.run_to_completion()[-1][0]
        self.assertEqual(status_colour(status), 'green')


if __name__ == '__main__':
    unittest.main()
