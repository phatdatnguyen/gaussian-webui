"""Spectrum builders in utils.py and the Result tab helpers."""
import os
import unittest

import numpy as np
import pandas as pd

from helpers import TempWorkingDirTestCase
from result import _parse_reference, on_export_data, on_show_absorption_spectrum
from result import on_working_directory_file_list_change as result_file_list_change
from utils import (gaussian, generate_absorption_emission_spectrum_interactive,
                   generate_ir_spectrum_interactive, generate_nmr_spectrum_interactive,
                   lorentzian_ir, lorentzian_nmr)


class LineShapeTests(unittest.TestCase):
    def test_lorentzian_peaks_at_its_centre(self):
        self.assertAlmostEqual(lorentzian_ir(1000.0, 1000.0, 5.0, width=10), 5.0)
        self.assertLess(lorentzian_ir(1100.0, 1000.0, 5.0, width=10), 5.0)

    def test_lorentzian_is_symmetric(self):
        left = lorentzian_ir(990.0, 1000.0, 1.0, width=10)
        right = lorentzian_ir(1010.0, 1000.0, 1.0, width=10)
        self.assertAlmostEqual(left, right)

    def test_wider_lines_are_broader(self):
        narrow = lorentzian_ir(1020.0, 1000.0, 1.0, width=5)
        wide = lorentzian_ir(1020.0, 1000.0, 1.0, width=40)
        self.assertLess(narrow, wide)

    def test_gaussian_peaks_at_its_centre(self):
        self.assertAlmostEqual(gaussian(300.0, 300.0, 20.0), 1.0)
        self.assertLess(gaussian(340.0, 300.0, 20.0), 1.0)

    def test_nmr_lorentzian_peaks_at_its_centre(self):
        self.assertAlmostEqual(lorentzian_nmr(7.2, 7.2, 0.5), 1.0)
        self.assertLess(lorentzian_nmr(8.0, 7.2, 0.5), 1.0)


class IrSpectrumTests(unittest.TestCase):
    def test_a_figure_is_built_from_frequencies_and_intensities(self):
        figure = generate_ir_spectrum_interactive([1000.0, 1700.0, 3000.0], [10.0, 50.0, 5.0])
        self.assertIsNotNone(figure)
        self.assertTrue(figure.data)

    def test_no_frequencies_gives_no_figure(self):
        self.assertIsNone(generate_ir_spectrum_interactive([], []))
        self.assertIsNone(generate_ir_spectrum_interactive(None, None))

    def test_the_requested_range_is_respected(self):
        figure = generate_ir_spectrum_interactive([1000.0], [10.0], plot_range=(500, 1500))
        x = np.asarray(figure.data[0].x)
        self.assertGreaterEqual(x.min(), 500)
        self.assertLessEqual(x.max(), 1500)

    def test_the_point_count_is_respected(self):
        figure = generate_ir_spectrum_interactive([1000.0], [10.0], points=250)
        self.assertEqual(len(figure.data[0].x), 250)

    def test_transmittance_and_absorbance_differ(self):
        peaks, intensities = [1000.0], [10.0]
        transmittance = generate_ir_spectrum_interactive(peaks, intensities, transmittance=True)
        absorbance = generate_ir_spectrum_interactive(peaks, intensities, transmittance=False)
        self.assertFalse(np.allclose(np.asarray(transmittance.data[0].y),
                                     np.asarray(absorbance.data[0].y)))


class UvVisSpectrumTests(unittest.TestCase):
    def test_a_figure_is_built_from_wavelengths_and_strengths(self):
        figure = generate_absorption_emission_spectrum_interactive(
            wavelengths=np.array([250.0, 320.0]), oscs=np.array([0.4, 0.1]))
        self.assertIsNotNone(figure)
        self.assertTrue(figure.data)

    def test_the_requested_range_is_respected(self):
        figure = generate_absorption_emission_spectrum_interactive(
            wavelengths=np.array([300.0]), oscs=np.array([1.0]),
            points=500, plot_range=(200, 400))
        x = np.asarray(figure.data[0].x)
        self.assertGreaterEqual(x.min(), 200)
        self.assertLessEqual(x.max(), 400)


class NmrSpectrumTests(unittest.TestCase):
    def test_a_figure_is_built_for_the_requested_element(self):
        shifts = [(1, 'H', 1.2), (2, 'H', 3.6), (3, 'C', 20.0)]
        figure = generate_nmr_spectrum_interactive(shifts, 'H', points=500)
        self.assertIsNotNone(figure)
        self.assertTrue(figure.data)

    def test_an_element_with_no_nuclei_gives_no_figure(self):
        self.assertIsNone(generate_nmr_spectrum_interactive([(1, 'H', 1.2)], 'N', points=100))


class ResultFileListTests(unittest.TestCase):
    def test_only_logs_are_offered_and_they_are_sorted(self):
        update = result_file_list_change(['run_10.log', 'run_2.log', 'a.gjf', 'b.xyz'])
        self.assertEqual(update['choices'], ['run_2.log', 'run_10.log'])
        self.assertEqual(update['value'], 'run_2.log')

    def test_no_logs_selects_nothing(self):
        update = result_file_list_change(['a.gjf'])
        self.assertEqual(update['choices'], [])
        self.assertIsNone(update['value'])


class ReferenceShieldingTests(unittest.TestCase):
    def test_numbers_are_parsed(self):
        self.assertEqual(_parse_reference('32.9', 0.0), 32.9)
        self.assertEqual(_parse_reference('  203.4 ', 0.0), 203.4)

    def test_blanks_and_junk_fall_back_to_the_default(self):
        for value in ('', '   ', 'abc', None):
            with self.subTest(value=value):
                self.assertEqual(_parse_reference(value, 32.9), 32.9)


class ExportTests(TempWorkingDirTestCase):
    def test_a_dataframe_is_written_as_csv_and_announced(self):
        frame = pd.DataFrame({'Wavelength': [250.0, 320.0], 'Strength': [0.4, 0.1]})
        status, files = on_export_data(self.working_directory, 'peaks', frame)
        self.assertIn('green', status)
        self.assertIn('peaks.csv', files)
        written = pd.read_csv(self.path('peaks.csv'))
        self.assertEqual(list(written.columns), ['Wavelength', 'Strength'])
        self.assertEqual(len(written), 2)

    def test_an_export_failure_is_reported(self):
        status, _ = on_export_data(self.working_directory, 'bad', None)
        self.assertIn('red', status)
        self.assertIn('Error exporting data', status)


class AbsorptionSpectrumHandlerTests(unittest.TestCase):
    class FakeData:
        etenergies = np.array([40000.0, 33000.0])
        etoscs = np.array([0.42, 0.02])
        etsecs = [[((10, 0), (11, 0), 0.7)], [((9, 0), (11, 0), 0.6)]]
        etsyms = ['Singlet-A', 'Singlet-B']

    def test_peaks_are_tabulated_and_sorted_by_wavelength(self):
        frame, figure = on_show_absorption_spectrum(self.FakeData())
        self.assertIsNotNone(figure)
        self.assertEqual(list(frame.columns),
                         ['Absorption Wavelength (nm)', 'Oscillator Strength',
                          'Transitions', 'Symmetry'])
        wavelengths = list(frame['Absorption Wavelength (nm)'])
        self.assertEqual(wavelengths, sorted(wavelengths))
        self.assertAlmostEqual(wavelengths[0], 1e7 / 40000.0, places=3)

    def test_data_without_excited_states_is_handled(self):
        frame, figure = on_show_absorption_spectrum(object())
        self.assertIsNone(frame)
        self.assertIsNone(figure)


if __name__ == '__main__':
    unittest.main()
