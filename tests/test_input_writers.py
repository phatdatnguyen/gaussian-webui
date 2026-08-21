"""The six Gaussian input writers in utils.py."""
import os
import unittest

from helpers import BASIS_WITH_ECP, BSE_STYLE_BASIS, TempWorkingDirTestCase, make_molecule
from utils import (parse_element_list, write_fluorescence_gaussian_input, write_nmr_gaussian_input,
                   write_opt_freq_gaussian_input, write_opt_gaussian_input, write_sp_gaussian_input,
                   write_uv_vis_gaussian_input)


def route_lines(text):
    return [line for line in text.splitlines() if line.startswith('#P')]


def molecule_spec_lines(text):
    """The atom lines between the charge/multiplicity line and the terminating blank line."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == '0 1') + 1
    atoms = []
    for line in lines[start:]:
        if not line.strip():
            break
        atoms.append(line)
    return atoms


class WriterStructureTests(TempWorkingDirTestCase):
    def setUp(self):
        super().setUp()
        self.mol = make_molecule('CCO')
        self.stem = self.path('job')

    def write_all(self, **kwargs):
        """Every writer with the same arguments, keyed by the file it produces."""
        write_sp_gaussian_input(self.mol, self.stem, 'DFT', '', **kwargs)
        produced = {'sp': self.read_file('job.gjf')}
        write_opt_gaussian_input(self.mol, self.stem, 'DFT', '', **kwargs)
        produced['opt'] = self.read_file('job.gjf')
        write_opt_freq_gaussian_input(self.mol, self.stem, 'DFT', '', **kwargs)
        produced['freq'] = self.read_file('job.gjf')
        write_uv_vis_gaussian_input(self.mol, self.stem, 'DFT', '', **kwargs)
        produced['uv'] = self.read_file('job.gjf')
        write_nmr_gaussian_input(self.mol, self.stem, 'DFT', **{
            k: v for k, v in kwargs.items() if k != 'method_name'})
        produced['nmr'] = self.read_file('job.gjf')
        write_fluorescence_gaussian_input(self.mol, self.stem, 'DFT', '', **kwargs)
        produced['em_opt'] = self.read_file('job_S1_Opt.gjf')
        produced['em_sp'] = self.read_file('job_S1_SP.gjf')
        return produced

    def test_every_writer_emits_link0_route_title_and_geometry(self):
        for name, text in self.write_all().items():
            with self.subTest(writer=name):
                self.assertIn('%NProcShared=', text)
                self.assertIn('%Mem=', text)
                self.assertIn('%Chk=', text)
                self.assertTrue(route_lines(text), 'no route section')
                self.assertTrue(text.endswith('\n'))

    def test_geometry_matches_the_molecule(self):
        write_sp_gaussian_input(self.mol, self.stem, 'DFT', '')
        atoms = molecule_spec_lines(self.read_file('job.gjf'))
        self.assertEqual(len(atoms), self.mol.GetNumAtoms())
        self.assertEqual([line.split()[0] for line in atoms],
                         [atom.GetSymbol() for atom in self.mol.GetAtoms()])

    def test_charge_and_multiplicity_are_written(self):
        write_sp_gaussian_input(self.mol, self.stem, 'DFT', '', charge=-1, multiplicity=3)
        self.assertIn('\n-1 3\n', self.read_file('job.gjf'))

    def test_calculation_keywords(self):
        expected = {'sp': ' sp', 'opt': ' opt', 'freq': ' opt freq',
                    'uv': 'TD(NStates=', 'nmr': 'NMR=GIAO'}
        produced = self.write_all()
        for name, keyword in expected.items():
            with self.subTest(writer=name):
                self.assertIn(keyword, produced[name])

    def test_nmr_spin_spin_is_optional(self):
        write_nmr_gaussian_input(self.mol, self.stem, 'DFT')
        self.assertIn('NMR=GIAO', self.read_file('job.gjf'))
        write_nmr_gaussian_input(self.mol, self.stem, 'DFT', spin_spin_coupling=True)
        self.assertIn('NMR=(GIAO,spinspin)', self.read_file('job.gjf'))

    def test_nmr_rejects_methods_it_cannot_use(self):
        for method_type in ('CCSD', 'MP4', 'Semi-empirical', 'Compound'):
            with self.subTest(method_type=method_type):
                with self.assertRaises(Exception):
                    write_nmr_gaussian_input(self.mol, self.stem, method_type)

    def test_solvation_is_added_to_every_route_section(self):
        produced = self.write_all(solvation=True, solvation_model='smd', solvent='water')
        for name, text in produced.items():
            with self.subTest(writer=name):
                for route in route_lines(text):
                    self.assertIn('scrf=(smd,solvent=water)', route)

    def test_solvation_off_leaves_no_scrf(self):
        for name, text in self.write_all().items():
            with self.subTest(writer=name):
                self.assertNotIn('scrf', text)

    def test_resources_are_honoured(self):
        write_sp_gaussian_input(self.mol, self.stem, 'DFT', '', n_proc=8, memory=16)
        text = self.read_file('job.gjf')
        self.assertIn('%NProcShared=8', text)
        self.assertIn('%Mem=16GB', text)

    def test_excited_state_count_is_honoured(self):
        write_uv_vis_gaussian_input(self.mol, self.stem, 'DFT', '', n_states=42)
        self.assertIn('TD(NStates=42)', self.read_file('job.gjf'))

    def test_uv_vis_writes_two_linked_jobs(self):
        write_uv_vis_gaussian_input(self.mol, self.stem, 'DFT', '')
        text = self.read_file('job.gjf')
        self.assertEqual(text.count('--Link1--'), 1)
        self.assertEqual(len(route_lines(text)), 2)
        self.assertIn('Geom=AllCheck', route_lines(text)[1])

    def test_emission_writes_two_separate_files_sharing_a_checkpoint(self):
        write_fluorescence_gaussian_input(self.mol, self.stem, 'DFT', '')
        opt = self.read_file('job_S1_Opt.gjf')
        single_point = self.read_file('job_S1_SP.gjf')
        self.assertIn('Root=1', opt)
        self.assertIn('Geom=AllCheck', single_point)
        self.assertIn('%Chk=' + self.stem + '.chk', opt)
        self.assertIn('%Chk=' + self.stem + '.chk', single_point)


class WriterCustomBasisTests(TempWorkingDirTestCase):
    def setUp(self):
        super().setUp()
        self.mol = make_molecule('CCO')
        self.stem = self.path('job')

    def test_gen_keyword_and_section_are_written_together(self):
        write_sp_gaussian_input(self.mol, self.stem, 'DFT', '', basis='Custom',
                                custom_basis_set=BSE_STYLE_BASIS)
        text = self.read_file('job.gjf')
        self.assertIn('b3lyp/gen', route_lines(text)[0])
        headers = [line for line in text.splitlines() if parse_element_list(line)]
        self.assertEqual(sorted(headers), ['C     0', 'H     0', 'O     0'])

    def test_basis_section_follows_the_molecule_specification(self):
        write_sp_gaussian_input(self.mol, self.stem, 'DFT', '', basis='Custom',
                                custom_basis_set=BSE_STYLE_BASIS)
        lines = self.read_file('job.gjf').splitlines()
        last_atom = max(i for i, line in enumerate(lines)
                        if line[:2].strip() in ('C', 'H', 'O') and len(line.split()) == 4)
        self.assertEqual(lines[last_atom + 1].strip(), '',
                         'a blank line must close the molecule specification')
        self.assertIsNotNone(parse_element_list(lines[last_atom + 2]),
                             'the basis section must start right after it')

    def test_genecp_is_used_when_the_molecule_needs_a_pseudopotential(self):
        write_sp_gaussian_input(make_molecule('[Cu]C'), self.stem, 'DFT', '', basis='Custom',
                                custom_basis_set=BASIS_WITH_ECP)
        text = self.read_file('job.gjf')
        self.assertIn('b3lyp/genecp', route_lines(text)[0])
        self.assertIn('CU-ECP', text)

    def test_checkpoint_jobs_read_the_basis_back_instead_of_repeating_it(self):
        write_uv_vis_gaussian_input(self.mol, self.stem, 'DFT', '', basis='Custom',
                                    custom_basis_set=BSE_STYLE_BASIS)
        routes = route_lines(self.read_file('job.gjf'))
        self.assertIn('b3lyp/gen ', routes[0] + ' ')
        self.assertIn('chkbasis', routes[1])

        write_fluorescence_gaussian_input(self.mol, self.stem, 'DFT', '', basis='Custom',
                                          custom_basis_set=BSE_STYLE_BASIS)
        self.assertIn('gen', route_lines(self.read_file('job_S1_Opt.gjf'))[0])
        self.assertIn('chkbasis', route_lines(self.read_file('job_S1_SP.gjf'))[0])

    def test_every_writer_accepts_a_custom_basis_set(self):
        writers = [
            (write_sp_gaussian_input, 'job.gjf'),
            (write_opt_gaussian_input, 'job.gjf'),
            (write_opt_freq_gaussian_input, 'job.gjf'),
            (write_uv_vis_gaussian_input, 'job.gjf'),
        ]
        for writer, produced in writers:
            with self.subTest(writer=writer.__name__):
                writer(self.mol, self.stem, 'DFT', '', basis='Custom',
                       custom_basis_set=BSE_STYLE_BASIS)
                self.assertIn('gen', route_lines(self.read_file(produced))[0])
        write_nmr_gaussian_input(self.mol, self.stem, 'DFT', basis='Custom',
                                 custom_basis_set=BSE_STYLE_BASIS)
        self.assertIn('gen', route_lines(self.read_file('job.gjf'))[0])

    def test_semi_empirical_ignores_the_custom_basis_set(self):
        write_sp_gaussian_input(self.mol, self.stem, 'Semi-empirical', 'PM6', basis='Custom',
                                custom_basis_set=BSE_STYLE_BASIS)
        text = self.read_file('job.gjf')
        self.assertIn('pm6', route_lines(text)[0])
        self.assertNotIn('gen', route_lines(text)[0])
        self.assertNotIn('****', text)

    def test_a_basis_set_missing_an_element_leaves_no_partial_file(self):
        with self.assertRaises(ValueError):
            write_sp_gaussian_input(make_molecule('CI'), self.stem, 'DFT', '', basis='Custom',
                                    custom_basis_set=BSE_STYLE_BASIS)
        self.assertFalse(os.path.exists(self.path('job.gjf')))

    def test_named_basis_set_output_is_unaffected(self):
        write_sp_gaussian_input(self.mol, self.stem, 'DFT', '', basis='6-31G(d,p)')
        with_named = self.read_file('job.gjf')
        self.assertIn('b3lyp/6-31g(d,p)', with_named)
        self.assertNotIn('****', with_named)


if __name__ == '__main__':
    unittest.main()
