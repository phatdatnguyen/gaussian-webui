"""Custom basis set handling: parsing a Basis Set Exchange file into a Gaussian gen section."""
import unittest

from helpers import (BASIS_WITH_ECP, BSE_STYLE_BASIS, SHARED_BLOCK_BASIS,
                     TempWorkingDirTestCase, make_molecule)
from utils import (build_custom_basis_section, custom_basis_set_uses_ecp, gaussian_method_string,
                   keep_blocks_for_elements, parse_element_list, prepare_custom_basis_section,
                   resolve_basis_keyword, split_basis_blocks, split_custom_basis_sections,
                   split_ecp_blocks, uses_basis_set)


class ParseElementListTests(unittest.TestCase):
    def test_reads_a_single_element_header(self):
        self.assertEqual(parse_element_list('H     0'), ['H'])

    def test_reads_a_shared_header(self):
        self.assertEqual(parse_element_list('C  H  O     0'), ['C', 'H', 'O'])

    def test_accepts_the_dash_prefix_some_files_use(self):
        self.assertEqual(parse_element_list('-Cu     0'), ['Cu'])

    def test_rejects_shell_and_coefficient_lines(self):
        for line in ('S    4   1.00', 'SP   3   1.00', 'P    1   1.00',
                     '      0.754732D+02           0.427821D-02',
                     'CU-ECP     3     10', 'f potential', '  1',
                     '2      1.0000000        0.0000000', '****', ''):
            with self.subTest(line=line):
                self.assertIsNone(parse_element_list(line))


class SectionSplittingTests(unittest.TestCase):
    def test_comment_header_and_blank_lines_are_dropped(self):
        # Gaussian ends the gen section at the first blank line, so a Basis Set Exchange
        # header would otherwise truncate the basis to nothing.
        basis_lines, ecp_lines = split_custom_basis_sections(BSE_STYLE_BASIS)
        self.assertEqual(ecp_lines, [])
        self.assertEqual(basis_lines[0], 'H     0')
        self.assertFalse([line for line in basis_lines if not line.strip()])
        self.assertFalse([line for line in basis_lines if line.lstrip().startswith('!')])

    def test_pseudopotentials_are_separated_from_basis_functions(self):
        basis_lines, ecp_lines = split_custom_basis_sections(BASIS_WITH_ECP)
        self.assertIn('CU-ECP     3     10', ecp_lines)
        self.assertNotIn('CU-ECP     3     10', basis_lines)
        self.assertEqual(basis_lines[0], 'H     0')
        self.assertEqual(ecp_lines[0], 'Cu     0')

    def test_basis_blocks_split_on_the_terminator(self):
        basis_lines, _ = split_custom_basis_sections(BSE_STYLE_BASIS)
        blocks = split_basis_blocks(basis_lines)
        self.assertEqual(len(blocks), 4)
        self.assertTrue(all(block[-1].strip() == '****' for block in blocks))

    def test_a_leading_terminator_is_tolerated(self):
        # some downloads open with a bare **** before the first element, which forms a
        # headerless block that must be dropped rather than confuse the filter
        leading = '****\n' + BSE_STYLE_BASIS
        section = build_custom_basis_section(make_molecule('CCO'), leading)
        self.assertTrue(section.startswith('H     0'))
        self.assertEqual(sorted(line for line in section.splitlines() if parse_element_list(line)),
                         ['C     0', 'H     0', 'O     0'])

    def test_ecp_blocks_split_on_each_element_header(self):
        two_entries = BASIS_WITH_ECP + 'Zn     0\nZN-ECP     3     10\nf potential\n  1\n'
        _, ecp_lines = split_custom_basis_sections(two_entries)
        blocks = split_ecp_blocks(ecp_lines)
        self.assertEqual([block[0] for block in blocks], ['Cu     0', 'Zn     0'])


class ElementFilteringTests(unittest.TestCase):
    def test_only_the_requested_elements_survive(self):
        basis_lines, _ = split_custom_basis_sections(BSE_STYLE_BASIS)
        blocks, defined = keep_blocks_for_elements(split_basis_blocks(basis_lines), {'C', 'H'})
        self.assertEqual(defined, {'C', 'H'})
        self.assertEqual([block[0] for block in blocks], ['H     0', 'C     0'])

    def test_a_shared_header_is_rewritten_to_the_matching_elements(self):
        basis_lines, _ = split_custom_basis_sections(SHARED_BLOCK_BASIS)
        blocks, defined = keep_blocks_for_elements(split_basis_blocks(basis_lines), {'C', 'O'})
        self.assertEqual(defined, {'C', 'O'})
        # H must not be dragged in by the block it shared with C and O
        self.assertEqual(blocks[0][0], 'C O     0')


class BuildSectionTests(unittest.TestCase):
    def test_section_covers_exactly_the_molecule_elements(self):
        section = build_custom_basis_section(make_molecule('CCO'), BSE_STYLE_BASIS)
        headers = [line for line in section.splitlines() if parse_element_list(line)]
        self.assertEqual(sorted(headers), ['C     0', 'H     0', 'O     0'])
        # N is in the file but not in ethanol
        self.assertNotIn('N     0', section)

    def test_section_has_no_blank_or_comment_lines(self):
        section = build_custom_basis_section(make_molecule('CCO'), BSE_STYLE_BASIS)
        self.assertFalse([line for line in section.splitlines() if not line.strip()])
        self.assertFalse([line for line in section.splitlines() if line.lstrip().startswith('!')])

    def test_ecp_section_keeps_exactly_one_separating_blank_line(self):
        section = build_custom_basis_section(make_molecule('[Cu]C'), BASIS_WITH_ECP)
        blanks = [i for i, line in enumerate(section.splitlines()) if not line.strip()]
        self.assertEqual(len(blanks), 1, "basis and ECP must be separated by a single blank line")
        lines = section.splitlines()
        self.assertIn('CU-ECP     3     10', lines[blanks[0]:])
        self.assertNotIn('CU-ECP     3     10', lines[:blanks[0]])

    def test_pseudopotentials_for_absent_elements_are_dropped(self):
        section = build_custom_basis_section(make_molecule('C'), BASIS_WITH_ECP)
        self.assertNotIn('CU-ECP', section)
        self.assertNotIn('Cu', section)
        self.assertFalse([line for line in section.splitlines() if not line.strip()])

    def test_missing_element_is_reported(self):
        with self.assertRaises(ValueError) as caught:
            build_custom_basis_section(make_molecule('CI'), BSE_STYLE_BASIS)
        self.assertIn('I', str(caught.exception))

    def test_file_covering_nothing_is_reported(self):
        with self.assertRaises(ValueError) as caught:
            build_custom_basis_section(make_molecule('[Cu]'), BSE_STYLE_BASIS)
        self.assertIn('does not define any element', str(caught.exception))


class BasisKeywordTests(unittest.TestCase):
    def test_named_basis_set_is_passed_through(self):
        self.assertEqual(resolve_basis_keyword('6-31G(d,p)', None), '6-31G(d,p)')

    def test_plain_custom_basis_uses_gen(self):
        section = build_custom_basis_section(make_molecule('CCO'), BSE_STYLE_BASIS)
        self.assertEqual(resolve_basis_keyword('Custom', section), 'gen')

    def test_custom_basis_with_pseudopotentials_uses_genecp(self):
        section = build_custom_basis_section(make_molecule('[Cu]C'), BASIS_WITH_ECP)
        self.assertEqual(resolve_basis_keyword('Custom', section), 'genecp')

    def test_keyword_follows_the_filtered_section_not_the_raw_file(self):
        # The raw file has an ECP, but a molecule without Cu keeps none of it, so the
        # route line has to say gen or Gaussian will look for a section that is not there.
        self.assertTrue(custom_basis_set_uses_ecp(BASIS_WITH_ECP))
        section = build_custom_basis_section(make_molecule('C'), BASIS_WITH_ECP)
        self.assertEqual(resolve_basis_keyword('Custom', section), 'gen')

    def test_semi_empirical_and_compound_methods_take_no_basis(self):
        self.assertFalse(uses_basis_set('Semi-empirical'))
        self.assertFalse(uses_basis_set('Compound'))
        for method_type in ('HF', 'DFT', 'MP2', 'MP4', 'CCSD', 'BD'):
            self.assertTrue(uses_basis_set(method_type))

    def test_no_section_is_prepared_for_a_method_without_a_basis(self):
        mol = make_molecule('CCO')
        self.assertIsNone(prepare_custom_basis_section(mol, BSE_STYLE_BASIS, 'Semi-empirical'))
        self.assertIsNone(prepare_custom_basis_section(mol, None, 'DFT'))
        self.assertIsNotNone(prepare_custom_basis_section(mol, BSE_STYLE_BASIS, 'DFT'))


class MethodStringTests(unittest.TestCase):
    def test_each_method_type_builds_the_expected_route_fragment(self):
        cases = {
            'HF': 'hf/3-21g',
            'DFT': 'b3lyp/3-21g',
            'MP2': 'mp2/3-21g',
            'CCSD': 'ccsd/3-21g',
            'BD': 'bd/3-21g',
            'MP4': 'mp4(sdtq)/3-21g',
        }
        for method_type, expected in cases.items():
            with self.subTest(method_type=method_type):
                self.assertEqual(
                    gaussian_method_string(method_type, 'CNDO', 'B3LYP', '3-21G'), expected)

    def test_semi_empirical_and_compound_use_the_method_name_alone(self):
        self.assertEqual(gaussian_method_string('Semi-empirical', 'PM6', 'B3LYP', '3-21G'), 'pm6')
        self.assertEqual(gaussian_method_string('Compound', 'G3', 'B3LYP', '3-21G'), 'g3')


class RealBasisSetExchangeFileTests(TempWorkingDirTestCase):
    """The pcSseg-2.txt the maintainer downloaded, if it is still in the repo."""

    def setUp(self):
        super().setUp()
        import os
        from helpers import REPO_ROOT
        self.basis_path = os.path.join(REPO_ROOT, 'pcSseg-2.txt')
        if not os.path.exists(self.basis_path):
            self.skipTest('pcSseg-2.txt is not in the repo')
        with open(self.basis_path) as handle:
            self.basis_text = handle.read()

    def test_whole_periodic_table_file_is_trimmed_to_the_molecule(self):
        section = build_custom_basis_section(make_molecule('CCO'), self.basis_text)
        headers = sorted(line.split()[0] for line in section.splitlines() if parse_element_list(line))
        self.assertEqual(headers, ['C', 'H', 'O'])
        self.assertLess(len(section.splitlines()), len(self.basis_text.splitlines()) / 10)

    def test_it_is_recognised_as_an_all_electron_basis(self):
        section = build_custom_basis_section(make_molecule('CCO'), self.basis_text)
        self.assertEqual(resolve_basis_keyword('Custom', section), 'gen')


if __name__ == '__main__':
    unittest.main()
