"""Working directory column: file typing, selection, and the text viewer."""
import os
import unittest

from helpers import SelectEvent, TempWorkingDirTestCase
from working_directory import (STRUCTURE_FILE_EXTENSIONS, TEXT_FILE_EXTENSIONS, get_working_directories,
                               on_delete_file, on_file_list_change, on_open_working_directory,
                               on_save_text_file, on_select_file,
                               on_selected_structure_file_state_change,
                               on_selected_text_file_state_change, on_upload_file, on_view_text_file)


class FileTypeTests(TempWorkingDirTestCase):
    def types_for(self, names):
        for name in names:
            self.write_file(name, 'x')
        frame = on_file_list_change(self.working_directory)
        return dict(zip(frame['File'], frame['Type']))

    def test_each_extension_gets_its_label(self):
        types = self.types_for(['a.xyz', 'a.pdb', 'a.mol', 'a.mol2', 'a.gjf', 'a.chk', 'a.fchk',
                                'a.log', 'a.cube', 'a.txt', 'a.gbs', 'mystery.bin'])
        self.assertEqual(types['a.xyz'], 'Structure file')
        self.assertEqual(types['a.pdb'], 'Structure file')
        self.assertEqual(types['a.mol'], 'Structure file')
        self.assertEqual(types['a.mol2'], 'Structure file')
        self.assertEqual(types['a.gjf'], 'Input file')
        self.assertEqual(types['a.chk'], 'Check file')
        self.assertEqual(types['a.fchk'], 'Check file')
        self.assertEqual(types['a.log'], 'Log file')
        self.assertEqual(types['a.cube'], 'Cube file')
        self.assertEqual(types['mystery.bin'], 'Other File')

    def test_text_files_are_labelled_text_file(self):
        types = self.types_for(['notes.txt', 'basis.gbs', 'b.bas', 'b.basis', 'b.dat'])
        for name in ('notes.txt', 'basis.gbs', 'b.bas', 'b.basis', 'b.dat'):
            with self.subTest(name=name):
                self.assertEqual(types[name], 'Text file')

    def test_the_table_carries_name_type_and_timestamp(self):
        self.write_file('a.xyz', 'x')
        frame = on_file_list_change(self.working_directory)
        self.assertEqual(list(frame.columns), ['File', 'Type', 'Modified'])
        self.assertEqual(len(frame), 1)

    def test_newest_files_are_listed_first(self):
        import time
        self.write_file('old.xyz', 'x')
        time.sleep(1.05)
        self.write_file('new.xyz', 'x')
        frame = on_file_list_change(self.working_directory)
        self.assertEqual(list(frame['File']), ['new.xyz', 'old.xyz'])


class SelectionTests(unittest.TestCase):
    def test_text_viewable_files(self):
        viewable = ['a.xyz', 'a.pdb', 'a.mol', 'a.mol2', 'a.gjf', 'a.log',
                    'a.txt', 'a.gbs', 'a.bas', 'a.basis', 'a.dat']
        for name in viewable:
            with self.subTest(name=name):
                _, _, text_file, _ = on_select_file(SelectEvent(name))
                self.assertEqual(text_file, name)

    def test_binary_files_are_not_text_viewable(self):
        for name in ('a.chk', 'a.fchk', 'a.cube', 'mystery.bin'):
            with self.subTest(name=name):
                _, _, text_file, _ = on_select_file(SelectEvent(name))
                self.assertIsNone(text_file)

    def test_only_structures_and_logs_open_in_the_3d_viewer(self):
        for name in ('a.xyz', 'a.pdb', 'a.mol', 'a.mol2', 'a.log'):
            with self.subTest(name=name):
                _, structure_file, _, _ = on_select_file(SelectEvent(name))
                self.assertEqual(structure_file, name)
        for name in ('a.gjf', 'a.txt', 'a.chk'):
            with self.subTest(name=name):
                _, structure_file, _, _ = on_select_file(SelectEvent(name))
                self.assertIsNone(structure_file)

    def test_the_selected_name_is_always_returned(self):
        selected, _, _, delete_button = on_select_file(SelectEvent('mystery.bin'))
        self.assertEqual(selected, 'mystery.bin')
        self.assertTrue(delete_button['interactive'], 'anything selected can be deleted')

    def test_view_buttons_track_the_selection_state(self):
        self.assertTrue(on_selected_structure_file_state_change('a.xyz')['interactive'])
        self.assertFalse(on_selected_structure_file_state_change(None)['interactive'])
        self.assertTrue(on_selected_text_file_state_change('a.txt')['interactive'])
        self.assertFalse(on_selected_text_file_state_change(None)['interactive'])


class TextViewerTests(TempWorkingDirTestCase):
    def test_viewing_loads_the_content_and_names_the_file(self):
        self.write_file('notes.txt', 'line one\nline two\n')
        viewer, save_button = on_view_text_file(self.working_directory, 'notes.txt')
        self.assertEqual(viewer['value'], 'line one\nline two\n')
        self.assertIn('notes.txt', viewer['label'])
        self.assertTrue(viewer['interactive'], 'the viewer must be editable')
        self.assertTrue(save_button['interactive'])

    def test_saving_writes_the_edited_content_back(self):
        self.write_file('notes.txt', 'before\n')
        files = on_save_text_file(self.working_directory, 'notes.txt', 'after\n')
        self.assertEqual(self.read_file('notes.txt'), 'after\n')
        self.assertIn('notes.txt', files)

    def test_a_basis_set_file_round_trips_through_the_editor(self):
        from helpers import BSE_STYLE_BASIS
        self.write_file('basis.gbs', BSE_STYLE_BASIS)
        viewer, _ = on_view_text_file(self.working_directory, 'basis.gbs')
        self.assertEqual(viewer['value'], BSE_STYLE_BASIS)
        on_save_text_file(self.working_directory, 'basis.gbs', viewer['value'].replace('test-set', 'edited'))
        self.assertIn('edited', self.read_file('basis.gbs'))

    def test_viewing_a_missing_file_warns_instead_of_raising(self):
        viewer, save_button = on_view_text_file(self.working_directory, 'nope.txt')
        self.assertEqual(viewer['__type__'], 'update')

    def test_saving_without_a_selection_warns(self):
        files = on_save_text_file(self.working_directory, None, 'content')
        self.assertEqual(files, [])


class FileOperationTests(TempWorkingDirTestCase):
    def test_uploading_copies_the_file_in(self):
        source = os.path.join(self.temp_root, 'incoming.xyz')
        with open(source, 'w') as handle:
            handle.write('0 1\nH 0.0 0.0 0.0\n')
        files = on_upload_file(self.working_directory, source)
        self.assertIn('incoming.xyz', files)
        self.assertEqual(self.read_file('incoming.xyz'), '0 1\nH 0.0 0.0 0.0\n')

    def test_deleting_removes_the_file_and_refreshes_the_list(self):
        self.write_file('a.xyz', 'x')
        self.write_file('b.xyz', 'x')
        files = on_delete_file(self.working_directory, 'a.xyz')
        self.assertEqual(files, ['b.xyz'])
        self.assertFalse(os.path.exists(self.path('a.xyz')))

    def test_deleting_nothing_is_harmless(self):
        self.write_file('a.xyz', 'x')
        self.assertEqual(on_delete_file(self.working_directory, None), ['a.xyz'])

    def test_deleting_a_missing_file_warns_instead_of_raising(self):
        self.assertEqual(on_delete_file(self.working_directory, 'ghost.xyz'), [])


class OpenWorkingDirectoryTests(TempWorkingDirTestCase):
    def setUp(self):
        super().setUp()
        # get_working_directories reads ./data relative to the process working directory
        self.previous_cwd = os.getcwd()
        os.chdir(self.temp_root)
        os.makedirs(os.path.join(self.temp_root, 'data'), exist_ok=True)
        self.addCleanup(os.chdir, self.previous_cwd)

    def test_directories_are_listed_in_ascending_order(self):
        for name in ('wd10', 'wd2', 'Alpha', 'wd1'):
            os.makedirs(os.path.join('data', name), exist_ok=True)
        self.assertEqual(get_working_directories(), ['Alpha', 'wd1', 'wd2', 'wd10'])

    def test_only_directories_are_listed(self):
        os.makedirs(os.path.join('data', 'real'), exist_ok=True)
        with open(os.path.join('data', 'stray.txt'), 'w') as handle:
            handle.write('x')
        self.assertEqual(get_working_directories(), ['real'])

    def test_opening_creates_the_directory_and_returns_its_path(self):
        dropdown, path, files, upload_button = on_open_working_directory('fresh')
        self.assertTrue(os.path.isdir(os.path.join('data', 'fresh')))
        self.assertEqual(path, os.path.join('./data/', 'fresh'))
        self.assertEqual(files, [])
        self.assertTrue(upload_button['interactive'])
        self.assertIn('fresh', dropdown['choices'])

    def test_a_blank_name_is_refused(self):
        dropdown, path, files, upload_button = on_open_working_directory('   ')
        self.assertIsNone(path)
        self.assertIsNone(files)

    def test_reopening_keeps_existing_files(self):
        on_open_working_directory('again')
        with open(os.path.join('data', 'again', 'a.xyz'), 'w') as handle:
            handle.write('x')
        _, _, files, _ = on_open_working_directory('again')
        self.assertEqual(files, ['a.xyz'])


class ExtensionConstantTests(unittest.TestCase):
    def test_the_extension_lists_do_not_overlap(self):
        self.assertFalse(set(STRUCTURE_FILE_EXTENSIONS) & set(TEXT_FILE_EXTENSIONS))

    def test_txt_is_treated_as_text(self):
        self.assertIn('.txt', TEXT_FILE_EXTENSIONS)


if __name__ == '__main__':
    unittest.main()
