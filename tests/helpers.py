"""Shared fixtures and utilities for the test suite.

Plain unittest, no third party test runner needed:

    ./gaussian-env/bin/python -m unittest discover -s tests -t .

Tests that need Gaussian are skipped automatically when g16 is not on PATH.
"""
import os
import shutil
import stat
import sys
import tempfile
import unittest

# matplotlib is imported by result.py, keep it headless
os.environ.setdefault('MPLBACKEND', 'Agg')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rdkit import Chem
from rdkit.Chem import AllChem

from utils import conformer_to_xyz_file


def gaussian_available():
    return shutil.which('g16') is not None


requires_gaussian = unittest.skipUnless(
    gaussian_available(),
    "g16 is not on PATH (source $g16root/g16/bsd/g16.profile to enable these tests)",
)


class FakeProgress:
    """Stand-in for gr.Progress, which handlers call but which needs a live request."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def tqdm(self, iterable, **kwargs):
        return iterable


class SelectEvent:
    """Stand-in for gr.SelectData as on_select_file consumes it."""

    def __init__(self, file_name):
        self.row_value = [file_name]


def make_molecule(smiles, seed=0xf00d, optimize=False):
    """Build an RDKit molecule with explicit hydrogens and one embedded conformer."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    if optimize:
        AllChem.MMFFOptimizeMolecule(mol)
    return mol


def status_text(html):
    """Strip the colour span the handlers wrap their status messages in."""
    if not isinstance(html, str):
        return html
    for prefix in ("<span style='color:green;'>", "<span style='color:red;'>",
                   "<span style='color:orange;'>", "<span style='color:#1e90ff;'>"):
        if html.startswith(prefix):
            return html[len(prefix):-len("</span>")]
    return html


def status_colour(html):
    if not isinstance(html, str) or not html.startswith("<span style='color:"):
        return None
    return html.split("color:", 1)[1].split(";", 1)[0]


class TempWorkingDirTestCase(unittest.TestCase):
    """Gives each test an empty working directory under a private temp root."""

    def setUp(self):
        self.temp_root = tempfile.mkdtemp(prefix='gwui-test-')
        self.working_directory = os.path.join(self.temp_root, 'wd')
        os.makedirs(self.working_directory)
        self.addCleanup(shutil.rmtree, self.temp_root, True)

    def path(self, *parts):
        return os.path.join(self.working_directory, *parts)

    def write_structure(self, smiles, file_name, charge=0, multiplicity=1, seed=0xf00d):
        mol = make_molecule(smiles, seed=seed)
        conformer_to_xyz_file(mol, 0, self.path(file_name), charge, multiplicity)
        return mol

    def write_file(self, file_name, content=''):
        with open(self.path(file_name), 'w') as handle:
            handle.write(content)
        return self.path(file_name)

    def read_file(self, file_name):
        with open(self.path(file_name)) as handle:
            return handle.read()

    def install_fake_g16(self, script_body):
        """Put an executable fake g16 first on PATH, restored when the test ends."""
        bin_dir = os.path.join(self.temp_root, 'bin')
        os.makedirs(bin_dir, exist_ok=True)
        fake = os.path.join(bin_dir, 'g16')
        with open(fake, 'w') as handle:
            handle.write(script_body)
        os.chmod(fake, os.stat(fake).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        previous_path = os.environ['PATH']
        os.environ['PATH'] = bin_dir + os.pathsep + previous_path
        self.addCleanup(os.environ.__setitem__, 'PATH', previous_path)
        return fake


# --- sample basis sets -------------------------------------------------------
# Shaped exactly like a Basis Set Exchange download in Gaussian format: a "!" comment
# header, then blank lines, then the element blocks closed by ****.
BSE_STYLE_BASIS = """!----------------------------------------------------------------------
! Basis Set Exchange
! Version 0.12
!   Basis set: test-set
!----------------------------------------------------------------------


H     0
S    2   1.00
      0.187311D+02           0.334946D-01
      0.282539D+01           0.234727D+00
S    1   1.00
      0.161278D+00           1.0000000
****
C     0
S    2   1.00
      0.304752D+04           0.183474D-02
      0.457370D+03           0.140373D-01
P    1   1.00
      0.159599D+01           1.0000000
****
O     0
S    1   1.00
      0.548467D+04           0.183107D-02
****
N     0
S    1   1.00
      0.417351D+04           0.183477D-02
****
"""

# A basis set carrying a pseudopotential block, the shape that requires genecp.
BASIS_WITH_ECP = """! test set with an ECP
H     0
S    1   1.00
      0.187311D+02           0.334946D-01
****
C     0
S    1   1.00
      0.304752D+04           0.183474D-02
****
Cu     0
S    1   1.00
      0.514000D+01          -0.704000D+00
****

Cu     0
CU-ECP     3     10
f potential
  1
2      1.0000000        0.0000000
s-f potential
  3
0     30.2200000       -3.0000000
"""

# One block covering several elements at once, to check the header gets rewritten.
SHARED_BLOCK_BASIS = """C  H  O     0
S    1   1.00
      0.100000D+01           1.0000000
****
"""


# --- sample Gaussian output --------------------------------------------------
# Formats below were copied from a real g16 HF/STO-3G NMR=(GIAO,spinspin) run.
NMR_SHIELDING_LOG = """ Some earlier output that must be ignored.
 SCF GIAO Magnetic shielding tensor (ppm):
      1  C    Isotropic =   225.1454   Anisotropy =    18.2528
   XX=   233.7343   YX=    -4.9600   ZX=    -1.1922
   XY=    -9.3001   YY=   222.7863   ZY=    -0.6965
   XZ=    -1.9781   YZ=    -0.8186   ZZ=   218.9155
   Eigenvalues:   217.6673   220.4549   237.3139
      2  C    Isotropic =   195.2693   Anisotropy =    46.0402
   XX=   211.0768   YX=    20.6295   ZX=     2.6602
   Eigenvalues:   175.1662   184.6791   225.9628
      3  O    Isotropic =   360.8253   Anisotropy =    89.2816
   XX=   336.6154   YX=   -29.3269   ZX=     9.2434
   Eigenvalues:   300.0000   310.0000   320.0000
      4  H    Isotropic =    31.5000   Anisotropy =     8.0000
   XX=    30.0000   YX=     0.0000   ZX=     0.0000
   Eigenvalues:    28.0000    31.0000    35.0000

 End of shielding section.
"""

# Two column blocks, which is how Gaussian wraps the lower triangle past five nuclei.
NMR_JCOUPLING_LOG = """ Total nuclear spin-spin coupling K (Hz):
                1             2
      1  0.000000D+00
      2  0.999999D+02  0.000000D+00
 Total nuclear spin-spin coupling J (Hz):
                1             2             3             4             5
      1  0.000000D+00
      2  0.418515D+02  0.000000D+00
      3 -0.420058D-01  0.871706D+01  0.000000D+00
      4  0.133245D+03 -0.108619D+02 -0.244496D+00  0.000000D+00
      5  0.134370D+03 -0.116225D+02  0.167309D+00 -0.296581D+02  0.000000D+00
      6  0.132612D+03 -0.113061D+02 -0.210182D+00 -0.295024D+02 -0.298381D+02
                6             7
      6  0.000000D+00
      7  0.398266D+01  0.000000D+00
 End of Minotr F.D. properties file.
"""

GEOMETRY_LOG = """                         Standard orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          8           0        0.000000    0.000000   -0.110000
      2          1           0        0.000000    0.780000    0.440000
      3          1           0        0.000000   -0.780000    0.440000
 ---------------------------------------------------------------------
                         Standard orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          8           0        0.000000    0.000000   -0.120000
      2          1           0        0.000000    0.760000    0.450000
      3          1           0        0.000000   -0.760000    0.450000
 ---------------------------------------------------------------------
"""
