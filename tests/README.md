# Tests

Plain `unittest`, no extra dependency to install. Run from the repository root:

```bash
./gaussian-env/bin/python -m unittest discover -s tests -t .
```

Useful variations:

```bash
# verbose, one line per test
./gaussian-env/bin/python -m unittest discover -s tests -t . -v

# a single module, class or test
./gaussian-env/bin/python -m unittest tests.test_custom_basis
./gaussian-env/bin/python -m unittest tests.test_custom_basis.BuildSectionTests
./gaussian-env/bin/python -m unittest tests.test_custom_basis.BuildSectionTests.test_missing_element_is_reported

# stop at the first failure
./gaussian-env/bin/python -m unittest discover -s tests -t . -f
```

`pytest` also runs these unchanged (`pytest tests`) if you ever install it.

## Gaussian

Most tests are pure logic and need nothing installed. The ones in
`test_gaussian_integration.py` run real `g16` jobs and skip themselves when `g16` is not
on `PATH`, so the suite is green either way.

An interactive shell already has Gaussian set up from `~/.bashrc`. A non-interactive one
(a CI step, `wsl -- bash script.sh`) does not, so set it up first:

```bash
export g16root=$HOME/gaussian
. $g16root/g16/bsd/g16.profile
export GAUSS_SCRDIR=$HOME/gaussian/scratch
```

Every Gaussian job in the suite is deliberately tiny (water or ethanol at HF/STO-3G), so
enabling them adds roughly 30 seconds. Killing a live job is part of the coverage, so
expect Gaussian to print register dumps and "Error: software termination" to the console
during `GaussianStopTests` and the broken-basis test. Those are the tests working.

## Layout

| File | Covers |
| --- | --- |
| `helpers.py` | Shared base class, temp working directories, fake `g16`, sample basis sets and log excerpts |
| `test_custom_basis.py` | Parsing a Basis Set Exchange file into a `gen`/`genecp` section, element filtering |
| `test_input_writers.py` | The six `write_*_gaussian_input` writers |
| `test_structure_io.py` | The non-standard `.xyz` format, bond perception, natural sort |
| `test_nmr_analysis.py` | Shielding and J-coupling parsing, chemical equivalence, multiplet table |
| `test_spectra_and_result.py` | IR/UV-Vis/NMR spectrum builders, Result tab helpers |
| `test_conformer_generation.py` | Minimisation, duplicate removal, conformer files and table |
| `test_calculation_tab.py` | Form wiring, input generation, run/stop lifecycle against a fake `g16` |
| `test_working_directory.py` | File typing, selection, text viewer, working directory management |
| `test_gaussian_integration.py` | Real `g16` runs end to end (skipped without Gaussian) |

`webui.py` is deliberately not imported anywhere: importing it deletes stale `*.log` and
`static/**` files and binds a port, which is not something a test run should do.
