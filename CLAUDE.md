# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Gradio web UI that drives **Gaussian 16** for computational chemistry: build a structure, generate a `.gjf` input file, run `g16`, then parse and visualize the `.log`. Supported calculations: single-point, geometry optimization, frequency/IR, absorption/emission (TD-DFT), NMR.

## Environment

The repo lives inside WSL (Ubuntu distro) and is Linux-only — it shells out to `g16`, `formchk`, and `cubegen`, which must be on `PATH`.

```bash
source gaussian-env/bin/activate
python3 webui.py
```

From a Windows host, `-d Ubuntu` is required — the default WSL distro is `docker-desktop`, and `--cd` with a `\\wsl.localhost\...` cwd fails to translate:

```powershell
wsl.exe -d Ubuntu --cd /home/datnguyen/github/gaussian-webui -- ./gaussian-env/bin/python webui.py
```

`git` run from the Windows side over the UNC path fails with "dubious ownership" — run git inside WSL instead.

There is **no test suite, linter, or `requirements.txt`**. Dependencies are pip-installed by hand per `Readme.md`; the versions that matter are pinned there (`gradio==5.50.0`, `nglview==4.0`, and `gradio_molecule2d --no-deps`). To check a change, run an ad-hoc script through the venv interpreter (`./gaussian-env/bin/python`) — handler functions are plain functions and can be called directly with a stub `progress` object that implements `__call__` and `tqdm`.

`webui.py` picks the first free port from 7860 upward and deletes stale `*.log`, `static/**/*.cube`, `static/**/*.xyz`, and `static/**/*.html` on startup.

## Architecture

`webui.py` builds one `gr.Blocks`, mounts it on a FastAPI app, and serves `./static` at `/static` (needed by the 3D viewers). Layout: a left column of working-directory controls, and a right column holding one shared status bar above three tabs.

Each tab module exposes `<name>_tab_content(working_directory_path_state, working_directory_file_list_state, status_markdown)` which builds its own UI *and* wires its own `.click`/`.change` handlers, then returns the tab. `working_directory.py` mirrors this with `working_directory_blocks()`, which returns the two states everything else consumes.

### The file-list state is the app's event bus

`working_directory_file_list_state` (a plain list of filenames) is how tabs communicate. Any handler that writes a file returns `get_files_in_working_directory(path)` into it. Its `.change` event then fans out to: the file table in the left column, the structure/input dropdowns in the Calculation tab, and the result-file dropdown in the Result tab. **A new handler that creates files must return this state or the rest of the UI won't see the files.**

Working directories are `./data/<name>/` (gitignored, as are all chemistry file types).

### Status reporting convention

Handlers return an HTML string into `status_markdown` — `<span style='color:green;'>` on success, `red` on failure — and catch exceptions rather than raising. `gr.Warning(...)` is used for transient popups (validation, file operations). Note that `conformer_generation.py` shadows the `status_markdown` parameter with a local `gr.Markdown()`, so its status renders inside the tab rather than in the shared top bar.

### Pipeline

`conformer_generation.py` (SMILES → RDKit conformers → structure files; candidates are MMFF/UFF minimized, then treated as duplicates only when they are *both* within the energy threshold *and* below the heavy-atom RMSD threshold — energy alone over-prunes badly, and skipping the minimization makes the energies meaningless) → `calculation.py` (structure file → `.gjf` via the `write_*_gaussian_input` writers → `subprocess.run(["g16", in, out])`) → `result.py` (`cclib.io.ccopen` on the `.log`; which result accordions become visible is driven by which attributes cclib found, e.g. `scfenergies` + `moenergies` → Energy, `len(scfenergies) > 1` → optimization plot).

`utils.py` (~1000 lines) is the shared library behind all three: structure I/O, the six Gaussian input writers, plotly spectrum builders, and the NMR parsing/chemical-equivalence code. `calculation.py` does `from utils import *`.

### Two things that will bite you

**XYZ files here are not standard XYZ.** `conformer_to_xyz_file` writes `"<charge> <multiplicity>"` as line 1, then atom lines — no atom count, no comment line. `mol_from_xyz_file` expects exactly that. Files from outside tools will not parse.

**Structures loaded from `.xyz`/`.log` have no bonds.** Both paths go through `add_bonds(...)`, which infers connectivity from covalent radii × a distance factor, followed by `Chem.SanitizeMol`. Perception failures on unusual geometries surface here.

### 3D visualization

`nglview` widgets are written to standalone HTML under `./static/` and returned to Gradio as an `<iframe src="/static/...?ts={timestamp}">`. The timestamp is cache-busting — keep it. Orbital/density surfaces additionally run `formchk` then `cubegen` on the `.chk` file.
