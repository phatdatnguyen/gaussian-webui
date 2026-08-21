"""Test package bootstrap.

Puts the repository root and this directory on sys.path so the test modules can import
the application modules (utils, calculation, ...) and the shared helpers regardless of
where the runner was started from.
"""
import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)

for _path in (_REPO_ROOT, _TESTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# result.py imports matplotlib, keep it off any display
os.environ.setdefault('MPLBACKEND', 'Agg')
