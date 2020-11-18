#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for tutorial notebooks."""

import os
import pytest
import subprocess
import tempfile

current_path = os.path.abspath(gw_sky.__path__[0])
data_dir = os.path.join(current_path,'data/')

def _exec_notebook(path):
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=1000",
                "--output", fout.name, path]
        subprocess.check_call(args)

def test_pta_smbbh():
    _exec_notebook('_static/notebooks/pta_smbbh.ipynb')
