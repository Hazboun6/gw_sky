#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for tutorial notebooks."""

import pytest
import subprocess as sp

def test_pta_smbbh():
    sp.call('jupyter nbconvert --to notebook --inplace --execute ../docs/_static/notebooks/pta_smbbh.ipynb',
            shell=True)
