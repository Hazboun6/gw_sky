#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `gw_sky` package."""

import pytest
import numpy as np

from gw_sky import gwsky

def test_smbbh():
    """Sample pytest test function with the pytest fixture as an argument."""
    L = 200
    costh = np.random.uniform(-1,1,size= L)
    th = np.arccos(costh)
    ph = np.random.uniform(0,2*np.pi,size= L)
    gw = []
    for ii in range(L):
        log10freq = np.random.normal(-8.5,1)
        freq = 10**log10freq
        log10_h = np.random.uniform(-18,-16)
        h = 10**log10_h
        gw.append(gwsky.SMBBH(freq,h,th[ii],ph[ii],))

    theta = np.linspace(0,np.pi,100)
    phi = np.linspace(0,2*np.pi,100)
    sky = gwsky.GWSky(gw,theta,phi)
    t = np.linspace(0,12.9*365.25*24*3600,300)
    str = sky.strain(t)
    res = sky.residuals(t)
