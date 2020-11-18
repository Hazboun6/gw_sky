# -*- coding: utf-8 -*-
"""Main module."""
import os
import numpy as np
import astropy.units as u
import astropy.constants as c
import gw_sky

__all__ = ['GW',
           'GWSky',
           'h_circ',
           ]

current_path = os.path.abspath(gw_sky.__path__[0])
data_dir = os.path.join(current_path,'data/')

day_sec = 24*3600
yr_sec = 365.25*24*3600

class GWBase():
    def __init__(self, src_class='none'):
        self.src_class = src_class

    def h_plus(self, t):
        err_msg = 'Method must be implemented by GW subclass.'
        raise NotImplementedError(err_msg)

    def h_cross(self, t):
        err_msg = 'Method must be implemented by GW subclass.'
        raise NotImplementedError(err_msg)

    def s_plus(self, t):
        err_msg = 'Method must be implemented by GW subclass.'
        raise NotImplementedError(err_msg)

    def s_cross(self, t):
        err_msg = 'Method must be implemented by GW subclass.'
        raise NotImplementedError(err_msg)


class SMBBH(GWBase):
    def __init__(self, fgw, h0, theta, phi,
                 iota=None, psi=None, phase0=None, src_class='smbbh'):
        self.fgw= fgw
        self.h0 = h0
        self.theta = theta
        self.phi = phi
        self.iota = iota if iota is not None else np.random.uniform(0,2*np.pi)
        self.psi = psi if psi is not None else np.random.uniform(0,2*np.pi)
        self.phase0 = phase0 if phase0 is not None else np.random.uniform(0,2*np.pi)
        self.src_class = src_class

    def h_plus(self, t):
        if self.src_class=='smbbh':
            return self._h_plus_smbbh(t)

    def h_cross(self, t):
        if self.src_class=='smbbh':
            return self._h_cross_smbbh(t)

    def s_plus(self, t):
        if self.src_class=='smbbh':
            return -self._h_plus_smbbh(t)/(4*np.pi*self.fgw)

    def s_cross(self, t):
        if self.src_class=='smbbh':
            return -self._h_cross_smbbh(t)/(4*np.pi*self.fgw)

    def _h_plus_smbbh(self,t):
        h1 = 0.5 * (1 + np.cos(self.iota)**2) * np.cos(2*self.psi)
        h1 *= np.sin(2*np.pi*self.fgw*t+self.phase0)
        h2 = np.cos(self.iota) * np.sin(2*self.psi)
        h2 *= np.cos(2*np.pi*self.fgw*t+self.phase0)
        h = self.h0 * (h1 + h2)
        return h

    def _h_cross_smbbh(self,t):
        h1 = 0.5 * (1 + np.cos(self.iota)**2) * np.sin(2*self.psi)
        h1 *= np.sin(2*np.pi*self.fgw*t+self.phase0)
        h2 = np.cos(self.iota) * np.cos(2*self.psi)
        h2 *= np.cos(2*np.pi*self.fgw*t+self.phase0)
        h = self.h0 * (h1 - h2)
        return h


class GWSky():
    r'''
    Class to make sky maps for deterministic PTA gravitational wave signals.
    Calculated in terms of :math:`\hat{n}=-\hat{k}`.

    Parameters
    ----------
    sources : list
        List of `skymap.GW` objects that contain the various gravitational wave
        sources for analysis.

    theta : list, array
        Pulsar sky location colatitude at which to calculate sky map.

    phi : list, array
        Pulsar sky location longitude at which to calculate sky map.

    pol: str, optional ['gr','scalar-trans','scalar-long','vector-long']
        Polarization of gravitational waves to be used in pulsar antenna
        patterns. Only one can be used at a time.
    '''
    def __init__(self, sources, theta, phi, pol='gr'):
        self.theta_gw = [gw.theta for gw in sources]
        self.phi_gw = [gw.phi for gw in sources]
        self.theta = theta
        self.phi = phi
        self.sources = sources
        self.pos = - khat(self.theta, self.phi)

        #Return 3xN array of k,l,m GW position vectors.
        self.K = khat(self.theta_gw, self.phi_gw)
        self.L = lhat(self.theta_gw, self.phi_gw)
        self.M = mhat(self.theta_gw, self.phi_gw)
        LL = np.einsum('ij, kj->ikj', self.L, self.L)
        MM = np.einsum('ij, kj->ikj', self.M, self.M)
        KK = np.einsum('ij, kj->ikj', self.K, self.K)
        LM = np.einsum('ij, kj->ikj', self.L, self.M)
        ML = np.einsum('ij, kj->ikj', self.M, self.L)
        KM = np.einsum('ij, kj->ikj', self.K, self.M)
        MK = np.einsum('ij, kj->ikj', self.M, self.K)
        KL = np.einsum('ij, kj->ikj', self.K, self.L)
        LK = np.einsum('ij, kj->ikj', self.L, self.K)
        self.eplus = MM - LL
        self.ecross = LM + ML
        self.e_b = LL + MM
        self.e_ell = KK # np.sqrt(2)*
        self.e_x = KL + LK
        self.e_y = KM + MK
        num = 0.5 * np.einsum('ij, kj->ikj', self.pos, self.pos)
        denom = 1 + np.einsum('ij, il->jl', self.pos, self.K)
        self.D = num[:,:,:,np.newaxis]/denom[np.newaxis, np.newaxis,:,:]

        if pol=='gr':
            self.Rplus = np.einsum('ijkl, ijl ->kl',self.D, self.eplus)
            self.Rcross = np.einsum('ijkl, ijl ->kl',self.D, self.ecross)
        elif pol=='scalar-trans':
            self.Rbreathe = np.einsum('ijkl, ijl ->kl',self.D, self.e_b)
        elif pol=='scalar-long':
            self.Rlong = np.einsum('ijkl, ijl ->kl',self.D, self.e_ell)
        elif pol=='vector-long':
            self.Rx = np.einsum('ijkl, ijl ->kl',self.D, self.e_x)
            self.Ry = np.einsum('ijkl, ijl ->kl',self.D, self.e_y)

    def residuals(self, t):
        """"Pulsar timing time-of-arrival residuals due to GWs"""
        res = np.einsum('ij, jk ->ijk', self.Rplus, self.s_plus(t))
        res += np.einsum('ij, jk ->ijk', self.Rcross, self.s_cross(t))
        return res

    def strain(self, t):
        """"Total strain for given GW sources."""
        strain = np.einsum('ij, jk ->ijk', self.Rplus, self.h_plus(t))
        strain += np.einsum('ij, jk ->ijk', self.Rcross, self.h_cross(t))
        return strain

    def h_cross(self, t):
        """Returns cross polarization strain for all GW sources. """
        return np.array([gw.h_cross(t) for gw in self.sources])

    def h_plus(self, t):
        """Returns plus polarization strain for all GW sources. """
        return np.array([gw.h_plus(t) for gw in self.sources])

    def s_cross(self, t):
        """Returns cross polarization GW-induced residuals for all GW sources. """
        return np.array([gw.s_cross(t) for gw in self.sources])

    def s_plus(self, t):
        """Returns plus polarization GW-induced residuals for all GW sources. """
        return np.array([gw.s_plus(t) for gw in self.sources])

def h0_circ(M_c, D_L, f0):
    """Amplitude of a circular super-massive binary black hole."""
    return (4*c.c / (D_L * u.Mpc)
            * np.power(c.G * M_c * u.Msun/c.c**3, 5/3)
            * np.power(np.pi * f0 * u.Hz, 2/3))

def khat(theta, phi):
    r'''
    Returns :math:`\hat{k}` from Hazboun, et al., 2019 `[1]`_.
    Also equal to :math:`-\hat{r}=-\hat{n}`.

    .. _[1]: https://arxiv.org/abs/1907.04341
    '''
    return np.array([-np.sin(theta)*np.cos(phi),
                     -np.sin(theta)*np.sin(phi),
                     -np.cos(theta)])

def lhat(theta, phi):
    r'''
    Returns :math:`\hat{l}` from Hazboun, et al., 2019 `[1]`_.
    Also equal to :math:`-\hat{\phi}`.

    .. _[1]: https://arxiv.org/abs/1907.04341
    '''
    return np.array([np.sin(phi), -np.cos(phi), np.zeros_like(theta)])

def mhat(theta, phi):
    r'''
    Returns :math:`\hat{m}` from Hazboun, et al., 2019 `[1]`_.
    Also equal to :math:`-\hat{\theta}`.

    .. _[1]: https://arxiv.org/abs/1907.04341
    '''
    return np.array([-np.cos(theta)*np.cos(phi),
                     -np.cos(theta)*np.sin(phi),
                     np.sin(theta)])

def smbbh_pop():
    '''
    Returns a `numpy.recarray` with a representative example of a supermassive
    binary black hole population.
    '''
    return np.load(data_dir+'smbbh_pop_gsmf.npz')
