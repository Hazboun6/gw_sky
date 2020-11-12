# -*- coding: utf-8 -*-
"""Main module."""
import numpy as np
import astropy.units as u
import astropy.constants as c

__all__ = ['GW',
           'GWSky',
           'h_circ',
           ]

day_sec = 24*3600
yr_sec = 365.25*24*3600

class GW():
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

    pulsar_term : bool, str, optional [True, False, 'explicit']
        Flag for including the pulsar term in sky map sensitivity. True
        includes an idealized factor of two from Equation (36) of `[1]`_.
        The `'explicit'` flag turns on an explicit calculation of
        pulsar terms using pulsar distances. (This option takes
        considerably more computational resources.)

        .. _[1]: https://arxiv.org/abs/1907.04341

    pol: str, optional ['gr','scalar-trans','scalar-long','vector-long']
        Polarization of gravitational waves to be used in pulsar antenna
        patterns. Only one can be used at a time.
    '''
    def __init__(self, sources, theta, phi, pulsar_term=False, pol='gr'):
        self.theta_gw = [gw.theta for gw in sources]
        self.phi_gw = [gw.phi for gw in sources]
        self.pulsar_term = pulsar_term
        self.theta = theta
        self.phi = phi
        self.sources = sources
        self.pos = - khat(self.theta, self.phi)
        # if pulsar_term == 'explicit':
        #     self.pdists = np.array([(sp.pdist/c.c).to('s').value
        #                             for sp in spectra]) #pulsar distances

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
        # if pulsar_term == 'explicit':
        #     Dp = self.pdists[:,np.newaxis] * denom
        #     Dp = self.freqs[:,np.newaxis,np.newaxis] * Dp[np.newaxis,:,:]
        #     pt = 1-np.exp(-1j*2*np.pi*Dp)
        #     pt /= 2*np.pi*1j*self.freqs[:,np.newaxis,np.newaxis]
        #     self.pt = pt
        #     self.pt_sqr = np.abs(pt)**2

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

        if pulsar_term == 'explicit':
            self.sky_response = (0.5 * self.sky_response[np.newaxis,:,:]
                                 * self.pt_sqr)

    def residuals(self, t):
        """"Pulsar timing time-of-arrival residuals due to GWs"""
        res = np.einsum('ij, jk ->ijk', self.Rplus, self.s_plus(t))
        res += np.einsum('ij, jk ->ijk', self.Rcross, self.s_cross(t))
        return res

    def strain(self, t):
        """"Pulsar timing time-of-arrival residuals due to GWs"""
        strain = np.einsum('ij, jk ->ijk', self.Rplus, self.h_plus(t))
        strain += np.einsum('ij, jk ->ijk', self.Rcross, self.h_cross(t))
        return strain

    def h_cross(self, t):
        """Strain power sensitivity. """
        return np.array([gw.h_cross(t) for gw in self.sources])

    def h_plus(self, t):
        """Strain power sensitivity. """
        return np.array([gw.h_plus(t) for gw in self.sources])

    def s_cross(self, t):
        """Strain power sensitivity. """
        return np.array([gw.s_cross(t) for gw in self.sources])

    def s_plus(self, t):
        """Strain power sensitivity. """
        return np.array([gw.s_plus(t) for gw in self.sources])

def h0_circ(M_c, D_L, f0):
    """Amplitude of a circular super-massive binary black hole."""
    return (4*c.c / (D_L * u.Mpc)
            * np.power(c.G * M_c * u.Msun/c.c**3, 5/3)
            * np.power(np.pi * f0 * u.Hz, 2/3))

def khat(theta, phi):
    r'''Returns :math:`\hat{k}` from paper.
    Also equal to :math:`-\hat{r}=-\hat{n}`.'''
    return np.array([-np.sin(theta)*np.cos(phi),
                     -np.sin(theta)*np.sin(phi),
                     -np.cos(theta)])

def lhat(theta, phi):
    r'''Returns :math:`\hat{l}` from paper. Also equal to :math:`-\hat{\phi}`.'''
    return np.array([np.sin(phi), -np.cos(phi), np.zeros_like(theta)])

def mhat(theta, phi):
    r'''Returns :math:`\hat{m}` from paper. Also equal to :math:`-\hat{\theta}`.'''
    return np.array([-np.cos(theta)*np.cos(phi),
                     -np.cos(theta)*np.sin(phi),
                     np.sin(theta)])
