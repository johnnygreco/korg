from __future__ import division, print_function

import numpy as np
from scipy.special import gammaincinv, gamma
import matplotlib.pyplot as plt
from astropy import units as u
import galsim
from .utils import *

__all__ = ['InclinedExponentialParameters', 'SersicParameters']


class ParametersBase(object):

    _unit_equiv = [(u.pixel, u.arcsec, lambda x: 0.168*x, lambda x: x/0.168)]

    @property
    def galsim_kw(self):
        raise NotImplemented()


class InclinedExponentialParameters(ParametersBase):
    
    _units = dict(
        incl = u.degree,
        r_e = u.arcsec,
        scale_height = u.arcsec,
        scale_h_over_r = 1.0*u.dimensionless_unscaled,
        flux = u.count/u.s/u.cm**2,
        PA = u.degree
    )
    
    def __init__(self, **kwargs):

        if 'm_tot' not in kwargs.keys():
            raise Exception('must give total magnitude')
        
        self.incl = kwargs.pop('incl', 0.0) 
        self.r_e = kwargs.pop('r_e', 5.0) 
        self.scale_height = kwargs.pop('scale_height', None)
        self.scale_radius = kwargs.pop('scale_radius', None)
        self.scale_h_over_r = kwargs.pop('scale_h_over_r', 0.1)
        self.PA = kwargs.pop('PA', 0.0)

        for k, v in self._units.items():
            if k == 'flux':
                continue
            value = getattr(self, k)
            if (value is not None) and (type(value) !=  u.Quantity):
                setattr(self, k, value * v)
                    
        self.m_tot = kwargs['m_tot']
        area = np.pi * self.r_e.to('arcsec').value**2
        self.mu_e_ave = self.m_tot + 2.5 * np.log10(2 * area)
        self.mu_e = self.mu_e_ave + 0.699
        self.mu_0 = self.mu_e - 1.822
        self.fnu, self.flam, self.flux, self.E_lam = mtot_to_flux(
            self.m_tot)
        self.phot_flux_to_fnu = (self.fnu/self.flux).\
            decompose().to('erg/Hz')
        
    @property
    def galsim_kw(self):
        kwargs = {'inclination': self.incl.to('deg').value * galsim.degrees,
                  'flux': self.flux.value}
        if self.r_e is not None:
            kwargs['half_light_radius'] = self.r_e.to('arcsec').value 
        else:
            assert self.scale_radius is not None
            kwargs['scale_radius'] = self.scale_radius.to('arcsec').value 
        if self.scale_height is not None:
            kwargs['scale_height'] = self.scale_height.to('arcsec').value 
        else:
            assert self.scale_h_over_r is not None
            kwargs['scale_h_over_r'] = self.scale_h_over_r.value
        return kwargs


class SersicParameters(ParametersBase):
    
    _units = dict(
        r_e = u.arcsec,
        scale_radius = u.arcsec,
        flux = u.count/u.s/u.cm**2,
        trunc = u.arcsec,
        ell = 1.0,
        n  = 1.0,
        PA = u.degree
    )
    
    def __init__(self, **kwargs):

        if 'm_tot' not in kwargs.keys():
            raise Exception('must give total magnitude')
        
        self.r_e = kwargs.pop('r_e', 5.0) 
        self.scale_radius = kwargs.pop('scale_radius', None)
        self.PA = kwargs.pop('PA', 0.0)

        self.n = kwargs.pop('n', 1.0)
        self.trunc = kwargs.pop('trunc', 0)

        if 'b_a' in kwargs.keys():
            self.b_a = kwargs['b_a']
            self.ell = 1 - self.b_a
        elif 'ell' in kwargs.keys():
            self.ell =kwargs['ell']
            self.b_a = 1 - self.ell
        else:
            self.b_a = 1.0
            self.ell = 0.0


        for k, v in self._units.items():
            if k == 'flux':
                continue
            value = getattr(self, k)
            if (value is not None) and (type(value) !=  u.Quantity):
                setattr(self, k, value * v)
                    
        self.m_tot = kwargs['m_tot'] 
        area = np.pi * self.r_e.to('arcsec').value**2
        result = sersic_surface_brightness(self.m_tot, self.r_e, self.n)
        self.mu_0, self.mu_e, self.mu_e_ave = result
        self.fnu, self.flam, self.flux, self.E_lam = mtot_to_flux(
            self.m_tot)
        self.phot_flux_to_fnu = (self.fnu/self.flux).\
            decompose().to('erg/Hz')
        
    @property
    def galsim_kw(self):
        kwargs = {'flux': self.flux.value}
        if self.r_e is not None:
            kwargs['half_light_radius'] = self.r_e.to('arcsec').value 
        else:
            assert self.scale_radius is not None
            kwargs['scale_radius'] = self.scale_radius.to('arcsec').value 
        kwargs['n'] = self.n 
        kwargs['trunc'] = self.trunc.value
        return kwargs
