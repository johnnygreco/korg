from __future__ import division, print_function

import numpy as np
from scipy.special import gammaincinv, gamma
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants
from photutils import CircularAperture, aperture_photometry
import galsim
from .utils import *
from .sample import rejection_sersic_xy 

__all__ = ['InclinedExponentialModel', 
           'SersicModel', 'subaru_area']

subaru_area = np.pi * (8.2 * 100 * u.cm /2)**2

class ModelBase(object):

    _pixscale = 0.168 * u.arcsec/u.pixel
    
    def __init__(self, pars):
        
        self.pars = pars
        self._is_psf_conv = False
        self.psf_fwhm = 0 
        self.pixscale = 0 
        self._original_gal = None
        self._setup_gal(pars)
        self._reset_gal()

    def _setup_gal(self, pars):
        raise NotImplementedError()

    def _reset_gal(self):
        if self._original_gal is None:
            self._original_gal = self.gal
        else:
            self.gal = self._original_gal

    def observe(self, psf_fwhm=None, exp_time=60*20*u.s, sky_sb=20,
                area=subaru_area, pixscale=0.168*u.arcsec/u.pixel, 
                img_dims=[None, None]):
        
        if psf_fwhm is not None:
            if psf_fwhm != self.psf_fwhm or pixscale != self.pixscale:
                self.psf_fwhm = psf_fwhm
                self.pixscale = pixscale
                self.gauss_psf = galsim.Gaussian(
                    flux=1.0, fwhm=psf_fwhm.to('arcsec').value)
                self._reset_gal()
                self.gal = galsim.Convolve([self.gal, self.gauss_psf])
                nx = 2*int(8 * self.gauss_psf.sigma/pixscale.value) - 1
                self.psf_array = self.gauss_psf.drawImage(
                    nx=nx, ny=nx, scale=pixscale.value).array
                self.psf_array /= self.psf_array.sum()

        self.image = self.gal.drawImage(area=area.to('cm2').value,
                                        exptime=exp_time.to('s').value, 
                                        scale=pixscale.value, 
                                        nx=img_dims[0], ny=img_dims[1])
        
        self.effective_exp_time = exp_time *  area.to('cm2').value
        self.sky_level_flux = sb_to_photon_flux_per_pixel(
            sky_sb, pixscale=pixscale)
        self.sky_level_count = self.sky_level_flux.value *\
            self.effective_exp_time.to('s').value
        self.image.addNoise(
            galsim.PoissonNoise(sky_level=self.sky_level_count))
        
        self.image = self.image / self.effective_exp_time.value

        max_sb = self.gal.max_sb * self.pars.phot_flux_to_fnu.value 
        self.mu_0_model = -2.5 * np.log10(max_sb) - mAB0

    def mu_in_aperture(self, r_pix, pixscale=0.168*u.arcsec/u.pixel):
        y, x = (np.array(self.array.shape)-1)/2
        ap = CircularAperture((x, y), r=r_pix)
        f = aperture_photometry(self.array, ap)['aperture_sum'][0]
        r_arcsec = r_pix * pixscale.value
        mu = -2.5 * np.log10(f*self.pars.phot_flux_to_fnu.value)
        mu += 2.5 * np.log10(np.pi*r_arcsec**2) - mAB0
        return mu

    @property
    def mu_0_psf(self):
        if self.psf_fwhm == 0:
            raise Exception('Must observe source first!')
        else:
            r_pix = self.psf_fwhm.to('arcsec') / self.pixscale / 2
            mu = self.mu_in_aperture(r_pix=r_pix.value, pixscale=self.pixscale)
            return mu
        
    def draw(self, ax=None, cmap='magma', **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.imshow(self.array, cmap=cmap, **kwargs)
        ax.set(xticks=[], yticks=[])
        return ax
        
    @property
    def array(self):
        try:
            array = self.image.array
        except AttributeError:
            array = self.gal.drawImage(scale=self._pixscale.value).array
        return array


class HIIRegions(galsim.GSObject):
    
    def __init__(self, n_regions, flux, r_e, PA=0.0, ell=0.0, 
                 points_per_region=100, gaussian_sigma=1e-6, 
                 r_min=0.0*u.arcsec, cloud_spread=1, sersic_n=1.0):

        self._flux = flux
        self.r_e = r_e.to('arcsec')
        self.n_regions = int(n_regions)
        self.r_min = r_min.to('arcsec')
        self.points_per_region = points_per_region
        self.cloud_spread = cloud_spread
        self.PA = PA - 90
        self.ell = ell
        self.n = sersic_n

        self._gaussian_sigma = gaussian_sigma

        self._points = self._get_points()
        self._gaussians = self._get_gaussians(self._points)
        self._sbp = galsim._galsim.SBAdd(self._gaussians)
        
    def _get_points(self):
        x, y = rejection_sersic_xy(
	    self.n_regions, r_e=self.r_e.value, PA=self.PA,
	    ell=self.ell, sersic_n=self.n, 
	    r_min=self.r_min.value)
        return np.vstack([x, y]).T
    
    def _sample_ball(self, p, std, size=1):
        return np.vstack(
            [p + std * np.random.normal(size=len(p)) for i in range(size)])
        
    def _get_gaussians(self, points):
        
        gaussians = []
        
        total_points = self.n_regions * self.points_per_region
        flux_per_point = self._flux/total_points
        
        for p in points:
            
            ball = self._sample_ball(
                p, self.cloud_spread, size=self.points_per_region)
            
            for bp in ball:
                g = galsim._galsim.SBGaussian(
		    sigma=self._gaussian_sigma,
		    flux=flux_per_point)
                pos = galsim.PositionD(bp[0], bp[1])
                g = galsim._galsim.SBTransform(
                    g, 1.0, 0.0, 0.0, 1.0, pos, 1.0)
                gaussians.append(g)
        
        return gaussians


class InclinedExponentialModel(ModelBase):
    
    def _setup_gal(self, pars):
        self.gal = galsim.InclinedExponential(**pars.galsim_kw)
        self.gal = self.gal.rotate(pars.PA.to('degree').value * galsim.degrees)


class SersicModel(ModelBase):
    
    def _setup_gal(self, pars):

        self.gal = galsim.Sersic(**pars.galsim_kw)
        self.gal = self.gal.shear(q=pars.b_a, beta=(0.0 * galsim.degrees))
        self.gal = self.gal.rotate(pars.PA.to('degree').value * galsim.degrees)

    def add_point_sources(self, n_points, frac_gal_flux=0.05, frac_r_e=1.0, 
                         total_flux=None, point_fwhm=None):

        r_e = self.pars.r_e.to('arcsec').value * frac_r_e
        flux = total_flux if total_flux else self.gal.flux * frac_gal_flux
        points = galsim.RandomWalk(n_points, half_light_radius=r_e, flux=flux)
        points = points.shear(q=self.pars.b_a, beta=(0.0 * galsim.degrees))
        points = points.rotate(
            self.pars.PA.to('degree').value * galsim.degrees)
        if total_flux is None:
            self.gal = self.gal.withFlux(self.gal.flux * (1-frac_gal_flux))
        if point_fwhm is not None:
            point_psf = galsim.Gaussian(
                flux=1.0, fwhm=point_fwhm.to('arcsec').value)
            points = galsim.Convolve([points, point_psf])
        self.gal = galsim.Add([self.gal, points])
        self._original_gal = self.gal

    def add_lumps(self, n_regions, frac_gal_flux=0.05, frac_r_e=1.0, 
                  **kwargs):

        flux = self.gal.flux * frac_gal_flux
        r_e = self.pars.r_e.to('arcsec') * frac_r_e
        PA = self.pars.PA.value
        ell = self.pars.ell
        n = self.pars.n

        lumps = HIIRegions(
            n_regions=n_regions, flux=flux, PA=PA, r_e=r_e, ell=ell, 
            sersic_n=n, **kwargs)

        self.gal = self.gal.withFlux(self.gal.flux * (1-frac_gal_flux))
        self.gal = galsim.Add([self.gal, lumps])
        self._original_gal = self.gal
