import numpy as np
from astropy import units as u
from astropy import constants
from scipy.special import gammaincinv, gamma
from toolbox import phot

__all__ = [
    'sb_to_photon_flux_per_pixel',
    'mtot_to_flux',
    'mu_0_to_mu_e_ave',
    'mAB0', 
    'sersic_surface_brightness',
    'mu0_to_mtot',
    'mtot_to_mu0'
]


mAB0 = 48.6


def sb_to_photon_flux_per_pixel(mu, pixscale=0.168*u.arcsec/u.pixel):
    # SDSS i-band (ref: martini's page)
    delta_lam =  1064.0 * u.angstrom
    lam_eff = 7670 * u.angstrom
    E_lam = (constants.h * constants.c / lam_eff).decompose().to('erg')
    fnu_per_square_arcsec = phot.fnu_from_AB_mag(mu) / u.arcsec**2
    flam_per_square_arcsec = fnu_per_square_arcsec *\
        constants.c.to('angstrom/s') / lam_eff**2
    flam_per_pixel = flam_per_square_arcsec * pixscale**2
    photon_flux_per_pixel = (flam_per_pixel * delta_lam / E_lam).\
        decompose().to('1/(cm2*pix2*s)')
    return photon_flux_per_pixel


def mtot_to_flux(mtot, pixscale=0.168*u.arcsec/u.pixel):
    # SDSS i-band (ref: martini's page)
    delta_lam =  1064.0 * u.angstrom
    lam_eff = 7670 * u.angstrom
    E_lam = (constants.h * constants.c / lam_eff).decompose().to('erg')
    fnu = phot.fnu_from_AB_mag(mtot)
    flam = fnu * constants.c.to('angstrom/s') / lam_eff**2
    flam = flam.decompose().to('erg/(angstrom * cm2 * s)')
    photon_flux = (flam * delta_lam / E_lam).decompose().to('1/(cm2*s)')
    return fnu, flam, photon_flux, E_lam


def b_n(n):
    return gammaincinv(2.*n, 0.5)


def f_n(n):
    _bn = b_n(n)
    return gamma(2*n)*n*np.exp(_bn)/_bn**(2*n)


def mu_0_to_mu_e_ave(mu_0, n):
    mu_e = mu_0 + 2.5*b_n(n)/np.log(10)
    mu_e_ave = mu_e - 2.5*np.log10(f_n(n))
    return mu_e_ave


def sersic_surface_brightness(m_tot, r_e, n):
    area = np.pi * r_e.to('arcsec').value**2
    mu_e_ave = m_tot + 2.5 * np.log10(2 * area)
    mu_e = mu_e_ave + 2.5*np.log10(f_n(n))
    mu_0 = mu_e - 2.5*b_n(n)/np.log(10)
    return mu_0, mu_e, mu_e_ave


def mu0_to_mtot(mu_0, r_e, n):
    area = np.pi * r_e.to('arcsec').value**2
    mu_e = mu_0 + 2.5*b_n(n)/np.log(10) 	
    mu_e_ave = mu_e - 2.5*np.log10(f_n(n))
    m_tot = mu_e_ave - 2.5 * np.log10(2 * area)
    return m_tot


def mtot_to_mu0(m_tot, r_e, n):
    area = np.pi * r_e.to('arcsec').value**2
    mu_e_ave = m_tot + 2.5 * np.log10(2 * area)
    mu_e = mu_e_ave + 2.5*np.log10(f_n(n))
    mu_0 = mu_e - 2.5*b_n(n)/np.log(10)
    return mu_0
