import numpy as np
from astropy.io import fits
from scipy.special import gammaincinv, gamma

import pymfit

__all__ = ['imfit_out_to_sersic_params', 'fit_disk']


img_fn = '/Users/jgreco/temp-files/model-disk.fits'
psf_fn = '/Users/jgreco/temp-files/psf.fits'


def imfit_out_to_sersic_params(params, pixscale=0.168):
    I_e = params['I_e']
    r_e = params['r_e']
    n = params['n']
    PA = params['PA']
    ell = params['ell']
    q = 1 - ell
    r_circ = r_e*np.sqrt(q)*pixscale
    b_n = gammaincinv(2.*n, 0.5)
    mu_e = -2.5*np.log10(I_e/pixscale**2)
    mu_0 = mu_e - 2.5*b_n/np.log(10)
    f_n = gamma(2*n)*n*np.exp(b_n)/b_n**(2*n)
    mu_e_ave = mu_e - 2.5*np.log10(f_n)
    A_eff = np.pi*r_circ**2
    m_tot = mu_e_ave - 2.5*np.log10(2*A_eff)
    params['mu_0'] = mu_0
    params['mu_e'] = mu_e
    params['mu_e_ave'] = mu_e_ave
    params['r_e'] = r_e * pixscale
    params['m_tot'] = m_tot

    return params


def fit_disk(model):    
    
    fits.writeto(img_fn, model.array, overwrite=True)
    fits.writeto(psf_fn, model.psf_array, overwrite=True)
    
    init_params = dict(PA=45, n=[1.0, 0.01, 5.0], ell=0.1, I_e=[1, 0, 1000]) 
    config = pymfit.sersic_config(init_params, img_shape=img_fn)
    options = '--sky={} --exptime={}'.format(model.sky_level_flux.value, 
                                             model.effective_exp_time.value)
    fit = pymfit.run(img_fn, config_fn='/Users/jgreco/temp-files/config.txt', 
                     config=config, psf_fn=psf_fn, 
                     out_fn='/Users/jgreco/temp-files/best-fit.dat',
                     options=options)
    
    if fit['ell'] < 0:
        # a & b are flipped
        a = (1.0 - fit['ell'])*fit['r_e']
        b = fit['r_e']
        fit['ell'] = 1.0 - b/a
        fit['r_e'] = a
        fit['PA'] -= 90.0
    if (fit['PA'] > 180) or (fit['PA'] < 0):
        wrapped = fit['PA'] % 360.0
        wrapped = wrapped - 180 * (wrapped > 180)
        fit['PA'] = wrapped

    sersic_params = fit.copy()
    sersic_params['I_e'] = 10**(0.4 * 48.6) * fit['I_e']
    sersic_params['I_e'] *= model.pars.phot_flux_to_fnu.value
    sersic_params = imfit_out_to_sersic_params(sersic_params)

    image_params = fit.copy()

    return image_params, sersic_params
