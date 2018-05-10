import numpy as np
from scipy.special import gammaincinv
from scipy.interpolate import interp1d
import warnings

__all__ = ['r_scaled', 'sersic_prob_xy', 'rejection_sersic_xy']


r_scaled = lambda x, y, q: np.sqrt(x**2 + (y/q)**2)


def sersic_prob_xy(x, y, n, q=1.0, r_e=4.0):
    b_n = gammaincinv(2.0 * n, 0.5)
    r = r_scaled(x, y, q)
    prob = np.exp(-b_n * (r/r_e)**(1.0/n))
    return prob


def rejection_sersic_xy(num_draws=100, sersic_n=1.0, r_e=2.0, ell=0.0, PA=90, 
                        centroid=(0, 0), r_max_num_r_e=5, verbose=False, 
                        return_all_draws=False, sample_factor=100, r_min=0):

    q = 1 - ell
    r_max = r_max_num_r_e * r_e

    # rejection sampling
    while True:
        try:
            num_sample = sample_factor * num_draws
            if num_sample < 5e5:
                num_sample += 5e5
            x_sample = 2*r_max*np.random.random(int(num_sample)) - r_max
            y_sample = 2*r_max*np.random.random(int(num_sample)) - r_max
            z_sample = np.random.random(int(num_sample))
            keep = z_sample < sersic_prob_xy(x_sample, y_sample, sersic_n, 
                                          q=q, r_e=r_e)
            if r_min > 0:
                r_cut = np.sqrt(x_sample**2 + y_sample**2) > r_min
                cut = keep & r_cut
            else:
                cut = keep
            x_draws = x_sample[cut]
            y_draws = y_sample[cut]
            rand_i = np.random.choice(
                np.arange(len(y_draws)), size=int(num_draws), replace=False)
            break
        except ValueError:
            warnings.warn('sample_factor too low')
            print('***** multiplying sample_factor by 5 *****')
            sample_factor *= 5

    if return_all_draws:
        _x = x_draws.copy()
        _y = y_draws.copy()

    x_draws = x_draws[rand_i]
    y_draws = y_draws[rand_i]
	
    # rotate about origin
    theta = (90 - PA) * np.pi / 180
    xp = x_draws * np.cos(theta) + y_draws * np.sin(theta)
    yp = -x_draws * np.sin(theta) + y_draws * np.cos(theta)
    
    # now translate
    x0, y0 = centroid
    xp = xp + x0
    yp = yp + y0

    if verbose:
        accept_frac = keep.sum()/float(num_sample)
        print('{} samples drawn from {} successful draws out of {}: '\
              'acceptance fraction = {:.2f}'.\
              format(len(xp), keep.sum(), num_sample, accept_frac))

    return (xp, yp, _x, _y) if return_all_draws else (xp, yp)
