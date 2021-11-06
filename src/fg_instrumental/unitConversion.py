import numpy as np
from astropy import constants as const
from astropy import units as un
from astropy.cosmology import Planck15 as cosmo

def mK_to_Jy_per_sr(z):#, cellsize, distances):
    """
    Conversion factor to convert a pixel of mK to Jy/sr (and vice versa via division)
    Taken from http://w.astro.berkeley.edu/~wright/school_2012.pdf
    Parameters
    ----------
    nu : float array, optional
        The mean readshift of the observation.
    
    Returns
    -------
    conversion_factor : float or array
        The conversion factor(s) (per frequency) which convert temperature in Kelvin to flux density in Jy.
    """

    nu = redshifts_to_frequencies(z)

    wvlngth = const.c / (nu / un.s)

    intensity = 2 * const.k_B * 1e-3 * un.K / wvlngth ** 2

    flux_density = 1e26 * intensity.to(un.W / (un.Hz * un.m ** 2))
    
    return flux_density / (1 * un.sr) #* (( cellsize ) / distances)**2

def redshifts_to_frequencies(z):
    """The cosmological redshift (of signal) associated with each frequency"""
    return 1420e6 / (z + 1)

def frequencies_to_redshifts(frequency):
    """The cosmological redshift (of signal) associated with each frequency"""
    return 1420e6 / frequency - 1

def k_perpendicular(r, z):
    '''
    The conversion factor to find the perpendicular scale in Mpc given the angular scales and redshift
    
    Parameters
    ----------
    r : float or array-like
        The radius in u,v Fourier space
    z : float or array-like
        The redshifts
        
    Returns
    -------
    k_perpendicular : float or array-like
        The scale in h Mpc^1
    '''
    k_perpendicular = 2 * np.pi * r / cosmo.comoving_transverse_distance([z]) / cosmo.h
    return k_perpendicular ## [h Mpc^1]

def Gz(z):
    f_21 = 1420e6 * un.Hz
    E_z = cosmo.efunc(z)
    H_0 = (cosmo.H0).to(un.m/(un.Mpc * un.s))
    return H_0 / cosmo.h * f_21 * E_z / (const.c * (1 + z )**2)

def k_parallel(eta, z):
    '''
    The conversion factor to find the parallel scale in Mpc given the frequency scale in Hz^-1 and redshift
    
    Parameters
    ----------
    eta : float or array-like
        The frequency scale in Hz^-1
    z : float or array-like
        The redshifts
        
    Returns
    -------
    k_perpendicular : float or array-like
        The scale in h Mpc^1
    '''

    k_parallel = 2 * np.pi * Gz(z) * eta / (1 * un.Hz)
    return k_parallel ## [h Hz Mpc^-1]

def hz_to_mpc(nu_min, nu_max):
    """
    Convert a frequency range in Hz to a distance range in Mpc.
    """
    z_max = frequencies_to_redshifts(nu_min)
    z_min = frequencies_to_redshifts(nu_max)

    return 1 / (Gz(z_max) - Gz(z_min))

def sr_to_mpc2(z_mid):
    """
    Conversion factor from steradian to Mpc^2 at a given redshift.
    Parameters
    ----------
    z_mid: mean readshift of observation
    Returns
    -------
    """
    return cosmo.comoving_transverse_distance(z_mid)**2 / (1 * un.sr)

def degree_to_mpc(degree, z):# use minimum redshift

    radian = np.deg2rad(degree)
    boxsize = cosmo.comoving_transverse_distance(z).value * radian
    
    return boxsize

def theta_phi_to_lm(theta, phi):
    '''
    Convert theta phi (radian) to lm (unitless)

    '''

    l = np.sin(theta) * np.cos(phi)
    m = np.sin(theta) * np.sin(phi)

    return l, m

def lm_to_theta_phi(l, m):
    '''
    Convert lm (unitless) to theta phi (radian) by solving for:

    l = sin(theta) * cos(phi)
    m = sin(theta) * sin(phi)

    '''
    # solve for phi first so we can plug-in
    phi = np.arctan(m / l)

    theta = np.arcsin(l / np.cos(phi))

    #phi is undefined for theta = 0 so need to correct for this
    index = np.where(theta == 0)
    phi[index] = 0

    return theta, phi