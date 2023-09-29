import numpy as np
from astropy import constants as const
from astropy import units as un
from astropy.cosmology import Planck15 as cosmo
from scipy.integrate import quad 
from scipy import signal

def mK_to_Jy_per_sr(frequencies, tile_diameter=None):#, cellsize, distances):
    """
    Conversion factor to convert a pixel of mK to Jy/sr (and vice versa via division)
    Taken from http://w.astro.berkeley.edu/~wright/school_2012.pdf
    Parameters
    ----------
    frequencies : float array, optional
        The mean readshift of the observation.
    
    Returns
    -------
    conversion_factor : float or array
        The conversion factor(s) (per frequency) which convert temperature in Kelvin to flux density in Jy.
    """
    if tile_diameter==None:
        wvlngth = const.c / (frequencies / un.s)

        intensity = 2 * const.k_B * 1e-3 * un.K / wvlngth ** 2

        flux_density = 1e26 * intensity.to(un.W / (un.Hz * un.m ** 2))

        return (flux_density / (1 * un.sr))#.value
    else:
        return ( 2 * const.k_B * 1e26 * 1e-3) / (tile_diameter)**2 

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
    return H_0 / cosmo.h * f_21 * E_z / (const.c * (1 + z )**2) # h / Mpc Hz

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

def hz_to_mpc_per_h(nu):
    """
    Convert a frequency range in Hz to a distance range in Mpc.
    """
    z = frequencies_to_redshifts(nu)

    return 1 / (Gz(z))#- Gz(z_min))

def sr_to_mpc2_per_h2(z_mid):
    """
    Conversion factor from steradian to Mpc^2 at a given redshift.
    Parameters
    ----------
    z_mid: mean readshift of observation
    Returns
    -------
    """
    return cosmo.comoving_transverse_distance(z_mid)**2 / (1 * un.sr) * (cosmo.h)**2

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
    
def obsUnits_to_cosmoUnits(power, frequencies, uv, eta, sky_size=None, tile_diameter=None, include_taper= False):
    z_mid = np.mean(frequencies_to_redshifts(frequencies))

    power_cosmo = power * (un.W / un.m**2 / un.Hz ) **2 * un.Hz**2 / mK_to_Jy_per_sr(np.mean(frequencies))**2 * (hz_to_mpc_per_h(frequencies[0]))**2 * (sr_to_mpc2_per_h2(z_mid))**2 #, tile_diameter = tile_diameter

    # divide by volume
    if tile_diameter == None:
        volume_obs = (sky_size**2 * un.sr * sr_to_mpc2_per_h2(z_mid) * (frequencies[-1] - frequencies[0]) * un.Hz * hz_to_mpc_per_h(np.mean(frequencies)))#[0]
    else:
        volume_obs = volume(frequencies, tile_diameter=tile_diameter, include_taper=include_taper)
    
    power_cosmo = power_cosmo / volume_obs #/ Gz(z_mid))

    kperp = k_perpendicular(uv, z_mid).value
    kpar = k_parallel(eta, z_mid).value

    return power_cosmo.value, kperp, kpar # mK^2 Mpc^3 h^-3, h Mpc^-1, h Mpc^-1

def kparallel_wedge(k_perp, field_of_view, redshift):

    E_z = cosmo.efunc(redshift)
    H_0 = (cosmo.H0).to(un.m/(un.Mpc * un.s))

    functionDc_z = lambda x: 1 / cosmo.efunc(x)
    Dc_z = quad(functionDc_z, 0, np.max(redshift))[0]

    return k_perp * np.sin(field_of_view) * E_z * Dc_z / (1 + redshift) #* H_0 / const.c #* H_0 * E_z * cosmo.comoving_transverse_distance(redshift) / const.c / (1+redshift) #

def volume(frequencies, tile_diameter=20, include_taper= False):
    """
    Calculate the effective volume of an observation in Mpc**3, when co-ordinates are provided in Hz.
    Parameters
    ----------
    z_mid : float
        Mid-point redshift of the observation.
    nu_min, nu_max : float
        Min/Max frequency of observation, in Hz.
    tile_diameter : float
        Diameter of the station.
    Returns
    -------
    vol : float
        The volume.
    """
    # TODO: update the notes in the docs above.
    if include_taper == True:
        diff_nu = np.sum(signal.blackmanharris(len(frequencies)) * (np.max(frequencies) - np.min(frequencies)) / len(frequencies))
    else:
        diff_nu = np.max(frequencies) - np.min(frequencies)

    z_mid = np.mean(frequencies_to_redshifts(frequencies))

    field_of_view = const.c / (tile_diameter * un.m  * np.max(frequencies) * (1 / un.s))

    Vol = np.pi * (field_of_view / 2) ** 2 * (1 / un.s) * cosmo.comoving_transverse_distance([z_mid]) ** 2  * cosmo.h ** 2 * diff_nu / (Gz(z_mid))

    return Vol.value