from typing import Union

import numpy as np


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def planck_law(x: np.ndarray,
               temp: Union[float, np.ndarray],
               mode: str):
    """
    Calculates the photon flux emitted from a black body according to Planck's law in the
    wavelength or frequency regime

    Parameters
    ----------
    x : np.ndarray
        The frequency of wavelength at which the photon fluxes are calculated in [Hz] or [m]
    temp : Union[float, np.ndarray]
        The temperature of the black body
    mode : str
        If ``x`` is given in [Hz], set ``mode = 'frequency'. If ``x`` is given in [m], set
        ``mode = 'wavelength'

    Raises
    ------
    ValueError
        If the mode is not recognized

    Returns
    -------
    fgamma : np.ndarray
        The photon flux at the respective wavelengths or frequencies
    """

    h = 6.62607e-34
    k = 1.380649e-23
    c = 2.99792e+8

    # select the correct mode
    if mode == 'wavelength':

        # account for the temperature being zero at some pixels
        with np.errstate(divide='ignore'):

            # the Planck law divided by the photon energy to obtain the photon flux
            fgamma = 2 * c / (x**4) / \
               (np.exp(h * c / x / k / temp) - 1)
    elif mode == 'frequency':

        # account for the temperature being zero at some pixels
        with np.errstate(divide='ignore'):

            # the Planck law divided by the photon energy to obtain the photon flux
            fgamma = np.where(temp == 0,
                              0,
                              2 * x**2 / (c**2) /
                              (np.exp(h * x / k / temp))-1.)
    else:
        raise ValueError('Mode not recognised')

    return fgamma


def black_body(mode: str,
               bins: np.ndarray,
               width: np.ndarray,
               temp: Union[float, np.ndarray],
               radius: float = None,
               distance: float = None):
    """
    Calculates the black body photon flux in wavelength or frequency as well as for planetary or
    stellar sources

    Parameters
    ----------
    mode : str
        Defines the mode of the ``black_body`` function.
            - ``mode = 'wavelength'`` : Clean photon flux black body spectrum over wavelength is
              returned. Parameters used are ``bins``, ``width`` and ``temp``
            - ``mode = 'frequency'`` : Clean photon flux black body spectrum over frequency is
              returned. Parameters used are ``bins``, ``width`` and ``temp``
            - ``mode = 'star'`` : Photon flux black body spectrum received from a star of specified
              radius from the specified distance. All parameters are used. In this mode, the
              parameter ``bins`` needs to be in wavelength
            - ``mode = 'planet'`` : Photon flux black body spectrum received from a planet of
              specified radius from the specified distance. All parameters are used. In this mode,
              the parameter ``bins`` needs to be in wavelength
    bins : np.ndarray
        The wavelength or frequency bins at which the black body is evaluated in [m] or [Hz]
        respectively
    width : np.ndarray
        The width of the wavelength or frequency bins to integrate over the black body spectrum in
        [m] or [Hz] respectively
    temp : Union[float, np.ndarray]
        The temperature of the black body
    radius : float
        The radius of the spherical black body object. For ``mode = 'star'`` in [sun_radii], for
        ``mode = 'planet'`` in [earth_radii]
    distance : float
        The distance between the instrument and the observed object in [pc]

    Raises
    ------
    ValueError
        If the mode is not recognized

    Returns
    -------
    fgamma : np.ndarray
        The photon flux at the respective wavelengths or frequencies
    """

    radius_sun = 6.947e+8
    radius_earth = 6.371e+6
    m_per_pc = 3.086e+16

    if mode == 'star':
        fgamma = planck_law(x=bins,
                            temp=temp,
                            mode='wavelength') * width \
                 * np.pi * ((radius * radius_sun) / (distance * m_per_pc)) ** 2
    elif mode == 'planet':
        fgamma = planck_law(x=bins,
                            temp=temp,
                            mode='wavelength') * width \
                 * np.pi * ((radius * radius_earth) / (distance * m_per_pc)) ** 2
    elif mode == 'wavelength':
        fgamma = planck_law(x=bins,
                            temp=temp,
                            mode='wavelength') * width
    elif mode == 'frequency':
        # TODO remove hardcoded np.newaxis solution. The redim is needed for the PhotonNoiseExozodi
        #   class
        fgamma = planck_law(x=bins,
                            temp=temp,
                            mode='frequency') * width[:, np.newaxis, np.newaxis]
    else:
        raise ValueError('Mode not recognised')

    return fgamma
