from typing import Union
from functools import wraps
import inspect

import numpy as np
from scipy.fft import rfft

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

def harmonic_number_approximation(n):
    """Returns an approximate value of n-th harmonic number.

       http://en.wikipedia.org/wiki/Harmonic_number
    """
    # Euler-Mascheroni constant
    gamma = 0.57721566490153286060651209008240243104215933593992

    return gamma + np.log(n) + 0.5/n - 1./(12*n**2) + 1./(120*n**4)

def temp2freq_fft(time_series: np.ndarray,
                  total_time: float):
    """
    Fourier series from time to frequency space convention using fft

    Parameters
    ----------
    time_series : np.ndarray
        The time series that is to be converted to Fourier space
    total_time : float
        The total time of the time series in [s]

    Returns
    -------
    fourier_series : np.ndarray
        The Fourier series of the time series
    """

    # note to self: rfft automatically does the Fourier transform on the -1
    # axis
    fourier_series = rfft(time_series)
    fourier_series = np.concatenate((
        np.conjugate(
            np.flip(
                np.delete(
                    fourier_series, obj=0, axis=-1
                ), axis=-1
            )
        ), fourier_series
    ),
    axis=-1)
    fourier_series *= (total_time / fourier_series.shape[-1])

    return fourier_series

def freq2temp_fft(fourier_series: np.ndarray,
                  total_time: float):
    """
    Fourier series from frequency to time space convention using fft

    Parameters
    ----------
    fourier_series : np.ndarray
        The Fourier series that is to be converted to temporal space
    total_time : float
        The total time of the time series in [s]

    Returns
    -------
    time_series : np.ndarray
        The time series of the Fourier series
    """
    n = fourier_series.shape[-1]
    fourier_series = np.delete(fourier_series,
                               obj=np.arange(int(fourier_series.shape[-1]/2)),
                               axis=-1)
    time_series = np.fft.irfft(fourier_series, n=n)
    time_series *= (time_series.shape[-1] / total_time)

    return time_series


def freq2temp_ft(fourier_series: np.ndarray,
                 total_time: float):
    """
    Exact Fourier series convention from frequency to time space

    Parameters
    ----------
    fourier_series : np.ndarray
        The Fourier series that is to be converted to time space
    total_time : float
        The total time of the time series in [s]

    Returns
    -------
    time_series : np.ndarray
        The time series of the Fourier series
    """
    time_series = [
        1 / total_time * np.sum(
            fourier_series * np.exp(
                2 * np.pi * 1j * n
                * np.arange(-(fourier_series.shape[-1]-1)/2,
                            (fourier_series.shape[-1])/2, 1)
                / fourier_series.shape[-1]
            ),
            axis=-1
        ) for n in np.arange(0, fourier_series.shape[-1], 1)]

    return np.swapaxes(np.array(time_series).real, 0, -1)


def dict_sumover(d: dict,
                    keys: list):
        """
        Sums over the values of a dictionary for a given set of keys

        Parameters
        ----------
        d : dict
            The dictionary that is to be summed over
        keys : list
            The keys for which the values are summed over

        Returns
        -------
        sum : float
            The sum of the values of the dictionary for the given keys
        """

        sum = np.zeros_like(d[keys[0]])
        for key in keys:
            sum += d[key]

        return sum


def remove_non_increasing(arr1, arr2):
    """
    This function takes two numpy arrays as input and returns two new arrays
    that only include the elements such that the first array is strictly
    increasing.

    Parameters:
    arr1 (numpy.ndarray): The first input array.
    arr2 (numpy.ndarray): The second input array.

    Returns:
    tuple: A tuple containing the new arrays with strictly increasing elements.
    """
    # Initialize an empty list to store the strictly increasing values
    arr1_inc = []
    arr2_inc = []

    arr1_inc.append(arr1[0])
    arr2_inc.append(arr2[0])

    # Iterate over the array
    for i in range(len(arr1)-1):
        # If this is the first element or the current element is greater than
        # the previous one
        if arr1[i+1] > arr1_inc[-1]:
            # Add the current element to the list
            arr1_inc.append(arr1[i+1])
            arr2_inc.append(arr2[i+1])

    # Convert the list back to a numpy array
    arr1_inc = np.array(arr1_inc)
    arr2_inc = np.array(arr2_inc)

    return arr1_inc, arr2_inc


def combine_to_full_observation(arr,
                                t_total,
                                t_rot,
                                t_exp):

    if len(arr.shape) > 1:
        result = np.tile(A=arr,
                         reps=(1, int(t_total / t_rot)))

        # add the last not-finished rotation
        result = np.concatenate(
            (result,
             arr[:, :int(
                 np.round(
                     (t_total
                      - t_rot * (t_total // t_rot))
                     / t_exp))]
             ),
            axis=1
        )

    else:
        result = np.tile(A=arr,
                         reps=int(t_total / t_rot))

        # add the last not-finished rotation
        result = np.concatenate(
            (result,
             arr[:int(np.round(
                     (t_total - t_rot * (t_total // t_rot)) / t_exp))]
             ),
        )

    return result
