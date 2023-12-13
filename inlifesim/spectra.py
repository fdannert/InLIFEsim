from typing import Union

import numpy as np

def rms_frequency_adjust(rms_mode: str,
                         wl: float,
                         d_a_rms: Union[float, type(None)],
                         d_phi_rms: Union[float, type(None)],
                         d_pol_rms: Union[float, type(None)],
                         d_x_rms: Union[float, type(None)],
                         d_y_rms: Union[float, type(None)]):
    '''
    Adjust the RMS values to the wavelength in use.

    Parameters
    ----------
    rms_mode
        'lay': uses the rms values specified in lay 2004 and ignores the input
        rms values
        'static': returns the input rms values
        'wavelength': scales the input rms values to the wavelength. For
        amplitude noise, the scaling is lamnda^-1.5, for phase noise, the
        scaling is lambda^-1
    wl
        wavelength in m
    d_a_rms
        amplitude noise spectrum rms
    d_phi_rms
        phase noise spectrum rms

    Returns
    -------
    d_a_rms
        amplitude noise spectrum rms
    d_phi_rms
        phase noise spectrum rms
    d_pol_rms
        polarization noise spectrum rms
    d_x_rms
        collector position noise spectrum rms, x-direction
    d_y_rms
        collector position noise spectrum rms, y-direction
    '''

    # create noise PSD
    if rms_mode == 'lay':
        d_a_rms_0 = 0.001
        d_a_rms = d_a_rms_0 * (wl / 10e-6) ** (-1.5)

        d_phi_rms_0 = 0.001
        d_phi_rms = d_phi_rms_0 * (wl / 10e-6) ** (-1)

        d_pol_rms = 0.001
        d_x_rms = 0.01
        d_y_rms = 0.01

    elif rms_mode == 'static':
        if (d_a_rms is None) or (d_phi_rms is None):
            raise ValueError('RMS values need to be specified in static mode')
    elif rms_mode == 'wavelength':
        if (d_a_rms is None) or (d_phi_rms is None):
            raise ValueError('RMS values need to be specified in wavelength '
                             'mode')
        d_a_rms = d_a_rms * (wl / 10e-6) ** (-1.5)
        d_phi_rms = d_phi_rms * (wl / 10e-6) ** (-1)
    else:
        raise ValueError('RMS mode not recongnized')

    return d_a_rms, d_phi_rms, d_pol_rms, d_x_rms, d_y_rms

def create_pink_psd(t_rot: float,
                    n_sampling_max: int,
                    harmonic_number_n_cutoff: int,
                    rms: float,
                    num_a: int):
    '''
    Create a pink noise power spectral density (PSD)

    Parameters
    ----------
    t_rot
        Rotation period in [s]
    n_sampling_max
        Number of positive frequency samples, should be chosen to be the same
        as half of the number of samples in the time domain series
    harmonic_number_n_cutoff
        The PSD is defined via a rms within a certain frequency range. The
        high frequency cutoff can be converted to a cutoff at the n-th
        harmonic. Supply here the Harmonic number of n (= sum of the
        reciprocals of the first n natural numbers)
    rms
        rms that the PSD should have between 0 and the cutoff frequency
    num_a
        number of apertures

    Returns
    -------
    psd
        The power spectral density
    avg_2
        The squared average of the power spectral density
        # TODO: this is not the correct description
    b_2
        The power of the Fourier components corresponding to the PSD
    '''

    psd = (rms ** 2 * t_rot
           / harmonic_number_n_cutoff / np.arange(0, n_sampling_max+1))
    psd[0] = 0
    psd = np.concatenate((np.flip(psd[1:]), psd))

    if num_a != 1:
        psd = np.tile(psd, (num_a, 1))

    b_2 = t_rot * psd

    avg_2 = np.sum(b_2, axis=-1) / t_rot ** 4

    return psd, avg_2, b_2