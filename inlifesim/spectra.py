from typing import Union

import numpy as np

from inlifesim.util import freq2temp_fft

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


def create_pink_psd(t_total: float,
                    n_sampling_max: int,
                    harmonic_number_n_cutoff: int,
                    rms: float,
                    num_a: int,
                    n_rot: Union[int, type(None)] = None,
                    hyperrot_noise: Union[str, type(None)] = None):
    '''
    Create a pink noise power spectral density (PSD)

    Parameters
    ----------
    t_total
        Total integration time in [s]
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
    n_rot
        number of rotations, if None, the PSD is calculated for a single
        rotation

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

    # freq = np.arange(1, n_sampling_max + 1) / t_total
    # freq = np.concatenate((np.flip(-freq), np.array([0]), freq))

    if (n_rot is None) or (n_rot == 1):
        psd = (
                2
                * rms ** 2
                * t_total ** 3
                # / (2 * n_sampling_max) ** 2
                / harmonic_number_n_cutoff
                / np.arange(1, n_sampling_max + 1)
        )
        psd = np.insert(arr=psd, obj=0, values=0)
        psd = np.concatenate((np.flip(psd[1:]), psd))

    else:
        if hyperrot_noise is None:
            print('Hyperrot noise not specified, using default')
            hyperrot_noise = 'zero'
        psd = (
                2
                * rms ** 2
                * t_total ** 3
                # / (2 * n_sampling_max) ** 2
                / harmonic_number_n_cutoff
                * n_rot
                / np.arange(n_rot-1, n_sampling_max)
        )
        if hyperrot_noise == 'zero':
            hyper_psd=np.zeros(n_rot-1)
        elif hyperrot_noise == 'max':
            hyper_psd = np.ones(n_rot-1) * np.max(psd)
        elif hyperrot_noise == '1/f_r':
            hyper_psd = 1 / np.arange(n_rot, 1, -1) * np.max(psd)
        elif hyperrot_noise == '1/f':
            hyper_psd = 1 / np.arange(1, n_rot) * np.max(psd) * (n_rot)
        else:
            raise ValueError('Hyperrotational noise mode not recognized')

        psd = np.insert(arr=psd, obj=0, values=hyper_psd)

        psd = np.insert(arr=psd, obj=0, values=0)
        psd = np.concatenate((np.flip(psd[1:]), psd))

    if num_a != 1:
        psd = np.tile(psd, (num_a, 1))

    b_2 = (2 * n_sampling_max) ** 2 / t_total * psd

    # b_2 = psd / 2 / t_rot

    avg_2 = np.sum(b_2, axis=-1) / t_total ** 4

    return psd, avg_2, b_2

