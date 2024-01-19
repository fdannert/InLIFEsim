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

    psd = (2 * rms ** 2 * t_rot ** 3 / (2 * n_sampling_max) ** 2
           / harmonic_number_n_cutoff / np.arange(0, n_sampling_max + 1))
    psd[0] = 0
    psd = np.concatenate((np.flip(psd[1:]), psd))

    if num_a != 1:
        psd = np.tile(psd, (num_a, 1))

    b_2 = (2 * n_sampling_max) ** 2 / t_rot * psd

    avg_2 = np.sum(b_2, axis=-1) / t_rot ** 4

    return psd, avg_2, b_2

def draw_fourier_noise(psd: np.ndarray,
                       n_sampling_rot: int,
                       t_rot: float,
                       n_draws: int):
    '''
    Draw Fourier series from a given power spectral density (PSD). The output
    shape will be (n_draws, n_sampling_rot). Can be used to generate a
    realisation of the perturbation random variables.

    Parameters
    ----------
    psd
        Power spectral density
    n_sampling_rot
        Number of samples (exposures) per rotation of the array
    t_rot
        Array rotation period in [s]
    n_draws
        Number of times the experiment is drawn

    Returns
    -------
    x
        The temporal series.
    x_ft
        The Fourier series in frequency space
    '''

    x_ft = (np.random.normal(loc=0,
                             scale=n_sampling_rot * np.sqrt(
                                 psd[np.newaxis,
                                 int((psd.shape[-1] - 1) / 2)::]
                                 / 2 / t_rot
                             ),
                             size=(n_draws,
                                   int((psd.shape[-1] + 1) / 2)))
            + 1j * np.random.normal(loc=0,
                                    scale=n_sampling_rot * np.sqrt(
                                        psd[np.newaxis,
                                        int((psd.shape[-1] - 1) / 2)::]
                                        / 2 / t_rot
                                    ),
                                    size=(n_draws,
                                          int((psd.shape[-1] + 1) / 2)))
            )

    x_ft = np.concatenate((np.flip(x_ft[:, 1:], axis=-1), x_ft),
                          axis=-1)

    x = freq2temp_fft(fourier_series=x_ft,
                      total_time=t_rot)

    return x, x_ft