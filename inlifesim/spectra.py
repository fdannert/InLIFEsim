from typing import Union

import numpy as np

from inlifesim.util import freq2temp_fft, harmonic_number_approximation

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
                    n_sampling_total: int,
                    rms: float,
                    num_a: int,
                    harmonic_number_n_cutoff: Union[int, type(None)] = None,
                    period_bin: Union[tuple, type(None)] = None,
                    n_rot: Union[int, type(None)] = None,
                    hyperrot_noise: Union[str, type(None)] = None):
    """
    Generates a Power Spectral Density (PSD) following a pink noise spectrum.

    The function creates a PSD for 1/f noise (pink noise) based on given parameters, optionally
    applying additional constraints and noise adjustments for hyperrotational components. The PSD
    is shaped depending on the defined number of time series or segments (num_a), and the harmonic
    number is calculated using either `period_bin` or `harmonic_number_n_cutoff`. The resulting PSD
    is symmetric around zero, and additional outputs related to the spectral properties, `avg_2` and
    `b_2`, are currently placeholders (not computed).

    Parameters
    ----------
    t_total : float
        Total duration of the signal in seconds.
    n_sampling_total : int
        Total number of data points sampled in the signal.
    rms : float
        Root Mean Square (RMS) value for normalizing the resultant PSD.
    num_a : int
        Number of PSD spectra or segments to tile in the final output.
    harmonic_number_n_cutoff : int or None, optional
        Harmonic number of the cutoff frequency cutoff. Defines the highest frequency component to consider if
        provided. Should not be provided along with `period_bin`.
    period_bin : tuple or None, optional
        A two-element tuple defining the range of periods in which the RMS is defined (in seconds). The harmonic
        number is approximated based on this range. Should not be provided along with
        `harmonic_number_n_cutoff`.
    n_rot : int or None, optional
        Optional number of array rotations. When specified (greater than 1), hyperrotational
        noise adjustments are applied on the PSD, further controlled using the `hyperrot_noise`
        parameter.
    hyperrot_noise : str or None, optional
        Defines the hyperrotational noise mode when `n_rot > 1`. Options are:
            - 'zero': Sets hyperrotational noise to zero.
            - 'max': Uses maximum value of the PSD for hyperrotational noise.
            - '1/f_r': Creates inversely proportional noise to rotational frequencies.
            - '1/f': Creates inversely proportional noise with scaling based on `n_rot`.

        If left unspecified, defaults to 'zero'. Only used if `n_rot` is greater than 1.

    Returns
    -------
    psd : ndarray
        The 1/f pink noise PSD array generated based on input specifications. For `num_a > 1`,
        the PSD is tiled for multiple beams.
    avg_2 : None
        Placeholder, not currently computed.
    b_2 : None
        Placeholder, not currently computed.

    Raises
    ------
    ValueError
        If both `period_bin` and `harmonic_number_n_cutoff` are provided, or if neither of them is
        provided, an error will be raised.
    ValueError
        When the lower frequency bound in `period_bin` is less than or equal to the rotation
        frequency (`n_rot`), an error will be raised.
    ValueError
        If the specified `hyperrot_noise` option is not recognized, an error will be raised.

    Notes
    -----
    - The PSD is symmetric around zero. The first half corresponds to negative frequencies, and
      the second half to positive frequencies.
    - The outputs `avg_2` and `b_2` are placeholders and are not currently implemented in the
      function.
    - When `n_rot` and `hyperrot_noise` are specified, the PSD low-frequency components are
      adjusted based on the hyperrotational noise mode.
    """

    hn_samples = int(n_sampling_total / 2)

    if (period_bin is not None) and (harmonic_number_n_cutoff is None):
        harmonic_number = (harmonic_number_approximation(t_total / period_bin[1])
                           - harmonic_number_approximation(t_total / period_bin[0]))
        if (not ((n_rot is None) or (n_rot == 1))) and ((t_total / period_bin[0]) <= n_rot):
            raise ValueError('Lower frequency bin must be larger than rotation frequency.')
    elif (period_bin is None) and (harmonic_number_n_cutoff is not None) and ((n_rot is None) or (n_rot == 1)):
        harmonic_number = harmonic_number_n_cutoff
    elif (period_bin is None) and (harmonic_number_n_cutoff is not None) and (not((n_rot is None) or (n_rot == 1))):
        harmonic_number = (harmonic_number_n_cutoff
                           - harmonic_number_approximation(n_rot))
    elif (period_bin is not None) and (harmonic_number_n_cutoff is not None):
        raise ValueError('Both period_bin and harmonic_number_n_cutoff are specified. Please choose only one.')
    else:
        raise ValueError('Neither period_bin nor harmonic_number_n_cutoff are specified. Please choose one.')

    psd = (
                rms ** 2
                * t_total
                / harmonic_number
                / 2
                / np.arange(1, hn_samples + 1)
        )

    if not ((n_rot is None) or (n_rot == 1)):
        if hyperrot_noise is None:
            print('Hyperrot noise not specified, using default')
            hyperrot_noise = 'zero'

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

        psd[:n_rot-1] = hyper_psd

    psd = np.insert(arr=psd, obj=0, values=0)
    psd = np.concatenate((np.flip(psd[1:]), psd))

    if num_a != 1:
        psd = np.tile(psd, (num_a, 1))

    # b_2 = (2 * n_sampling_max) ** 2 / t_total * psd
    b_2 = None

    # avg_2 = np.sum(b_2, axis=-1) / t_total ** 4
    avg_2 = np.sum(psd, axis=-1) / t_total

    return psd, avg_2, b_2

