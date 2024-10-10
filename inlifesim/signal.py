import numpy as np
from astropy.constants import au, pc
from tqdm import tqdm

from inlifesim.util import (temp2freq_fft, freq2temp_ft, freq2temp_fft,
                            combine_to_full_observation)
from inlifesim.debug import debug_planet_signal

def planet_response(flux_planet: np.ndarray,
                    A: np.ndarray,
                    phi: np.ndarray,
                    wl_bins: np.ndarray,
                    bl: np.ndarray,
                    num_a: int,
                    theta: np.ndarray,
                    phi_rot: np.ndarray):

    #  old version:
    # n_planet = np.swapaxes(np.array(
    #     [flux_planet
    #      * np.array(
    #         [np.array(
    #             [A[j] * A[k]
    #              * (np.cos(phi[j] - phi[k])
    #                 * np.cos(
    #                         2 * np.pi / wl_bins
    #                         * (bl[0, j, k] * theta[0, l] + bl[1, j, k] * theta[1, l])
    #                     )
    #                 - np.sin(phi[j] - phi[k])
    #                 * np.sin(
    #                         2 * np.pi / wl_bins
    #                         * (bl[0, j, k] * theta[0, l] + bl[1, j, k] * theta[1, l])))
    #              for k in range(num_a)]).sum(axis=0)
    #          for j in range(num_a)]).sum(axis=0)
    #      for l in range(len(phi_rot))]), 0, 1)
    #
    # return n_planet

    # vectorized version

    dp_matrix = np.tensordot(bl, theta, axes=([0], [0]))

    n_planet = np.sum(np.outer(A, A)[np.newaxis, :, :, np.newaxis] * (
        np.cos(
            phi[:, np.newaxis] - phi[np.newaxis, :]
        )[np.newaxis, :, :, np.newaxis]
        * np.cos(
        2 * np.pi / wl_bins[:, np.newaxis, np.newaxis, np.newaxis]
        * dp_matrix[np.newaxis, ]) -
        np.sin(
            phi[:, np.newaxis] - phi[np.newaxis, :]
        )[np.newaxis, :, :, np.newaxis]
        * np.sin(
        2 * np.pi / wl_bins[:, np.newaxis, np.newaxis, np.newaxis]
        * dp_matrix[np.newaxis, ])
    ), axis=(1, 2)) * flux_planet[:, np.newaxis]

    return n_planet


def planet_signal(separation_planet: float,
                  dist_star: float,
                  #t_rot: float,
                  t_exp: float,
                  t_total: float,
                  t_rot: float,
                  n_sampling_rot: int,
                  # phi_rot: np.ndarray,
                  flux_planet: np.ndarray,
                  A: np.ndarray,
                  phi: np.ndarray,
                  phi_r: np.ndarray,
                  wl_bins: np.ndarray,
                  bl: np.ndarray,
                  num_a: int,
                  simultaneous_chopping: bool,
                  phi_rot_start: float = 0,
                  ):
    """
    Calculates the planet signal and template function for the planet signal

    Parameters
    ----------
    separation_planet : float
        The separation of the planet in [au]
    dist_star : float
        The distance to the star in [pc]
    n_sampling_rot : int
        The number of samples per rotation
    t_rot : float
        The rotation period of the array in [s]
    t_int : float
        The integration time of the observation in [s]
    flux_planet : np.ndarray
        The flux of the planet in [ph m-2 s-1]
    A : np.ndarray
        The amplitude response of each collector aperture
    phi : np.ndarray
        The phase response of each collector aperture in [rad]
    phi_r : np.ndarray
        The phase response of each collector aperture in [rad] for the chopped
        planet signal
    wl_bins : np.ndarray
        The wavelength bins in [m]
    bl : np.ndarray
        The baseline vectors of each collector aperture pair in [m]
    num_a : int
        The number of collector apertures
    simultaneous_chopping : bool
        If True, it is assumed that the beamcombiner is set-up such that the
        chopped and normal signal are produced by two simulatneous outputs. If
        False, it is assumend that the chopping is done interatively in time
        (not yet implemented).

    Returns
    -------
    planet_template_nchop : np.ndarray
        The template function for the planet signal for the non-chopped case
    photon_rates_nchop_signal : np.ndarray
        The total photon rates of the planet signal for the non-chopped case
        in [ph s-1]
    planet_template_chop : np.ndarray
        The template function for the planet signal for the chopped case
    photon_rates_chop_signal : np.ndarray
        The total photon rates of the planet signal for the chopped case in
        [ph s-1]
    """

    if not simultaneous_chopping:
        raise ValueError(
            'Currently, only simultaneous chopping is implemented'
        )

    theta = separation_planet * au.value / (dist_star * pc.value)
    phi_rot = np.linspace(phi_rot_start,
                          phi_rot_start + 2 * np.pi,
                          n_sampling_rot)
    theta = np.array((-theta * np.cos(phi_rot), theta * np.sin(phi_rot)))

    # create planet signal via Eq (9)
    n_planet = planet_response(flux_planet=flux_planet,
                               A=A,
                               phi=phi,
                               wl_bins=wl_bins,
                               bl=bl,
                               num_a=num_a,
                               theta=theta,
                               phi_rot=phi_rot)

    # chopped planet signal
    n_planet_r = planet_response(flux_planet=flux_planet,
                               A=A,
                               phi=phi_r,
                               wl_bins=wl_bins,
                               bl=bl,
                               num_a=num_a,
                               theta=theta,
                               phi_rot=phi_rot)

    n_planet_nchop = n_planet

    # Fourier transform of planet signal equivalent to Eq (33)
    nfft = temp2freq_fft(n_planet_nchop, t_rot)

    # creation of template function
    # removal of even components and DC
    nfft_odd = nfft
    if not simultaneous_chopping:
        nfft_odd[:, int(nfft_odd.shape[1] / 2)::2] = 0
        nfft_odd[:, :int(nfft_odd.shape[1] / 2)] = np.flip(
            nfft_odd[:, int(nfft_odd.shape[1] / 2 + 1):], axis=1)

    # transform back into time domain
    planet_template_nchop = freq2temp_fft(nfft_odd, t_rot)

    # normalize the template function to rms of one
    planet_template_nchop = (planet_template_nchop
                             / np.std(planet_template_nchop,
                                      axis=1)[:, np.newaxis])

    # planet_template_nchop = np.abs(planet_template_nchop)

    n_planet_nchop = combine_to_full_observation(arr=n_planet_nchop,
                                                 t_total=t_total,
                                                 t_rot=t_rot,
                                                 t_exp=t_exp)

    planet_template_nchop = combine_to_full_observation(
        arr=planet_template_nchop,
        t_total=t_total,
        t_rot=t_rot,
        t_exp=t_exp
    )

    nchop_signal = t_exp * n_planet_nchop

    photon_rates_nchop_signal = (np.abs(
        (t_exp * planet_template_nchop * n_planet_nchop)).sum(axis=1))

    # ----- For chopped planet signal -----
    n_planet_chop = (n_planet - n_planet_r)

    if not simultaneous_chopping:
        n_planet_chop *= 0.5

    # Fourier transform of planet signal equivalent to Eq (33)
    nfft_chop = temp2freq_fft(n_planet_chop, t_rot)

    # creation of template function
    # removal of even components and DC
    nfft_odd_chop = nfft_chop
    nfft_odd_chop[:, int(nfft_odd_chop.shape[1] / 2)::2] = 0
    nfft_odd_chop[:, :int(nfft_odd_chop.shape[1] / 2)] = np.flip(
        nfft_odd_chop[:, int(nfft_odd_chop.shape[1] / 2 + 1):], axis=1)


    # transform back into time domain
    planet_template_chop = freq2temp_fft(nfft_odd_chop, t_rot)

    # normalize the template function to rms of one
    planet_template_chop = (planet_template_chop
                             / np.std(planet_template_chop,
                                      axis=1)[:, np.newaxis])

    #planet_template_chop = np.abs(planet_template_chop+np.min(planet_template_chop.real))-np.min(planet_template_chop.real)

    n_planet_chop = combine_to_full_observation(arr=n_planet_chop,
                                                 t_total=t_total,
                                                 t_rot=t_rot,
                                                 t_exp=t_exp)

    planet_template_chop = combine_to_full_observation(
        arr=planet_template_chop,
        t_total=t_total,
        t_rot=t_rot,
        t_exp=t_exp
    )

    chop_signal = t_exp * n_planet_chop

    photon_rates_chop_signal = (np.abs(
        (t_exp * planet_template_chop * n_planet_chop)).sum(axis=1))

    # photon_rates_chop_signal = (np.abs(
    #     (t_exp * n_planet_chop))).sum(axis=1)

    debug_planet_signal(n_planet_nchop=n_planet_nchop,
                        planet_template_nchop=planet_template_nchop,
                        n_planet_chop=n_planet_chop,
                        planet_template_chop=planet_template_chop,
                        wl_bins=wl_bins)

    return (planet_template_nchop, photon_rates_nchop_signal, nchop_signal,
            planet_template_chop, photon_rates_chop_signal, chop_signal)

def star_signal(A: np.ndarray,
                phi: np.ndarray,
                b_star: np.ndarray,
                num_a: int,
                t_int: float
                ):
    """
    Calculates the photon rate received from the star after nulling, that is
    the stellar geometric leakage

    Parameters
    ----------
    A : np.ndarray
        The amplitude response of each collector aperture
    phi : np.ndarray
        The phase response of each collector aperture in [rad]
    b_star : np.ndarray
        The Fourier transform of the stellar sky-brightness distribution
        (Bessel function) evaluated at the baselines of each collector aperture
        pair
    num_a : int
        The number of collector apertures
    t_int : float
        The integration time of the observation in [s]

    Returns
    -------
    photon_rates_nchop_pn_sgl : np.ndarray
        The total photon rates of the star signal for the non-chopped case in
        [ph s-1]
    """

    n_0_star = np.array([
        np.array([A[j] * A[k] * np.cos(phi[j] - phi[k]) * b_star[:, j, k]
                  for k in range(num_a)]).sum(axis=0)
        for j in range(num_a)]).sum(axis=0)
    photon_rates_nchop_pn_sgl = np.sqrt(n_0_star * t_int)

    return photon_rates_nchop_pn_sgl

def exozodi_signal(A: np.ndarray,
                   phi: np.ndarray,
                   b_ez: np.ndarray,
                   num_a: int,
                   t_int: float):
    """
    Calculates the photon rate received from the exozodi after nulling, that
    is the exozodi geometric leakage

    Parameters
    ----------
    A : np.ndarray
        The amplitude response of each collector aperture
    phi : np.ndarray
        The phase response of each collector aperture in [rad]
    b_ez : np.ndarray
        The Fourier transform of the exozodi sky-brightness distribution
        evaluated at the baselines of each collector aperture pair
    num_a : int
        The number of collector apertures
    t_int : float
        The integration time of the observation in [s]

    Returns
    -------
    photon_rates_nchop_pn_ez : np.ndarray
        The total photon rates of the exozodi signal for the non-chopped case
        in [ph s-1]
    """
    n_0_ez = np.array([
        np.array([A[j] * A[k] * np.cos(phi[j] - phi[k]) * b_ez[:, j, k]
                  for k in range(num_a)]).sum(axis=0)
        for j in range(num_a)]).sum(axis=0)
    photon_rates_nchop_pn_ez = np.sqrt(n_0_ez * t_int)

    return photon_rates_nchop_pn_ez

def localzodi_signal(A: np.ndarray,
                     omega: np.ndarray,
                     flux_localzodi: np.ndarray,
                     t_int:float):
    """
    Calculates the photon rate received from the local zodiacal light

    Parameters
    ----------
    A : np.ndarray
        The amplitude response of each collector aperture
    omega : np.ndarray
        The solid angle of each collector aperture in [sr]
    flux_localzodi : np.ndarray
        The flux of the local zodiacal light in [ph m-2 s-1 sr-1]
    t_int : float
        The integration time of the observation in [s]

    Returns
    -------
    photon_rates_nchop_pn_lz : np.ndarray
        The total photon rates of the local zodiacal light signal for the
        non-chopped case in [ph s-1]
    """
    n_0_lz = (
            flux_localzodi[:, np.newaxis] * A[np.newaxis, :] ** 2
            * omega[:, np.newaxis]
    ).sum(axis=1)
    photon_rates_nchop_pn_lz = np.sqrt(n_0_lz * t_int)
    return photon_rates_nchop_pn_lz

def fundamental_noise(A: np.ndarray,
                      phi: np.ndarray,
                      b_star: np.ndarray,
                      b_ez: np.ndarray,
                      flux_localzodi: np.ndarray,
                      num_a: int,
                      t_int: float,
                      omega: np.ndarray
                      ):
    photon_rates_nchop_pn_sgl = star_signal(A=A,
                                            phi=phi,
                                            b_star=b_star,
                                            num_a=num_a,
                                            t_int=t_int)

    photon_rates_nchop_pn_ez = exozodi_signal(A=A,
                                              phi=phi,
                                              b_ez=b_ez,
                                              num_a=num_a,
                                              t_int=t_int)

    photon_rates_nchop_pn_lz = localzodi_signal(A=A,
                                                omega=omega,
                                                flux_localzodi=flux_localzodi,
                                                t_int=t_int)

    return (photon_rates_nchop_pn_sgl, photon_rates_nchop_pn_ez,
            photon_rates_nchop_pn_lz)

def create_template_grid(
        max_ang_sep: float,
        n_grid: int,
        n_sampling_rot: int,
        A: np.ndarray,
        phi: np.ndarray,
        phi_r: np.ndarray,
        wl_bins: np.ndarray,
        bl: np.ndarray,
        num_a: int,
        flux_planet: np.ndarray,
        dist_star: float,
        t_total: float,
        t_rot: float,
        t_exp: float,
        simultaneous_chopping: bool = True):

    if not simultaneous_chopping:
        raise ValueError(
            'Currently, only simultaneous chopping is implemented'
        )

    theta_x, theta_y = np.meshgrid(
        np.linspace(-max_ang_sep, max_ang_sep, n_grid),
        np.linspace(-max_ang_sep, max_ang_sep, n_grid)
    )

    templates_nchop = np.zeros((n_grid, n_grid, len(wl_bins), n_sampling_rot))
    templates_chop = np.zeros((n_grid, n_grid, len(wl_bins), n_sampling_rot))

    for i in tqdm(range(n_grid)):
        for j in range(n_grid):

            sep = np.sqrt(theta_x[i, j] ** 2 + theta_y[i, j] ** 2) * dist_star
            ang = np.arctan2(theta_y[i, j], theta_x[i, j])

            template_nchop, _, _, template_chop, _, _ = planet_signal(
                separation_planet=sep,
            dist_star=dist_star,
            # t_rot: float,
            t_exp=t_exp,
            t_total=t_total,
            t_rot=t_rot,
            n_sampling_rot=n_sampling_rot,
            flux_planet=flux_planet,
            A=A,
            phi=phi,
            phi_r=phi_r,
            wl_bins=wl_bins,
            bl=bl,
            num_a=num_a,
            simultaneous_chopping=True,
            phi_rot_start=ang,
            )

            templates_nchop[i, j, :, :] = template_nchop
            templates_chop[i, j, :, :] = template_chop

    return templates_nchop, templates_chop



