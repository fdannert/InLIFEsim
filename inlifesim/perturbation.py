import numpy as np

from inlifesim.util import temp2freq_fft
from inlifesim.spectra import rms_frequency_adjust, create_pink_psd
from inlifesim.debug import debug_sys_noise_chop


def response(wl_bins: np.ndarray, phi: np.ndarray, bl: np.ndarray, num_a: int):
    """
    Calculates the response of the array to a point source

    Parameters
    ----------
    wl_bins : np.ndarray
        The wavelength bins in [m]
    phi : np.ndarray
        The phase response of each collector aperture in [rad]
    bl : np.ndarray
        The baseline of each collector apertures in [m]
    num_a : int
        The number of collector apertures
    """

    theta_x = np.linspace(-1e-6, 1e-6, 200)[np.newaxis, :]
    theta_y = np.linspace(-1e-6, 1e-6, 200)[np.newaxis, :]
    R = np.sum(
        np.array(
            [
                np.sum(
                    np.array(
                        [
                            np.cos(phi[j] - phi[k])
                            * np.cos(
                                2
                                * np.pi
                                / wl_bins[:, np.newaxis]
                                * (
                                    bl[0, j, k] * theta_x
                                    + bl[1, j, k] * theta_y
                                )
                            )
                            - np.sin(phi[j] - phi[k])
                            * np.sin(
                                2
                                * np.pi
                                / wl_bins[:, np.newaxis]
                                * (
                                    bl[0, j, k] * theta_x
                                    + bl[1, j, k] * theta_y
                                )
                            )
                            for k in range(num_a)
                        ]
                    ),
                    axis=0,
                )
                for j in range(num_a)
            ]
        ),
        axis=0,
    )  # Eq (5)

    return R


def stellar_leakage(
    A: np.ndarray,
    phi: np.ndarray,
    b_star: np.ndarray,
    db_star_dx: np.ndarray,
    db_star_dy: np.ndarray,
    num_a: int,
):
    """
    Calculates the gradient and Hessian coefficients for the stellar leakage

    Parameters
    ----------
    A : np.ndarray
        The amplitude of each collector aperture
    phi : np.ndarray
        The phase response of each collector aperture in [rad]
    b_star : np.ndarray
        The Fourier transform of the stellar sky-brightness distribution
        (Bessel function) evaluated at the baselines of each collector aperture
        pair
    db_star_dx : np.ndarray
        The derivative of the Fourier transform of the stellar sky-brightness
        distribution (Bessel function) evaluated at the baselines of each
        collector aperture pair in aperture displacement in the x direction
    db_star_dy : np.ndarray
        The derivative of the Fourier transform of the stellar sky-brightness
        distribution (Bessel function) evaluated at the baselines of each
        collector aperture pair in aperture displacement in the y direction
    num_a : int
        The number of collector apertures

    Returns
    -------
    grad_n_star_coeff : np.ndarray
        The gradient coefficients for the stellar leakage
    hess_n_star_coeff : np.ndarray
        The Hessian coefficients for the stellar leakage

    """
    # TODO: Check if all indices are correct
    # stellar sensitivity coefficients
    c_a_star = np.swapaxes(
        np.array(
            [
                2
                * A[j]
                * np.array(
                    [
                        A[k] * np.cos(phi[j] - phi[k]) * b_star[:, j, k]
                        for k in range(num_a)
                    ]
                ).sum(axis=0)
                for j in range(num_a)
            ]
        ),
        0,
        1,
    )  # Eq (15)

    c_phi_star = np.swapaxes(
        np.array(
            [
                -2
                * A[j]
                * np.array(
                    [
                        A[k] * np.sin(phi[j] - phi[k]) * b_star[:, j, k]
                        for k in range(num_a)
                        if (k != j)
                    ]
                ).sum(axis=0)
                for j in range(num_a)
            ]
        ),
        0,
        1,
    )  # Eq (16)

    c_x_star = np.swapaxes(
        np.array(
            [
                2
                * np.array(
                    [
                        A[j]
                        * A[k]
                        * np.cos(phi[j] - phi[k])
                        * db_star_dx[:, j, k]
                        for k in range(num_a)
                    ]
                ).sum(axis=0)
                for j in range(num_a)
            ]
        ),
        0,
        1,
    )  # Eq (17)

    c_y_star = np.swapaxes(
        np.array(
            [
                2
                * np.array(
                    [
                        A[j]
                        * A[k]
                        * np.cos(phi[j] - phi[k])
                        * db_star_dy[:, j, k]
                        for k in range(num_a)
                    ]
                ).sum(axis=0)
                for j in range(num_a)
            ]
        ),
        0,
        1,
    )  # Eq (18)

    grad_n_star_coeff = {
        "a": c_a_star,
        "phi": c_phi_star,
        "x": c_x_star,
        "y": c_y_star,
    }

    # Eq (20)
    c_aphi_star = np.swapaxes(
        np.array(
            [
                -2 * A[j] * A * np.sin(phi[j] - phi) * b_star[:, j, :]
                for j in range(num_a)
            ]
        ),
        0,
        1,
    )
    c_aphi_diag_star = np.swapaxes(
        np.array(
            [
                -2
                * A[j]
                * np.array(
                    [
                        A[l] * np.sin(phi[j] - phi[l]) * b_star[:, j, l]
                        for l in range(num_a)
                    ]
                ).sum(axis=0)
                for j in range(num_a)
            ]
        ),
        0,
        1,
    )
    for i in range(c_aphi_star.shape[0]):
        np.fill_diagonal(c_aphi_star[i,], c_aphi_diag_star[i,])

    c_aa_star = np.swapaxes(
        np.array(
            [
                A[j] * A * np.cos(phi[j] - phi) * b_star[:, j, :]
                for j in range(num_a)
            ]
        ),
        0,
        1,
    )  # Eq (19)

    # Eq (21)
    c_phiphi_star = np.swapaxes(
        np.array(
            [
                A[j] * A * np.cos(phi[j] - phi) * b_star[:, j, :]
                for j in range(num_a)
            ]
        ),
        0,
        1,
    )
    c_phiphi_diag_star = np.swapaxes(
        np.array(
            [
                -A[j]
                * np.array(
                    [
                        A[l] * np.cos(phi[j] - phi[l]) * b_star[:, j, l]
                        for l in range(num_a)
                        if (l != j)
                    ]
                ).sum(axis=0)
                for j in range(num_a)
            ]
        ),
        0,
        1,
    )
    for i in range(c_phiphi_star.shape[0]):
        np.fill_diagonal(c_phiphi_star[i,], c_phiphi_diag_star[i,])

    hess_n_star_coeff = {
        "aa": c_aa_star,
        "aphi": c_aphi_star,
        "phiphi": c_phiphi_star,
    }

    return grad_n_star_coeff, hess_n_star_coeff


def exozodi_leakage(
    A: np.ndarray, phi: np.ndarray, b_ez: np.ndarray, num_a: int
):
    """
    Calculates the gradient and Hessian coefficients for the exozodi leakage

    Parameters
    ----------
    A : np.ndarray
        The amplitude of each collector aperture
    phi : np.ndarray
        The phase response of each collector aperture in [rad]
    b_ez : np.ndarray
        The Fourier transform of the exozodi sky-brightness distribution
        evaluated at the baselines of each collector aperture pair
    num_a : int
        The number of collector apertures

    Returns
    -------
    grad_n_ez_coeff : np.ndarray
        The gradient coefficients for the exozodi leakage
    hess_n_ez_coeff : np.ndarray
        The Hessian coefficients for the exozodi leakage

    """

    c_a_ez = np.swapaxes(
        np.array(
            [
                2
                * A[j]
                * np.array(
                    [
                        A[k] * np.cos(phi[j] - phi[k]) * b_ez[:, j, k]
                        for k in range(num_a)
                    ]
                ).sum(axis=0)
                for j in range(num_a)
            ]
        ),
        0,
        1,
    )  # Eq (15)

    c_phi_ez = np.swapaxes(
        np.array(
            [
                -2
                * A[j]
                * np.array(
                    [
                        A[k] * np.sin(phi[j] - phi[k]) * b_ez[:, j, k]
                        for k in range(num_a)
                        if (k != j)
                    ]
                ).sum(axis=0)
                for j in range(num_a)
            ]
        ),
        0,
        1,
    )  # Eq (16)

    grad_n_ez_coeff = {
        "a": c_a_ez,
        "phi": c_phi_ez,
        "x": np.zeros_like(c_a_ez),
        "y": np.zeros_like(c_a_ez),
    }

    c_aa_ez = np.swapaxes(
        np.array(
            [
                A[j] * A * np.cos(phi[j] - phi) * b_ez[:, j, :]
                for j in range(num_a)
            ]
        ),
        0,
        1,
    )  # Eq (19)

    # Eq (20)
    c_aphi_ez = np.swapaxes(
        np.array(
            [
                -2 * A[j] * A * np.sin(phi[j] - phi) * b_ez[:, j, :]
                for j in range(num_a)
            ]
        ),
        0,
        1,
    )
    c_aphi_diag_ez = np.swapaxes(
        np.array(
            [
                -2
                * A[j]
                * np.array(
                    [
                        A[l] * np.sin(phi[j] - phi[l]) * b_ez[:, j, l]
                        for l in range(num_a)
                    ]
                ).sum(axis=0)
                for j in range(num_a)
            ]
        ),
        0,
        1,
    )
    for i in range(c_aphi_ez.shape[0]):
        np.fill_diagonal(c_aphi_ez[i,], c_aphi_diag_ez[i,])

    # Eq (21)
    c_phiphi_ez = np.swapaxes(
        np.array(
            [
                A[j] * A * np.cos(phi[j] - phi) * b_ez[:, j, :]
                for j in range(num_a)
            ]
        ),
        0,
        1,
    )
    c_phiphi_diag_ez = np.swapaxes(
        np.array(
            [
                -A[j]
                * np.array(
                    [
                        A[l] * np.cos(phi[j] - phi[l]) * b_ez[:, j, l]
                        for l in range(num_a)
                        if (l != j)
                    ]
                ).sum(axis=0)
                for j in range(num_a)
            ]
        ),
        0,
        1,
    )
    for i in range(c_phiphi_ez.shape[0]):
        np.fill_diagonal(c_phiphi_ez[i,], c_phiphi_diag_ez[i,])

    hess_n_ez_coeff = {"aa": c_aa_ez, "aphi": c_aphi_ez, "phiphi": c_phiphi_ez}

    return grad_n_ez_coeff, hess_n_ez_coeff


def localzodi_leakage(
    A: np.ndarray, omega: np.ndarray, flux_localzodi: np.ndarray
):
    """
    Computes the local zodiacal light leakage and its gradient coefficients.

    This function utilizes the input aperture parameters, omega values, and
    the local zodiacal flux to calculate the local zodiacal light leakage
    coefficient and its gradient with respect to various parameters. The
    results are returned as a dictionary structure.

    :param A: An array representing aperture parameters.
    :param omega: An array representing the solid angles associated with
        the apertures.
    :param flux_localzodi: An array representing the local zodiacal flux
        values.
    :return: A dictionary containing the local zodiacal light leakage
        coefficient and its gradients with respect to 'a', 'phi', 'x',
        and 'y' parameters.
    :rtype: dict
    """
    c_a_lz = (
        2
        * flux_localzodi[:, np.newaxis]
        * A[np.newaxis, :] ** 2
        * omega[:, np.newaxis]
    )

    grad_n_lz_coeff = {
        "a": c_a_lz,
        "phi": np.zeros_like(c_a_lz),
        "x": np.zeros_like(c_a_lz),
        "y": np.zeros_like(c_a_lz),
    }

    return grad_n_lz_coeff


def sys_noise_chop(mp_arg) -> dict:
    """
    Multiprocessing worker to calculate the systematic noise contribution for
    a single wavelength bin in the chopping mode

     Parameters
    ----------
    mp_arg : dict
        Dictionary containing the arguments for the multiprocessing worker
        function

    Returns
    -------
    noise_chop : dict
        Dictionary containing the systematic noise contributions for the
        wavelength bin
    """

    # TODO: Insert references to the equations once the publication is written

    flux_star = mp_arg['flux_star']
    A = mp_arg['A']
    wl = mp_arg['wl']
    num_a = mp_arg['num_a']
    planet_template_chop = mp_arg['planet_template_chop']
    grad_n_coeff = mp_arg['grad_n_coeff']
    hess_n_coeff = mp_arg['hess_n_coeff']
    grad_n_coeff_chop = mp_arg['grad_n_coeff_chop']
    hess_n_coeff_chop = mp_arg['hess_n_coeff_chop']
    # c_phi = mp_arg['c_phi']
    # c_aphi = mp_arg['c_aphi']
    # c_aa = mp_arg['c_aa']
    # c_phiphi = mp_arg['c_phiphi']
    rms_mode = mp_arg['rms_mode']
    # n_sampling_max = mp_arg['n_sampling_max']
    n_sampling_total = mp_arg['n_sampling_total']
    harmonic_number_n_cutoff = mp_arg['harmonic_number_n_cutoff']
    rms_period_bins = mp_arg['rms_period_bins']
    t_total = mp_arg['t_total']
    d_a_rms = mp_arg['d_a_rms']
    d_phi_rms = mp_arg['d_phi_rms']
    d_pol_rms = mp_arg['d_pol_rms']
    n_rot = mp_arg['n_rot']
    hyperrot_noise = mp_arg['hyperrot_noise']

    # calculate the Fourier components of the planet template
    planet_template_c_fft = temp2freq_fft(
        time_series=planet_template_chop
    )

    # adjust rms values
    d_a_rms, d_phi_rms, d_pol_rms, _, _ = rms_frequency_adjust(
        rms_mode=rms_mode,
        wl=wl,
        d_a_rms=d_a_rms,
        d_phi_rms=d_phi_rms,
        d_pol_rms=d_pol_rms,
        d_x_rms=None,
        d_y_rms=None,
    )

    # create PSDs
    d_a_psd, avg_d_a_2, d_a_b_2 = create_pink_psd(
        t_total=t_total,
        n_sampling_total=n_sampling_total,
        rms=d_a_rms,
        num_a=num_a,
        harmonic_number_n_cutoff=harmonic_number_n_cutoff['a'],
        period_bin=rms_period_bins['a'],
        n_rot=n_rot,
        hyperrot_noise=hyperrot_noise,
    )

    d_phi_psd, avg_d_phi_2, d_phi_b_2 = create_pink_psd(
        t_total=t_total,
        n_sampling_total=n_sampling_total,
        rms=d_phi_rms,
        num_a=1,
        harmonic_number_n_cutoff=harmonic_number_n_cutoff['phi'],
        period_bin=rms_period_bins['phi'],
        n_rot=n_rot,
        hyperrot_noise=hyperrot_noise,
    )

    d_pol_psd, avg_d_pol_2, d_pol_b_2 = create_pink_psd(
        t_total=t_total,
        n_sampling_total=n_sampling_total,
        rms=d_pol_rms,
        num_a=1,
        harmonic_number_n_cutoff=harmonic_number_n_cutoff['pol'],
        period_bin=rms_period_bins['pol'],
        n_rot=n_rot,
        hyperrot_noise=hyperrot_noise,
    )

    # noise contribution
    noise_chop = {'wl': wl}

    # polarization noise
    dn_pol = (flux_star * A**2 * avg_d_pol_2).sum()
    noise_chop['pn_pa'] = np.sqrt(dn_pol * t_total)


    # calculate fourier components
    # TODO: This still assumes the same PSD for the different input apertures

    # d_phi_j_hat_2_chop = np.full(
    #     shape=num_a,
    #     fill_value=(np.sum(d_phi_psd[int(len(d_phi_psd)/2):] * np.abs(planet_template_c_fft[int(len(planet_template_c_fft)/2):]) ** 2)
    #                 / t_total ** 5 * (len(planet_template_chop)/2) ** 2)
    # )

    d_phi_j_hat_2_chop = np.full(
        shape=num_a,
        fill_value=(
            np.sum(d_phi_psd * np.abs(planet_template_c_fft) ** 2)
            / t_total
            # / t_total**5
            # * len(planet_template_chop) ** 4
        ),
    )

    d_a_j_hat_2_chop = np.full(
        shape=num_a,
        fill_value=(
            np.sum(d_a_psd * np.abs(planet_template_c_fft) ** 2)
            / t_total
            # / t_total**5
            # * len(planet_template_chop) ** 4
        ),
    )

    # first order phase noise
    noise_chop['sn_fo_phi'] = np.sum(
        (grad_n_coeff['phi'] - grad_n_coeff_chop['phi']) ** 2
        * d_phi_j_hat_2_chop
    )

    # poisson noise from null floor perturbation
    dn_null_floor = np.sum(
        np.array(
            [
                (
                    (
                        hess_n_coeff['aa'][j, j]
                        # - hess_n_coeff_chop['aa'][j, j]
                    )
                    * d_a_j_hat_2_chop[j]
                    + (
                        hess_n_coeff['phiphi'][j, j]
                        # - hess_n_coeff_chop['phiphi'][j, j]
                    )
                    * d_phi_j_hat_2_chop[j]
                )
                for j in range(num_a)
            ]
        )
    )
    noise_chop['pn_snfl'] = np.sqrt(dn_null_floor * t_total)

    # second order dadphi
    # d_a_d_phi_j_hat_2_chop = np.full(
    #     shape=(num_a, num_a),
    #     fill_value=np.sum(
    #         np.convolve(d_a_psd[0], d_phi_psd, mode='same')
    #         * np.abs(planet_template_c_fft)**2
    #     ) * len(planet_template_chop) ** 4 / t_total ** 8 / n_rot
    # )

    # d_a_d_phi_j_hat_2_chop = np.full(
    #     shape=(num_a, num_a),
    #     fill_value=np.sum(
    #         np.convolve(d_a_psd[0][int(len(d_a_psd[0]) / 2):], d_phi_psd[int(len(d_phi_psd)/2):], mode='same')
    #         * np.abs(planet_template_c_fft[int(len(planet_template_c_fft)/2):])**2
    #     ) * (len(planet_template_chop)/2) ** 4 / t_total ** 8 / n_rot * 2
    # )

    d_a_d_phi_j_hat_2_chop = np.full(
        shape=(num_a, num_a),
        fill_value=np.sum(
            np.convolve(d_a_psd[0], d_phi_psd, mode='same')
            * np.abs(planet_template_c_fft) ** 2
        )
                   # * len(planet_template_chop) ** 2
                   / t_total ** 2
        # * len(planet_template_chop) ** 6
        # / t_total**8
        # / n_rot,
    )

    noise_chop['sn_so_aphi'] = np.sum(
        (hess_n_coeff['aphi'] - hess_n_coeff_chop['aphi']) ** 2
        * d_a_d_phi_j_hat_2_chop
    )

    noise_chop['sn_fo'] = noise_chop['sn_fo_phi']
    noise_chop['sn_so'] = noise_chop['sn_so_aphi']

    noise_chop['sn'] = np.sqrt(
        noise_chop['sn_fo_phi'] + noise_chop['sn_so_aphi']
    )

    debug_sys_noise_chop(
        d_phi_b_2=d_phi_b_2,
        planet_template_c_fft=planet_template_c_fft,
        d_phi_j_hat_2_chop=d_phi_j_hat_2_chop,
    )

    return noise_chop
