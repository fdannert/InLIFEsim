import numpy as np
from scipy.special import jn

from inlifesim.util import black_body

def create_star(wl_bins: np.ndarray,
                wl_bin_widths: np.ndarray,
                temp_star: float,
                radius_star: float,
                dist_star: float,
                bl: np.ndarray,
                col_pos: np.ndarray,
                num_a: int,
                ):
    """
    Creates the star black body spectrum and calculates the Bessel function
    and its derivative needed for the calculation of the stellar contribution
    to the noise.

    Parameters
    ----------
    wl_bins : np.ndarray
        The wavelength bins at which the black body is evaluated in [m]
    wl_bin_widths : np.ndarray
        The width of the wavelength bins in [m]
    temp_star : float
        The temperature of the star in [K]
    radius_star : float
        The radius of the star in [sun_radii]
    dist_star : float
        The distance between the instrument and the star in [pc]
    bl : np.ndarray
        The baseline matrix in [m]
    col_pos : np.ndarray
        The position of the collector apertures in [m]
    num_a : int
        The number of collector apertures

    Returns
    -------
    flux_star : np.ndarray
        The stellar photon flux at the specified wavelength bins [ph m-2 s-1]
    b_star : np.ndarray
        A matrix containing the Bessel function of the star evaluated at the
        baseline matrix and the wavelength bins [ph m-2 s-1]
    db_star_dx : np.ndarray
        The derivative of the Bessel function with respect to the x-coordinate
        of the aperture positions
    db_star_dy : np.ndarray
        The derivative of the Bessel function with respect to the y-coordinate
        of the aperture positions
    """
    flux_star = black_body(mode='star',
                                bins=wl_bins,
                                width=wl_bin_widths,
                                temp=temp_star,
                                radius=radius_star,
                                distance=dist_star)

    # angular extend of the star disk in rad divided by 2 to get radius
    ang_star = radius_star * 0.00465 / dist_star * np.pi / (180 * 3600)
    bl_mat = (bl[0] ** 2 + bl[1] ** 2) ** 0.5
    b_star = np.nan_to_num(
        np.divide(
            2 * flux_star[:, np.newaxis, np.newaxis]
            * jn(1, 2 * np.pi * bl_mat[np.newaxis, :] * ang_star
                 / wl_bins[:, np.newaxis, np.newaxis]),
            2 * np.pi * bl_mat[np.newaxis, :] * ang_star
            / wl_bins[:, np.newaxis, np.newaxis]))  # Eq (11)
    for i in range(b_star.shape[0]):
        np.fill_diagonal(b_star[i], flux_star[i])

    # derivative of the Bessel function needed for Eqs (17) & (18)
    a = 2 * np.pi * ang_star / wl_bins
    db_star_dx = np.swapaxes(np.nan_to_num(np.array(
        [2 * flux_star[:, np.newaxis]
         * ((col_pos[j, 0] - col_pos[:, 0]) / bl_mat[j, :] ** 2)[np.newaxis, :]
         * (0.5 * (jn(0, a[:, np.newaxis] * bl_mat[np.newaxis, j, :])
                   - jn(2, a[:, np.newaxis] * bl_mat[np.newaxis, j, :]))
            - jn(1, a[:, np.newaxis] * bl_mat[np.newaxis, j, :])
            / (a[:, np.newaxis] * bl_mat[np.newaxis, j, :]))
         for j in range(num_a)])), 0, 1)

    db_star_dy = np.swapaxes(np.nan_to_num(np.array(
        [2 * flux_star[:, np.newaxis]
         * ((col_pos[j, 1] - col_pos[:, 1]) / bl_mat[j, :] ** 2)[np.newaxis, :]
         * (0.5 * (jn(0, a[:, np.newaxis] * bl_mat[np.newaxis, j, :])
                   - jn(2, a[:, np.newaxis] * bl_mat[np.newaxis, j, :]))
            - jn(1, a[:, np.newaxis] * bl_mat[np.newaxis, j, :])
            / (a[:, np.newaxis] * bl_mat[np.newaxis, j, :]))
         for j in range(num_a)])), 0, 1)

    return flux_star, b_star, db_star_dx, db_star_dy

def create_planet(wl_bins: np.ndarray,
                  wl_bin_widths: np.ndarray,
                  temp_planet: float,
                  radius_planet: float,
                  dist_star: float):
    """
    Creates the planet black body spectrum.

    Parameters
    ----------
    wl_bins : np.ndarray
        The wavelength bins at which the black body is evaluated in [m]
    wl_bin_widths : np.ndarray
        The width of the wavelength bins in [m]
    temp_planet : float
        The temperature of the planet in [K]
    radius_planet : float
        The radius of the planet in [earth_radii]
    dist_star : float
        The distance between the instrument and the planet in [pc]

    Returns
    -------
    flux_planet : np.ndarray
        The planet photon flux at the specified wavelength bins [ph m-2 s-1]
    """
    flux_planet = black_body(mode='planet',
                                  bins=wl_bins,
                                  width=wl_bin_widths,
                                  temp=temp_planet,
                                  radius=radius_planet,
                                  distance=dist_star)

    return flux_planet

def create_localzodi(wl_bins: np.ndarray,
                     wl_bin_widths: np.ndarray,
                     lat: float,
                     long: float = 3 / 4 * np.pi):
    """
    Creates the local zodiacal light black body spectrum.

    Parameters
    ----------
    wl_bins : np.ndarray
        The wavelength bins at which the black body is evaluated in [m]
    wl_bin_widths : np.ndarray
        The width of the wavelength bins in [m]
    lat : float
        The ecliptic latitude of the target star in [rad]
    long : float
        The ecliptic longitude of the target star in [rad]

    Returns
    -------
    flux_localzodi : np.ndarray
        The local zodiacal light photon flux at the specified wavelength bins
        [ph m-2 s-1]
    """

    radius_sun_au = 0.00465047  # in AU
    tau = 4e-8
    temp_eff = 265
    temp_sun = 5777
    a = 0.22

    b_tot = black_body(mode='wavelength',
                       bins=wl_bins,
                       width=wl_bin_widths,
                       temp=temp_eff) + a \
            * black_body(mode='wavelength',
                         bins=wl_bins,
                         width=wl_bin_widths,
                         temp=temp_sun) \
            * (radius_sun_au / 1.5) ** 2
    flux_localzodi = tau * b_tot * np.sqrt(
        np.pi / np.arccos(np.cos(long) * np.cos(lat)) /
        (np.sin(lat) ** 2
         + (0.6 * (wl_bins / 11e-6) ** (-0.4) * np.cos(lat)) ** 2)
    )

    return flux_localzodi

def create_exozodi(wl_bins: np.ndarray,
                   wl_bin_widths: np.ndarray,
                   l_sun: float,
                   z: float,
                   r_au: np.ndarray,
                   image_size: int,
                   au_pix: np.ndarray,
                   rad_pix: np.ndarray,
                   radius_map: np.ndarray,
                   bl: np.ndarray,
                   hfov: np.ndarray
                   ):
    """
    Creates the exozodiacal dust black body spectrum.

    Parameters
    ----------
    wl_bins : np.ndarray
        The wavelength bins at which the black body is evaluated in [m]
    wl_bin_widths : np.ndarray
        The width of the wavelength bins in [m]
    l_sun : float
        The luminosity of the target star in [stellar luminosity]
    z : float
        The zodi number of the target system in [zodi]
    r_au : np.ndarray
        The radial distance of each pixel to the star in [AU]
    image_size : int
        The size of the image in pixels
    au_pix : np.ndarray
        The conversion factor between [AU] and [pix]
    rad_pix : np.ndarray
        The conversion factor between [rad] and [pix]
    radius_map : np.ndarray
        The radial distance of each pixel to the center of the image in [pix]
    bl : np.ndarray
        The baseline matrix in [m]
    hfov : np.ndarray
        The half field of view of the instrument in [rad]

    Returns
    -------
    b_ez : np.ndarray
        A matrix containing the Bessel function of the exozodi evaluated at the
        baseline matrix and the wavelength bins [ph m-2 s-1]

    """

    # calculate the parameters required by Kennedy2015
    alpha = 0.34
    r_in = 0.034422617777777775 * np.sqrt(l_sun)
    r_0 = np.sqrt(l_sun)
    # Sigma_{m,0} from Kennedy+2015 (doi:10.1088/0067-0049/216/2/23)
    sigma_zero = 7.11889e-8

    # identify all pixels where the radius is larger than the inner radius by
    # Kennedy+2015
    r_cond = ((r_au >= r_in)
              & (r_au <= image_size / 2 * au_pix[:, np.newaxis, np.newaxis]))

    # calculate the temperature at all pixel positions according to Eq. 2 in
    # Kennedy2015
    temp_map = np.where(r_cond,
                        np.divide(278.3 * (l_sun ** 0.25), np.sqrt(r_au),
                                  out=np.zeros_like(r_au),
                                  where=(r_au != 0.)),
                        0)

    # calculate the Sigma (Eq. 3) in Kennedy2015 and set everything inside the
    # inner radius to 0
    sigma = np.where(r_cond,
                     sigma_zero * z *
                     np.power(r_au / r_in, -alpha,
                              out=np.zeros_like(r_au),
                              where=(r_au != 0.)),
                     0)

    # get the black body radiation emitted by the interexoplanetary dust
    f_nu_disk = black_body(bins=wl_bins[:, np.newaxis, np.newaxis],
                           width=wl_bin_widths[:, np.newaxis, np.newaxis],
                           temp=temp_map,
                           mode='wavelength') \
                * sigma * rad_pix[:, np.newaxis, np.newaxis] ** 2

    ap = np.where(radius_map <= image_size / 2, 1, 0)
    flux_map_exozodi = f_nu_disk * ap

    # fourier transform done manually
    b_ez = np.zeros((wl_bins.shape[0], bl.shape[1], bl.shape[2]))
    for k in range(wl_bins.shape[0]):
        theta_x, theta_y = np.meshgrid(
            np.linspace(-hfov[k], hfov[k], image_size),
            np.linspace(-hfov[k], hfov[k], image_size)
        )

        # Compute the phase term
        phase = 2 * np.pi / wl_bins[k]

        # Broadcast and calculate for all `i, j` at once
        cos_term = np.cos(
            phase * (
                    bl[0][:, :, np.newaxis, np.newaxis] * theta_x +
                    bl[1][:, :, np.newaxis, np.newaxis] * theta_y
            )
        )  # Resulting shape: (bl.shape[1], bl.shape[2], image_size, image_size)

        # Compute flux for all elements
        b_ez[k] = (
                flux_map_exozodi[k, np.newaxis, np.newaxis, :, :] * cos_term
        ).sum(axis=(-2, -1))  # Reduce last two dimensions

        # for i in range(bl.shape[1]):
        #     for j in range(bl.shape[2]):
        #         b_ez[k, i, j] = (flux_map_exozodi[k, ] * np.cos(
        #             2 * np.pi / wl_bins[k]
        #             * (bl[0, i, j] * theta_x
        #                + bl[1, i, j] * theta_y))).sum()

    return b_ez
