from typing import Union
import multiprocessing as mp

import numpy as np
from scipy.special import jn, spherical_jn
from scipy.fft import fft, fftfreq, fftshift, rfft
import pandas as pd
from tqdm import tqdm

from inlifesim.util import black_body, find_nearest_idx

#TODO: Move calculation of the chopped planet signal into the planet signal method


class Instrument(object):

    def __init__(self,
                 wl_bins: np.ndarray,  # wavelength bins center position in m
                 wl_bin_widths: np.ndarray,  # wavelength bin widhts in m
                 image_size: int,  # size of image used to simulate exozodi in pix
                 diameter_ap: float,  # diameter of the primary mirrors in m
                 flux_division: np.ndarray,  # division of the flux between the primary mirrors, e.g. in baseline case
                                             # [0.25, 0.25, 0.25, 0.25]
                 throughput: float,  # fraction of light that is sustained through the optical train
                 dist_star: float,  # distance to the target system in pc
                 radius_star: float,  # radius of the star in stellar radii
                 col_pos: np.ndarray,  # collector position in m
                 phase_response: np.ndarray,  # phase response of each collector arm in rad
                 phase_response_chop: np.ndarray,  # phase response of each collector arm in the chopped state in rad
                 t_rot: float,  # rotation period of the array in seconds
                 pix_per_wl,  # pixels on detector used per wavelength channel
                 n_sampling_rot: int,  # number of sampling points per array rotation
                 pink_noise_co: int,  #
                 ):

        # setting simulation parameters
        self.wl_bins = wl_bins
        self.wl_bin_widths = wl_bin_widths
        self.image_size = image_size
        self.n_sampling_rot = n_sampling_rot
        self.pink_noise_co = pink_noise_co

        # setting instrument parameters
        self.col_pos = col_pos
        self.phi = phase_response
        self.phi_r = phase_response_chop

        self.R = None  # response function R
        self.bl_x = None  # baseline matrix in x-direction (i.e. x_jk in Lay2004)
        self.bl_y = None  # baseline matrix in y-direction

        self.t_rot = t_rot
        self.pix_per_wl = pix_per_wl

        # setting source parameters
        self.dist_star = dist_star
        self.radius_star = radius_star
        self.flux_star = None
        self.b_star = None
        self.db_star_dx = None
        self.db_star_dy = None

        self.flux_planet = None

        self.flux_localzodi = None

        self.b_ez = None

        # sensitivity coefficients
        self.c_a_star = None
        self.c_phi_star = None
        self.c_x_star = None
        self.c_y_star = None
        self.c_aphi_star = None
        self.c_aa_star = None
        self.c_phiphi_star = None
        self.c_thetatheta_star = None

        self.c_a_ez = None
        self.c_phi_ez = None
        self.c_aphi_ez = None
        self.c_aa_ez = None
        self.c_phiphi_ez = None

        self.c_a_lz = None

        self.c_a = None
        self.c_phi = None
        self.c_x = None
        self.c_y = None
        self.c_aphi = None
        self.c_aa = None
        self.c_phiphi = None
        self.c_thetatheta = None

        # planet signal
        self.sig_planet_nchop = None
        self.planet_template = None
        self.n_planet = None
        self.phi_rot = None
        self.theta_x = None
        self.theta_y = None

        # noise terms without chopping
        # self.pn_sgl_nchop = None
        # self.pn_ez_nchop = None
        # self.pn_lz_nchop = None
        #
        # self.pn_dc_nchop = None
        # self.pn_tbd_nchop = None
        #
        # self.pn_pa_nchop = None
        # self.pn_snfl_nchop = None
        # self.sn_fo_a_nchop = None
        # self.sn_fo_phi_nchop = None
        # self.sn_fo_x_nchop = None
        # self.sn_fo_y_nchop = None
        # self.sn_fo_nchop = None
        # self.sn_so_aa_nchop = None
        # self.sn_so_phiphi_nchop = None
        # self.sn_so_aphi_nchop = None
        # self.sn_so_polpol_nchop = None
        # self.sn_so_nchop = None
        # self.sn_nchop = None

        self.photon_rates = pd.DataFrame(columns=['nchop', 'chop'],
                                         index=['signal',  # planet signal
                                                'noise',  # overall noise contribution
                                                'wl',  # wavelength bin
                                                'pn_sgl',  # stellar geometric leakage
                                                'pn_ez',  # exozodi leakage
                                                'pn_lz',  # localzodi leakage
                                                'pn_dc',  # dark current
                                                'pn_tbd',  # thermal background detector
                                                'pn_pa',  # polarization angle
                                                'pn_snfl',  # stellar null floor leakage
                                                'pn',  # photon noise
                                                'sn_fo_a',  # first order amplitude
                                                'sn_fo_phi',  # first order phase
                                                'sn_fo_x',  # first order x position
                                                'sn_fo_y',  # first order y position
                                                'sn_fo',  # systematic noise first order
                                                'sn_so_aa',  # second order amplitude-amplitude term
                                                'sn_so_phiphi',  # second order phase-phase term
                                                'sn_so_aphi',  # amplitude phase cross term
                                                'sn_so_polpol',  # second order polarization-polarization term
                                                'sn_so',  # systematic noise second order
                                                'sn',  # systematic noise
                                                'snr'  # signal to noise ratio
                                                ])

        self.photon_rates.loc['wl', 'nchop'] = self.wl_bins
        self.photon_rates.loc['wl', 'chop'] = self.wl_bins

        # calculate some further instrumental parameters needed for Lay 2004 implementation
        self.A = np.sqrt(np.pi * (0.5 * diameter_ap) ** 2 * throughput * flux_division)  # area term A_j
        self.num_a = len(self.A)

        self.bl_x = np.array([(self.col_pos[:, 0] - self.col_pos[i, 0]) for i in range(self.num_a)])
        self.bl_y = np.array([(self.col_pos[:, 1] - self.col_pos[i, 1]) for i in range(self.num_a)])

        self.omega = 2 * np.pi * (wl_bins/(2. * diameter_ap))**2

        hfov = wl_bins / (2. * diameter_ap)

        hfov_mas = hfov * (3600000. * 180.) / np.pi
        self.rad_pix = (2 * hfov) / image_size  # Radians per pixel
        mas_pix = (2 * hfov_mas) / image_size  # mas per pixel
        self.au_pix = mas_pix / 1e3 * dist_star  # AU per pixel

        telescope_area = 4. * np.pi * (diameter_ap / 2.) ** 2

        x_map = np.tile(np.array(range(0, image_size)),
                        (image_size, 1))
        y_map = x_map.T
        r_square_map = ((x_map - (image_size - 1) / 2) ** 2
                        + (y_map - (image_size - 1) / 2) ** 2)
        self.radius_map = np.sqrt(r_square_map)
        self.r_au = self.radius_map[np.newaxis, :, :] * self.au_pix[:, np.newaxis, np.newaxis]

    def create_star(self,
                    temp_star: float,  # temperature of the host star in Kelvin
                    ) -> None:
        self.flux_star = black_body(mode='star',
                                    bins=self.wl_bins,
                                    width=self.wl_bin_widths,
                                    temp=temp_star,
                                    radius=self.radius_star,
                                    distance=self.dist_star)

        # angular extend of the star disk in rad
        ang_star = self.radius_star * 0.00465 / self.dist_star * np.pi / (180 * 3600)
        bl_mat = (self.bl_x ** 2 + self.bl_y ** 2) ** 0.5
        self.b_star = np.nan_to_num(
            np.divide(
                2 * self.flux_star[:, np.newaxis, np.newaxis]
                * jn(1, 2 * np.pi * bl_mat[np.newaxis, :] * ang_star / self.wl_bins[:, np.newaxis, np.newaxis]),
                2 * np.pi * bl_mat[np.newaxis, :] * ang_star / self.wl_bins[:, np.newaxis, np.newaxis]))  # Eq (11)
        for i in range(self.b_star.shape[0]):
            np.fill_diagonal(self.b_star[i], self.flux_star[i])

        # derivative of the Bessel function needed for Eqs (17) & (18)
        a = 2 * np.pi * ang_star / self.wl_bins
        self.db_star_dx = np.swapaxes(np.nan_to_num(np.array(
            [2 * self.flux_star[:, np.newaxis]
             * ((self.col_pos[j, 0] - self.col_pos[:, 0]) / bl_mat[j, :] ** 2)[np.newaxis, :]
             * (0.5 * (jn(0, a[:, np.newaxis] * bl_mat[np.newaxis, j, :])
                       - jn(2, a[:, np.newaxis] * bl_mat[np.newaxis, j, :]))
                - jn(1, a[:, np.newaxis] * bl_mat[np.newaxis, j, :])
                / (a[:, np.newaxis] * bl_mat[np.newaxis, j, :]))
             for j in range(self.num_a)])), 0, 1)

        self.db_star_dy = np.swapaxes(np.nan_to_num(np.array(
            [2 * self.flux_star[:, np.newaxis]
             * ((self.col_pos[j, 1] - self.col_pos[:, 1]) / bl_mat[j, :] ** 2)[np.newaxis, :]
             * (0.5 * (jn(0, a[:, np.newaxis] * bl_mat[np.newaxis, j, :])
                       - jn(2, a[:, np.newaxis] * bl_mat[np.newaxis, j, :]))
                - jn(1, a[:, np.newaxis] * bl_mat[np.newaxis, j, :])
                / (a[:, np.newaxis] * bl_mat[np.newaxis, j, :]))
             for j in range(self.num_a)])), 0, 1)

    def create_planet(self,
                      temp_planet: float,  # planet temperature in Kelvin
                      radius_planet: float,  # planet radius in earth radii
                      flux_planet: np.ndarray = None,  # substitute flux input in ph m-2 s-1
                      ) -> None:
        if flux_planet is None:
            self.flux_planet = black_body(mode='planet',
                                          bins=self.wl_bins,
                                          width=self.wl_bin_widths,
                                          temp=temp_planet,
                                          radius=radius_planet,
                                          distance=self.dist_star)
        else:
            self.flux_planet = flux_planet

    def create_localzodi(self,
                         lat_star: float,  # ecliptic latitude of the target star
                         ) -> None:
        long = 3 / 4 * np.pi
        lat = lat_star

        radius_sun_au = 0.00465047  # in AU
        tau = 4e-8
        temp_eff = 265
        temp_sun = 5777
        a = 0.22

        b_tot = black_body(mode='wavelength',
                           bins=self.wl_bins,
                           width=self.wl_bin_widths,
                           temp=temp_eff) + a \
                * black_body(mode='wavelength',
                             bins=self.wl_bins,
                             width=self.wl_bin_widths,
                             temp=temp_sun) \
                * (radius_sun_au / 1.5) ** 2
        self.flux_localzodi = tau * b_tot * np.sqrt(
            np.pi / np.arccos(np.cos(long) * np.cos(lat)) /
            (np.sin(lat) ** 2
             + (0.6 * (self.wl_bins / 11e-6) ** (-0.4) * np.cos(lat)) ** 2)
        )

    def create_exozodi(self,
                       l_sun: float,  # stellar luminosity in solar luminosities
                       z: float,  # zodi level: the exozodi dust is z-times denser than the localzodi dust
                       ) -> None:
        # calculate the parameters required by Kennedy2015
        alpha = 0.34
        r_in = 0.034422617777777775 * np.sqrt(l_sun)
        r_0 = np.sqrt(l_sun)
        sigma_zero = 7.11889e-8  # Sigma_{m,0} from Kennedy+2015 (doi:10.1088/0067-0049/216/2/23)

        # identify all pixels where the radius is larges than the inner radius by Kennedy+2015
        r_cond = ((self.r_au >= r_in)
                  & (self.r_au <= self.image_size / 2 * self.au_pix[:, np.newaxis, np.newaxis]))

        # calculate the temperature at all pixel positions according to Kennedy2015 Eq. 2
        temp_map = np.where(r_cond,
                            278.3 * (l_sun ** 0.25) / np.sqrt(self.r_au), 0)

        # calculate the Sigma (Eq. 3) in Kennedy2015 and set everything inside the inner radius to 0
        sigma = np.where(r_cond,
                         sigma_zero * z *
                         (self.r_au / r_0) ** (-alpha), 0)

        # get the black body radiation emitted by the interexoplanetary dust
        f_nu_disk = black_body(bins=self.wl_bins[:, np.newaxis, np.newaxis],
                               width=self.wl_bin_widths[:, np.newaxis, np.newaxis],
                               temp=temp_map,
                               mode='wavelength') \
                    * sigma * self.rad_pix[:, np.newaxis, np.newaxis] ** 2

        ap = np.where(self.radius_map <= self.image_size / 2, 1, 0)
        flux_map_exozodi = f_nu_disk * ap

        sampling_rate_rad = self.rad_pix

        ez_fft = np.fft.fftshift(np.fft.fft2(flux_map_exozodi))
        r_rad_fft = np.fft.fftshift(np.fft.fftfreq(self.image_size, sampling_rate_rad[:, np.newaxis]))

        bl_x_fft = 2 * np.pi * self.bl_x[np.newaxis, :, :] / self.wl_bins[:, np.newaxis, np.newaxis]
        bl_y_fft = 2 * np.pi * self.bl_y[np.newaxis, :, :] / self.wl_bins[:, np.newaxis, np.newaxis]

        bl_x_pix = np.zeros_like(bl_x_fft)
        bl_y_pix = np.zeros_like(bl_y_fft)
        self.b_ez = np.zeros_like(bl_x_fft)
        for k in range(bl_x_fft.shape[0]):
            for i in range(bl_x_fft.shape[1]):
                for j in range(bl_x_fft.shape[2]):
                    bl_x_pix[k, i, j] = find_nearest_idx(r_rad_fft[k, :], bl_x_fft[k, i, j])
                    bl_y_pix[k, i, j] = find_nearest_idx(r_rad_fft[k, :], bl_y_fft[k, i, j])
                    self.b_ez[k, i, j] = ez_fft[k,
                                           int(find_nearest_idx(r_rad_fft[k, :], bl_x_fft[k, i, j])),
                                           int(find_nearest_idx(r_rad_fft[k, :], bl_y_fft[k, i, j]))
                    ]

    def response(self) -> None:
        theta_x = np.linspace(-1e-6, 1e-6, 200)[np.newaxis, :]
        theta_y = np.linspace(-1e-6, 1e-6, 200)[np.newaxis, :]
        self.R = np.sum(np.array([
            np.sum(np.array([
                np.cos(self.phi[j] - self.phi[k])
                * np.cos(
                    2 * np.pi / self.wl_bins[:, np.newaxis]
                    * (self.bl_x[j, k] * theta_x + self.bl_y[j, k] * theta_y))
                - np.sin(self.phi[j] - self.phi[k])
                * np.sin(
                    2 * np.pi / self.wl_bins[:, np.newaxis]
                    * (self.bl_x[j, k] * theta_x + self.bl_y[j, k] * theta_y))
                for k in range(self.num_a)]), axis=0)
            for j in range(self.num_a)]), axis=0)  # Eq (5)

    def stellar_leakage(self) -> None:
        # stellar sensitivity coefficients
        self.c_a_star = np.swapaxes(np.array(
            [2 * self.A[j]
             * np.array(
                [self.A[k] * np.cos(self.phi[j] - self.phi[k]) * self.b_star[:, j, k] for k in range(self.num_a)]
            ).sum(axis=0) for j in range(self.num_a)]), 0, 1)  # Eq (15), linear Taylor coefficient wrt amplitude

        self.c_phi_star = np.swapaxes(np.array(
            [-2 * self.A[j]
             * np.array([
                self.A[k] * np.sin(self.phi[j] - self.phi[k]) * self.b_star[:, j, k]
                for k in range(self.num_a) if (k != j)]
            ).sum(axis=0) for j in range(self.num_a)]), 0, 1)  # Eq (16)

        self.c_aa_star = np.swapaxes(np.array(
            [self.A[j] * self.A * np.cos(self.phi[j] - self.phi) * self.b_star[:, j, :] for j in range(self.num_a)]
        ), 0, 1)  # Eq (19)


        # Eq (20)
        self.c_aphi_star = np.swapaxes(np.array(
            [-2 * self.A[j] * self.A * np.sin(self.phi[j] - self.phi) * self.b_star[:, j, :] for j in range(self.num_a)]
        ), 0, 1)
        c_aphi_diag_star = np.swapaxes(np.array(
            [-2 * self.A[j]
             * np.array([
                self.A[l] * np.sin(self.phi[j] - self.phi[l]) * self.b_star[:, j, l]
                for l in range(self.num_a)]
            ).sum(axis=0) for j in range(self.num_a)]), 0, 1)
        for i in range(self.c_aphi_star.shape[0]):
            np.fill_diagonal(self.c_aphi_star[i, ], c_aphi_diag_star[i, ])

        # Eq (21)
        self.c_phiphi_star = np.swapaxes(np.array(
            [self.A[j] * self.A * np.cos(self.phi[j] - self.phi) * self.b_star[:, j, :] for j in range(self.num_a)]
        ), 0, 1)
        c_phiphi_diag_star = np.swapaxes(np.array(
            [-self.A[j] * np.array([
                self.A[l] * np.cos(self.phi[j] - self.phi[l]) * self.b_star[:, j, l]
                for l in range(self.num_a) if (l != j)]
            ).sum(axis=0) for j in range(self.num_a)]), 0, 1)
        for i in range(self.c_phiphi_star.shape[0]):
            np.fill_diagonal(self.c_phiphi_star[i, ], c_phiphi_diag_star[i, ])

        self.c_x_star = np.swapaxes(np.array([2 * np.array([
            self.A[j] * self.A[k] * np.cos(self.phi[j] - self.phi[k]) * self.db_star_dx[:, j, k]
            for k in range(self.num_a)]
        ).sum(axis=0) for j in range(self.num_a)]), 0, 1)  # Eq (17)

        self.c_y_star = np.swapaxes(np.array([2 * np.array([
            self.A[j] * self.A[k] * np.cos(self.phi[j] - self.phi[k]) * self.db_star_dy[:, j, k]
            for k in range(self.num_a)]
        ).sum(axis=0) for j in range(self.num_a)]), 0, 1)  # Eq (18)

        self.c_thetatheta_star = self.c_aa_star

    def exozodi_leakage(self) -> None:
        self.c_a_ez = np.swapaxes(np.array(
            [2 * self.A[j]
             * np.array(
                [self.A[k] * np.cos(self.phi[j] - self.phi[k]) * self.b_ez[:, j, k] for k in range(self.num_a)]
            ).sum(axis=0) for j in range(self.num_a)]), 0, 1)  # Eq (15)

        self.c_phi_ez = np.swapaxes(np.array(
            [-2 * self.A[j]
             * np.array([
                self.A[k] * np.sin(self.phi[j] - self.phi[k]) * self.b_ez[:, j, k]
                for k in range(self.num_a) if (k != j)]
            ).sum(axis=0) for j in range(self.num_a)]), 0, 1)  # Eq (16)

        self.c_aa_ez = np.swapaxes(np.array(
            [self.A[j] * self.A * np.cos(self.phi[j] - self.phi) * self.b_ez[:, j, :] for j in range(self.num_a)]
        ), 0, 1)  # Eq (19)

        # Eq (20)
        self.c_aphi_ez = np.swapaxes(np.array(
            [-2 * self.A[j] * self.A * np.sin(self.phi[j] - self.phi) * self.b_ez[:, j, :] for j in range(self.num_a)]
        ), 0, 1)
        c_aphi_diag_ez = np.swapaxes(np.array(
            [-2 * self.A[j]
             * np.array([self.A[l] * np.sin(self.phi[j] - self.phi[l]) * self.b_ez[:, j, l] for l in range(self.num_a)]
                        ).sum(axis=0) for j in range(self.num_a)]), 0, 1)
        for i in range(self.c_aphi_ez.shape[0]):
            np.fill_diagonal(self.c_aphi_ez[i, ], c_aphi_diag_ez[i, ])

        # Eq (21)
        self.c_phiphi_ez = np.swapaxes(np.array(
            [self.A[j] * self.A * np.cos(self.phi[j] - self.phi) * self.b_ez[:, j, :] for j in range(self.num_a)]
        ), 0, 1)
        c_phiphi_diag_ez = np.swapaxes(np.array(
            [-self.A[j] * np.array([
                self.A[l] * np.cos(self.phi[j] - self.phi[l]) * self.b_ez[:, j, l]
                for l in range(self.num_a) if (l != j)]
            ).sum(axis=0) for j in range(self.num_a)]), 0, 1)
        for i in range(self.c_phiphi_ez.shape[0]):
            np.fill_diagonal(self.c_phiphi_ez[i, ], c_phiphi_diag_ez[i, ])


    def localzodi_leakage(self) -> None:
        self.c_a_lz = 2 * self.flux_localzodi[:, np.newaxis] * self.A[np.newaxis, :] ** 2 * self.omega[:, np.newaxis]

    def sensitivity_coefficients(self) -> None:
        self.stellar_leakage()
        self.exozodi_leakage()
        self.localzodi_leakage()

        self.c_a = self.c_a_star + self.c_a_ez + self.c_a_lz
        self.c_phi = self.c_phi_star + self.c_phi_ez
        self.c_x = self.c_x_star
        self.c_y = self.c_y_star

        self.c_aphi = self.c_aphi_star + self.c_aphi_ez

        self.c_aa = self.c_aa_star + self.c_aa_ez
        self.c_phiphi = self.c_phiphi_star + self.c_phiphi_ez
        self.c_thetatheta = self.c_thetatheta_star

    def planet_signal(self,
                      separation_planet: float,  # separation of target planet from host star in AU
                      ) -> None:
        theta = separation_planet * 1.496e11 / (self.dist_star * 3.086e16)  # theta_x/y in Fig. 1
        self.phi_rot = np.linspace(0, 2 * np.pi, self.n_sampling_rot)
        self.theta_x = -theta * np.cos(self.phi_rot)
        self.theta_y = theta * np.sin(self.phi_rot)

        time_per_bin = self.t_rot / len(self.phi_rot)

        # create planet signal via Eq (9)
        self.n_planet = np.swapaxes(np.array(
            [self.flux_planet
             * np.array(
                [np.array(
                    [self.A[j] * self.A[k]
                     * (np.cos(self.phi[j] - self.phi[k])
                        * np.cos(
                                2 * np.pi / self.wl_bins
                                * (self.bl_x[j, k] * self.theta_x[l] + self.bl_y[j, k] * self.theta_y[l])
                            )
                        - np.sin(self.phi[j] - self.phi[k])
                        * np.sin(
                                2 * np.pi / self.wl_bins
                                * (self.bl_x[j, k] * self.theta_x[l] + self.bl_y[j, k] * self.theta_y[l])))
                     for k in range(self.num_a)]).sum(axis=0)
                 for j in range(self.num_a)]).sum(axis=0)
             for l in range(len(self.phi_rot))]), 0, 1)

        # Fourier transform of planet signal equivalent to Eq (33)
        nf = rfft(self.n_planet)
        nfft = nf / self.n_sampling_rot
        nfft[:, 1:] *= 2

        # creation of template function
        # removal of even components and DC
        nfft_odd = nfft
        nfft_odd[:, :2] = 0

        # transform back into time domain
        self.planet_template = np.zeros((self.wl_bins.shape[0], len(self.phi_rot)))
        for k in range(self.wl_bins.shape[0]):
            ret = []
            for n in range(len(nfft_odd[k, :])):
                s = nfft_odd[k, n] * np.exp(1j * 2 * np.pi * n * self.phi_rot / (2 * np.pi))
                ret.append(s)
            ret = np.array(ret).sum(axis=0).real
            self.planet_template[k, :] = ret

            # normalize the template function to rms of one
            self.planet_template[k, :] = self.planet_template[k, :] / np.std(self.planet_template[k, :])

        self.photon_rates.loc['signal', 'nchop'] = np.abs(
            (time_per_bin * self.planet_template * self.n_planet).sum(axis=1)
        ) / self.t_rot

        # chopped planet signal
        self.n_planet_r = np.swapaxes(np.array(
            [self.flux_planet
             * np.array(
                [np.array(
                    [self.A[j] * self.A[k]
                     * (np.cos(self.phi_r[j] - self.phi_r[k])
                        * np.cos(
                                2 * np.pi / self.wl_bins
                                * (self.bl_x[j, k] * self.theta_x[l] + self.bl_y[j, k] * self.theta_y[l])
                            )
                        - np.sin(self.phi_r[j] - self.phi_r[k])
                        * np.sin(
                                2 * np.pi / self.wl_bins
                                * (self.bl_x[j, k] * self.theta_x[l] + self.bl_y[j, k] * self.theta_y[l])))
                     for k in range(self.num_a)]).sum(axis=0)
                 for j in range(self.num_a)]).sum(axis=0)
             for l in range(len(self.phi_rot))]), 0, 1)

        self.n_planet_chop = 0.5 * (self.n_planet - self.n_planet_r)

        # Fourier transform of planet signal equivalent to Eq (33)
        nf_chop = rfft(self.n_planet_chop)
        nfft_chop = nf_chop / self.n_sampling_rot
        nfft_chop[:, 1:] *= 2

        # creation of template function
        # removal of even components and DC
        nfft_odd_chop = nfft_chop
        nfft_odd_chop[:, :2] = 0

        # transform back into time domain
        self.planet_template_chop = np.zeros((self.wl_bins.shape[0], len(self.phi_rot)))
        for k in range(self.wl_bins.shape[0]):
            ret = []
            for n in range(len(nfft_odd[k, :])):
                s = nfft_odd_chop[k, n] * np.exp(1j * 2 * np.pi * n * self.phi_rot / (2 * np.pi))
                ret.append(s)
            ret = np.array(ret).sum(axis=0).real
            self.planet_template_chop[k, :] = ret

            # normalize the template function to rms of one
            self.planet_template_chop[k, :] = self.planet_template_chop[k, :] / np.std(self.planet_template_chop[k, :])

        self.photon_rates.loc['signal', 'chop'] = np.abs(
            (time_per_bin * self.planet_template_chop * self.n_planet_chop).sum(axis=1)
        ) / self.t_rot


    def fundamental_noise(self) -> None:
        n_0_star = np.array([
            np.array([self.A[j] * self.A[k] * np.cos(self.phi[j] - self.phi[k]) * self.b_star[:, j, k]
                      for k in range(self.num_a)]).sum(axis=0)
            for j in range(self.num_a)]).sum(axis=0)
        self.photon_rates.loc['pn_sgl', 'nchop'] = np.sqrt(n_0_star / self.t_rot)

        n_0_lz = (
                self.flux_localzodi[:, np.newaxis] * self.A[np.newaxis, :] ** 2 * self.omega[:, np.newaxis]
        ).sum(axis=1)
        self.photon_rates.loc['pn_lz', 'nchop'] = np.sqrt(n_0_lz / self.t_rot)

        n_0_ez = np.array([
            np.array([self.A[j] * self.A[k] * np.cos(self.phi[j] - self.phi[k]) * self.b_ez[:, j, k]
                      for k in range(self.num_a)]).sum(axis=0)
            for j in range(self.num_a)]).sum(axis=0)
        self.photon_rates.loc['pn_ez', 'nchop'] = np.sqrt(n_0_ez / self.t_rot)

    def pn_dark_current(self,
                        detector: str,
                        dark_current_pix: Union[float, type(None)],
                        ) -> None:
        if detector == 'MIRI':
            dark_current_pix = 0.2
            self.photon_rates.loc['pn_dc', 'nchop'] = (np.sqrt(dark_current_pix * self.pix_per_wl)
                                                       * np.ones((self.wl_bins.shape[0])))
        elif detector == 'manual':
            if dark_current_pix == None:
                raise ValueError('Dark current per pixel needs to be specified in manual mode')
            self.photon_rates.loc['pn_dc', 'nchop'] = (np.sqrt(dark_current_pix * self.pix_per_wl)
                                                       * np.ones((self.wl_bins.shape[0])))
        else:
            raise ValueError('Unkown detector type')

    def pn_thermal_background_detector(self,
                                       detector: str,
                                       det_temp: float,  # temperature of the detector environment in K
                                       wl_resolution: int = 200,  # number of wavelength bins simulated for the thermal
                                                                  # background
                                       ) -> None:
        h = 6.62607e-34
        k = 1.380649e-23
        c = 2.99792e+8
        if detector == 'MIRI':
            # pitch - gap
            area_pixel = ((25 - 2) * 1e-6) ** 2
            det_wl_min = 5e-6
            det_wl_max = 28e-6
        else:
            raise ValueError('Unkown detector type')
        wl_bins = np.linspace(start=det_wl_min, stop=det_wl_max, num=wl_resolution, endpoint=True)
        B_photon = 2 * c / wl_bins ** 4 / (np.exp(h * c / (wl_bins * k * det_temp)) - 1)
        B_photon_int = np.trapz(y=B_photon, x=wl_bins)
        thermal_emission_det = 2 * np.pi * area_pixel * B_photon_int

        self.photon_rates.loc['pn_tbd', 'nchop'] = (np.sqrt(thermal_emission_det * self.pix_per_wl)
                                                    * np.ones((self.wl_bins.shape[0])))

    def sn_nchop(self,
                 n_cpu: int,
                 rms_mode: str,
                 n_sampling_max: int = 10000000,
                 d_a_rms: Union[float, type(None)] = None,
                 d_phi_rms: Union[float, type(None)] = None,
                 d_pol_rms: Union[float, type(None)] = None,
                 d_x_rms: Union[float, type(None)] = None,
                 d_y_rms: Union[float, type(None)] = None):

        mp_args = []
        for i in range(self.wl_bins.shape[0]):
            mp_args.append({'rms_mode': rms_mode,
                            'wl': self.wl_bins[i],
                            't_rot': self.t_rot,
                            'num_a': self.num_a,
                            'flux_star': self.flux_star[i],
                            'A': self.A,
                            'c_a': self.c_a[i, :],
                            'c_phi': self.c_phi[i, :],
                            'c_x': self.c_x[i, :],
                            'c_y': self.c_y[i, :],
                            'c_aa': self.c_aa[i, :, :],
                            'c_phiphi': self.c_phiphi[i, :, :],
                            'c_aphi': self.c_aphi[i, :, :],
                            'c_thetatheta': self.c_thetatheta[i, :, :],
                            'template': self.planet_template[i, :],
                            'n_sampling_max': n_sampling_max,
                            'd_a_rms': d_a_rms,
                            'd_phi_rms': d_phi_rms,
                            'd_pol_rms': d_pol_rms,
                            'd_x_rms': d_x_rms,
                            'd_y_rms': d_y_rms,
                            'pink_noise_co': self.pink_noise_co})
        if n_cpu == 1:
            res = []
            for i in tqdm(range(self.wl_bins.shape[0])):
                rr = instrumental_noise_single_wav_nchop(mp_args[i])
                res.append(rr)
        else:
            # collect arguments for multiprocessing
            pool = mp.Pool(n_cpu)
            results = pool.map(instrumental_noise_single_wav_nchop, mp_args)
            res = []
            for wl in self.wl_bins:
                for r in results:
                    if np.round(r['wl'], 10) == np.round(wl, 10):
                        res.append(r)

        self.save_to_results(data=res,
                             column_results='nchop')

        self.photon_rates.loc['pn', 'nchop'] = np.sqrt(self.photon_rates.loc['pn_sgl', 'nchop'] ** 2
                                                       + self.photon_rates.loc['pn_ez', 'nchop'] ** 2
                                                       + self.photon_rates.loc['pn_lz', 'nchop'] ** 2
                                                       + self.photon_rates.loc['pn_dc', 'nchop'] ** 2
                                                       + self.photon_rates.loc['pn_tbd', 'nchop'] ** 2
                                                       + self.photon_rates.loc['pn_pa', 'nchop'] ** 2
                                                       + self.photon_rates.loc['pn_snfl', 'nchop'] ** 2)

        self.photon_rates.loc['noise', 'nchop'] = np.sqrt(self.photon_rates.loc['pn', 'nchop'] ** 2
                                                          + self.photon_rates.loc['sn', 'nchop'] ** 2)

        self.photon_rates.loc['snr', 'nchop'] = (self.photon_rates.loc['signal', 'nchop']
                                                 / self.photon_rates.loc['noise', 'nchop'])

    def save_to_results(self,
                        data,
                        column_results):
        for k in data[0].keys():
            self.photon_rates.loc[k, column_results] = []
        for d in data:
            for k in d.keys():
                self.photon_rates.loc[k, column_results].append(d[k])
        for k in data[0].keys():
            self.photon_rates.loc[k, column_results] = np.array(self.photon_rates.loc[k, column_results])

    def sn_chop(self,
                n_cpu: int,
                rms_mode: str,
                n_sampling_max: int = 10000000,
                d_a_rms: Union[float, type(None)] = None,
                d_phi_rms: Union[float, type(None)] = None,
                d_pol_rms: Union[float, type(None)] = None,
                ):
        mp_args = []
        for i in range(self.wl_bins.shape[0]):
            mp_args.append({'A': self.A,
                            'wl': self.wl_bins[i],
                            'num_a': self.num_a,
                            'planet_template_chop': self.planet_template_chop[i, :],
                            'c_phi': self.c_phi[i, :],
                            'c_aphi': self.c_aphi[i, :, :],
                            'c_aa' : self.c_aa[i, :, :],
                            'c_phiphi': self.c_phiphi[i, :, :],
                            'rms_mode': rms_mode,
                            'n_sampling_max': n_sampling_max,
                            't_rot': self.t_rot,
                            'd_a_rms': d_a_rms,
                            'd_phi_rms': d_phi_rms,
                            'd_pol_rms': d_pol_rms,
                            'pink_noise_co': self.pink_noise_co,
                            'flux_star': self.flux_star[i]
                            })
        if n_cpu == 1:
            res = []
            for i in tqdm(range(self.wl_bins.shape[0])):
                rr = instrumental_noise_single_wav_chop(mp_args[i])
                res.append(rr)
            print(res)
        else:
            # collect arguments for multiprocessing
            pool = mp.Pool(n_cpu)
            results = pool.map(instrumental_noise_single_wav_chop, mp_args)
            res = []
            for wl in self.wl_bins:
                for r in results:
                    if np.round(r['wl'], 10) == np.round(wl, 10):
                        res.append(r)

        self.save_to_results(data=res,
                             column_results='chop')

        self.photon_rates.loc['pn_sgl', 'chop'] = self.photon_rates.loc['pn_sgl', 'nchop']
        self.photon_rates.loc['pn_ez', 'chop'] = self.photon_rates.loc['pn_ez', 'nchop']
        self.photon_rates.loc['pn_lz', 'chop'] = self.photon_rates.loc['pn_lz', 'nchop']
        self.photon_rates.loc['pn_dc', 'chop'] = self.photon_rates.loc['pn_dc', 'nchop']
        self.photon_rates.loc['pn_tbd', 'chop'] = self.photon_rates.loc['pn_tbd', 'nchop']

        self.photon_rates.loc['pn', 'chop'] = np.sqrt(self.photon_rates.loc['pn_sgl', 'chop'] ** 2
                                                      + self.photon_rates.loc['pn_ez', 'chop'] ** 2
                                                      + self.photon_rates.loc['pn_lz', 'chop'] ** 2
                                                      + self.photon_rates.loc['pn_dc', 'chop'] ** 2
                                                      + self.photon_rates.loc['pn_tbd', 'chop'] ** 2
                                                      + self.photon_rates.loc['pn_pa', 'chop'] ** 2
                                                      + self.photon_rates.loc['pn_snfl', 'chop'] ** 2)

        self.photon_rates.loc['noise', 'chop'] = np.sqrt(self.photon_rates.loc['pn', 'chop'] ** 2
                                                         + self.photon_rates.loc['sn', 'chop'] ** 2)

        self.photon_rates.loc['snr', 'chop'] = (self.photon_rates.loc['signal', 'chop']
                                                / self.photon_rates.loc['noise', 'chop'])

    def cleanup(self):
        if self.wl_bins.shape[0] == 1:
            for i in self.photon_rates.index:
                if type(self.photon_rates.loc[i, 'nchop']) == np.ndarray:
                    self.photon_rates.loc[i, 'nchop'] = self.photon_rates.loc[i, 'nchop'][0]
                if type(self.photon_rates.loc[i, 'chop']) == np.ndarray:
                    self.photon_rates.loc[i, 'chop'] = self.photon_rates.loc[i, 'chop'][0]


    def run_multiwav(self,
                     temp_star,
                     temp_planet,
                     radius_planet,
                     lat_star,
                     l_sun,
                     z,
                     dark_current_pix,
                     det_temp,
                     rms_mode,
                     n_cpu,
                     separation_planet) -> None:
        self.create_star(temp_star=temp_star)
        self.create_planet(temp_planet=temp_planet,
                           radius_planet=radius_planet)
        self.create_localzodi(lat_star=lat_star)
        self.create_exozodi(l_sun=l_sun,
                            z=z)

        self.response()
        self.sensitivity_coefficients()

        self.planet_signal(separation_planet=separation_planet)

        self.fundamental_noise()
        self.pn_dark_current(detector='manual',
                             dark_current_pix=dark_current_pix)
        self.pn_thermal_background_detector(detector='MIRI',
                                            det_temp=det_temp)

        self.sn_nchop(n_cpu=n_cpu,
                      rms_mode=rms_mode)

        self.sn_chop(n_cpu=n_cpu,
                     rms_mode=rms_mode)

        self.cleanup()

    def run_singlewav_chop(self,
                           temp_star,
                           temp_planet,
                           radius_planet,
                           lat_star,
                           l_sun,
                           z,
                           dark_current_pix,
                           det_temp,
                           rms_mode,
                           n_cpu,
                           separation_planet,
                           d_a_rms,
                           d_phi_rms,
                           d_pol_rms):

        self.create_star(temp_star=temp_star)
        self.create_planet(temp_planet=temp_planet,
                           radius_planet=radius_planet)
        self.create_localzodi(lat_star=lat_star)
        self.create_exozodi(l_sun=l_sun,
                            z=z)

        self.response()
        self.sensitivity_coefficients()

        self.planet_signal(separation_planet=separation_planet)

        self.fundamental_noise()
        self.pn_dark_current(detector='manual',
                             dark_current_pix=dark_current_pix)
        self.pn_thermal_background_detector(detector='MIRI',
                                            det_temp=det_temp)

        self.sn_chop(n_cpu=n_cpu,
                     rms_mode=rms_mode,
                     d_a_rms=d_a_rms,
                     d_phi_rms=d_phi_rms,
                     d_pol_rms=d_pol_rms)

        self.cleanup()


def instrumental_noise_single_wav_nchop(mp_arg) -> dict:
    rms_mode = mp_arg['rms_mode']
    wl = mp_arg['wl']
    num_a = mp_arg['num_a']
    flux_star = mp_arg['flux_star']
    A = mp_arg['A']
    c_a = mp_arg['c_a']
    c_phi = mp_arg['c_phi']
    c_x = mp_arg['c_x']
    c_y = mp_arg['c_y']
    c_aa = mp_arg['c_aa']
    c_phiphi = mp_arg['c_phiphi']
    c_aphi = mp_arg['c_aphi']
    c_thetatheta = mp_arg['c_thetatheta']
    template = mp_arg['template']
    n_sampling_max = mp_arg['n_sampling_max']
    d_a_rms = mp_arg['d_a_rms']
    d_phi_rms = mp_arg['d_phi_rms']
    d_pol_rms = mp_arg['d_pol_rms']
    d_x_rms = mp_arg['d_x_rms']
    d_y_rms = mp_arg['d_y_rms']
    t_rot = mp_arg['t_rot']
    pink_noise_co = mp_arg['pink_noise_co']

    if rms_mode == 'lay':
        d_a_rms_0 = 0.001
        d_a_rms = d_a_rms_0 * (wl / 10e-6) ** (-1.5)

        d_phi_rms_0 = 0.001
        d_phi_rms = d_phi_rms_0 * (wl / 10e-6) ** (-1)

        d_pol_rms = 0.001

        d_x_rms = 0.01

        d_y_rms = 0.01
    elif rms_mode == 'static':
        if (d_a_rms is None) or (d_phi_rms is None) or (d_pol_rms is None) or (d_x_rms is None) or (d_y_rms is None):
            raise ValueError('RMS values need to be specified in static mode')
    elif rms_mode == 'wavelength':
        if (d_a_rms is None) or (d_phi_rms is None) or (d_pol_rms is None) or (d_x_rms is None) or (d_y_rms is None):
            raise ValueError('RMS values need to be specified in wavelength mode')
        d_a_rms = d_a_rms * (wl / 10e-6) ** (-1.5)
        d_phi_rms = d_phi_rms * (wl / 10e-6) ** (-1)
    else:
        raise ValueError('RMS mode not recongnized')

    d_a_co = pink_noise_co
    d_phi_co = pink_noise_co
    d_pol_co = pink_noise_co
    d_x_co = 0.64e-3
    d_y_co = 0.64e-3

    comp_factor = 1  # factor for the conversion of periodigram to Fourier components

    # create pink noise PSDs
    d_a_freq = np.arange(0, n_sampling_max + 1) * 1 / t_rot
    d_a_psd = np.append([0], 1 / d_a_freq[1:])
    d_a_psd = np.array(
        [d_a_psd * ((d_a_rms / 1) ** 2 / (np.sum(d_a_psd / t_rot)
                                          + np.log(d_a_co / d_a_freq[int(n_sampling_max)])))
         for j in range(num_a)])
    avg_d_a_2 = comp_factor / t_rot * d_a_psd.sum(axis=1)  # by parseval theorem
    d_a_b_2 = comp_factor / t_rot * d_a_psd

    d_phi_freq = np.arange(0, n_sampling_max + 1) * 1 / t_rot
    d_phi_psd = np.append([0], 1 / d_phi_freq[1:])
    d_phi_psd = d_phi_psd * (
            d_phi_rms ** 2 / (np.sum(d_phi_psd / t_rot) + np.log(d_phi_co / d_phi_freq[int(n_sampling_max)])))
    avg_d_phi_2 = comp_factor / t_rot * d_phi_psd.sum()
    d_phi_b_2 = comp_factor / t_rot * d_phi_psd

    d_pol_freq = np.arange(0, n_sampling_max + 1) * 1 / t_rot
    d_pol_psd = np.append([0], 1 / d_pol_freq[1:])
    d_pol_psd = d_pol_psd * (
            d_pol_rms ** 2 / (np.sum(d_pol_psd / t_rot) + np.log(d_pol_co / d_pol_freq[int(n_sampling_max)])))
    avg_d_pol_2 = comp_factor / t_rot * d_pol_psd.sum()
    d_pol_b_2 = comp_factor / t_rot * d_pol_psd

    # create white noise PSD
    d_x_freq = np.arange(0, d_x_co, step=1 / t_rot)
    d_x_psd = np.ones_like(d_x_freq) * d_x_rms ** 2 * t_rot / len(d_x_freq - 1)
    d_x_psd[0] = 0
    d_x_b_2 = comp_factor / t_rot * d_x_psd

    d_y_freq = np.arange(0, d_y_co, step=1 / t_rot)
    d_y_psd = np.ones_like(d_y_freq) * d_y_rms ** 2 * t_rot / len(d_y_freq - 1)
    d_y_psd[0] = 0
    d_y_b_2 = comp_factor / t_rot * d_y_psd

    noise_nchop = {'wl': wl}

    dn_pol = (flux_star * A ** 2 * avg_d_pol_2).sum()
    noise_nchop['pn_pa'] = np.sqrt(dn_pol / t_rot)
    dn_null_floor = np.array([c_aa[j, j] * avg_d_a_2[j] + c_phiphi[j, j] * avg_d_phi_2 for j in range(num_a)]).sum()
    noise_nchop['pn_snfl'] = np.sqrt(dn_null_floor / t_rot)

    template_fft = rfft(template)
    template_fft = template_fft / len(template)

    # first order terms
    d_a_j_hat_2 = np.array([(np.abs(template_fft) ** 2 * d_a_b_2[j, :len(template_fft)]).sum() for j in range(num_a)])
    d_phi_j_hat_2 = np.array([(np.abs(template_fft) ** 2 * d_phi_b_2[:len(template_fft)]).sum() for j in range(num_a)])
    d_x_j_hat_2 = np.array([(np.abs(template_fft[:len(d_x_b_2)]) ** 2 * d_x_b_2).sum() for j in range(num_a)])
    d_y_j_hat_2 = np.array([(np.abs(template_fft[:len(d_y_b_2)]) ** 2 * d_y_b_2).sum() for j in range(num_a)])

    noise_nchop['sn_fo_a'] = np.sqrt((c_a ** 2 * d_a_j_hat_2).sum())

    noise_nchop['sn_fo_phi'] = np.sqrt((c_phi ** 2 * d_phi_j_hat_2).sum())

    noise_nchop['sn_fo_x'] = np.sqrt((c_x ** 2 * d_x_j_hat_2).sum())

    noise_nchop['sn_fo_y'] = np.sqrt((c_y ** 2 * d_y_j_hat_2).sum())

    noise_nchop['sn_fo'] = np.sqrt(noise_nchop['sn_fo_a'] ** 2
                                         + noise_nchop['sn_fo_phi'] ** 2
                                         + noise_nchop['sn_fo_x'] ** 2
                                         + noise_nchop['sn_fo_y'] ** 2)

    # second order terms
    nt = len(template_fft)
    d_a_b_2_f = np.vstack((np.concatenate((np.flip(d_a_b_2[0, :nt]), d_a_b_2[0, 1:], np.zeros(1))),
                           np.concatenate((np.flip(d_a_b_2[1, :nt]), d_a_b_2[1, 1:], np.zeros(1))),
                           np.concatenate((np.flip(d_a_b_2[2, :nt]), d_a_b_2[2, 1:], np.zeros(1))),
                           np.concatenate((np.flip(d_a_b_2[3, :nt]), d_a_b_2[3, 1:], np.zeros(1)))))

    d_a_j_hat_2_2 = np.array([
        (np.sum(
            np.abs(template_fft[1:]) ** 2 * (
                    #np.array([np.sum(d_a_b_2[j, 1:] * d_a_b_2_f[j, nt - r:-r - 1]) for r in range(1, nt)])
                    np.convolve(np.flip(d_a_b_2_f[j, 1:-2]), d_a_b_2[j, 1:], mode='valid')
                    + np.array([np.sum(d_a_b_2[j, 1:nt - r] * d_a_b_2[j, 1 + r:nt]) for r in range(1, nt)])
                    + 2 * np.array([d_a_b_2[j, r] * d_a_b_2[j, 0] for r in range(1, nt)])
            )
        )
         + (np.sum(d_a_b_2[j, 1:] ** 2) + d_a_b_2[j, 0] ** 2 / 2) * np.abs(template_fft[0]) ** 2)
        for j in range(num_a)])[0]

    d_a_j_a_k_hat_2 = d_a_j_hat_2_2 / 2
    d_a_hat_2 = np.ones((num_a, num_a)) * d_a_j_a_k_hat_2
    np.fill_diagonal(d_a_hat_2, d_a_j_hat_2_2)

    d_phi_b_2_f = np.concatenate((np.flip(d_phi_b_2[:nt]), d_phi_b_2[1:], np.zeros(1)))
    d_phi_j_hat_2_2 = (
            np.sum(
                np.abs(template_fft[1:]) ** 2
                * (
                        #np.array([np.sum(d_phi_b_2[1:] * d_phi_b_2_f[nt - r:-r - 1]) for r in range(1, nt)])
                        np.convolve(np.flip(d_phi_b_2_f[1:-2]), d_phi_b_2[1:], mode='valid')
                        + np.array([np.sum(d_phi_b_2[1:nt - r] * d_phi_b_2[1 + r:nt]) for r in range(1, nt)])
                        + np.array([d_phi_b_2[r] * d_phi_b_2[0] for r in range(1, nt)])
                )
            )
            + (np.sum(d_phi_b_2[1:] ** 2) + d_phi_b_2[0] ** 2 / 2) * np.abs(template_fft[0]) ** 2
    )
    d_phi_j_phi_k_hat_2 = d_phi_j_hat_2_2 / 2
    d_phi_hat_2 = np.ones((num_a, num_a)) * d_phi_j_phi_k_hat_2
    np.fill_diagonal(d_phi_hat_2, d_phi_j_hat_2_2)

    d_a_j_phi_k_hat_2 = np.array([0.5 * (np.sum(np.abs(template_fft[1:]) ** 2 * (
            # np.array([np.sum(d_a_b_2[j, 1:] * d_phi_b_2_f[nt - r:-r - 1]) for r in range(1, nt)])
            np.convolve(np.flip(d_phi_b_2_f[1:-2]), d_a_b_2[j, 1:], mode='valid')
            + np.array([np.sum(d_a_b_2[j, 1:nt - r] * d_phi_b_2[1 + r:nt]) for r in range(1, nt)])
            + np.array([d_a_b_2[j, 0] * d_phi_b_2[r] for r in range(1, nt)])))
                                         + (np.sum(d_a_b_2[j, 1:] * d_phi_b_2[1:]) + d_a_b_2[j, 0] * d_phi_b_2[
                0] / 2) * np.abs(template_fft[0]) ** 2) for j in range(num_a)])[0]
    d_a_phi_hat_2 = np.ones((num_a, num_a)) * d_a_j_phi_k_hat_2

    d_pol_b_2_f = np.concatenate((np.flip(d_pol_b_2[:nt]), d_pol_b_2[1:], np.zeros(1)))
    d_pol_j_hat_2_2 = (np.sum(np.abs(template_fft[1:]) ** 2 * (
            # np.array([np.sum(d_pol_b_2[1:] * d_pol_b_2_f[nt - r:-r - 1]) for r in range(1, nt)])
            np.convolve(np.flip(d_pol_b_2_f[1:-2]), d_pol_b_2[1:], mode='valid')
            + np.array([np.sum(d_pol_b_2[1:nt - r] * d_pol_b_2[1 + r:nt]) for r in range(1, nt)])
            + np.array([d_pol_b_2[r] * d_pol_b_2[0] for r in range(1, nt)])))
                       + (np.sum(d_pol_b_2[1:] ** 2) + d_pol_b_2[0] ** 2 / 2) * np.abs(template_fft[0]) ** 2)
    d_pol_j_pol_k_hat_2 = d_pol_j_hat_2_2 / 2
    d_pol_hat_2 = np.ones((num_a, num_a)) * d_pol_j_pol_k_hat_2
    np.fill_diagonal(d_pol_hat_2, d_pol_j_hat_2_2)

    noise_nchop['sn_so_aa'] = np.sqrt(np.sum(c_aa ** 2 * d_a_hat_2))

    noise_nchop['sn_so_phiphi'] = np.sqrt(np.sum(c_phiphi ** 2 * d_phi_hat_2))

    noise_nchop['sn_so_aphi'] = np.sqrt(np.sum(c_aphi ** 2 * d_a_phi_hat_2))

    noise_nchop['sn_so_polpol'] = np.sqrt(np.sum(c_thetatheta ** 2 * d_pol_hat_2))

    noise_nchop['sn_so'] = np.sqrt(
        noise_nchop['sn_so_aa'] ** 2
        + noise_nchop['sn_so_aphi'] ** 2
        + noise_nchop['sn_so_phiphi'] ** 2
        + noise_nchop['sn_so_polpol'] ** 2)

    noise_nchop['sn'] = np.sqrt(noise_nchop['sn_fo'] ** 2 + noise_nchop['sn_so'] ** 2)

    return noise_nchop


def instrumental_noise_single_wav_chop(mp_arg) -> dict:
    flux_star = mp_arg['flux_star']
    A = mp_arg['A']
    wl = mp_arg['wl']
    num_a = mp_arg['num_a']
    planet_template_chop = mp_arg['planet_template_chop']
    c_phi = mp_arg['c_phi']
    c_aphi = mp_arg['c_aphi']
    c_aa = mp_arg['c_aa']
    c_phiphi = mp_arg['c_phiphi']
    rms_mode = mp_arg['rms_mode']
    n_sampling_max = mp_arg['n_sampling_max']
    t_rot = mp_arg['t_rot']
    d_a_rms = mp_arg['d_a_rms']
    d_phi_rms = mp_arg['d_phi_rms']
    d_pol_rms = mp_arg['d_pol_rms']
    pink_noise_co = mp_arg['pink_noise_co']

    planet_template_c_fft = rfft(planet_template_chop)
    planet_template_c_fft = planet_template_c_fft / len(planet_template_chop)

    # create noise PSD
    if rms_mode == 'lay':
        d_a_rms_0 = 0.001
        d_a_rms = d_a_rms_0 * (wl / 10e-6) ** (-1.5)

        d_phi_rms_0 = 0.001
        d_phi_rms = d_phi_rms_0 * (wl / 10e-6) ** (-1)

        d_pol_rms = 0.001
    elif rms_mode == 'static':
        if (d_a_rms is None) or (d_phi_rms is None):
            raise ValueError('RMS values need to be specified in static mode')
    elif rms_mode == 'wavelength':
        if (d_a_rms is None) or (d_phi_rms is None):
            raise ValueError('RMS values need to be specified in wavelength mode')
        d_a_rms = d_a_rms * (wl / 10e-6) ** (-1.5)
        d_phi_rms = d_phi_rms * (wl / 10e-6) ** (-1)
    else:
        raise ValueError('RMS mode not recongnized')

    d_a_co = pink_noise_co
    d_phi_co = pink_noise_co
    d_pol_co = pink_noise_co

    comp_factor = 1

    # create pink noise PSDs
    d_a_freq = np.arange(0, n_sampling_max + 1) * 1 / t_rot
    d_a_psd = np.append([0], 1 / d_a_freq[1:])
    d_a_psd = np.array(
        [d_a_psd * ((d_a_rms / 1) ** 2 / (np.sum(d_a_psd / t_rot)
                                          + np.log(d_a_co / d_a_freq[int(n_sampling_max)])))
         for j in range(num_a)])
    avg_d_a_2 = comp_factor / t_rot * d_a_psd.sum(axis=1)  # by parseval theorem
    d_a_b_2 = comp_factor / t_rot * d_a_psd

    comp_factor = 1  # factor for the conversion of periodigram to Fourier components
    d_phi_freq = np.arange(0, n_sampling_max + 1) * 1 / t_rot
    d_phi_psd = np.append([0], 1 / d_phi_freq[1:])
    d_phi_psd = d_phi_psd * (
            d_phi_rms ** 2 / (np.sum(d_phi_psd / t_rot) + np.log(d_phi_co / d_phi_freq[int(n_sampling_max)])))
    avg_d_phi_2 = comp_factor / t_rot * d_phi_psd.sum()
    d_phi_b_2 = comp_factor / t_rot * d_phi_psd

    d_pol_freq = np.arange(0, n_sampling_max + 1) * 1 / t_rot
    d_pol_psd = np.append([0], 1 / d_pol_freq[1:])
    d_pol_psd = d_pol_psd * (
            d_pol_rms ** 2 / (np.sum(d_pol_psd / t_rot) + np.log(d_pol_co / d_pol_freq[int(n_sampling_max)])))
    avg_d_pol_2 = comp_factor / t_rot * d_pol_psd.sum()
    d_pol_b_2 = comp_factor / t_rot * d_pol_psd

    # noise contribution
    noise_chop = {'wl': wl}

    dn_pol = (flux_star * A ** 2 * avg_d_pol_2).sum()
    noise_chop['pn_pa'] = np.sqrt(dn_pol / t_rot)
    dn_null_floor = np.array([c_aa[j, j] * avg_d_a_2[j] + c_phiphi[j, j] * avg_d_phi_2 for j in range(num_a)]).sum()
    noise_chop['pn_snfl'] = np.sqrt(dn_null_floor / t_rot)

    # first order dphi
    d_phi_j_hat_2_chop = np.array([(np.abs(planet_template_c_fft) ** 2
                                    * d_phi_b_2[:len(planet_template_c_fft)]).sum() for j in range(num_a)])
    noise_chop['sn_fo_phi'] = np.sqrt((c_phi ** 2 * d_phi_j_hat_2_chop).sum())

    # second order dadphi
    nt = len(planet_template_c_fft)
    d_phi_b_2_f = np.concatenate((np.flip(d_phi_b_2[:nt]), d_phi_b_2[1:], np.zeros(1)))

    # Eq (45)
    d_a_j_phi_k_hat_2_chop = np.array([0.5 * (np.sum(np.abs(planet_template_c_fft[1:]) ** 2 * (
            np.convolve(np.flip(d_phi_b_2_f[1:-2]), d_a_b_2[j, 1:], mode='valid')
            + np.array([np.sum(d_a_b_2[j, 1:nt - r] * d_phi_b_2[1 + r:nt]) for r in range(1, nt)])
            + np.array([d_a_b_2[j, 0] * d_phi_b_2[r] for r in range(1, nt)])))
                                              + (np.sum(d_a_b_2[j, 1:] * d_phi_b_2[1:]) + d_a_b_2[j, 0] * d_phi_b_2[
                0] / 2) * np.abs(planet_template_c_fft[0]) ** 2) for j in range(1)])[0]
    d_a_phi_hat_2_chop = np.ones((num_a, num_a)) * d_a_j_phi_k_hat_2_chop

    noise_chop['sn_so_aphi'] = np.sqrt(np.sum(c_aphi ** 2 * d_a_phi_hat_2_chop))

    noise_chop['sn_fo'] = noise_chop['sn_fo_phi']
    noise_chop['sn_so'] = noise_chop['sn_so_aphi']

    noise_chop['sn'] = np.sqrt(noise_chop['sn_fo_phi'] ** 2 + noise_chop['sn_so_aphi'] ** 2)

    return noise_chop
