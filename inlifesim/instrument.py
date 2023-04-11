from typing import Union
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jn, spherical_jn
from scipy.fft import fft, fftfreq, fftshift, rfft
import pandas as pd
import xarray as xr

from inlifesim.util import black_body, find_nearest_idx

#TODO: Move calculation of the chopped planet signal into the planet signal method


class Instrument(object):

    def __init__(self,
                 wl_bins: np.ndarray,  # wavelength bins center position in m
                 wl_bin_widths: np.ndarray,  # wavelength bin widhts in m
                 integration_time: float,  # integration time in s
                 image_size: int,  # size of image used to simulate exozodi in pix
                 diameter_ap: float,  # diameter of the primary mirrors in m
                 flux_division: np.ndarray,  # division of the flux between the primary mirrors, e.g. in baseline case
                                             # [0.25, 0.25, 0.25, 0.25]
                 throughput: float,  # fraction of light that is sustained through the optical train
                 dist_star: float,  # distance to the target system in pc
                 radius_star: float,  # radius of the star in stellar radii
                 temp_star: float,  # temperature of the host star in Kelvin
                 lat_star: float,  # ecliptic latitude of the target star
                 l_sun: float,  # stellar luminosity in solar luminosities
                 z: float,  # zodi level: the exozodi dust is z-times denser than the localzodi dust
                 temp_planet: float,  # planet temperature in Kelvin
                 radius_planet: float,  # planet radius in earth radii
                 separation_planet: float,  # separation of target planet from host star in AU
                 col_pos: np.ndarray,  # collector position in m
                 phase_response: np.ndarray,  # phase response of each collector arm in rad
                 phase_response_chop: np.ndarray,  # phase response of each collector arm in the chopped state in rad
                 t_rot: float,  # rotation period of the array in seconds
                 chopping: str,  # run calculation with or without chopping, 'chop', 'nchop', 'both'
                 pix_per_wl,  # pixels on detector used per wavelength channel
                 detector_dark_current: str,  # detector type, 'MIRI' or 'manual'. Specify dark_current_pix in 'manual'
                 dark_current_pix: Union[float, type(None)],  # detector dark current in electrons s-1 px-1
                 detector_thermal: str,  # detector type, 'MIRI'
                 det_temp: float,  # temperature of the detector environment in K
                 magnification: float,  # telescope magnification
                 f_number: float,  # telescope f-number, i.e. ratio of focal length to aperture size
                 secondary_primary_ratio: float,  # ratio of secondary to primary mirror sizes
                 primary_emissivity: float,  # emissivity epsilon of the primary mirror
                 primary_temp: float,  # temperature of the primary mirror in K
                 n_sampling_rot: int,  # number of sampling points per array rotation
                 pink_noise_co: int,  # cutoff frequency for the pink noise spectra
                 n_cpu: int,  # number of cores used in the simulation
                 rms_mode: str,  # mode for rms values, 'lay', 'static', 'wavelength'
                 agnostic_mode: bool = False,  # derive instrumental photon noise from agnostic mode
                 eps_cold: Union[float, type(None)] = None,  # scaling constant for cold agnostic photon noise spectrum
                 eps_hot: Union[float, type(None)] = None,  # scaling constant for hot agnostic photon noise spectrum
                 eps_white: Union[float, type(None)] = None,  # scaling constant white agnostic photon noise spectrum
                 agnostic_spacecraft_temp: Union[float, type(None)] = None,  # cold-side spacecraft temperature in the 
                                                                             # agnostic case
                 n_sampling_max: int = 10000000,  # largest fourier mode used in noise sampling
                 d_a_rms: Union[float, type(None)] = None,  # relative amplitude error rms
                 d_phi_rms: Union[float, type(None)] = None,  # phase error rms
                 d_pol_rms: Union[float, type(None)] = None,  # polarization error rms
                 d_x_rms: Union[float, type(None)] = None,  # collector position rms, x-direction
                 d_y_rms: Union[float, type(None)] = None,  # collector position rms, y-direction
                 wl_resolution: int = 200,  # number of wavelength bins simulated for the thermal background
                 flux_planet: np.ndarray = None,  # substitute flux input in ph m-2 s-1
                 simultaneous_chopping: bool = False,  # true if the two chop states are produced at the same time
                 ):

        # setting simulation parameters
        self.wl_bins = wl_bins
        self.wl_bin_widths = wl_bin_widths
        # TODO: image size deprecated, remove
        self.image_size = image_size
        self.n_sampling_rot = n_sampling_rot
        self.pink_noise_co = pink_noise_co
        self.n_cpu = n_cpu
        self.n_sampling_max = n_sampling_max
        self.chopping = chopping
        self.simultaneous_chopping = simultaneous_chopping
        self.t_int = integration_time

        # setting instrument parameters
        self.col_pos = col_pos
        self.phi = phase_response
        self.phi_r = phase_response_chop
        self.diameter_ap = diameter_ap

        self.throughput = throughput
        self.flux_division = flux_division

        self.R = None  # response function R
        self.bl_x = None  # baseline matrix in x-direction (i.e. x_jk in Lay2004)
        self.bl_y = None  # baseline matrix in y-direction

        self.t_rot = t_rot
        self.pix_per_wl = pix_per_wl
        self.detector_dark_current = detector_dark_current
        self.dark_current_pix = dark_current_pix
        self.detector_thermal = detector_thermal
        self.det_temp = det_temp
        self.wl_resolution = wl_resolution

        self.magnification = magnification
        self.f_number = f_number
        self.secondary_primary_ratio = secondary_primary_ratio
        self.primary_temp = primary_temp
        self.primary_emmisivity = primary_emissivity
        
        self.agnostic_mode = agnostic_mode
        self.eps_cold = eps_cold
        self.eps_hot = eps_hot
        self.eps_white = eps_white
        self.agnostic_spacecraft_temp = agnostic_spacecraft_temp

        self.rms_mode = rms_mode
        self.d_a_rms = d_a_rms
        self.d_phi_rms = d_phi_rms
        self.d_pol_rms = d_pol_rms
        self.d_x_rms = d_x_rms
        self.d_y_rms = d_y_rms

        # setting source parameters
        self.dist_star = dist_star
        self.radius_star = radius_star
        self.temp_star = temp_star
        self.lat_star = lat_star
        self.l_sun = l_sun
        self.z = z
        self.temp_planet = temp_planet
        self.radius_planet = radius_planet
        self.separation_planet = separation_planet
        self.flux_planet = flux_planet
        self.flux_star = None
        self.b_star = None
        self.db_star_dx = None
        self.db_star_dy = None

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

        # self.photon_rates = pd.DataFrame(columns=['nchop', 'chop'],
        #                                  index=['signal',  # planet signal
        #                                         'noise',  # overall noise contribution
        #                                         'wl',  # wavelength bin
        #                                         'pn_sgl',  # stellar geometric leakage
        #                                         'pn_ez',  # exozodi leakage
        #                                         'pn_lz',  # localzodi leakage
        #                                         'pn_dc',  # dark current
        #                                         'pn_tbd',  # thermal background detector
        #                                         'pn_tbpm',  # thermal background primary mirror
        #                                         'pn_pa',  # polarization angle
        #                                         'pn_snfl',  # stellar null floor leakage
        #                                         'pn_ag_cld',  # agnostic cold instrumental photon noise
        #                                         'pn_ag_ht',  # agnostic hot instrumental photon noise
        #                                         'pn_ag_wht',  # agnostic white instrumental photon noise
        #                                         'pn',  # photon noise
        #                                         'sn_fo_a',  # first order amplitude
        #                                         'sn_fo_phi',  # first order phase
        #                                         'sn_fo_x',  # first order x position
        #                                         'sn_fo_y',  # first order y position
        #                                         'sn_fo',  # systematic noise first order
        #                                         'sn_so_aa',  # second order amplitude-amplitude term
        #                                         'sn_so_phiphi',  # second order phase-phase term
        #                                         'sn_so_aphi',  # amplitude phase cross term
        #                                         'sn_so_polpol',  # second order polarization-polarization term
        #                                         'sn_so',  # systematic noise second order
        #                                         'sn',  # systematic noise
        #                                         'fundamental',  # fundamental noise (astrophysical)
        #                                         'instrumental',  # instrumental noise
        #                                         'snr'  # signal to noise ratio
        #                                         ])

        self.photon_rates_chop = pd.DataFrame(columns=['signal',  # planet signal
                                                       'noise',  # overall noise contribution
                                                       'wl',  # wavelength bin
                                                       'pn_sgl',  # stellar geometric leakage
                                                       'pn_ez',  # exozodi leakage
                                                       'pn_lz',  # localzodi leakage
                                                       'pn_dc',  # dark current
                                                       'pn_tbd',  # thermal background detector
                                                       'pn_tbpm',  # thermal background primary mirror
                                                       'pn_pa',  # polarization angle
                                                       'pn_snfl',  # stellar null floor leakage
                                                       'pn_ag_cld',  # agnostic cold instrumental photon noise
                                                       'pn_ag_ht',  # agnostic hot instrumental photon noise
                                                       'pn_ag_wht',  # agnostic white instrumental photon noise
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
                                                       'fundamental',  # fundamental noise (astrophysical)
                                                       'instrumental',  # instrumental noise
                                                       'snr'  # signal to noise ratio
                                                       ],
                                              index=[str(np.round(wl*1e6, 1)) for wl in self.wl_bins]
                                              )

        self.photon_rates_nchop = pd.DataFrame(columns=['signal',  # planet signal
                                                        'noise',  # overall noise contribution
                                                        'wl',  # wavelength bin
                                                        'pn_sgl',  # stellar geometric leakage
                                                        'pn_ez',  # exozodi leakage
                                                        'pn_lz',  # localzodi leakage
                                                        'pn_dc',  # dark current
                                                        'pn_tbd',  # thermal background detector
                                                        'pn_tbpm',  # thermal background primary mirror
                                                        'pn_pa',  # polarization angle
                                                        'pn_snfl',  # stellar null floor leakage
                                                        'pn_ag_cld',  # agnostic cold instrumental photon noise
                                                        'pn_ag_ht',  # agnostic hot instrumental photon noise
                                                        'pn_ag_wht',  # agnostic white instrumental photon noise
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
                                                        'fundamental',  # fundamental noise (astrophysical)
                                                        'instrumental',  # instrumental noise
                                                        'snr'  # signal to noise ratio
                                                        ],
                                               index=[str(np.round(wl*1e6, 1)) for wl in self.wl_bins]
                                               )

        self.photon_rates_nchop['wl'] = self.wl_bins
        self.photon_rates_chop['wl'] = self.wl_bins

        np.seterr(invalid='ignore')

    def instrumental_parameters(self):
        # calculate some further instrumental parameters needed for Lay 2004 implementation
        self.A = np.sqrt(np.pi * (0.5 * self.diameter_ap) ** 2 * self.throughput * self.flux_division)  # area term A_j
        self.num_a = len(self.A)

        self.bl_x = np.array([(self.col_pos[:, 0] - self.col_pos[i, 0]) for i in range(self.num_a)])
        self.bl_y = np.array([(self.col_pos[:, 1] - self.col_pos[i, 1]) for i in range(self.num_a)])

        # set image size to appropriate sampling
        # TODO: Add scaling factor for image size, currently hardcoded to 2
        self.image_size = (4 * np.pi * np.max(np.abs(np.concatenate((self.bl_x.flatten(), self.bl_y.flatten()))))
                           / self.diameter_ap + 2) * 2
        self.image_size = int(np.ceil(self.image_size / 2.) * 2)

        # TODO: Factor 2 or factor 1 here?
        self.omega = 1 * np.pi * (self.wl_bins/(2. * self.diameter_ap))**2

        hfov = self.wl_bins / (2. * self.diameter_ap)

        hfov_mas = hfov * (3600000. * 180.) / np.pi
        self.rad_pix = (2 * hfov) / self.image_size  # Radians per pixel
        mas_pix = (2 * hfov_mas) / self.image_size  # mas per pixel
        self.au_pix = mas_pix / 1e3 * self.dist_star  # AU per pixel

        telescope_area = 4. * np.pi * (self.diameter_ap / 2.) ** 2

        x_map = np.tile(np.array(range(0, self.image_size)),
                        (self.image_size, 1))
        y_map = x_map.T
        r_square_map = ((x_map - (self.image_size - 1) / 2) ** 2
                        + (y_map - (self.image_size - 1) / 2) ** 2)
        self.radius_map = np.sqrt(r_square_map)
        self.r_au = self.radius_map[np.newaxis, :, :] * self.au_pix[:, np.newaxis, np.newaxis]

    def create_star(self) -> None:
        self.flux_star = black_body(mode='star',
                                    bins=self.wl_bins,
                                    width=self.wl_bin_widths,
                                    temp=self.temp_star,
                                    radius=self.radius_star,
                                    distance=self.dist_star)

        # angular extend of the star disk in rad divided by 2 to get radius
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
                      force: bool = False) -> None:
        if (self.flux_planet is None) or force:
            self.flux_planet = black_body(mode='planet',
                                          bins=self.wl_bins,
                                          width=self.wl_bin_widths,
                                          temp=self.temp_planet,
                                          radius=self.radius_planet,
                                          distance=self.dist_star)
        else:
            self.flux_planet = self.flux_planet

    def create_localzodi(self) -> None:
        long = 3 / 4 * np.pi
        lat = self.lat_star

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

    def create_exozodi(self) -> None:
        # calculate the parameters required by Kennedy2015
        alpha = 0.34
        r_in = 0.034422617777777775 * np.sqrt(self.l_sun)
        r_0 = np.sqrt(self.l_sun)
        sigma_zero = 7.11889e-8  # Sigma_{m,0} from Kennedy+2015 (doi:10.1088/0067-0049/216/2/23)

        # identify all pixels where the radius is larges than the inner radius by Kennedy+2015
        r_cond = ((self.r_au >= r_in)
                  & (self.r_au <= self.image_size / 2 * self.au_pix[:, np.newaxis, np.newaxis]))

        # calculate the temperature at all pixel positions according to Kennedy2015 Eq. 2
        temp_map = np.where(r_cond,
                            278.3 * (self.l_sun ** 0.25) / np.sqrt(self.r_au), 0)

        # calculate the Sigma (Eq. 3) in Kennedy2015 and set everything inside the inner radius to 0
        sigma = np.where(r_cond,
                         sigma_zero * self.z *
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

        # TODO: Check the transformation coefficients here
        ez_fft = np.fft.fftshift(np.fft.fft2(flux_map_exozodi), axes=(-2, -1)) / 2
        # ez_fft = np.fft.fft2(flux_map_exozodi) / 2
        # ez_fft = np.fft.fftshift(np.fft.fft2(flux_map_exozodi)) / np.pi
        r_rad_fft = np.fft.fftshift(np.fft.fftfreq(self.image_size, sampling_rate_rad[:, np.newaxis]), axes=(-1))
        # r_rad_fft = np.fft.fftfreq(self.image_size, sampling_rate_rad[:, np.newaxis])


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
                    self.b_ez[k, i, j] = np.real(ez_fft[k,
                                                        int(find_nearest_idx(r_rad_fft[k, :], bl_x_fft[k, i, j])),
                                                        int(find_nearest_idx(r_rad_fft[k, :], bl_y_fft[k, i, j]))
                                                 ])

        # plot bl_x_pix and bl_y_pix over the ez_fft image for the wavelength bin 13, 14 & 15
        # plt_wls = [12, 13, 14, 15, 16]
        # fig, ax = plt.subplots(nrows=len(plt_wls), figsize=(2,2*len(plt_wls)))
        # for i, wl in enumerate(plt_wls):
        #     ax[i].imshow(np.log10(np.abs(ez_fft[wl, :, :])))
        #     ax[i].scatter(bl_x_pix[wl, :, :], bl_y_pix[wl, :, :], s=1)
        #     ax[i].set_title(f'Wavelength bin {wl}')
        # fig.tight_layout()
        # plt.show()

        # plot the mean of bl_y_pix for each wavelenght bin
        # plt.plot(self.wl_bins, np.mean(bl_y_pix, axis=(1, 2)))
        # plt.show()

        # plot the mean of self.b_ez for each wavelenght bin and compare it to the mean of bl_y_pix
        # plt.plot(np.mean(np.abs(bl_y_pix - np.mean(bl_y_pix)), axis=(1, 2)), label='bl_y_pix')
        # plt.plot(np.mean(self.b_ez*10, axis=(1, 2)), label='b_ez')
        # plt.xlabel('Wavelength bin')
        # plt.ylabel('Mean of ...')
        # plt.legend()
        # plt.show()

        # plot a grid of all ez_fft images
        # fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
        # for i in range(5):
        #     for j in range(5):
        #         ax[i, j].imshow(np.log10(np.abs(ez_fft[i*5+j, :, :])))
        #         ax[i, j].set_title(f'Wavelength bin {i*5+j}')
        # fig.tight_layout()
        # plt.show()
        #
        # a=1

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

    def sensitivity_coefficients(self,
                                 exozodi_only: bool = False) -> None:
        self.exozodi_leakage()
        if not exozodi_only:
            self.stellar_leakage()
            self.localzodi_leakage()

        self.c_a = self.c_a_star + self.c_a_ez + self.c_a_lz
        self.c_phi = self.c_phi_star + self.c_phi_ez
        self.c_x = self.c_x_star
        self.c_y = self.c_y_star

        self.c_aphi = self.c_aphi_star + self.c_aphi_ez

        self.c_aa = self.c_aa_star + self.c_aa_ez
        self.c_phiphi = self.c_phiphi_star + self.c_phiphi_ez
        self.c_thetatheta = self.c_thetatheta_star

    def planet_signal(self) -> None:
        theta = self.separation_planet * 1.496e11 / (self.dist_star * 3.086e16)  # theta_x/y in Fig. 1
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
        nfft_odd[:, ::2] = 0

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

        self.photon_rates_nchop['signal'] = np.abs(
            (time_per_bin * self.planet_template * self.n_planet).sum(axis=1)
        ) / self.t_rot * self.t_int

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

        self.n_planet_chop = (self.n_planet - self.n_planet_r)

        if not self.simultaneous_chopping:
            self.n_planet_chop *= 0.5

        # Fourier transform of planet signal equivalent to Eq (33)
        nf_chop = rfft(self.n_planet_chop)
        nfft_chop = nf_chop / self.n_sampling_rot
        nfft_chop[:, 1:] *= 2

        # creation of template function
        # removal of even components and DC
        nfft_odd_chop = nfft_chop
        nfft_odd_chop[:, ::2] = 0

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

        self.photon_rates_chop['signal'] = np.abs(
            (time_per_bin * self.planet_template_chop * self.n_planet_chop).sum(axis=1)
        ) / self.t_rot * self.t_int


    def fundamental_noise(self,
                          exozodi_only: bool = False) -> None:
        if not exozodi_only:
            n_0_star = np.array([
                np.array([self.A[j] * self.A[k] * np.cos(self.phi[j] - self.phi[k]) * self.b_star[:, j, k]
                          for k in range(self.num_a)]).sum(axis=0)
                for j in range(self.num_a)]).sum(axis=0)
            self.photon_rates_nchop['pn_sgl'] = np.sqrt(n_0_star * self.t_int)

            n_0_lz = (
                    self.flux_localzodi[:, np.newaxis] * self.A[np.newaxis, :] ** 2 * self.omega[:, np.newaxis]
            ).sum(axis=1)
            self.photon_rates_nchop['pn_lz'] = np.sqrt(n_0_lz * self.t_int)


        n_0_ez = np.array([
            np.array([self.A[j] * self.A[k] * np.cos(self.phi[j] - self.phi[k]) * self.b_ez[:, j, k]
                      for k in range(self.num_a)]).sum(axis=0)
            for j in range(self.num_a)]).sum(axis=0)
        self.photon_rates_nchop['pn_ez'] = np.sqrt(n_0_ez * self.t_int)

    def pn_dark_current(self) -> None:
        if self.detector_dark_current == 'MIRI':
            self.dark_current_pix = 0.2
            self.photon_rates_nchop['pn_dc'] = (np.sqrt(self.dark_current_pix * self.pix_per_wl)
                                                       * np.ones((self.wl_bins.shape[0])))
        elif self.detector_dark_current == 'manual':
            if self.dark_current_pix == None:
                raise ValueError('Dark current per pixel needs to be specified in manual mode')
            self.photon_rates_nchop['pn_dc'] = (np.sqrt(self.dark_current_pix * self.pix_per_wl)
                                                       * np.ones((self.wl_bins.shape[0])))
        else:
            raise ValueError('Unkown detector type')

    def pn_thermal_background_detector(self) -> None:
        h = 6.62607e-34
        k = 1.380649e-23
        c = 2.99792e+8
        if self.detector_thermal == 'MIRI':
            # pitch - gap
            area_pixel = ((25 - 2) * 1e-6) ** 2
            det_wl_min = 5e-6
            det_wl_max = 28e-6
        else:
            raise ValueError('Unkown detector type')
        wl_bins = np.linspace(start=det_wl_min, stop=det_wl_max, num=self.wl_resolution, endpoint=True)
        B_photon = 2 * c / wl_bins ** 4 / (np.exp(h * c / (wl_bins * k * self.det_temp)) - 1)
        B_photon_int = np.trapz(y=B_photon, x=wl_bins)
        thermal_emission_det = 2 * np.pi * area_pixel * B_photon_int

        self.photon_rates_nchop['pn_tbd'] = (np.sqrt(thermal_emission_det * self.pix_per_wl)
                                                    * np.ones((self.wl_bins.shape[0])))

    def pn_thermal_primary_mirror(self) -> None:
        prefactor = 4 * self.primary_emmisivity * (
                np.pi * self.diameter_ap * self.magnification / self.f_number
                * self.secondary_primary_ratio / (1 - self.secondary_primary_ratio)
        ) ** 2
        thermal_emission_primary = prefactor * black_body(mode='wavelength',
                                                          bins=self.wl_bins,
                                                          width=self.wl_bin_widths,
                                                          temp=self.primary_temp)
        self.photon_rates_nchop['pn_tbpm'] = np.sqrt(thermal_emission_primary)
        
    def pn_agnostic(self) -> None:
        if (self.eps_white is None) or (self.eps_cold is None) or (self.eps_hot is None):
            raise ValueError('Agnostic scaling variables need to be specified in agnostic mode')

        self.photon_rates_nchop['pn_ag_ht'] = (self.eps_hot * 0.342 * self.diameter_ap ** 2
                                                      * black_body(mode='wavelength',
                                                                   bins=self.wl_bins,
                                                                   width=self.wl_bin_widths,
                                                                   temp=self.temp_star)
                                                      / 4 / black_body(mode='wavelength',
                                                                       bins=np.array((0.5e-6)),
                                                                       width=np.array((0.05e-6)),
                                                                       temp=self.temp_star)
                                                      )

        self.photon_rates_nchop['pn_ag_cld'] = (self.eps_cold * 0.947 * self.diameter_ap ** 2
                                                       * black_body(mode='wavelength',
                                                                    bins=self.wl_bins,
                                                                    width=self.wl_bin_widths,
                                                                    temp=self.agnostic_spacecraft_temp)
                                                       / 4 / black_body(mode='wavelength',
                                                                        bins=np.array((58e-6)),
                                                                        width=np.array((0.05e-6)),
                                                                        temp=50.)
                                                       )

        self.photon_rates_nchop['pn_ag_wht'] = self.eps_white * np.ones_like(self.wl_bins)

    def fundamental_collect(self):
        self.photon_rates_nchop['fundamental'] = np.sqrt(self.photon_rates_nchop['pn_sgl'] ** 2
                                                                 + self.photon_rates_nchop['pn_ez'] ** 2
                                                                 + self.photon_rates_nchop['pn_lz'] ** 2)

        # because of the incoherent combination of the final outputs, see Mugnier 2006
        if self.simultaneous_chopping:
            self.photon_rates_chop['fundamental'] *= np.sqrt(2)

        self.photon_rates_nchop['snr'] = (self.photon_rates_nchop['signal']
                                                 / self.photon_rates_nchop['fundamental'])

        self.photon_rates_chop['pn_sgl'] = self.photon_rates_nchop['pn_sgl']
        self.photon_rates_chop['pn_ez'] = self.photon_rates_nchop['pn_ez']
        self.photon_rates_chop['pn_lz'] = self.photon_rates_nchop['pn_lz']
        self.photon_rates_chop['pn_dc'] = self.photon_rates_nchop['pn_dc']
        self.photon_rates_chop['pn_tbd'] = self.photon_rates_nchop['pn_tbd']
        self.photon_rates_chop['pn_tbpm'] = self.photon_rates_nchop['pn_tbpm']
        self.photon_rates_chop['pn_ag_ht'] = self.photon_rates_nchop['pn_ag_ht']
        self.photon_rates_chop['pn_ag_cld'] = self.photon_rates_nchop['pn_ag_cld']
        self.photon_rates_chop['pn_ag_wht'] = self.photon_rates_nchop['pn_ag_wht']
        self.photon_rates_chop['fundamental'] = self.photon_rates_nchop['fundamental']
        self.photon_rates_chop['snr'] = self.photon_rates_nchop['snr']

    def sn_nchop(self):

        mp_args = []
        for i in range(self.wl_bins.shape[0]):
            mp_args.append({'rms_mode': self.rms_mode,
                            'wl': self.wl_bins[i],
                            't_rot': self.t_rot,
                            't_int': self.t_int,
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
                            'n_sampling_max': self.n_sampling_max,
                            'd_a_rms': self.d_a_rms,
                            'd_phi_rms': self.d_phi_rms,
                            'd_pol_rms': self.d_pol_rms,
                            'd_x_rms': self.d_x_rms,
                            'd_y_rms': self.d_y_rms,
                            'pink_noise_co': self.pink_noise_co})
        if self.n_cpu == 1:
            res = []
            for i in range(self.wl_bins.shape[0]):
                rr = instrumental_noise_single_wav_nchop(mp_args[i])
                res.append(rr)
        else:
            # collect arguments for multiprocessing
            pool = mp.Pool(self.n_cpu)
            results = pool.map(instrumental_noise_single_wav_nchop, mp_args)
            res = []
            for wl in self.wl_bins:
                for r in results:
                    if np.round(r['wl'], 10) == np.round(wl, 10):
                        res.append(r)

        self.save_to_results(data=res,
                             chop='nchop')

        self.photon_rates_nchop['pn'] = np.sqrt(np.array(self.photon_rates_nchop['pn_sgl'] ** 2
                                                         + self.photon_rates_nchop['pn_ez'] ** 2
                                                         + self.photon_rates_nchop['pn_lz'] ** 2
                                                         + self.photon_rates_nchop['pn_pa'] ** 2
                                                         + self.photon_rates_nchop['pn_snfl'] ** 2).astype(float))

        self.photon_rates_nchop['instrumental'] = np.sqrt((self.photon_rates_nchop['sn'] ** 2
                                                           + self.photon_rates_nchop['pn_pa'] ** 2
                                                           + self.photon_rates_nchop['pn_snfl'] ** 2).astype(float))

        if not self.agnostic_mode:
            self.photon_rates_nchop['pn'] = np.sqrt((self.photon_rates_nchop['pn'] ** 2
                                                     + self.photon_rates_nchop['pn_dc'] ** 2
                                                     + self.photon_rates_nchop['pn_tbd'] ** 2
                                                     + self.photon_rates_nchop['pn_tbpm'] ** 2).astype(float))

            self.photon_rates_nchop['instrumental'] = np.sqrt((self.photon_rates_nchop['instrumental'] ** 2
                                                               + self.photon_rates_nchop['pn_dc'] ** 2
                                                               + self.photon_rates_nchop['pn_tbd'] ** 2
                                                               + self.photon_rates_nchop['pn_tbpm'] ** 2).astype(float))

        else:
            self.photon_rates_nchop['pn'] = np.sqrt((self.photon_rates_nchop['pn'] ** 2
                                                     + self.photon_rates_nchop['pn_ag_ht'] ** 2
                                                     + self.photon_rates_nchop['pn_ag_cld'] ** 2
                                                     + self.photon_rates_nchop['pn_ag_wht'] ** 2).astype(float))

            self.photon_rates_nchop['instrumental'] = np.sqrt((self.photon_rates_nchop['instrumental'] ** 2
                                                               + self.photon_rates_nchop['pn_ag_ht'] ** 2
                                                               + self.photon_rates_nchop['pn_ag_cld'] ** 2
                                                               + self.photon_rates_nchop['pn_ag_wht'] ** 2).astype(float))

        self.photon_rates_nchop['noise'] = np.sqrt((self.photon_rates_nchop['pn'] ** 2
                                                    + self.photon_rates_nchop['sn'] ** 2).astype(float))

        self.photon_rates_nchop['fundamental'] = np.sqrt((self.photon_rates_nchop['pn_sgl'] ** 2
                                                          + self.photon_rates_nchop['pn_ez'] ** 2
                                                          + self.photon_rates_nchop['pn_lz'] ** 2).astype(float))

        # because of the incoherent combination of the final outputs, see Mugnier 2006
        if self.simultaneous_chopping:
            self.photon_rates_chop['noise'] *= np.sqrt(2)
            self.photon_rates_chop['fundamental'] *= np.sqrt(2)
            self.photon_rates_chop['instrumental'] *= np.sqrt(2)

        self.photon_rates_nchop['snr'] = (self.photon_rates_nchop['signal']
                                                 / self.photon_rates_nchop['noise'])

    def save_to_results(self,
                        data,
                        chop):
        # for k in data[0].keys():
        #     self.photon_rates.loc[k, column_results] = []
        # for d in data:
        #     for k in d.keys():
        #         self.photon_rates.loc[k, column_results].append(d[k])
        # for k in data[0].keys():
        #     self.photon_rates.loc[k, column_results] = np.array(self.photon_rates.loc[k, column_results])
        for d in data:
            for k in d.keys():
                if chop == 'nchop':
                    self.photon_rates_nchop.loc[self.photon_rates_nchop['wl'] == d['wl'], k] = d[k]
                elif chop == 'chop':
                    self.photon_rates_chop.loc[self.photon_rates_chop['wl'] == d['wl'], k] = d[k]

    def sn_chop(self):
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
                            'rms_mode': self.rms_mode,
                            'n_sampling_max': self.n_sampling_max,
                            't_rot': self.t_rot,
                            't_int': self.t_int,
                            'd_a_rms': self.d_a_rms,
                            'd_phi_rms': self.d_phi_rms,
                            'd_pol_rms': self.d_pol_rms,
                            'pink_noise_co': self.pink_noise_co,
                            'flux_star': self.flux_star[i]
                            })
        if self.n_cpu == 1:
            res = []
            for i in range(self.wl_bins.shape[0]):
                rr = instrumental_noise_single_wav_chop(mp_args[i])
                res.append(rr)
        else:
            # collect arguments for multiprocessing
            pool = mp.Pool(self.n_cpu)
            results = pool.map(instrumental_noise_single_wav_chop, mp_args)
            res = []
            for wl in self.wl_bins:
                for r in results:
                    if np.round(r['wl'], 10) == np.round(wl, 10):
                        res.append(r)

        self.save_to_results(data=res,
                             chop='chop')

        self.photon_rates_chop['pn_sgl'] = self.photon_rates_nchop['pn_sgl']
        self.photon_rates_chop['pn_ez'] = self.photon_rates_nchop['pn_ez']
        self.photon_rates_chop['pn_lz'] = self.photon_rates_nchop['pn_lz']
        self.photon_rates_chop['pn_dc'] = self.photon_rates_nchop['pn_dc']
        self.photon_rates_chop['pn_tbd'] = self.photon_rates_nchop['pn_tbd']
        self.photon_rates_chop['pn_tbpm'] = self.photon_rates_nchop['pn_tbpm']
        self.photon_rates_chop['pn_ag_ht'] = self.photon_rates_nchop['pn_ag_ht']
        self.photon_rates_chop['pn_ag_cld'] = self.photon_rates_nchop['pn_ag_cld']
        self.photon_rates_chop['pn_ag_wht'] = self.photon_rates_nchop['pn_ag_wht']

        self.photon_rates_chop['pn'] = np.sqrt((self.photon_rates_chop['pn_sgl'] ** 2
                                                + self.photon_rates_chop['pn_ez'] ** 2
                                                + self.photon_rates_chop['pn_lz'] ** 2
                                                + self.photon_rates_chop['pn_pa'] ** 2
                                                + self.photon_rates_chop['pn_snfl'] ** 2).astype(float))

        self.photon_rates_chop['instrumental'] = np.sqrt((self.photon_rates_chop['sn'] ** 2
                                                          + self.photon_rates_chop['pn_pa'] ** 2
                                                          + self.photon_rates_chop['pn_snfl'] ** 2).astype(float))

        if not self.agnostic_mode:
            self.photon_rates_chop['pn'] = np.sqrt((self.photon_rates_chop['pn'] ** 2
                                                    + self.photon_rates_chop['pn_dc'] ** 2
                                                    + self.photon_rates_chop['pn_tbd'] ** 2
                                                    + self.photon_rates_chop['pn_tbpm'] ** 2).astype(float))

            self.photon_rates_chop['instrumental'] = np.sqrt((self.photon_rates_chop['instrumental'] ** 2
                                                              + self.photon_rates_chop['pn_dc'] ** 2
                                                              + self.photon_rates_chop['pn_tbd'] ** 2
                                                              + self.photon_rates_chop['pn_tbpm'] ** 2).astype(float))

        else:
            self.photon_rates_chop['pn'] = np.sqrt((self.photon_rates_chop['pn'] ** 2
                                                    + self.photon_rates_chop['pn_ag_ht'] ** 2
                                                    + self.photon_rates_chop['pn_ag_cld'] ** 2
                                                    + self.photon_rates_chop['pn_ag_wht'] ** 2).astype(float))

            self.photon_rates_chop['instrumental'] = np.sqrt((self.photon_rates_chop['instrumental'] ** 2
                                                              + self.photon_rates_chop['pn_ag_ht'] ** 2
                                                              + self.photon_rates_chop['pn_ag_cld'] ** 2
                                                              + self.photon_rates_chop['pn_ag_wht'] ** 2).astype(float))

        self.photon_rates_chop['noise'] = np.sqrt((self.photon_rates_chop['pn'] ** 2
                                                   + self.photon_rates_chop['sn'] ** 2).astype(float))

        self.photon_rates_chop['fundamental'] = np.sqrt((self.photon_rates_chop['pn_sgl'] ** 2
                                                         + self.photon_rates_chop['pn_ez'] ** 2
                                                         + self.photon_rates_chop['pn_lz'] ** 2).astype(float))

        # because of the incoherent combination of the final outputs, see Mugnier 2006
        if self.simultaneous_chopping:
            self.photon_rates_chop['noise'] *= np.sqrt(2)
            self.photon_rates_chop['fundamental'] *= np.sqrt(2)
            self.photon_rates_chop['instrumental'] *= np.sqrt(2)

        self.photon_rates_chop['snr'] = (self.photon_rates_chop['signal']
                                                / self.photon_rates_chop['noise'])

    # def cleanup(self):
    #     if self.wl_bins.shape[0] == 1:
    #         for i in self.photon_rates.index:
    #             if type(self.photon_rates_nchop[i]) == np.ndarray:
    #                 self.photon_rates_nchop[i] = self.photon_rates_nchop[i][0]
    #             if type(self.photon_rates_chop[i]) == np.ndarray:
    #                 self.photon_rates_chop[i, 'chop'] = self.photon_rates_chop[i, 'chop'][0]


    def run(self) -> None:
        self.instrumental_parameters()

        self.create_star()
        self.create_planet()
        self.create_localzodi()
        self.create_exozodi()

        self.sensitivity_coefficients()

        self.planet_signal()

        self.fundamental_noise()

        if self.agnostic_mode:
            self.pn_agnostic()
        else:
            self.pn_dark_current()
            self.pn_thermal_background_detector()
            self.pn_thermal_primary_mirror()

        if (self.chopping == 'nchop') or (self.chopping == 'both'):
            self.sn_nchop()
        if (self.chopping == 'chop') or (self.chopping == 'both'):
            self.sn_chop()
        if self.chopping not in ['chop', 'nchop', 'both']:
            raise ValueError('Invalid chopping selection')

        # self.cleanup()


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
    t_int = mp_arg['t_int']
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
    noise_nchop['pn_pa'] = np.sqrt(dn_pol * t_int)
    dn_null_floor = np.array([c_aa[j, j] * avg_d_a_2[j] + c_phiphi[j, j] * avg_d_phi_2 for j in range(num_a)]).sum()
    noise_nchop['pn_snfl'] = np.sqrt(dn_null_floor * t_int)

    template_fft = rfft(template)
    template_fft = template_fft / len(template)

    # first order terms
    d_a_j_hat_2 = np.array([(np.abs(template_fft) ** 2 * d_a_b_2[j, :len(template_fft)]).sum() for j in range(num_a)])
    d_phi_j_hat_2 = np.array([(np.abs(template_fft) ** 2 * d_phi_b_2[:len(template_fft)]).sum() for j in range(num_a)])
    d_x_j_hat_2 = np.array([(np.abs(template_fft[:len(d_x_b_2)]) ** 2 * d_x_b_2).sum() for j in range(num_a)])
    d_y_j_hat_2 = np.array([(np.abs(template_fft[:len(d_y_b_2)]) ** 2 * d_y_b_2).sum() for j in range(num_a)])

    noise_nchop['sn_fo_a'] = np.sqrt((c_a ** 2 * d_a_j_hat_2).sum() * t_int ** 2)

    noise_nchop['sn_fo_phi'] = np.sqrt((c_phi ** 2 * d_phi_j_hat_2).sum() * t_int ** 2)

    noise_nchop['sn_fo_x'] = np.sqrt((c_x ** 2 * d_x_j_hat_2).sum() * t_int ** 2)

    noise_nchop['sn_fo_y'] = np.sqrt((c_y ** 2 * d_y_j_hat_2).sum() * t_int ** 2)

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

    noise_nchop['sn_so_aa'] = np.sqrt(np.sum(c_aa ** 2 * d_a_hat_2) * t_int ** 2)

    noise_nchop['sn_so_phiphi'] = np.sqrt(np.sum(c_phiphi ** 2 * d_phi_hat_2) * t_int ** 2)

    noise_nchop['sn_so_aphi'] = np.sqrt(np.sum(c_aphi ** 2 * d_a_phi_hat_2) * t_int ** 2)

    noise_nchop['sn_so_polpol'] = np.sqrt(np.sum(c_thetatheta ** 2 * d_pol_hat_2) * t_int ** 2)

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
    t_int = mp_arg['t_int']
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
    noise_chop['pn_pa'] = np.sqrt(dn_pol * t_int)
    dn_null_floor = np.array([c_aa[j, j] * avg_d_a_2[j] + c_phiphi[j, j] * avg_d_phi_2 for j in range(num_a)]).sum()
    noise_chop['pn_snfl'] = np.sqrt(dn_null_floor * t_int)

    # first order dphi
    d_phi_j_hat_2_chop = np.array([(np.abs(planet_template_c_fft) ** 2
                                    * d_phi_b_2[:len(planet_template_c_fft)]).sum() for j in range(num_a)])
    noise_chop['sn_fo_phi'] = np.sqrt((c_phi ** 2 * d_phi_j_hat_2_chop).sum() * t_int ** 2)


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

    noise_chop['sn_so_aphi'] = np.sqrt(np.sum(c_aphi ** 2 * d_a_phi_hat_2_chop) * t_int ** 2)

    noise_chop['sn_fo'] = noise_chop['sn_fo_phi']
    noise_chop['sn_so'] = noise_chop['sn_so_aphi']

    noise_chop['sn'] = np.sqrt(noise_chop['sn_fo_phi'] ** 2 + noise_chop['sn_so_aphi'] ** 2)

    return noise_chop
