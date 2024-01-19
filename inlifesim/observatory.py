from typing import Union
import multiprocessing as mp

import numpy as np
import pandas as pd

from inlifesim.util import harmonic_number_approximation
from inlifesim.sources import (create_star, create_planet, create_localzodi,
                               create_exozodi)
from inlifesim.perturbation import (stellar_leakage, exozodi_leakage,
                                    localzodi_leakage, sys_noise_chop)
from inlifesim.signal import planet_signal, fundamental_noise
from inlifesim.spectra import rms_frequency_adjust

class Instrument(object):


    def __init__(self,
                 wl_bins: np.ndarray,
                 wl_bin_widths: np.ndarray,
                 integration_time: float,
                 image_size: int,
                 diameter_ap: float,
                 flux_division: np.ndarray,
                 throughput: float,
                 dist_star: float,
                 radius_star: float,
                 temp_star: float,
                 lat_star: float,
                 l_sun: float,
                 z: float,
                 temp_planet: float,
                 radius_planet: float,
                 separation_planet: float,
                 col_pos: np.ndarray,
                 phase_response: np.ndarray,
                 phase_response_chop: np.ndarray,
                 t_rot: float,
                 chopping: str,
                 pix_per_wl,
                 detector_dark_current: str,
                 dark_current_pix: Union[float, type(None)],
                 detector_thermal: str,
                 det_temp: float,
                 magnification: float,
                 f_number: float,
                 secondary_primary_ratio: float,
                 primary_emissivity: float,
                 primary_temp: float,
                 n_sampling_rot: int,
                 n_cpu: int,
                 rms_mode: str,
                 agnostic_mode: bool = False,
                 eps_cold: Union[float, type(None)] = None,
                 eps_hot: Union[float, type(None)] = None,
                 eps_white: Union[float, type(None)] = None,
                 agnostic_spacecraft_temp: Union[float, type(None)] = None,
                 n_sampling_max: int = 10000000,
                 d_a_rms: Union[float, type(None)] = None,
                 d_phi_rms: Union[float, type(None)] = None,
                 d_pol_rms: Union[float, type(None)] = None,
                 d_x_rms: Union[float, type(None)] = None,
                 d_y_rms: Union[float, type(None)] = None,
                 d_a_co: Union[float, type(None)] = None,
                 d_phi_co: Union[float, type(None)] = None,
                 d_pol_co: Union[float, type(None)] = None,
                 d_x_co: Union[float, type(None)] = None,
                 d_y_co: Union[float, type(None)] = None,
                 wl_resolution: int = 200,
                 flux_planet: np.ndarray = None,
                 simultaneous_chopping: bool = False,
                 verbose: bool = False,
                 draw_samples: bool = False
                 ):
        '''
        Observatory instance to calculate instrumental noise. TODO: add example

        Parameters
        ----------
        wl_bins : np.ndarray
            wavelength bins center position in [m]
        wl_bin_widths : np.ndarray
            wavelength bin widths in [m]
        integration_time : float
            integration time in [s]
        image_size : int
            size of image used to simulate exozodi in [pix]
        diameter_ap : float
            diameter of the primary mirrors in [m]
        flux_division : np.ndarray
            division of the flux between the primary mirrors, e.g. in baseline
            case [0.25, 0.25, 0.25, 0.25]
        throughput : float
            fraction of light that is sustained through the optical train
        dist_star : float
            distance to the target system in [pc]
        radius_star : float
            radius of the star in [solar radii]
        temp_star : float
            temperature of the host star in [K]
        lat_star : float
            ecliptic latitude of the target star in [rad]
        l_sun : float
            stellar luminosity in solar luminosities
        z : float
            zodi level: the exozodi dust is z-times denser than the localzodi
            dust
        temp_planet : float
            planet temperature in [K]
        radius_planet : float
            planet radius in [earth radii]
        separation_planet : float
            separation of target planet from host star in [AU]
        col_pos : np.ndarray
            collector position in [m]
        phase_response : np.ndarray
            phase response of each collector arm in [rad]
        phase_response_chop : np.ndarray
            phase response of each collector arm in the chopped state in [rad]
        t_rot : float
            rotation period of the array in [s]
        chopping : str
            run calculation with or without chopping, 'chop', 'nchop', 'both'
        pix_per_wl : TYPE
            pixels on detector used per wavelength channel
        detector_dark_current : str
            detector type, 'MIRI' or 'manual'. Specify dark_current_pix in
            'manual'
        dark_current_pix : Union[float, type(None)]
            detector dark current in [electrons s-1 px-1]
        detector_thermal : str
            detector type, 'MIRI'
        det_temp : float
            temperature of the detector environment in [K]
        magnification : float
            telescope magnification
        f_number : float
            telescope f-number, i.e. ratio of focal length to aperture size
        secondary_primary_ratio : float
            ratio of secondary to primary mirror sizes
        primary_emissivity : float
            emissivity epsilon of the primary mirror
        primary_temp : float
            temperature of the primary mirror in K
        n_sampling_rot : int
            number of sampling points per array rotation, should be odd
        pink_noise_co : int
            cutoff frequency for the pink noise spectra
        n_cpu : int
            number of cores used in the simulation
        rms_mode : str
            mode for rms values, 'lay', 'static', 'wavelength'
        agnostic_mode : bool, optional
            derive instrumental photon noise from agnostic mode
        eps_cold : Union[float, type(None)], optional
            scaling constant for cold agnostic photon noise spectrum
        eps_hot : Union[float, type(None)], optional
            scaling constant for hot agnostic photon noise spectrum
        eps_white : Union[float, type(None)], optional
            scaling constant white agnostic photon noise spectrum
        agnostic_spacecraft_temp : Union[float, type(None)], optional
            cold-side spacecraft temperature in the agnostic case
        n_sampling_max : int, optional
            largest fourier mode used in noise sampling
        d_a_rms : Union[float, type(None)], optional
            relative amplitude error rms
        d_phi_rms : Union[float, type(None)], optional
            phase error rms
        d_pol_rms : Union[float, type(None)], optional
            polarization error rms
        d_x_rms : Union[float, type(None)], optional
            collector position rms, x-direction
        d_y_rms : Union[float, type(None)], optional
            collector position rms, y-direction
        d_a_co : Union[float, type(None)], optional
            cutoff frequency for the definition of the amplitude PSD rms
        d_phi_co : Union[float, type(None)], optional
            cutoff frequency for the definition of the phase PSD rms
        d_pol_co : Union[float, type(None)], optional
            cutoff frequency for the definition of the polarization PSD rms
        d_x_co : Union[float, type(None)], optional
            cutoff frequency for the definition of the x-position PSD rms
        d_y_co : Union[float, type(None)], optional
            cutoff frequency for the definition of the y-position PSD rms
        wl_resolution : int, optional
            number of wavelength bins simulated for the thermal background
        flux_planet : np.ndarray, optional
            substitute flux input in ph m-2 s-1
        simultaneous_chopping : bool, optional
            true if the two chop states are produced at the same time
        verbose : bool, optional
            print progress to terminal
        '''

        self.verbose = verbose
        self.draw_samples = draw_samples

        # setting simulation parameters
        self.wl_bins = wl_bins
        self.wl_bin_widths = wl_bin_widths
        self.image_size = image_size
        self.n_sampling_rot = n_sampling_rot

        if self.n_sampling_rot % 2 == 0:
            self.n_sampling_rot += 1
            print('Sampling rate was adjusted to be odd')

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
        # baseline matrix in x-direction (i.e. x_jk in Lay2004)
        self.bl_x = None
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

        self.d_a_co = d_a_co
        self.d_phi_co = d_phi_co
        self.d_pol_co = d_pol_co
        self.d_x_co = d_x_co
        self.d_y_co = d_y_co

        if self.rms_mode == 'lay':
            self.d_a_co = 10e3
            self.d_phi_co = 10e3
            self.d_pol_co = 10e3
            self.d_x_co = 0.64e-3
            self.d_y_co = 0.64e-3

        self.harmonic_number_n_cutoff = {
            'a': harmonic_number_approximation(self.d_a_co*self.t_rot),
            'phi': harmonic_number_approximation(self.d_phi_co*self.t_rot),
            'pol': harmonic_number_approximation(self.d_pol_co*self.t_rot),
            'x': harmonic_number_approximation(self.d_x_co*self.t_rot),
            'y': harmonic_number_approximation(self.d_y_co*self.t_rot)
        }

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
        self.planet_template_nchop = None
        self.planet_template_chop = None
        self.n_planet = None
        self.phi_rot = None
        self.theta_x = None
        self.theta_y = None

        self.photon_rates_chop = pd.DataFrame(
            columns=['signal',  # planet signal
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
                     # second order polarization-polarization term
                     'sn_so_polpol',
                     'sn_so',  # systematic noise second order
                     'sn',  # systematic noise
                     'fundamental',  # fundamental noise (astrophysical)
                     'instrumental',  # instrumental noise
                     'snr'  # signal to noise ratio
                     ],
            index=[str(np.round(wl * 1e6, 2)) for wl in self.wl_bins]
        )

        self.photon_rates_nchop = pd.DataFrame(
            columns=['signal',  # planet signal
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
                     # second order polarization-polarization term
                     'sn_so_polpol',
                     'sn_so',  # systematic noise second order
                     'sn',  # systematic noise
                     'fundamental',  # fundamental noise (astrophysical)
                     'instrumental',  # instrumental noise
                     'snr'  # signal to noise ratio
                     ],
            index=[str(np.round(wl * 1e6, 2)) for wl in self.wl_bins]
        )

        self.photon_rates_nchop['wl'] = self.wl_bins
        self.photon_rates_chop['wl'] = self.wl_bins

        np.seterr(invalid='ignore')

    def instrumental_parameters(self):
        '''
        Calculate instrumental parameters
        :return:
        '''
        self.A = np.sqrt(np.pi * (0.5 * self.diameter_ap) ** 2
                         * self.throughput * self.flux_division)
        # area term A_j
        self.num_a = len(self.A)

        self.bl = np.array((
            np.subtract.outer(self.col_pos[:, 0], self.col_pos[:, 0]).T,
            np.subtract.outer(self.col_pos[:, 1], self.col_pos[:, 1]).T
        ))

        self.omega = 1 * np.pi * (self.wl_bins/(2. * self.diameter_ap))**2

        self.hfov = self.wl_bins / (2. * self.diameter_ap)

        hfov_mas = self.hfov * (3600000. * 180.) / np.pi
        self.rad_pix = (2 * self.hfov) / self.image_size  # Radians per pixel
        mas_pix = (2 * hfov_mas) / self.image_size  # mas per pixel
        self.au_pix = mas_pix / 1e3 * self.dist_star  # AU per pixel

        telescope_area = 4. * np.pi * (self.diameter_ap / 2.) ** 2

        x_map = np.tile(np.array(range(0, self.image_size)),
                        (self.image_size, 1))
        y_map = x_map.T
        r_square_map = ((x_map - (self.image_size - 1) / 2) ** 2
                        + (y_map - (self.image_size - 1) / 2) ** 2)
        self.radius_map = np.sqrt(r_square_map)
        self.r_au = (self.radius_map[np.newaxis, :, :]
                     * self.au_pix[:, np.newaxis, np.newaxis])

    def combine_coefficients(self):
        self.grad_n_coeff = []
        self.hess_n_coeff = []
        for i in range(len(self.wl_bins)):
            self.grad_n_coeff.append({k: self.grad_star[k][i]
                                         + self.grad_ez[k][i]
                                         + self.grad_lz[k][i]
                                      for k in self.grad_star.keys()})
            self.hess_n_coeff.append({k: self.hess_star[k][i]
                                         + self.hess_ez[k][i]
                                      for k in self.hess_star.keys()})

    def build_gradient_hessian(
            self,
            gradient: dict,
            hessian: dict,
            order_gradient: tuple = ('a', 'phi', 'x', 'y'),
            order_hessian: tuple = (('aa', 'aphi'), ('aphi', 'phiphi'))
    ):
        """
        Build gradient vector and Hessian matrix from gradient and Hessian
        dictionaries

        Parameters
        ----------
        gradient : np.ndarray
            gradient dictionary
        hessian : np.ndarray
            hessian dictionary
        order_gradient : tuple, optional
            order of gradient terms, by default ('a', 'phi', 'x', 'y')
        order_hessian : tuple, optional
            order of hessian terms, by default
            (('aa', 'aphi'), ('aphi', 'phiphi'))

        Returns
        -------
        gradient_vector : np.ndarray
            gradient vector
        hessian_matrix : np.ndarray
            hessian matrix
        """

        gradient_vector = np.concatenate((gradient[order_gradient[0]],
                                          gradient[order_gradient[1]],
                                          gradient[order_gradient[2]],
                                          gradient[order_gradient[3]]),
                                         axis = 1)
        hessian_matrix = np.concatenate((
            np.concatenate((hessian[order_hessian[0][0]],
                            hessian[order_hessian[0][1]]), axis=2),
            np.concatenate((
                np.transpose(hessian[order_hessian[1][0]], axes=(0, 2, 1)),
                            hessian[order_hessian[1][1]]), axis=2)
        ), axis=1)

        return gradient_vector, hessian_matrix

    def combine_fundamental(self):
        self.photon_rates_nchop['fundamental'] = np.sqrt(
            self.photon_rates_nchop['pn_sgl'] ** 2
            + self.photon_rates_nchop['pn_ez'] ** 2
            + self.photon_rates_nchop['pn_lz'] ** 2
        )

        # because of the incoherent combination of the final outputs, see
        # Mugnier 2006
        if self.simultaneous_chopping:
            self.photon_rates_chop['fundamental'] *= np.sqrt(2)

        self.photon_rates_nchop['snr'] = (
                self.photon_rates_nchop['signal']
                / self.photon_rates_nchop['fundamental']
        )

        self.photon_rates_chop['pn_sgl'] = self.photon_rates_nchop['pn_sgl']
        self.photon_rates_chop['pn_ez'] = self.photon_rates_nchop['pn_ez']
        self.photon_rates_chop['pn_lz'] = self.photon_rates_nchop['pn_lz']
        self.photon_rates_chop['fundamental'] = self.photon_rates_nchop[
            'fundamental'
        ]
        self.photon_rates_chop['snr'] = self.photon_rates_nchop['snr']

    def sn_chop(self):

        # prepare variable dictionary to send to multiprocessing workers
        mp_args = []
        for i in range(self.wl_bins.shape[0]):
            mp_args.append({
                'A': self.A,
                'wl': self.wl_bins[i],
                'num_a': self.num_a,
                'planet_template_chop': self.planet_template_chop[i, :],
                'grad_n_coeff': self.grad_n_coeff[i],
                'hess_n_coeff': self.hess_n_coeff[i],
                'rms_mode': self.rms_mode,
                'n_sampling_max': self.n_sampling_max,
                'harmonic_number_n_cutoff':
                    self.harmonic_number_n_cutoff,
                't_rot': self.t_rot,
                't_int': self.t_int,
                'd_a_rms': self.d_a_rms,
                'd_phi_rms': self.d_phi_rms,
                'd_pol_rms': self.d_pol_rms,
                'flux_star': self.flux_star[i]
            })
        if self.n_cpu == 1:
            res = []
            for i in range(self.wl_bins.shape[0]):
                rr = sys_noise_chop(mp_args[i])
                res.append(rr)
        else:
            # collect arguments for multiprocessing
            pool = mp.Pool(self.n_cpu)
            results = pool.map(sys_noise_chop, mp_args)
            res = []
            for wl in self.wl_bins:
                for r in results:
                    if np.round(r['wl'], 10) == np.round(wl, 10):
                        res.append(r)

        '''
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
                                                
        '''

    def run(self) -> None:
        self.instrumental_parameters()

        if self.verbose:
            print('Creating astrophysical sources ...', end=' ')
        self.flux_star, self.b_star, self.db_star_dx, self.db_star_dy = (
            create_star(wl_bins=self.wl_bins,
                        wl_bin_widths=self.wl_bin_widths,
                        temp_star=self.temp_star,
                        radius_star=self.radius_star,
                        dist_star=self.dist_star,
                        bl=self.bl,
                        col_pos=self.col_pos,
                        num_a=self.num_a,
                        ))

        if self.flux_planet is None:
            self.flux_planet = create_planet(wl_bins=self.wl_bins,
                                             wl_bin_widths=self.wl_bin_widths,
                                             temp_planet=self.temp_planet,
                                             radius_planet=self.radius_planet,
                                             dist_star=self.dist_star)

        self.flux_localzodi = create_localzodi(
            wl_bins=self.wl_bins,
            wl_bin_widths=self.wl_bin_widths,
            lat=self.lat_star
        )

        self.b_ez = create_exozodi(wl_bins=self.wl_bins,
                                   wl_bin_widths=self.wl_bin_widths,
                                   z=self.z,
                                   l_sun=self.l_sun,
                                   r_au=self.r_au,
                                   image_size=self.image_size,
                                   au_pix=self.au_pix,
                                   rad_pix=self.rad_pix,
                                   radius_map=self.radius_map,
                                   bl=self.bl,
                                   hfov=self.hfov)

        if self.verbose:
            print('[Done]')
            print('Calculating gradient and Hessian coefficients ...', end=' ')

        self.grad_star, self.hess_star = stellar_leakage(
            A=self.A,
            phi=self.phi,
            b_star=self.b_star,
            db_star_dx=self.db_star_dx,
            db_star_dy=self.db_star_dy,
            num_a=self.num_a
        )

        self.grad_ez, self.hess_ez = exozodi_leakage(A=self.A,
                                                     phi=self.phi,
                                                     b_ez=self.b_ez,
                                                     num_a=self.num_a)

        self.grad_lz = localzodi_leakage(A=self.A,
                                         omega=self.omega,
                                         flux_localzodi=self.flux_localzodi)

        self.combine_coefficients()

        if self.verbose:
            print('[Done]')
            print('Generating planet signal ...', end=' ')

        (self.planet_template_nchop,
         self.photon_rates_nchop['signal'],
         _,
         self.planet_template_chop,
         self.photon_rates_chop['signal'],
         _) = planet_signal(
            flux_planet=self.flux_planet,
            A=self.A,
            phi=self.phi,
            phi_r=self.phi_r,
            wl_bins=self.wl_bins,
            bl=self.bl,
            num_a=self.num_a,
            t_rot=self.t_rot,
            n_sampling_rot=self.n_sampling_rot,
            simultaneous_chopping=self.simultaneous_chopping,
            separation_planet=self.separation_planet,
            dist_star=self.dist_star,
            t_int=self.t_int,
        )

        if self.verbose:
            print('[Done]')
            print('Calculating fundamental noise ...', end=' ')

        (self.photon_rates_nchop['pn_sgl'],
         self.photon_rates_nchop['pn_ez'],
         self.photon_rates_nchop['pn_lz']) = fundamental_noise(
            A=self.A,
            phi=self.phi,
            num_a=self.num_a,
            t_int=self.t_int,
            flux_localzodi=self.flux_localzodi,
            b_star=self.b_star,
            b_ez=self.b_ez,
            omega=self.omega
        )

        self.combine_fundamental()

        if self.verbose:
            print('[Done]')
            print('Doing the next thing ...', end=' ')

        self.sn_chop()


        a=1
        # self.create_planet()
        # self.create_localzodi()
        # self.create_exozodi()
        #
        # self.sensitivity_coefficients()
        #
        # self.planet_signal()
        #
        # self.fundamental_noise()
        #
        # if self.agnostic_mode:
        #     self.pn_agnostic()
        # else:
        #     self.pn_dark_current()
        #     self.pn_thermal_background_detector()
        #     self.pn_thermal_primary_mirror()
        #
        # if (self.chopping == 'nchop') or (self.chopping == 'both'):
        #     self.sn_nchop()
        # if (self.chopping == 'chop') or (self.chopping == 'both'):
        #     self.sn_chop()
        # if self.chopping not in ['chop', 'nchop', 'both']:
        #     raise ValueError('Invalid chopping selection')
        #
        # self.calculate_null_depth()

        # self.cleanup()





