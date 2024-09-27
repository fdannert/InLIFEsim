from typing import Union
import multiprocessing as mp
from copy import deepcopy

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from joblib_progress import joblib_progress

from inlifesim.util import (harmonic_number_approximation,
                            combine_to_full_observation)
from inlifesim.sources import (create_star, create_planet, create_localzodi,
                               create_exozodi)
from inlifesim.perturbation import (stellar_leakage, exozodi_leakage,
                                    localzodi_leakage, sys_noise_chop)
from inlifesim.signal import planet_signal, fundamental_noise
from inlifesim.spectra import rms_frequency_adjust, create_pink_psd
from inlifesim.statistics import draw_sample


class Instrument(object):

    def __init__(self,
                 wl_bins: np.ndarray,
                 wl_bin_widths: np.ndarray,
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
                 n_rot: float,
                 t_total: float,
                 t_exp: float,
                 # n_sampling_rot: int,
                 n_cpu: int,
                 rms_mode: str,
                 hyperrot_noise: Union[str, type(None)] = None,
                 n_sampling_max: int = int(1e7),
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
                 n_draws: Union[int, type(None)] = None,
                 n_draws_per_run: Union[int, type(None)] = None,
                 time_series_return_values: Union[str, list] = 'all',
                 flux_planet: np.ndarray = None,
                 simultaneous_chopping: bool = False,
                 verbose: bool = False,
                 draw_samples: bool = False,
                 get_single_bracewell: bool = False,
                 instrumental_source: Union[None, str] = None,
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
        n_rot: int
            number of array rotations over the observation. Needs to be an odd
            number
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
        self.n_draws = n_draws
        self.n_draws_per_run = n_draws_per_run
        self.time_samples_return_values = time_series_return_values
        self.get_single_bracewell = get_single_bracewell
        self.instrumental_source = instrumental_source

        if self.draw_samples and (n_draws is None):
            raise ValueError('Sample size must be set in sampling mode')

        # setting simulation parameters
        self.wl_bins = wl_bins
        self.wl_bin_widths = wl_bin_widths
        self.image_size = image_size

        self.n_cpu = n_cpu
        self.n_sampling_max = n_sampling_max

        self.simultaneous_chopping = simultaneous_chopping

        # setting instrument parameters
        self.col_pos = col_pos
        self.phi = phase_response
        self.phi_r = phase_response_chop
        self.diameter_ap = diameter_ap

        # currently hardcoded to a Double-Bracewell beam combiner design, but
        # implemented to be changed in the future
        self.n_outputs = 2

        self.throughput = throughput
        self.flux_division = flux_division

        self.t_total = t_total
        self.t_exp = t_exp
        self.n_rot = int(n_rot)

        self.t_rot = self.t_total / self.n_rot

        if self.n_rot % 2 == 0:
            # todo: make even number of rotations possible
            raise ValueError('Number of rotations must be odd')


        # adjust exposure time to be a multiple of the rotation period
        t_exp_old = deepcopy(self.t_exp)
        t_total_old = deepcopy(self.t_total)
        self.t_exp = self.t_rot / np.ceil(self.t_rot / self.t_exp)
        if int(self.t_rot / self.t_exp) % 2 == 0:
            self.t_exp = self.t_rot / (int(self.t_rot / self.t_exp) + 1)

        # resulting numbers of samples
        self.n_sampling_total = int(np.round(self.t_total / self.t_exp))
        self.n_sampling_rot = int(np.round(self.t_rot / self.t_exp))

        if self.verbose:
            print('Adjusted exposure time from {} s to {} s'.format(
                np.round(t_exp_old, 2),
                np.round(self.t_exp, 2)))
            print('Will simulate {} rotations in {} days'.format(
                self.n_rot,
                np.round(self.t_total / (24 * 60 * 60), 2)
            ))
            print('Total number of samples: {}'.format(self.n_sampling_total))

        # create the array rotation angles
        phi_rot_single = np.linspace(0,
                                     2 * np.pi,
                                     self.n_sampling_rot,
                                     endpoint=False)

        self.phi_rot = combine_to_full_observation(arr=phi_rot_single,
                                                   t_total=self.t_total,
                                                   t_rot=self.t_rot,
                                                   t_exp=self.t_exp)

        if self.verbose:
            print('Number of rotation angles: {}'.format(len(self.phi_rot)))

        self.hyperrot_noise = hyperrot_noise

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
            'a': harmonic_number_approximation(self.d_a_co*self.t_total),
            'phi': harmonic_number_approximation(self.d_phi_co*self.t_total),
            'pol': harmonic_number_approximation(self.d_pol_co*self.t_total),
            'x': harmonic_number_approximation(self.d_x_co*self.t_total),
            'y': harmonic_number_approximation(self.d_y_co*self.t_total)
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

        # planet signal
        self.planet_template_nchop = None
        self.planet_template_chop = None

        # instrumental parameters
        self.A = None
        self.num_a = None
        self.bl = None
        self.omega = None
        self.hfov = None
        self.rad_pix = None
        self.au_pix = None
        self.radius_map = None
        self.r_au = None

        # sensitivity coefficients
        self.grad_star = None
        self.hess_star = None

        self.grad_star_chop = None
        self.hess_star_chop = None

        self.grad_ez = None
        self.hess_ez = None

        self.grad_ez_chop = None
        self.hess_ez_chop = None

        self.grad_lz = None

        self.grad_n_coeff = None
        self.hess_n_coeff = None
        self.grad_n_coeff_chop = None
        self.hess_n_coeff_chop = None

        # statistics
        self.d_a_psd = None
        self.d_phi_psd = None
        self.time_samples = None

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
                     'snr' # signal to noise ratio
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
        self.grad_n_coeff_chop = []
        self.hess_n_coeff_chop = []
        for i in range(len(self.wl_bins)):
            self.grad_n_coeff.append({k: self.grad_star[k][i]
                                         + self.grad_ez[k][i]
                                         + self.grad_lz[k][i]
                                      for k in self.grad_star.keys()})
            self.hess_n_coeff.append({k: self.hess_star[k][i]
                                         + self.hess_ez[k][i]
                                      for k in self.hess_star.keys()})

            self.grad_n_coeff_chop.append({k: self.grad_star_chop[k][i]
                                              + self.grad_ez_chop[k][i]
                                              + self.grad_lz[k][i]
                                           for k in self.grad_star.keys()})
            self.hess_n_coeff_chop.append({k: self.hess_star_chop[k][i]
                                              + self.hess_ez_chop[k][i]
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
        # because of the incoherent combination of the final outputs, see
        # Mugnier 2006
        if self.simultaneous_chopping:
            # Thank you Philipp!
            self.photon_rates_nchop['pn_sgl'] *= np.sqrt(2)
            self.photon_rates_nchop['pn_ez'] *= np.sqrt(2)
            self.photon_rates_nchop['pn_lz'] *= np.sqrt(2)

        self.photon_rates_nchop['fundamental'] = np.sqrt(
            self.photon_rates_nchop['pn_sgl'] ** 2
            + self.photon_rates_nchop['pn_ez'] ** 2
            + self.photon_rates_nchop['pn_lz'] ** 2
        )

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
            arg = {
                'A': self.A,
                'wl': self.wl_bins[i],
                'num_a': self.num_a,
                'planet_template_chop': self.planet_template_chop[i, :],
                'rms_mode': self.rms_mode,
                'n_sampling_max': self.n_sampling_max,
                'harmonic_number_n_cutoff':
                    self.harmonic_number_n_cutoff,
                't_total': self.t_total,
                'd_a_rms': self.d_a_rms,
                'd_phi_rms': self.d_phi_rms,
                'd_pol_rms': self.d_pol_rms,
                'flux_star': self.flux_star[i],
                'n_rot': self.n_rot,
                'hyperrot_noise': self.hyperrot_noise
            }

            if self.instrumental_source is None:
                arg['grad_n_coeff'] = self.grad_n_coeff[i]
                arg['hess_n_coeff'] = self.hess_n_coeff[i]
                arg['grad_n_coeff_chop'] = self.grad_n_coeff_chop[i]
                arg['hess_n_coeff_chop'] = self.hess_n_coeff_chop[i]

            elif self.instrumental_source == 'star':
                arg['grad_n_coeff'] = {k: self.grad_star[k][i]
                                       for k in self.grad_star.keys()}
                arg['hess_n_coeff'] = {k: self.hess_star[k][i]
                                       for k in self.hess_star.keys()}
                arg['grad_n_coeff_chop'] = {
                    k: self.grad_star_chop[k][i]
                    for k in self.grad_star_chop.keys()
                }
                arg['hess_n_coeff_chop'] = {
                    k: self.hess_star_chop[k][i]
                    for k in self.hess_star_chop.keys()
                }

            elif self.instrumental_source == 'ez':
                arg['grad_n_coeff'] = {k: self.grad_ez[k][i]
                                       for k in self.grad_ez.keys()}
                arg['hess_n_coeff'] = {k: self.hess_ez[k][i]
                                       for k in self.hess_ez.keys()}
                arg['grad_n_coeff_chop'] = {
                    k: self.grad_ez_chop[k][i]
                    for k in self.grad_ez_chop.keys()
                }
                arg['hess_n_coeff_chop'] = {
                    k: self.hess_ez_chop[k][i]
                    for k in self.hess_ez_chop.keys()
                }

            else:
                raise ValueError('Instrumental source not recognized')
            mp_args.append(arg)
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
                    if np.round(a=r['wl'],
                                decimals=10) == np.round(a=wl,
                                                         decimals=10):
                        res.append(r)

        # update the results to the photon rate table
        res = pd.DataFrame.from_dict(res)
        res['wl'] = np.round(res['wl'] * 1e6, 2).astype(str)
        res.set_index(keys='wl', inplace=True)
        self.photon_rates_chop.update(res)

    def combine_instrumental(self):
        self.photon_rates_chop['pn'] = np.sqrt(
            (self.photon_rates_chop['pn_sgl'] ** 2
            # the sqaure is missing on purpose, see perturbation.py
             + self.photon_rates_chop['pn_snfl'] ** 2
             + self.photon_rates_chop['pn_lz'] ** 2
             + self.photon_rates_chop['pn_ez'] ** 2
             + self.photon_rates_chop['pn_pa'] ** 2).astype(float)
        )

        self.photon_rates_chop['instrumental'] = np.sqrt(
            (self.photon_rates_chop['sn'] ** 2
             + self.photon_rates_chop['pn_pa'] ** 2
             + self.photon_rates_chop['pn_snfl'] ** 2).astype(float)
        )

        self.photon_rates_chop['noise'] = np.sqrt(
            (self.photon_rates_chop['pn'] ** 2
             + self.photon_rates_chop['sn'] ** 2).astype(float)
        )

        self.photon_rates_chop['snr'] = (self.photon_rates_chop['signal']
                                         / self.photon_rates_chop['noise'])

        '''
        # because of the incoherent combination of the final outputs, see 
        # Mugnier 2006
        if self.simultaneous_chopping:
            self.photon_rates_chop['noise'] *= np.sqrt(2)
            self.photon_rates_chop['fundamental'] *= np.sqrt(2)
            self.photon_rates_chop['instrumental'] *= np.sqrt(2)                                    
        '''

    def draw_time_series(self):
        '''
        Draw samples from noise distributions
        :return:
        '''

        if len(self.wl_bins) != 1:
            raise ValueError('Time series sampling is currently only supported'
                             'in single wavelength channels')

        d_a_rms, d_phi_rms, _, _, _ = rms_frequency_adjust(
            rms_mode=self.rms_mode,
            wl=self.wl_bins[0],
            d_a_rms=self.d_a_rms,
            d_phi_rms=self.d_phi_rms,
            d_pol_rms=self.d_pol_rms,
            d_x_rms=self.d_x_rms,
            d_y_rms=self.d_y_rms
        )

        # calculate the PSDs of the perturbation terms
        self.d_a_psd, _, _ = create_pink_psd(
            t_total=self.t_total,
            n_sampling_max=int(self.n_sampling_total / 2),
            harmonic_number_n_cutoff=self.harmonic_number_n_cutoff['a'],
            rms=d_a_rms,
            num_a=self.num_a,
            n_rot=self.n_rot,
            hyperrot_noise=self.hyperrot_noise
        )

        self.d_phi_psd, _, _ = create_pink_psd(
            t_total=self.t_total,
            n_sampling_max=int(self.n_sampling_total / 2),
            harmonic_number_n_cutoff=self.harmonic_number_n_cutoff['phi'],
            rms=d_phi_rms,
            num_a=self.num_a,
            n_rot=self.n_rot,
            hyperrot_noise=self.hyperrot_noise
        )

        if self.get_single_bracewell:
            ps = self.planet_signal_nchop
        else:
            ps = self.planet_signal_chop

        params = {'n_sampling_rot': self.n_sampling_total,
                  'n_outputs': self.n_outputs,
                  'pn_sgl': self.photon_rates_nchop['pn_sgl'],
                  'pn_ez': self.photon_rates_nchop['pn_ez'],
                  'pn_lz': self.photon_rates_nchop['pn_lz'],
                  'd_a_psd': self.d_a_psd,
                  'd_phi_psd': self.d_phi_psd,
                  't_rot': self.t_total,
                  'gradient': self.grad_n_coeff[0],
                  'gradient_chop': self.grad_n_coeff_chop[0],
                  'hessian': self.hess_n_coeff[0],
                  'hessian_chop': self.hess_n_coeff_chop[0],
                  'planet_signal': ps,
                  'planet_template': self.planet_template_chop}

        if self.n_cpu == 1:
            params['n_draws'] = self.n_draws
            self.time_samples = draw_sample(
                params=params,
                return_variables=self.time_samples_return_values
            )
        else:
            if self.verbose:
                print('')
                print('Drawing time series in multiprocessing')

            params['n_draws'] = self.n_draws_per_run
            with parallel_config(
                    backend="loky",
                    inner_max_num_threads=1
            ), joblib_progress(
                description="Calculating time series ...",
                total=int(self.n_draws / self.n_draws_per_run)
            ):
                results = Parallel(
                    n_jobs=self.n_cpu
                )(delayed(draw_sample)(
                    params=params,
                    return_variables=self.time_samples_return_values
                ) for _ in range(int(self.n_draws / self.n_draws_per_run)))

            if self.verbose:
                print('Combining results ...', end=' ')

            # combine the results dicts into single dict
            # create empty dict with the same keys as the first result and
            # properly sized arrays
            self.time_samples = {}
            time_samples_head = {}
            for k in results[0].keys():
                size = np.array(results[0][k].shape)
                size[np.argwhere(size == self.n_draws_per_run)] = self.n_draws
                self.time_samples[k] = np.zeros(size)
                time_samples_head[k] = 0

            # fill the arrays with the results
            for r in results:
                for k in r.keys():
                    size = np.array(r[k].shape)
                    axis = np.argwhere(size == self.n_draws_per_run)

                    put = [slice(None) for _ in range(len(size))]
                    put[axis[0][0]] = slice(
                        time_samples_head[k],
                        time_samples_head[k] + self.n_draws_per_run
                    )
                    put = tuple(put)

                    self.time_samples[k][put] = r[k]
                    time_samples_head[k] += self.n_draws_per_run

            if self.verbose:
                print('[Done]')

    def cleanup(self):
        self.photon_rates_chop = self.photon_rates_chop.astype(float)
        self.photon_rates_nchop = self.photon_rates_nchop.astype(float)

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

        self.grad_star_chop, self.hess_star_chop = stellar_leakage(
            A=self.A,
            phi=self.phi_r,
            b_star=self.b_star,
            db_star_dx=self.db_star_dx,
            db_star_dy=self.db_star_dy,
            num_a=self.num_a
        )

        self.grad_ez, self.hess_ez = exozodi_leakage(
            A=self.A,
            phi=self.phi,
            b_ez=self.b_ez,
            num_a=self.num_a
        )

        self.grad_ez_chop, self.hess_ez_chop = exozodi_leakage(
            A=self.A,
            phi=self.phi_r,
            b_ez=self.b_ez,
            num_a=self.num_a
        )

        self.grad_lz = localzodi_leakage(A=self.A,
                                         omega=self.omega,
                                         flux_localzodi=self.flux_localzodi)

        self.combine_coefficients()

        if self.verbose:
            print('[Done]')
            print('Generating planet signal ...', end=' ')

        (self.planet_template_nchop,
         self.photon_rates_nchop['signal'],
         self.planet_signal_nchop,
         self.planet_template_chop,
         self.photon_rates_chop['signal'],
         self.planet_signal_chop) = planet_signal(
            flux_planet=self.flux_planet,
            t_exp=self.t_exp,
            t_total=self.t_total,
            t_rot=self.t_rot,
            A=self.A,
            phi=self.phi,
            phi_r=self.phi_r,
            wl_bins=self.wl_bins,
            bl=self.bl,
            num_a=self.num_a,
            n_sampling_rot=self.n_sampling_rot,
            simultaneous_chopping=self.simultaneous_chopping,
            separation_planet=self.separation_planet,
            dist_star=self.dist_star,
        )

        if self.verbose:
            print('[Done]')
            print('Shape of the planet template: {}'.format(
                self.planet_template_chop.shape
            ))
            print('Calculating fundamental noise ...', end=' ')

        (self.photon_rates_nchop['pn_sgl'],
         self.photon_rates_nchop['pn_ez'],
         self.photon_rates_nchop['pn_lz']) = fundamental_noise(
            A=self.A,
            phi=self.phi,
            num_a=self.num_a,
            t_int=self.t_total,
            flux_localzodi=self.flux_localzodi,
            b_star=self.b_star,
            b_ez=self.b_ez,
            omega=self.omega
        )

        self.combine_fundamental()

        if self.verbose:
            print('[Done]')

        if self.draw_samples:
            if self.verbose:
                print('Drawing the time series ...', end=' ')
            self.draw_time_series()
            if self.verbose:
                print('[Done]')

        if self.verbose:
            print('Calculating systematics noise (chopping) ...', end=' ')
        self.sn_chop()
        self.combine_instrumental()
        if self.verbose:
            print('[Done]')

        self.cleanup()
