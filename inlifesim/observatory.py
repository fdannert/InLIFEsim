from typing import Union
import multiprocessing as mp
from copy import deepcopy

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from joblib_progress import joblib_progress

from inlifesim.util import (
    harmonic_number_approximation,
    combine_to_full_observation,
)
from inlifesim.sources import (
    create_star,
    create_planet,
    create_localzodi,
    create_exozodi,
)
from inlifesim.perturbation import (
    stellar_leakage,
    exozodi_leakage,
    localzodi_leakage,
    sys_noise_chop,
)
from inlifesim.signal import planet_signal, fundamental_noise
from inlifesim.spectra import rms_frequency_adjust, create_pink_psd
from inlifesim.statistics import draw_sample


class Instrument(object):
    """
    Represents a simulated observatory instrument for simulating astrophysical
    sources and noise within a telescope system. The `Instrument` class handles
    calculations such as instrumental noise, photon noise, and systematic
    effects while allowing configurations for different observational
    scenarios.

    :Attributes:
        General Simulation Parameters:
            wl_bins (np.ndarray): Wavelength bins center positions in [m].
            wl_bin_widths (np.ndarray): Wavelength bin widths in [m].
            image_size (int): Simulation image size for exozodiacal light in
            [pix].
            diameter_ap (float): Diameter of primary mirrors in [m].
            flux_division (np.ndarray): Flux distribution among primary
            mirrors.
            throughput (float): Fraction of light transmission through optics.

        Star Parameters:
            dist_star (float): Distance to the target star system in [pc].
            radius_star (float): Radius of the host star in [solar radii].
            temp_star (float): Temperature of the host star in [K].
            lat_star (float): Ecliptic latitude of target star in [rad].
            l_sun (float): Stellar luminosity in [solar luminosities].

        Planet Parameters:
            temp_planet (float): Temperature of the planet in [K].
            radius_planet (float): Planet radius in [Earth radii].
            separation_planet (float): Star-planet orbital separation in [AU].

        Instrument Configuration:
            col_pos (np.ndarray): Positions of collectors in [m].
            phase_response (np.ndarray): Phase response of collector arms
            [rad].
            phase_response_chop (np.ndarray): Phase response in the chopped
             state.
            n_rot (float): Number of array rotations during observation.
            t_total (float): Total observation time in [s].
            t_exp (float): Exposure time per observation segment in [s].

        Performance and Simulation Parameters:
            n_cpu (int): Number of CPU cores for simulation processing.
            rms_mode (str): Mode for RMS values ('lay', 'static', or
            'wavelength').
            n_sampling_max (int): Maximum Fourier sampling points (default:
            1e7).
            simultaneous_chopping (bool): Whether both chop states are
            simulated (default: False).
            verbose (bool): If True, prints progress to the console (default:
            False).

        Noise Control Parameters:
            hyperrot_noise (str or None): Type of hyperrotation noise (default:
             None).
            d_a_rms, d_phi_rms, d_pol_rms, d_x_rms, d_y_rms (float): RMS values
             for noise components.
            d_a_co, d_phi_co, d_pol_co, d_x_co, d_y_co (float): Cutoff
            frequencies for noise PSD components.

        Time Series Sampling:
            draw_samples (bool): If True, enables time-series sampling
            (default: False).
            n_draws (int or None): Number of time-series draws to generate.
            n_draws_per_run (int or None): Number of draws per subprocess run.
            time_series_return_values (str or list): Values to return for
            time-series draws ('all' by default).

        Custom Inputs and Outputs:
            flux_planet (np.ndarray): Custom flux input in units of
            [ph m⁻² s⁻¹].
            instrumental_source (str or None): Source of instrumental noise
                                               ('star', 'ez', or None for
                                               general).

    :Example:
        instrument = Instrument(
            wl_bins=np.array([...]),
            wl_bin_widths=np.array([...]),
            image_size=128,
            diameter_ap=6.5,
            flux_division=np.array([...]),
            throughput=0.2,
            dist_star=10.0,
            radius_star=1.0,
            temp_star=5800,
            lat_star=0.2,
            l_sun=1.0,
            z=1.0,
            temp_planet=300,
            radius_planet=1.0,
            separation_planet=1.0,
            col_pos=np.array([...]),
            phase_response=np.array([...]),
            phase_response_chop=np.array([...]),
            n_rot=25,
            t_total=3600,
            t_exp=10,
            n_cpu=4,
        )
        instrument.run()
        photon_data = instrument.photon_rates_chop  # Access simulation
        results.
    """

    def __init__(
        self,
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
        time_series_return_values: Union[str, list] = "all",
        flux_planet: np.ndarray = None,
        simultaneous_chopping: bool = False,
        verbose: bool = False,
        draw_samples: bool = False,
        get_single_bracewell: bool = False,
        instrumental_source: Union[None, str] = None,
    ):
      
        self.verbose = verbose
        self.draw_samples = draw_samples
        self.n_draws = n_draws
        self.n_draws_per_run = n_draws_per_run
        self.time_samples_return_values = time_series_return_values
        self.get_single_bracewell = get_single_bracewell
        self.instrumental_source = instrumental_source

        # If `draw_samples` mode is enabled, ensure the `n_draws` is specified;
        # otherwise, raise an error.
        if self.draw_samples and (n_draws is None):
            raise ValueError("Sample size must be set in sampling mode")

        # setting simulation parameters
        # Initialize general simulation parameters (e.g., wavelength bins,
        # image size).
        self.wl_bins = wl_bins
        self.wl_bin_widths = wl_bin_widths
        self.image_size = image_size

        if self.n_sampling_rot % 2 == 0:
            self.n_sampling_rot += 1
            if self.verbose:
                print('Sampling rate was adjusted to be odd')

        self.n_cpu = n_cpu
        self.n_sampling_max = n_sampling_max

        self.simultaneous_chopping = simultaneous_chopping

        # setting instrument parameters
        # Initialize instrument-specific parameters, such as collector
        # positions, phase responses, and aperture diameter.
        self.col_pos = col_pos
        self.phi = phase_response
        self.phi_r = phase_response_chop
        self.diameter_ap = diameter_ap

        # currently hardcoded to a Double-Bracewell beam combiner design, but
        # implemented to be changed in the future
        # Hardcoded for Double-Bracewell beam combiner design with two outputs;
        # extensibility planned for future.
        self.n_outputs = 2

        self.throughput = throughput
        self.flux_division = flux_division

        self.t_total = t_total
        self.t_exp = t_exp
        self.n_rot = int(n_rot)

        # Calculate the rotation duration based on total observation time and
        # number of rotations.
        self.t_rot = self.t_total / self.n_rot

        # Ensure the number of rotations is odd for compatibility with current
        # implementation requirements.
        if self.n_rot % 2 == 0:
            # todo: make even number of rotations possible
            raise ValueError("Number of rotations must be odd")

        # adjust exposure time to be a multiple of the rotation period
        t_exp_old = deepcopy(self.t_exp)
        self.t_exp = self.t_rot / np.ceil(self.t_rot / self.t_exp)
        if int(self.t_rot / self.t_exp) % 2 == 0:
            self.t_exp = self.t_rot / (int(self.t_rot / self.t_exp) + 1)

        # resulting numbers of samples
        self.n_sampling_total = int(np.round(self.t_total / self.t_exp))
        self.n_sampling_rot = int(np.round(self.t_rot / self.t_exp))

        if self.verbose:
            print(
                "Adjusted exposure time from {} s to {} s".format(
                    np.round(t_exp_old, 2), np.round(self.t_exp, 2)
                )
            )
            print(
                "Will simulate {} rotations in {} days".format(
                    self.n_rot, np.round(self.t_total / (24 * 60 * 60), 2)
                )
            )
            print("Total number of samples: {}".format(self.n_sampling_total))

        # create the array rotation angles
        # Generate single rotation angles evenly spaced between 0 and 2π for
        # rotation simulation.
        phi_rot_single = np.linspace(
            0, 2 * np.pi, self.n_sampling_rot, endpoint=False
        )

        self.phi_rot = combine_to_full_observation(
            arr=phi_rot_single,
            t_total=self.t_total,
            t_rot=self.t_rot,
            t_exp=self.t_exp,
        )

        if self.verbose:
            print("Number of rotation angles: {}".format(len(self.phi_rot)))

        # Hyperrotation noise model and root mean square (RMS) perturbation
        # settings for instrumental errors.
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

        if self.rms_mode == "lay":
            self.d_a_co = 10e3
            self.d_phi_co = 10e3
            self.d_pol_co = 10e3
            self.d_x_co = 0.64e-3
            self.d_y_co = 0.64e-3

        self.harmonic_number_n_cutoff = {

            "a": harmonic_number_approximation(self.d_a_co * self.t_total),
            "phi": harmonic_number_approximation(self.d_phi_co * self.t_total),
            "pol": harmonic_number_approximation(self.d_pol_co * self.t_total),
            "x": harmonic_number_approximation(self.d_x_co * self.t_total),
            "y": harmonic_number_approximation(self.d_y_co * self.t_total),

        }

        # Initialize source parameters, including stellar (e.g., distance,
        # radius, temperature)
        # and planetary parameters (e.g., size, temperature, and orbital
        # separation).
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

        # Initialize statistical properties, including power spectral density
        # vectors
        # and placeholders for sampled time-series data.
        self.d_a_psd = None
        self.d_phi_psd = None
        self.time_samples = None

        self.photon_rates_chop = pd.DataFrame(
            columns=[
                "signal",  # planet signal
                "noise",  # overall noise contribution
                "wl",  # wavelength bin
                "pn_sgl",  # stellar geometric leakage
                "pn_ez",  # exozodi leakage
                "pn_lz",  # localzodi leakage
                "pn_dc",  # dark current
                "pn_tbd",  # thermal background detector
                "pn_tbpm",  # thermal background primary mirror
                "pn_pa",  # polarization angle
                "pn_snfl",  # stellar null floor leakage
                "pn_ag_cld",  # agnostic cold instrumental photon noise
                "pn_ag_ht",  # agnostic hot instrumental photon noise
                "pn_ag_wht",  # agnostic white instrumental photon noise
                "pn",  # photon noise
                "sn_fo_a",  # first order amplitude
                "sn_fo_phi",  # first order phase
                "sn_fo_x",  # first order x position
                "sn_fo_y",  # first order y position
                "sn_fo",  # systematic noise first order
                "sn_so_aa",  # second order amplitude-amplitude term
                "sn_so_phiphi",  # second order phase-phase term
                "sn_so_aphi",  # amplitude phase cross term
                # second order polarization-polarization term
                "sn_so_polpol",
                "sn_so",  # systematic noise second order
                "sn",  # systematic noise
                "fundamental",  # fundamental noise (astrophysical)
                "instrumental",  # instrumental noise
                "snr",  # signal to noise ratio
            ],
            index=[str(np.round(wl * 1e6, 2)) for wl in self.wl_bins],
        )

        # Setup photon rate data tables (e.g., signal, noise, SNR) for
        # simulations with/without chopping.
        self.photon_rates_nchop = pd.DataFrame(
            columns=[
                "signal",  # planet signal
                "noise",  # overall noise contribution
                "wl",  # wavelength bin
                "pn_sgl",  # stellar geometric leakage
                "pn_ez",  # exozodi leakage
                "pn_lz",  # localzodi leakage
                "pn_dc",  # dark current
                "pn_tbd",  # thermal background detector
                "pn_tbpm",  # thermal background primary mirror
                "pn_pa",  # polarization angle
                "pn_snfl",  # stellar null floor leakage
                "pn_ag_cld",  # agnostic cold instrumental photon noise
                "pn_ag_ht",  # agnostic hot instrumental photon noise
                "pn_ag_wht",  # agnostic white instrumental photon noise
                "pn",  # photon noise
                "sn_fo_a",  # first order amplitude
                "sn_fo_phi",  # first order phase
                "sn_fo_x",  # first order x position
                "sn_fo_y",  # first order y position
                "sn_fo",  # systematic noise first order
                "sn_so_aa",  # second order amplitude-amplitude term
                "sn_so_phiphi",  # second order phase-phase term
                "sn_so_aphi",  # amplitude phase cross term
                # second order polarization-polarization term
                "sn_so_polpol",
                "sn_so",  # systematic noise second order
                "sn",  # systematic noise
                "fundamental",  # fundamental noise (astrophysical)
                "instrumental",  # instrumental noise
                "snr",  # signal to noise ratio
            ],
            index=[str(np.round(wl * 1e6, 2)) for wl in self.wl_bins],
        )

        self.photon_rates_nchop["wl"] = self.wl_bins
        self.photon_rates_chop["wl"] = self.wl_bins

        np.seterr(invalid="ignore")

    def instrumental_parameters(self):
        """
        Calculates and sets various instrumental parameters for a telescope
        simulation. These parameters include metrics such as effective area,
        field of view, resolution
        in various units (radians, milli-arcseconds, astronomical units), and
        pixel-related
        values used for further computations. These calculations assume
        geometric properties
        of the telescope and observational setup.

        :raises ValueError: Raised when division by zero occurs in the aperture
        calculations or other parameters depend on values that could result in
        invalid computations.

        Attributes
        ----------
        A : numpy.ndarray
            Effective area of the telescope aperture in square meters,
            considering flux division and throughput.

        num_a : int
            Number of elements in the effective area array.

        bl : numpy.ndarray
            Baseline separations in two dimensions for the input array of
            collector positions in meters.

        omega : numpy.ndarray
            Solid angle in steradians corresponding to the wavelength bins and
            the aperture diameter.

        hfov : numpy.ndarray
            Half field of view in radians for each wavelength bin in radians.

        rad_pix : numpy.ndarray
            Angular resolution in radians per pixel.

        radius_map : numpy.ndarray
            Map of radii for each pixel in the image grid.

        r_au : numpy.ndarray
            Three-dimensional array giving radial distances in astronomical units
            based on each pixel for various wavelength bins.

        Other Attributes
        ----------------
        hfov_mas : numpy.ndarray
            Half field of view in milli-arcseconds for each wavelength bin.

        mas_pix : numpy.ndarray
            Angular resolution in milli-arcseconds per pixel.

        au_pix : numpy.ndarray
            Angular resolution converted to astronomical units per pixel,
            as a function of the telescope distance to the star.

        telescope_area : float
            Total geometric area of the telescope aperture.
        """

        self.A = np.sqrt(
            np.pi
            * (0.5 * self.diameter_ap) ** 2
            * self.throughput
            * self.flux_division
        )
        # area term A_j
        self.num_a = len(self.A)

        self.bl = np.array(
            (
                np.subtract.outer(self.col_pos[:, 0], self.col_pos[:, 0]).T,
                np.subtract.outer(self.col_pos[:, 1], self.col_pos[:, 1]).T,
            )
        )

        self.omega = 1 * np.pi * (self.wl_bins / (2.0 * self.diameter_ap)) ** 2

        self.hfov = self.wl_bins / (2.0 * self.diameter_ap)

        hfov_mas = self.hfov * (3600000.0 * 180.0) / np.pi
        self.rad_pix = (2 * self.hfov) / self.image_size  # Radians per pixel
        mas_pix = (2 * hfov_mas) / self.image_size  # mas per pixel
        self.au_pix = mas_pix / 1e3 * self.dist_star  # AU per pixel

        telescope_area = 4.0 * np.pi * (self.diameter_ap / 2.0) ** 2

        x_map = np.tile(
            np.array(range(0, self.image_size)), (self.image_size, 1)
        )
        y_map = x_map.T
        r_square_map = (x_map - (self.image_size - 1) / 2) ** 2 + (
            y_map - (self.image_size - 1) / 2
        ) ** 2
        self.radius_map = np.sqrt(r_square_map)
        self.r_au = (
            self.radius_map[np.newaxis, :, :]
            * self.au_pix[:, np.newaxis, np.newaxis]
        )

    def combine_coefficients(self):
        """
        Combines the gradient and Hessian coefficients by summing up contributions
        from multiple component sources. The method processes each wavelength bin
        to calculate the combined coefficients for gradients and Hessians. Additionally,
        it calculates the "chopped" versions of these coefficients.

        :Attributes:
            grad_n_coeff : list[dict]
                List of dictionaries containing combined gradient coefficients for
                each wavelength bin.
            hess_n_coeff : list[dict]
                List of dictionaries containing combined Hessian coefficients for
                each wavelength bin.
            grad_n_coeff_chop : list[dict]
                List of dictionaries containing combined "chopped" gradient
                coefficients for each wavelength bin.
            hess_n_coeff_chop : list[dict]
                List of dictionaries containing combined "chopped" Hessian
                coefficients for each wavelength bin.
        """
        self.grad_n_coeff = []
        self.hess_n_coeff = []
        self.grad_n_coeff_chop = []
        self.hess_n_coeff_chop = []
        for i in range(len(self.wl_bins)):
            self.grad_n_coeff.append(
                {
                    k: self.grad_star[k][i]
                    + self.grad_ez[k][i]
                    + self.grad_lz[k][i]
                    for k in self.grad_star.keys()
                }
            )
            self.hess_n_coeff.append(
                {
                    k: self.hess_star[k][i] + self.hess_ez[k][i]
                    for k in self.hess_star.keys()
                }
            )

            self.grad_n_coeff_chop.append(
                {
                    k: self.grad_star_chop[k][i]
                    + self.grad_ez_chop[k][i]
                    + self.grad_lz[k][i]
                    for k in self.grad_star.keys()
                }
            )
            self.hess_n_coeff_chop.append(
                {
                    k: self.hess_star_chop[k][i] + self.hess_ez_chop[k][i]
                    for k in self.hess_star.keys()
                }
            )

    def build_gradient_hessian(
        self,
        gradient: dict,
        hessian: dict,
        order_gradient: tuple = ("a", "phi", "x", "y"),
        order_hessian: tuple = (("aa", "aphi"), ("aphi", "phiphi")),
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

        gradient_vector = np.concatenate(
            (
                gradient[order_gradient[0]],
                gradient[order_gradient[1]],
                gradient[order_gradient[2]],
                gradient[order_gradient[3]],
            ),
            axis=1,
        )
        hessian_matrix = np.concatenate(
            (
                np.concatenate(
                    (
                        hessian[order_hessian[0][0]],
                        hessian[order_hessian[0][1]],
                    ),
                    axis=2,
                ),
                np.concatenate(
                    (
                        np.transpose(
                            hessian[order_hessian[1][0]], axes=(0, 2, 1)
                        ),
                        hessian[order_hessian[1][1]],
                    ),
                    axis=2,
                ),
            ),
            axis=1,
        )

        return gradient_vector, hessian_matrix

    def combine_fundamental(self):
        """
        Computes the combination of photon rate signals into a fundamental value and
        associated signal-to-noise ratio (SNR). This is done under the context of
        chopping and non-chopping modes for photon rate data. Modifications are
        made based on the simultaneous chopping property, which introduces
        specific scaling adjustments to photon rates for pn_sgl, pn_ez, and pn_lz.
        The method consolidates the photon rates into a single measure
        (fundamental) and calculates its SNR.

        :raises KeyError: If any required photon rate key is missing in
            `self.photon_rates_nchop` or `self.photon_rates_chop`.

        """
        # because of the incoherent combination of the final outputs, see
        # Mugnier 2006
        if self.simultaneous_chopping:
            # Thank you Philipp!
            self.photon_rates_nchop["pn_sgl"] *= np.sqrt(2)
            self.photon_rates_nchop["pn_ez"] *= np.sqrt(2)
            self.photon_rates_nchop["pn_lz"] *= np.sqrt(2)

        self.photon_rates_nchop["fundamental"] = np.sqrt(
            self.photon_rates_nchop["pn_sgl"] ** 2
            + self.photon_rates_nchop["pn_ez"] ** 2
            + self.photon_rates_nchop["pn_lz"] ** 2
        )

        self.photon_rates_nchop["snr"] = (
            self.photon_rates_nchop["signal"]
            / self.photon_rates_nchop["fundamental"]
        )

        self.photon_rates_chop["pn_sgl"] = self.photon_rates_nchop["pn_sgl"]
        self.photon_rates_chop["pn_ez"] = self.photon_rates_nchop["pn_ez"]
        self.photon_rates_chop["pn_lz"] = self.photon_rates_nchop["pn_lz"]
        self.photon_rates_chop["fundamental"] = self.photon_rates_nchop[
            "fundamental"
        ]
        self.photon_rates_chop["snr"] = self.photon_rates_nchop["snr"]

    def sn_chop(self):
        """
        Prepare arguments for multiprocessing and execute calculations for System Noise Chop.

        The `sn_chop` function orchestrates the preparation of arguments for various
        scenarios (e.g., different instrumental sources) and computes the system noise
        chop. It uses either multiprocessing or a single-core processing, depending
        on the value of `self.n_cpu`. Results are collected and formatted as a
        pandas DataFrame before being updated to the photon rate table.

        :raises ValueError: If the provided instrumental source is not recognized.

        :param mp_args: Arguments passed to multiprocessing workers.
        :type mp_args: list of dict
        :param res: Results computed from `sys_noise_chop` function.
        :type res: pandas.DataFrame

        :example:
            (No example provided as per given requirements)

        :note:
            This function modifies `self.photon_rates_chop` in-place with the computed
            results of the system noise chop.

        :return: None
        """
        # prepare variable dictionary to send to multiprocessing workers
        mp_args = []
        for i in range(self.wl_bins.shape[0]):

            arg = {
                "A": self.A,
                "wl": self.wl_bins[i],
                "num_a": self.num_a,
                "planet_template_chop": self.planet_template_chop[i, :],
                "rms_mode": self.rms_mode,
                "n_sampling_max": self.n_sampling_max,
                "harmonic_number_n_cutoff": self.harmonic_number_n_cutoff,
                "t_total": self.t_total,
                "d_a_rms": self.d_a_rms,
                "d_phi_rms": self.d_phi_rms,
                "d_pol_rms": self.d_pol_rms,
                "flux_star": self.flux_star[i],
                "n_rot": self.n_rot,
                "hyperrot_noise": self.hyperrot_noise,
            }

            if self.instrumental_source is None:
                arg["grad_n_coeff"] = self.grad_n_coeff[i]
                arg["hess_n_coeff"] = self.hess_n_coeff[i]
                arg["grad_n_coeff_chop"] = self.grad_n_coeff_chop[i]
                arg["hess_n_coeff_chop"] = self.hess_n_coeff_chop[i]

            elif self.instrumental_source == "star":
                arg["grad_n_coeff"] = {
                    k: self.grad_star[k][i] for k in self.grad_star.keys()
                }
                arg["hess_n_coeff"] = {
                    k: self.hess_star[k][i] for k in self.hess_star.keys()
                }
                arg["grad_n_coeff_chop"] = {
                    k: self.grad_star_chop[k][i]
                    for k in self.grad_star_chop.keys()
                }
                arg["hess_n_coeff_chop"] = {
                    k: self.hess_star_chop[k][i]
                    for k in self.hess_star_chop.keys()
                }

            elif self.instrumental_source == "ez":
                arg["grad_n_coeff"] = {
                    k: self.grad_ez[k][i] for k in self.grad_ez.keys()
                }
                arg["hess_n_coeff"] = {
                    k: self.hess_ez[k][i] for k in self.hess_ez.keys()
                }
                arg["grad_n_coeff_chop"] = {
                    k: self.grad_ez_chop[k][i] for k in self.grad_ez_chop.keys()
                }
                arg["hess_n_coeff_chop"] = {
                    k: self.hess_ez_chop[k][i] for k in self.hess_ez_chop.keys()
                }

            else:
                raise ValueError("Instrumental source not recognized")
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
                    if np.round(a=r["wl"], decimals=10) == np.round(
                        a=wl, decimals=10
                    ):
                        res.append(r)

        # update the results to the photon rate table
        res = pd.DataFrame.from_dict(res)
        res["wl"] = np.round(res["wl"] * 1e6, 2).astype(str)
        res.set_index(keys="wl", inplace=True)
        self.photon_rates_chop.update(res)

    def combine_instrumental(self):
        """
        Combines various photon rate components to compute aggregated metrics.

        This method computes several key photon rate metrics including
        pn (combined photon noise), instrumental noise, overall noise, and
        signal-to-noise ratio (snr). Each of these metrics is derived by
        aggregating or combining their respective components through a
        square root of squared sums. Additionally, in the case of
        simultaneous chopping, these aggregated results are scaled by a
        factor of square root of 2 to account for the incoherent
        combination associated with the final outputs.

        Reference
        ---------
        Mugnier 2006: For details about the incoherent combination
        applied in simultaneous chopping.

        Raises
        ------
        KeyError: If any of the required keys are missing from
        `self.photon_rates_chop`.

        Attributes
        ----------
        photon_rates_chop
            A dictionary structure holding individual photon rate
            components such as noise, signal, instrumentals, and their
            intermediate elements.

        simultaneous_chopping
            A boolean flag indicating whether simultaneous chopping is
            active, which adjusts the metrics' scaling.

        """
        self.photon_rates_chop["pn"] = np.sqrt(
            (
                self.photon_rates_chop["pn_sgl"] ** 2
                # the sqaure is missing on purpose, see perturbation.py
                + self.photon_rates_chop["pn_snfl"] ** 2
                + self.photon_rates_chop["pn_lz"] ** 2
                + self.photon_rates_chop["pn_ez"] ** 2
                + self.photon_rates_chop["pn_pa"] ** 2
            ).astype(float)
        )

        self.photon_rates_chop["instrumental"] = np.sqrt(
            (
                self.photon_rates_chop["sn"] ** 2
                + self.photon_rates_chop["pn_pa"] ** 2
                + self.photon_rates_chop["pn_snfl"] ** 2
            ).astype(float)
        )

        self.photon_rates_chop["noise"] = np.sqrt(
            (
                self.photon_rates_chop["pn"] ** 2
                + self.photon_rates_chop["sn"] ** 2
            ).astype(float)
        )

        self.photon_rates_chop["snr"] = (
            self.photon_rates_chop["signal"] / self.photon_rates_chop["noise"]
        )

        """
        # because of the incoherent combination of the final outputs, see 
        # Mugnier 2006
        if self.simultaneous_chopping:
            self.photon_rates_chop['noise'] *= np.sqrt(2)
            self.photon_rates_chop['fundamental'] *= np.sqrt(2)
            self.photon_rates_chop['instrumental'] *= np.sqrt(2)                                    
        """

    def draw_time_series(self):
        """
        Draws time series data for a single wavelength channel while handling the computation
        of perturbation terms, PSDs, and time sampling. Multi-processing capabilities for
        parallelizing the time series generation are also provided.

        This method generates time series samples based on the input parameters, computes the
        necessary perturbation terms, and evaluates PSDs while ensuring the results are
        organized into a consolidated format for further analysis.

        :raises ValueError:
            If the wavelength bins contain more than one channel, as
            time series generation is limited to a single wavelength
            channel.
        :param self:
            Instance containing the parameters and attributes needed
            for performing the time series computation.
        """

        if len(self.wl_bins) != 1:
            raise ValueError(
                "Time series sampling is currently only supported"
                "in single wavelength channels"
            )

        d_a_rms, d_phi_rms, _, _, _ = rms_frequency_adjust(
            rms_mode=self.rms_mode,
            wl=self.wl_bins[0],
            d_a_rms=self.d_a_rms,
            d_phi_rms=self.d_phi_rms,
            d_pol_rms=self.d_pol_rms,
            d_x_rms=self.d_x_rms,
            d_y_rms=self.d_y_rms,
        )

        # calculate the PSDs of the perturbation terms
        self.d_a_psd, _, _ = create_pink_psd(

            t_total=self.t_total,
            n_sampling_max=int(self.n_sampling_total / 2),
            harmonic_number_n_cutoff=self.harmonic_number_n_cutoff["a"],

            rms=d_a_rms,
            num_a=self.num_a,
            n_rot=self.n_rot,
            hyperrot_noise=self.hyperrot_noise,
        )

        self.d_phi_psd, _, _ = create_pink_psd(

            t_total=self.t_total,
            n_sampling_max=int(self.n_sampling_total / 2),
            harmonic_number_n_cutoff=self.harmonic_number_n_cutoff["phi"],

            rms=d_phi_rms,
            num_a=self.num_a,
            n_rot=self.n_rot,
            hyperrot_noise=self.hyperrot_noise,
        )

        if self.get_single_bracewell:
            ps = self.planet_signal_nchop
        else:
            ps = self.planet_signal_chop

        params = {
            "n_sampling_rot": self.n_sampling_total,
            "n_outputs": self.n_outputs,
            "pn_sgl": self.photon_rates_nchop["pn_sgl"],
            "pn_ez": self.photon_rates_nchop["pn_ez"],
            "pn_lz": self.photon_rates_nchop["pn_lz"],
            "d_a_psd": self.d_a_psd,
            "d_phi_psd": self.d_phi_psd,
            "t_rot": self.t_total,
            "gradient": self.grad_n_coeff[0],
            "gradient_chop": self.grad_n_coeff_chop[0],
            "hessian": self.hess_n_coeff[0],
            "hessian_chop": self.hess_n_coeff_chop[0],
            "planet_signal": ps,
            "planet_template": self.planet_template_chop,
        }


        if self.n_cpu == 1:
            params["n_draws"] = self.n_draws
            self.time_samples = draw_sample(
                params=params, return_variables=self.time_samples_return_values
            )
        else:
            if self.verbose:
                print("")
                print("Drawing time series in multiprocessing")

            params["n_draws"] = self.n_draws_per_run
            with parallel_config(
                backend="loky", inner_max_num_threads=1
            ), joblib_progress(
                description="Calculating time series ...",
                total=int(self.n_draws / self.n_draws_per_run),
            ):
                results = Parallel(n_jobs=self.n_cpu)(
                    delayed(draw_sample)(
                        params=params,
                        return_variables=self.time_samples_return_values,
                    )
                    for _ in range(int(self.n_draws / self.n_draws_per_run))
                )

            if self.verbose:
                print("Combining results ...", end=" ")

            # combine the results dicts into single dict
            # create empty dict with the same keys as the first result and
            # properly sized arrays
            self.time_samples = {}
            time_samples_head = {}

            dtype_c = ["d_a_ft", "d_phi_ft"]

            for k in results[0].keys():
                size = np.array(results[0][k].shape)
                size[np.argwhere(size == self.n_draws_per_run)] = self.n_draws
                if k in dtype_c:
                    dtype = complex
                else:
                    dtype = float
                self.time_samples[k] = np.zeros(size, dtype=dtype)
                time_samples_head[k] = 0

            # fill the arrays with the results
            for r in results:
                for k in r.keys():
                    size = np.array(r[k].shape)
                    axis = np.argwhere(size == self.n_draws_per_run)

                    put = [slice(None) for _ in range(len(size))]
                    put[axis[0][0]] = slice(
                        time_samples_head[k],
                        time_samples_head[k] + self.n_draws_per_run,
                    )
                    put = tuple(put)

                    self.time_samples[k][put] = r[k]
                    time_samples_head[k] += self.n_draws_per_run

            if self.verbose:
                print("[Done]")

    def cleanup(self):
        """
        Converts photon rates data types to float format for both chopped and non-chopped
        photon rates. This ensures compatibility and accuracy when performing further
        calculations or data processing.

        :return: None
        """
        self.photon_rates_chop = self.photon_rates_chop.astype(float)
        self.photon_rates_nchop = self.photon_rates_nchop.astype(float)

    def run(self) -> None:
        """
        Executes the main operational flow of the class.

        This method comprises several sequential steps that include the initialization
        of instrumental parameters, generation of astrophysical sources, calculation
        of gradients and Hessian coefficients, generation of planet signals, evaluation
        of fundamental noise, sampling of time series (if enabled), and computing
        systematic noise alongside the final cleanup process. Depending on the
        `verbose` flag, it provides status updates for each major step, aiding in runtime
        debugging or monitoring.

        :raises ValueError: If any required attributes are missing or improperly set.
        :raises RuntimeError: If an error occurs during external function calls or intermediate operations.
        """
        self.instrumental_parameters()

        if self.verbose:
            print("Creating astrophysical sources ...", end=" ")
        self.flux_star, self.b_star, self.db_star_dx, self.db_star_dy = (
            create_star(
                wl_bins=self.wl_bins,
                wl_bin_widths=self.wl_bin_widths,
                temp_star=self.temp_star,
                radius_star=self.radius_star,
                dist_star=self.dist_star,
                bl=self.bl,
                col_pos=self.col_pos,
                num_a=self.num_a,
            )
        )

        if self.flux_planet is None:
            self.flux_planet = create_planet(
                wl_bins=self.wl_bins,
                wl_bin_widths=self.wl_bin_widths,
                temp_planet=self.temp_planet,
                radius_planet=self.radius_planet,
                dist_star=self.dist_star,
            )

        self.flux_localzodi = create_localzodi(
            wl_bins=self.wl_bins,
            wl_bin_widths=self.wl_bin_widths,
            lat=self.lat_star,
        )

        self.b_ez = create_exozodi(
            wl_bins=self.wl_bins,
            wl_bin_widths=self.wl_bin_widths,
            z=self.z,
            l_sun=self.l_sun,
            r_au=self.r_au,
            image_size=self.image_size,
            au_pix=self.au_pix,
            rad_pix=self.rad_pix,
            radius_map=self.radius_map,
            bl=self.bl,
            hfov=self.hfov,
        )

        if self.verbose:
            print("[Done]")
            print("Calculating gradient and Hessian coefficients ...", end=" ")

        self.grad_star, self.hess_star = stellar_leakage(
            A=self.A,
            phi=self.phi,
            b_star=self.b_star,
            db_star_dx=self.db_star_dx,
            db_star_dy=self.db_star_dy,
            num_a=self.num_a,
        )

        self.grad_star_chop, self.hess_star_chop = stellar_leakage(
            A=self.A,
            phi=self.phi_r,
            b_star=self.b_star,
            db_star_dx=self.db_star_dx,
            db_star_dy=self.db_star_dy,
            num_a=self.num_a,
        )

        self.grad_ez, self.hess_ez = exozodi_leakage(
            A=self.A, phi=self.phi, b_ez=self.b_ez, num_a=self.num_a
        )

        self.grad_ez_chop, self.hess_ez_chop = exozodi_leakage(
            A=self.A, phi=self.phi_r, b_ez=self.b_ez, num_a=self.num_a
        )

        self.grad_lz = localzodi_leakage(
            A=self.A, omega=self.omega, flux_localzodi=self.flux_localzodi
        )

        self.combine_coefficients()

        if self.verbose:
            print("[Done]")
            print("Generating planet signal ...", end=" ")

        (
            self.planet_template_nchop,
            self.photon_rates_nchop["signal"],
            self.planet_signal_nchop,
            self.planet_template_chop,
            self.photon_rates_chop["signal"],
            self.planet_signal_chop,
        ) = planet_signal(
            flux_planet=self.flux_planet,
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
            t_int=self.t_int,
        )

        if self.verbose:
            print("[Done]")
            print(
                "Shape of the planet template: {}".format(
                    self.planet_template_chop.shape
                )
            )
            print("Calculating fundamental noise ...", end=" ")

        (
            self.photon_rates_nchop["pn_sgl"],
            self.photon_rates_nchop["pn_ez"],
            self.photon_rates_nchop["pn_lz"],
        ) = fundamental_noise(
            A=self.A,
            phi=self.phi,
            num_a=self.num_a,
            t_int=self.t_int,
            flux_localzodi=self.flux_localzodi,
            b_star=self.b_star,
            b_ez=self.b_ez,
            omega=self.omega,
        )

        self.combine_fundamental()

        if self.verbose:
            print("[Done]")

        if self.draw_samples:
            if self.verbose:
                print("Drawing the time series ...", end=" ")
            self.draw_time_series()
            if self.verbose:
                print("[Done]")

        if self.verbose:
            print("Calculating systematics noise (chopping) ...", end=" ")
        self.sn_chop()
        self.combine_instrumental()
        if self.verbose:
            print("[Done]")

        self.cleanup()
