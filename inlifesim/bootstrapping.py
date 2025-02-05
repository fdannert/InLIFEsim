from os import listdir
from os.path import isfile, join
import itertools
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from cmocean.cm import thermal, balance
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
import pickle

from inlifesim.statistics import get_sigma_lookup


def logistic(x, x0, k, ymin, ymax):
    """
    Computes the logistic function value for a given input. This implementation models
    an S-shaped curve, often used in classification problems and population growth
    models. The function is defined by its midpoint, growth rate, minimum value,
    and maximum value.

    The logistic function is defined as follows:
        f(x) = (ymax - ymin) / (1 + exp(-k * (x - x0))) + ymin

    Parameters control the shape and behavior of the curve:
    - x represents the input value.
    - x0 determines the midpoint of the curve.
    - k specifies how steep the curve is.
    - ymin and ymax set the lower and upper bounds of the function's range.

    :param x: Input value for which the logistic function needs to be calculated.
    :param x0: The midpoint of the S-curve, where its value is halfway between ymin and ymax.
    :param k: The steepness of the curve, controls the rate of growth in the function.
    :param ymin: The minimum value of the S-curve.
    :param ymax: The maximum value of the S-curve.

    :return: The computed value of the logistic function as a float.
    """
    return (ymax - ymin) / (1 + np.exp(-k * (x - x0))) + ymin


def inverse_logistic(y, x0, k, ymin, ymax):
    """
    Perform the inverse of a logistic transformation to compute the original value
    before being scaled with a logistic function. This function is useful for
    rescaling values that were mapped to a constrained output range via a logistic
    function back to their original domain.

    :param y: The transformed value or output of the logistic function with
        constraints applied. Assumes that the value of 'y' is within the range
        [ymin, ymax].
    :param x0: The midpoint parameter of the logistic function corresponding to the
        input value that yields the midpoint of the logistic output range.
    :param k: The steepness of the logistic curve, controlling how rapidly the
        logistic function scales its input values.
    :param ymin: The minimum bound for the logistic function's output range.
    :param ymax: The maximum bound for the logistic function's output range.
    :return: The original value before applying the logistic transformation, rescaled
        back from the constrained output range [ymin, ymax].
    :rtype: float
    """
    return np.log((y - ymin) / (ymax - y) - 1) / k + x0


def get_interest_area(k, rt):
    """
    Calculate the interest area based on the provided growth factor and retention rate.

    This function computes the interest area using the given growth factor ``k``
    and retention rate ``rt``. The calculation is based on a specified mathematical
    formula involving a logarithmic transformation.

    :param k: Growth factor used for the calculation.
    :type k: float
    :param rt: Retention rate, a value typically within the range (0, 1).
    :type rt: float
    :return: The computed interest area as a float value.
    :rtype: float
    """
    return -np.log(1 / rt - 1) / k


class EvaluateBootstrapping:
    """
    Class to evaluate the bootstrapping process through logistic fitting
    and extrapolation. This class is responsible for processing data,
    performing logistic fitting to the data, and extrapolating fitting
    parameters for different sigma values. The purpose is to describe the
    underlying relationship between sigma values using logistic functions.

    The class supports various features such as data loading, polynomial
    fitting, and visualization of fitted parameters and residuals.

    :ivar verbose: Flag indicating whether verbose output is enabled.
    :type verbose: bool
    :ivar paths: List of paths where the data is stored.
    :type paths: list
    :ivar crit_value: A critical value used for bootstrapping analysis.
    :type crit_value: float
    :ivar bootstrap_properties: Dictionary specifying properties related to
        bootstrapping.
    :type bootstrap_properties: dict
    :ivar sigma_gauss: Collection of Gaussian sigma values extracted from the
        loaded data.
    :type sigma_gauss: numpy.ndarray
    :ivar sigma_bessel: Collection of Bessel sigma values extracted from the
        loaded data.
    :type sigma_bessel: numpy.ndarray
    :ivar sigma_want_get: Dictionary mapping (Gaussian sigma, Bessel sigma)
        pairs to the corresponding analytical and actual sigma data arrays.
    :type sigma_want_get: dict
    :ivar logistic_extrapolation: List of polynomial fitting coefficients for
        extrapolated logistic parameters.
    :type logistic_extrapolation: list
    """

    def __init__(self, verbose, paths, crit_value, bootstrap_properties):
        """
        Initializes a new instance of the class.

        This constructor initializes the class with the given parameters and sets up
        the necessary attributes. The ``paths`` parameter is used to load data when
        initializing the object. The ``bootstrap_properties`` are set for any bootstrap-related
        functionalities that may be used later. The parameter ``crit_value`` initializes a
        threshold for critical operations or comparisons. Verbose option helps in controlling
        detailed output representation if implemented elsewhere.

        :param verbose: Determines whether to enable verbose mode for output representation.
        :type verbose: bool
        :param paths: List or location of file paths to load data from.
        :type paths: Any
        :param crit_value: Critical threshold value for comparison or evaluation.
        :type crit_value: float | int
        :param bootstrap_properties: Configuration or properties to initialize bootstrap-related behavior.
        :type bootstrap_properties: dict
        """
        self.verbose = verbose
        self.paths = paths
        self.load_data(paths)
        self.crit_value = crit_value
        self.bootstrap_properties = bootstrap_properties

    def load_data(self, paths: list):
        """
        Loads data from specified paths, processes numerical values from filenames and
        loads corresponding numpy arrays. The method combines the loaded data into
        shared attributes for further use.

        :param paths: A list of file directory paths where data files are located.
        :type paths: list
        :return: None. The processed data is stored in the instance attributes.
        """
        sigma_gauss = []
        sigma_bessel = []
        sigmas_want_get = []

        for path in paths:
            files = [f for f in listdir(path) if isfile(join(path, f))]

            sigmas = np.array(
                [
                    np.array(
                        f.replace(".npy", "").replace("-", ".").split("_")
                    )[-2:]
                    for f in files
                ]
            ).astype(float)

            sig_g = sigmas[:, 0]
            sig_b = sigmas[:, 1]

            sigma_want_get = {}
            for sg, si in zip(sig_g, sig_b):
                sg_str = str(sg).replace(".", "-")
                si_str = str(si).replace(".", "-")

                res = np.load(path + f"sigma_lookup_{sg_str}_{si_str}.npy")
                sigma_want_get[(sg, si)] = np.array(
                    (res[: int(len(res) / 2)], res[int(len(res) / 2) :])
                )

            sigma_gauss.append(sig_g)
            sigma_bessel.append(sig_b)
            sigmas_want_get.append(sigma_want_get)

        self.sigma_gauss = np.concatenate(sigma_gauss)
        self.sigma_bessel = np.concatenate(sigma_bessel)
        self.sigma_want_get = dict(
            itertools.chain.from_iterable(d.items() for d in sigmas_want_get)
        )

    def fit_logistic(
        self, sig_actual, plot=False, guess=None, bounds=(-np.inf, np.inf)
    ):
        """
        Fits a logistic model to the relationship between actual Gaussian noise and the
        Bessel-Gaussian noise ratios. The method uses a logistic function to interpret
        the trend between a logarithmic transformation of the noise ratio and the
        analytical estimation of noise. It optionally plots the data, the logistic fit,
        and residuals for visualization.

        :param sig_actual: The actual noise levels to estimate against, expressed as
                           a sequence of numerical values.
        :type sig_actual: list or numpy.ndarray
        :param plot: A boolean flag indicating whether to generate a plot showing
                     the fitted curve, data points, and residuals. Defaults to False.
        :type plot: bool
        :param guess: An optional parameter for providing an initial guess for the
                      logistic curve fitting parameters. Defaults to None.
        :type guess: list or tuple or None
        :param bounds: Bounds for the logistic curve fitting parameters, expressed
                       as a tuple with lower and upper bounds. Defaults to
                       (-numpy.inf, numpy.inf).
        :type bounds: tuple
        :return: Returns the optimized logistic model parameters as obtained by
                 the curve fitting process.
        :rtype: tuple
        """
        sig_analytical = []
        rat_bes_gaus = []

        for sg, si in zip(self.sigma_gauss, self.sigma_bessel):
            sig_analytical.append(
                np.interp(
                    sig_actual,
                    self.sigma_want_get[(sg, si)][0],
                    self.sigma_want_get[(sg, si)][1],
                )
            )
            rat_bes_gaus.append(si / sg)

        ratio_sig = np.array((sig_analytical, rat_bes_gaus)).T

        # noinspection PyTupleAssignmentBalance
        popt, _ = curve_fit(
            f=logistic,
            xdata=np.log10(ratio_sig[1:-1, 1]),
            ydata=ratio_sig[1:-1, 0],
            p0=guess,
            bounds=bounds,
        )

        if plot:
            x = np.log10(ratio_sig[1:-1, 1])
            y = ratio_sig[1:-1, 0]
            fig, ax = plt.subplots(nrows=2, height_ratios=[1, 0.3])
            # Plot the original data (with noise) and the fitted curve
            ax[0].scatter(x, y, label="Data", color="k")

            x_sample = np.linspace(np.min(x), np.max(x), 1000)
            ax[0].plot(
                x_sample,
                logistic(x_sample, *popt),
                "--",
                c="tab:orange",
                label="Logistic fit",
            )

            # print popt in the plot
            ax[0].text(
                0.5,
                0.5,
                f"x0={popt[0]:.2f}\nk={popt[1]:.2f}"
                f"\nymin={popt[2]:.2f}\nymax={popt[3]:.2f}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax[0].transAxes,
            )

            ax[0].set_xticks([])
            ax[0].legend()

            ax[1].set_xlabel(r"$\sigma_\mathcal{B} / \sigma_\mathcal{N}$")
            ax[0].set_ylabel(r"$\sigma_{\mathcal{N}, \mathrm{actual}}$")

            # draw horizontal line at y=0
            ax[1].axhline(0, color="gray", ls="--")
            # plot residual in lower axis
            ax[1].scatter(
                x, y - logistic(x, *popt), color="k", label="Data - Logistic"
            )
            ax[1].set_ylim([-0.05, 0.05])
            ax[1].set_ylabel("Residuals")

            plt.show()

        return popt

    def do_logistic_fit(
        self,
        sigma_actual_shape: tuple,
        make_n_plots: int = 10,
        initial_guess=None,
        bounds=(-np.inf, np.inf),
    ):
        """
        Performs a logistic fitting operation across a range of values for sigma_actual_shape,
        splitting the operation into `make_n_plots` segments for plotting if specified. This
        method attempts to fit logistic functions to simulated data using an optimization
        process. If plotting is enabled, plots are generated at specified intervals as the
        fits progress. The resulting fitted parameters for each sigma value are collected
        and returned as an array.

        :param sigma_actual_shape: Specifies the range of sigma values as a tuple of floating
            point numbers. The tuple should represent the range (min, max, step) of the
            sigma values to simulate.
        :type sigma_actual_shape: tuple
        :param make_n_plots: The number of equally spaced plotting segments. Determines how
            often plots are generated during the logistic fitting process. Defaults to 10.
        :type make_n_plots: int
        :param initial_guess: Initial guess for the optimization process. If None, the
            optimizer will attempt to infer a suitable initial condition. Defaults to None.
            Its structure typically depends on the specific logistic fitting function details.
        :type initial_guess: Optional[Any]
        :param bounds: Tuple specifying the lower and upper bounds for the parameters during
            optimization. Defaults to a range of (-np.inf, np.inf).
        :type bounds: tuple
        :return: A tuple containing the sigma_actual array (generated based on the provided
            sigma_actual_shape) and an array of fitted parameters for each sigma value. If a
            fit operation fails, NaN values are populated in the results.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        sig_actual = np.linspace(*sigma_actual_shape)
        fit_values = []

        plot_every_n = len(sig_actual) // make_n_plots

        for i, sa in enumerate(sig_actual):
            if i == 0:
                guess = initial_guess
            else:
                guess = fit_values[-1]

            try:
                if (i % plot_every_n == 0) and self.verbose:
                    fit_values.append(
                        self.fit_logistic(
                            sa, plot=True, guess=guess, bounds=bounds
                        )
                    )
                else:
                    fit_values.append(
                        self.fit_logistic(
                            sa, plot=False, guess=guess, bounds=bounds
                        )
                    )
            except:
                print(f"Failed at sigma_actual={sa}")
                fit_values.append([np.nan, np.nan, np.nan, np.nan])

        return sig_actual, np.array(fit_values)

    def extrapolate_logistic_parameters(
        self,
        sigma_actual_shape: tuple,
        start_fit_at_sigma: list,
        end_fit_at_sigma: list,
        polynomial_order: list,
        make_n_plots: int = 10,
    ):
        """
        Extrapolates the logistic fit parameters using a polynomial fit based on the given
        actual sigma shape and specified ranges. This method processes logistic parameters,
        fits polynomial curves to the data within specified ranges, and optionally visualizes
        the fits for each parameter.

        :param sigma_actual_shape: Actual shape of the sigma values to be used in the extrapolation process.
        :param start_fit_at_sigma: List containing the start points of sigma values for fitting each parameter.
        :param end_fit_at_sigma: List containing the end points of sigma values for fitting each parameter.
        :param polynomial_order: List specifying the polynomial order to be used for fitting each logistic parameter.
        :param make_n_plots: Number of plots to generate for visualizing the polynomial fit. Defaults to 10.
        :return: None
        """
        sig_actual, fit_values = self.do_logistic_fit(
            sigma_actual_shape=sigma_actual_shape, make_n_plots=make_n_plots
        )

        # fit a second order polynomial to all fit parameters
        self.logistic_extrapolation = []
        idx_starts = []
        idx_ends = []

        for i in range(4):
            idx_start = np.where(sig_actual > start_fit_at_sigma[i])[0][0]
            try:
                idx_end = np.where(sig_actual > end_fit_at_sigma[i])[0][0]
            except IndexError:
                idx_end = -1

            idx_starts.append(idx_start)
            idx_ends.append(idx_end)

            popt, _ = np.polyfit(
                sig_actual[idx_start:idx_end],
                fit_values[idx_start:idx_end, i],
                polynomial_order[i],
                cov=True,
            )
            self.logistic_extrapolation.append(popt)

        if self.verbose:
            fig, ax = plt.subplots(1, 4, figsize=(16, 3), dpi=200)

            param_names = [
                "$x_0$ [$\sigma_{\mathcal{N}, \mathrm{actual}}$]",
                r"$k$ [$\frac{1}{\sigma_{\mathcal{N}, \mathrm{actual}}}$]",
                "$y_\mathrm{min}$ [$\sigma_{\mathcal{N},"
                " \mathrm{analytical}}$]",
                "$y_\mathrm{max}$ [$\sigma_{\mathcal{N},"
                " \mathrm{analytical}}$]",
            ]

            titles = ["Center", "Steepness", "Minimum", "Maximum"]

            for i in range(4):
                x = np.linspace(2, 8, 100)
                y = np.polyval(self.logistic_extrapolation[i], x)
                ax[i].plot(x, y, c="tab:orange", label="Polynomial fit")

                # differentiate between data used in fitting and not
                ax[i].scatter(
                    sig_actual[: idx_starts[i]],
                    fit_values[: idx_starts[i], i],
                    c="gray",
                    s=2,
                    label="Data (not used in fit)",
                )
                ax[i].scatter(
                    sig_actual[idx_ends[i] :],
                    fit_values[idx_ends[i] :, i],
                    c="gray",
                    s=2,
                )
                ax[i].scatter(
                    sig_actual[idx_starts[i] : idx_ends[i]],
                    fit_values[idx_starts[i] : idx_ends[i], i],
                    c="k",
                    s=3,
                    label="Data",
                )
                ax[i].set_ylabel(param_names[i])
                ax[i].set_xlabel(r"$\sigma_{\mathcal{N}, \mathrm{actual}}$")
                ax[i].set_title(titles[i])
                ax[i].legend()

            plt.tight_layout()
            plt.show()

    def plot_extrapolation(self, sigma_actual_shape: tuple):
        """
        Generates and plots the extrapolation of analytical sigma values based on provided
        parameters and the logistic function. The function computes a series of extrapolated
        values using the logistic function fitted with parameters from predictions, and it
        visually compares these against the actual data.

        :param sigma_actual_shape: A tuple specifying the range and number of points for
            actual sigma values. This parameter is used to create the `np.linspace` of sigma
            values used in the plot.
        :return: None
        """
        sigma_ratios = deepcopy((self.sigma_bessel / self.sigma_gauss)[:-1])
        sigma_actual = np.linspace(*sigma_actual_shape)

        sigma_analytical = []

        for sr in sigma_ratios:
            sig_anas = []
            for sa in sigma_actual:
                # retrieve the fit parameters extrapolation
                y1 = np.polyval(self.logistic_extrapolation[0], sa)
                x0 = np.polyval(self.logistic_extrapolation[1], sa)
                k = np.polyval(self.logistic_extrapolation[2], sa)
                y0 = np.polyval(self.logistic_extrapolation[3], sa)

                # calculate the logistic function
                y = logistic(np.log10(sr), y1, x0, k, y0)

                sig_anas.append(y)
            sigma_analytical.append(sig_anas)

        sigma_analytical = np.array(sigma_analytical)

        fig, ax = plt.subplots()

        # extract N colors from the thermal colormap
        colors = thermal(np.linspace(0, 0.85, len(sigma_ratios)))

        for sg, si in zip(self.sigma_gauss, self.sigma_bessel):
            ax.plot(
                self.sigma_want_get[(sg, si)][1],
                self.sigma_want_get[(sg, si)][0],
                color="gray",
            )

        for i, c in enumerate(colors):
            ax.plot(
                sigma_analytical[i],
                sigma_actual,
                # format the labels in scientific notation
                label=f"${sigma_ratios[i]:.2e}$",
                color=c,
            )

        ax.legend(fontsize=5)
        plt.show()

    def interpolate_to_grid(
        self,
        points: np.ndarray,
        sigma_analytical_shape: tuple = (0, 8, 100),
        sigma_ratio_shape: tuple = (np.log10(6e-2), np.log10(5), 100),
        fill_nan: bool = True,
    ):
        """
        Interpolates the provided data points onto a grid defined by the specified
        analytical and ratio ranges for sigma. The interpolation can utilize nearest
        neighbor or linear methods. Missing values (`NaN`) can optionally be handled and
        filled by vertically propagating non-`NaN` values up and down along the grid.

        :param points: The input array containing data points to interpolate. Assumes
            the shape is (n, 3), where the first two columns represent x and y
            coordinates, and the third column contains the corresponding values.
        :type points: np.ndarray
        :param sigma_analytical_shape: A tuple representing the range and number of
            grid points for the analytical sigma axis. Default is (0, 8, 100).
        :param sigma_ratio_shape: A tuple specifying the logarithmic range and number
            of grid points for the sigma ratio axis. Default is
            (np.log10(6e-2), np.log10(5), 100).
        :param fill_nan: A boolean flag deciding whether missing (`NaN`) values in the
            interpolated result are to be filled. Default is True.
        :return: A tuple containing the interpolated grid, an array of x-coordinates,
            and an array of y-coordinates. The interpolated grid has the same number
            of points specified in the sigma analytical and ratio shape parameters.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        X, Y = np.meshgrid(
            np.linspace(*sigma_analytical_shape),
            np.logspace(*sigma_ratio_shape),
        )

        interp = griddata(
            points=points[:, :2],
            values=points[:, 2],
            xi=(X, Y),
            method="nearest",
        )

        if fill_nan:
            interp_ref = griddata(
                points=points[:, :2],
                values=points[:, 2],
                xi=(X, Y),
                method="linear",
            )

            interp[np.isnan(interp_ref).astype(bool)] = np.nan

            # copy nan values vertically up
            for i in range(interp.shape[0]):
                for j in range(interp.shape[1]):
                    if np.isnan(interp[i, j]):
                        interp[i, j] = interp[i - 1, j]

            # copy nan values vertically down
            for j in range(0, interp.shape[0]):
                for i in range(interp.shape[1] - 2, -1, -1):
                    if np.isnan(interp[i, j]):
                        interp[i, j] = interp[i + 1, j]

        return interp, X, Y

    def create_lookup_table(
        self,
        sigma_actual_shape: tuple,
        sigma_analytical_shape: tuple,
        sigma_ratio_shape: tuple,
        n_analytical: int,
        make_n_plots: int = 10,
    ):
        """
        Creates a lookup table for interpolating between analytical and actual sigma
        values, based on logistic fit of the relationship. This process involves
        fitting a logistic function to the given sigma values, generating analytical
        values, ratios, and generating the lookup table using gridded interpolation.
        Also includes optional visualization of intermediate and final results.

        :param sigma_actual_shape: Tuple defining the shape or range of actual sigma
            values for the lookup process.
        :type sigma_actual_shape: tuple
        :param sigma_analytical_shape: Tuple defining the shape or range of
            analytical sigma values to be used in the grid interpolation.
        :type sigma_analytical_shape: tuple
        :param sigma_ratio_shape: Tuple defining the shape or range of the sigma
            ratios to be used in the grid interpolation.
        :type sigma_ratio_shape: tuple
        :param n_analytical: Number of points to be generated for the analytical
            sigma values during interpolation.
        :type n_analytical: int
        :param make_n_plots: Optional number of diagnostic plots to generate for
            visualizing the fitting process and results. Defaults to 10.
        :type make_n_plots: int, optional
        :return: None
        """
        sigma_actual, fit_values = self.do_logistic_fit(
            sigma_actual_shape=sigma_actual_shape,
            make_n_plots=make_n_plots,
            initial_guess=[0, 10, 0.1, 0.11],
            bounds=([-np.inf, 0, 0, 0], [np.inf, 20, np.inf, np.inf]),
        )

        if self.verbose:
            # make a figure showing the evolution of all fit parameters
            fig, ax = plt.subplots(1, 4, figsize=(16, 3), dpi=200)

            param_names = [
                "$x_0$ [$\sigma_{\mathcal{N}, \mathrm{actual}}$]",
                r"$k$ [$\frac{1}{\sigma_{\mathcal{N}, \mathrm{actual}}}$]",
                "$y_\mathrm{min}$ [$\sigma_{\mathcal{N},"
                " \mathrm{analytical}}$]",
                "$y_\mathrm{max}$ [$\sigma_{\mathcal{N},"
                " \mathrm{analytical}}$]",
            ]

            titles = ["Center", "Steepness", "Minimum", "Maximum"]

            for i in range(4):
                # differentiate between data used in fitting and not
                ax[i].scatter(
                    sigma_actual, fit_values[:, i], c="k", s=3, label="Data"
                )
                ax[i].set_ylabel(param_names[i])
                ax[i].set_xlabel(r"$\sigma_{\mathcal{N}, \mathrm{actual}}$")
                ax[i].set_title(titles[i])
                ax[i].legend()

            plt.tight_layout()
            plt.show()

        crit_regions_size = np.array(
            [get_interest_area(fv[1], 0.99) for fv in fit_values]
        )  # half the size of the critical region
        crit_region = np.array(
            (
                fit_values[:, 0] - crit_regions_size,
                fit_values[:, 0] + crit_regions_size,
            )
        ).T

        sigma_analytical = []
        sigma_ratio = []

        for i in range(len(fit_values)):
            sig_r = np.linspace(
                crit_region[i, 0], crit_region[i, 1], n_analytical
            )
            sigma_ratio.append(sig_r)

            sigma_analytical.append(logistic(sig_r, *fit_values[i]))

        sigma_analytical = np.array(sigma_analytical)
        sigma_ratio = np.array(sigma_ratio)

        if self.verbose:
            fig, ax = plt.subplots()

            min_s = np.min(sigma_actual)
            max_s = np.max(sigma_actual)

            for i in range(len(sigma_actual)):
                ax.scatter(
                    sigma_analytical[i,],
                    10 ** sigma_ratio[i,],
                    c=thermal((sigma_actual[i] - min_s) / (max_s - min_s)),
                    s=3,
                )

            plt.xlabel(r"$\sigma_{\mathcal{N}, \mathrm{analytical}}$")
            plt.ylabel(r"$\sigma_{\mathcal{B} / \sigma_{\mathcal{N}}}$")

            plt.yscale("log")

        points_ratio = np.concatenate(
            [
                np.array(
                    (
                        sigma_analytical[i,],
                        10 ** sigma_ratio[i,],
                        sigma_actual[i]
                        * np.ones(sigma_analytical.shape[1])
                        / sigma_analytical[i,],
                    )
                ).T
                for i in range(len(sigma_actual))
            ]
        )
        points_ratio = points_ratio[~np.isnan(points_ratio).any(axis=1)]

        points = np.concatenate(
            [
                np.array(
                    (
                        sigma_analytical[i,],
                        10 ** sigma_ratio[i,],
                        sigma_actual[i] * np.ones(sigma_analytical.shape[1]),
                    )
                ).T
                for i in range(len(sigma_actual))
            ]
        )
        points = points[~np.isnan(points).any(axis=1)]

        self.lookup_ratio, X, Y = self.interpolate_to_grid(
            points=points_ratio,
            sigma_analytical_shape=sigma_analytical_shape,
            sigma_ratio_shape=sigma_ratio_shape,
            fill_nan=False,
        )
        self.lookup, self.lookup_X, self.lookup_Y = self.interpolate_to_grid(
            points=points,
            sigma_analytical_shape=sigma_analytical_shape,
            sigma_ratio_shape=sigma_ratio_shape,
            fill_nan=False,
        )

        if self.verbose:
            fig, ax = plt.subplots()

            plt.imshow(
                self.lookup_ratio,
                origin="lower",
                cmap=thermal,
                extent=(
                    sigma_analytical_shape[0],
                    sigma_analytical_shape[1],
                    sigma_ratio_shape[0],
                    sigma_ratio_shape[1],
                ),
                aspect="auto",
            )

            plt.colorbar()
            # add white contours
            plt.contour(
                X,
                np.log10(Y),
                self.lookup_ratio,
                levels=[0.8, 0.85, 0.9, 0.95, 1, 1.05],
                # levels=[1, 2, 3, 4, 5, 6, 7, 8],
                colors="w",
                linewidths=1,
            )

            # include the sigma_actual = 6 cutoff line
            sigma_ratios = np.logspace(-4, 4, 100)
            sigma_actual = np.linspace(6, 15, 1000)

            sigma_analytical = []

            for sr in sigma_ratios:
                sig_anas = []
                for sa in sigma_actual:
                    # retrieve the fit parameters extrapolation
                    y1 = np.polyval(self.logistic_extrapolation[0], sa)
                    x0 = np.polyval(self.logistic_extrapolation[1], sa)
                    k = np.polyval(self.logistic_extrapolation[2], sa)
                    y0 = np.polyval(self.logistic_extrapolation[3], sa)

                    # calculate the logistic function
                    y = logistic(np.log10(sr), y1, x0, k, y0)

                    sig_anas.append(y)
                sigma_analytical.append(sig_anas)

            sigma_analytical = np.array(sigma_analytical)
            ana_limits = np.array(
                (sigma_ratios, np.min(sigma_analytical, axis=1))
            )

            # plt.plot(ana_limits[1, ], np.log10(ana_limits[0, ]), color='r')

            plt.xlabel(r"$\sigma_{\mathcal{N}, \mathrm{analytical}}$")
            plt.ylabel(r"$\log(\sigma_{\mathcal{B} / \sigma_{\mathcal{N}}})$")

            # plt.ylim(-1.25, 0.75)

            plt.show()

    def create_extrapolated_lookup_table(
        self,
        sigma_actual_shape: tuple,
        sigma_ratio_shape: tuple,
        sigma_analytical_shape: tuple,
    ):
        """
        Create an extrapolated lookup table based on the provided sigma parameters.

        This method uses logistic fits and interpolation to compute an extrapolated
        lookup table for given shapes of sigma_actual, sigma_ratio, and sigma_analytical.
        The process involves evaluating logistic functions and projecting the output
        onto specified grids. Additionally, visualizations can be generated to inspect
        the results when verbose mode is enabled.

        :param sigma_actual_shape: A tuple defining the shape parameters to create a
            linear space for sigma_actual.
        :param sigma_ratio_shape: A tuple defining the shape parameters to create a
            logarithmic space for sigma_ratio.
        :param sigma_analytical_shape: A tuple defining the shape of the sigma_analytical
            grid for interpolation.
        :return: None
        """
        sigma_actual = np.linspace(*sigma_actual_shape)
        sigma_ratio = np.logspace(*sigma_ratio_shape)

        SAct, SR = np.meshgrid(sigma_actual, sigma_ratio)

        # which of the requested values are above the critical sigma_actual values
        # first, evaluate the data based on the logisitc fits
        x0 = np.polyval(self.logistic_extrapolation[0], SAct.flatten())
        k = np.polyval(self.logistic_extrapolation[1], SAct.flatten())
        ymin = np.polyval(self.logistic_extrapolation[2], SAct.flatten())
        ymax = np.polyval(self.logistic_extrapolation[3], SAct.flatten())

        # calculate the logistic function
        SAna = logistic(np.log10(SR.flatten()), x0, k, ymin, ymax)

        (
            self.lookup_extrapolation,
            self.logistic_extrapolation_X,
            self.lookup_extrapolation_Y,
        ) = self.interpolate_to_grid(
            points=np.array((SAna, SR.flatten(), SAct.flatten())).T,
            sigma_analytical_shape=sigma_analytical_shape,
            sigma_ratio_shape=sigma_ratio_shape,
            fill_nan=False,
        )

        if self.verbose:
            lookup_extrapolation_ratio, _, _ = self.interpolate_to_grid(
                points=np.array((SAna, SR.flatten(), SAct.flatten() / SAna)).T,
                sigma_analytical_shape=sigma_analytical_shape,
                sigma_ratio_shape=sigma_ratio_shape,
                fill_nan=False,
            )

            fig, ax = plt.subplots()
            plt.imshow(
                lookup_extrapolation_ratio,
                origin="lower",
                cmap=thermal,
                extent=(
                    sigma_analytical_shape[0],
                    sigma_analytical_shape[1],
                    sigma_ratio_shape[0],
                    sigma_ratio_shape[1],
                ),
                aspect="auto",
            )

            plt.contour(
                self.logistic_extrapolation_X,
                np.log10(self.lookup_extrapolation_Y),
                lookup_extrapolation_ratio,
                levels=[0.8, 0.85, 0.9, 0.95, 1, 1.05],
                # levels=[5, 10, 15, 20, 25],
                colors="w",
                linewidths=1,
            )
            plt.colorbar()
            plt.show()

    def save_lookup(self, path: str):
        """
        Saves the lookup data to the specified file path. The method serializes a
        dictionary containing data paths, bootstrap properties, lookup tables, and
        fit parameters into a pickle file.

        The resulting serialized data structure includes:
        - Critical value for lookup.
        - Data paths and bootstrap properties.
        - Lookup tables with corresponding coordinates and extrapolation data.
        - Fit parameters for logistic extrapolation.

        :param path: The file path where the serialized lookup data will be stored.
                     The data is written as a binary file.
        :type path: str

        :return: None
        """
        self.result = {
            "critical_value": 6,
            "data_path": self.paths,
            "bootstrap_properties": self.bootstrap_properties,
            "lookup_table": {
                "table": self.lookup,
                "X": self.lookup_X,
                "Y": self.lookup_Y,
                "coords": {
                    "X": "sigma_analytical",
                    "Y": "sigma_ratio",
                    "Z": "sigma_actual",
                },
            },
            "lookup_table_extrapolation": {
                "table": self.lookup_extrapolation,
                "X": self.logistic_extrapolation_X,
                "Y": self.lookup_extrapolation_Y,
                "coords": {
                    "X": "sigma_analytical",
                    "Y": "sigma_ratio",
                    "Z": "sigma_actual",
                },
            },
            "fit_parameters": {
                "x0": self.logistic_extrapolation[0],
                "k": self.logistic_extrapolation[1],
                "ymin": self.logistic_extrapolation[2],
                "ymax": self.logistic_extrapolation[3],
            },
        }
        with open(path, "wb") as f:
            pickle.dump(self.result, f)


class InterpretBootstrapping:
    """
    Manages the interpretation of bootstrapped data using a pre-loaded lookup
    table for mapping analytical sigma values and sigma ratios to actual sigma
    values. The class is designed to perform lookup operations for both large
    datasets and extrapolated data, handling edge cases where values fall
    below critical thresholds or reach non-positive levels.

    :ivar lookup_table: The loaded lookup table used for sigma calculations and
        data interpretation.
    :type lookup_table: dict
    """

    def __init__(self, lookup_table_path: str):
        """
        Initializes the class and loads a lookup table from a specified file.

        The constructor opens the file specified by the `lookup_table_path`
        parameter in binary read mode and loads the contents, expecting it to
        be a pickle-serialized lookup table. The loaded lookup table is stored
        as an instance variable.

        :param lookup_table_path: Path to the pickle file containing the lookup table
        :type lookup_table_path: str
        """
        with open(lookup_table_path, "rb") as f:
            self.lookup_table = pickle.load(f)

    def get_sigma_actual(
        self, sigma_analytical: np.ndarray, sigma_ratio: np.ndarray
    ):
        """
        Computes the actual sigma values based on analytical sigma and sigma ratio input arrays using a lookup table. The function performs a two-dimensional indexing on the lookup table to find the nearest lookup value or extrapolation value for each pair of input values.

        The method incorporates adjustments based on a critical value threshold, and if extrapolated sigma values are below specified thresholds, they are reassigned appropriate values from the lookup or the input analytical sigma array.

        :param sigma_analytical: An array of analytical sigma values.
        :type sigma_analytical: np.ndarray
        :param sigma_ratio: An array of sigma ratios corresponding to the analytical sigma values.
        :type sigma_ratio: np.ndarray
        :return: A numpy array containing the computed actual sigma values, combining lookup values, extrapolated values, and adjusted input values.
        :rtype: np.ndarray
        """
        idx = [
            np.unravel_index(
                np.argmin(
                    (
                        self.lookup_table["lookup_table"]["X"]["large"]
                        - sigma_analytical[i]
                    )
                    ** 2
                    + (
                        self.lookup_table["lookup_table"]["Y"]["large"]
                        - sigma_ratio[i]
                    )
                    ** 2
                ),
                self.lookup_table["lookup_table"]["table"]["large"].shape,
            )
            for i in range(len(sigma_analytical))
        ]

        sigma_actual_lookup = np.array(
            [
                self.lookup_table["lookup_table"]["table"]["large"][idx[i]]
                for i in range(len(idx))
            ]
        )

        idx = [
            np.unravel_index(
                np.argmin(
                    (
                        self.lookup_table["lookup_table"]["X"]["extrapolation"]
                        - sigma_analytical[i]
                    )
                    ** 2
                    + (
                        self.lookup_table["lookup_table"]["Y"]["extrapolation"]
                        - sigma_ratio[i]
                    )
                    ** 2
                ),
                self.lookup_table["lookup_table"]["table"][
                    "extrapolation"
                ].shape,
            )
            for i in range(len(sigma_analytical))
        ]

        sigma_actual_extrapolated = np.array(
            [
                self.lookup_table["lookup_table"]["table"]["extrapolation"][
                    idx[i]
                ]
                for i in range(len(idx))
            ]
        )

        sigma_actual_extrapolated[
            sigma_actual_extrapolated
            < self.lookup_table["critical_value"]["large"]
        ] = sigma_actual_lookup[
            sigma_actual_extrapolated
            < self.lookup_table["critical_value"]["large"]
        ]

        sigma_actual_extrapolated[sigma_actual_extrapolated <= 0.1] = (
            sigma_analytical[sigma_actual_extrapolated <= 0.1]
        )

        return sigma_actual_extrapolated


def run_bootstrapping(
    B: int,
    N: int,
    sigma_imb: np.ndarray,
    sigma_gauss: np.ndarray,
    path: str,
    B_per: int = int(1e7),
    n_cpu: int = 64,
    n_sigma: int = 10000,
):
    """
    Executes the bootstrapping process, computes sigma lookup tables for specified
    sigma Gaussian and imbalance values, and saves the computed results as files.
    This function utilizes parallel computing to improve performance for large
    datasets. Results are retrieved using `get_sigma_lookup` and later stored in
    the defined file structure in the provided path.

    :param B: The bootstrapping parameter representing the size of the ensemble.
              It determines the number of samples to be considered in each ensemble.
    :param N: The number of sample points considered for computations.
    :param sigma_imb: An array of sigma imbalance values for which sigma lookup tables
                      are to be computed.
    :param sigma_gauss: An array of sigma Gaussian values for which sigma lookup tables
                        are to be computed.
    :param path: The directory path where the computed results are saved. If the
                 provided path does not end with a '/', it will automatically append it.
    :param B_per: Determines the maximum size of batches to be processed per
                  iteration. Default: 10^7.
    :param n_cpu: Number of CPU cores used for parallel processing to speed up
                  computation. Default: 64.
    :param n_sigma: Number of sigma values sampled during lookup. Default: 10,000.

    :return: None. Results are saved as .npy files in the specified `path`.
    """
    i = 0

    for sg, si in zip(sigma_gauss, sigma_imb):
        print(
            "Working on sigma_gauss =",
            sg,
            "sigma_imb =",
            si,
            "(",
            i,
            "/",
            len(sigma_gauss),
            ")",
        )
        i += 1
        sigma_want, sigma_get = get_sigma_lookup(
            sigma_gauss=sg,
            sigma_imb=si,
            B=B,
            N=N,
            B_per=B_per,
            n_sigma=n_sigma,
            n_cpu=n_cpu,
            verbose=True,
            parallel=True,
        )
        print("Saving the results...", end=" ", flush=True)
        results = np.concatenate((sigma_want, sigma_get))

        sg_str = str(sg).replace(".", "-")
        si_str = str(si).replace(".", "-")

        # save the results
        if path[-1] != "/":
            path += "/"
        np.save(path + f"sigma_lookup_{sg_str}_{si_str}.npy", results)

        print("[Done]")
