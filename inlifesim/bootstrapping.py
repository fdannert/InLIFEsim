from os import listdir
from os.path import isfile, join
import itertools
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from cmocean.cm import thermal, thermal_r
import numpy as np
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
import pickle

from inlifesim.statistics import get_sigma_lookup
from inlifesim.util import (
    add_normalized_line_collection,
    HandlerColorLineCollection,
)


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
    This class provides functionality to evaluate statistical bootstrapping
    data and methods for plotting, data fitting, and analysis. It processes data
    generated with a specific bootstrap creation tool, and facilitates tasks like
    data loading, visualization of raw data, and fitting using logistic functions.

    Its primary purpose is to load and analyze different distributions, perform
    transformations, and apply logistic regression fitting for further statistical
    evaluation.

    :ivar verbose: Indicates whether verbose output is enabled.
    :type verbose: bool
    :ivar paths: A list of paths to datasets containing bootstrapping data.
    :type paths: list
    :ivar bootstrap_properties: Properties or parameters relevant to the bootstrapping process.
    :type bootstrap_properties: Any
    :ivar lookup: Internal dictionary for lookups in processes related to bootstrapping.
    :type lookup: dict
    :ivar lookup_X: Internal dictionary for X-coordinate lookups.
    :type lookup_X: dict
    :ivar lookup_Y: Internal dictionary for Y-coordinate lookups.
    :type lookup_Y: dict
    :ivar crit_value: Stores critical values used for bootstrapping computations.
    :type crit_value: dict
    :ivar shape: Stores the shape information of the analyzed distributions.
    :type shape: dict
    :ivar sigma_gauss: Gaussian standard deviations for all analyzed datasets.
    :type sigma_gauss: numpy.ndarray
    :ivar sigma_bessel: Besselian standard deviations for all analyzed datasets.
    :type sigma_bessel: numpy.ndarray
    :ivar sigma_want_get: Dictionary linking Gaussian and Bessel standard deviations
        to corresponding lookup data.
    :type sigma_want_get: dict
    """

    def __init__(self, verbose, paths, bootstrap_properties):
        """
        This class provides functionalities for initializing and handling data needed
        to create bootstrapping structures. It manages input properties, file paths,
        and storage for additional computed data structures.

        Attributes
        ----------
        verbose : bool
            A boolean indicating whether to display verbose output during processing.
        paths : str
            A string denoting the path to the input data for processing.
        bootstrap_properties : dict
            A dictionary containing the configuration properties for bootstrapping.
        lookup : dict
            A dictionary initialized to store lookup structures for data transformation.
        lookup_X : dict
            A dictionary initialized to store additional lookup structures for the
            independent variables or data attributes.
        lookup_Y : dict
            A dictionary initialized to store additional lookup structures for the
            dependent variables or target attributes.
        crit_value : dict
            A dictionary initialized to store any critical values derived during
            bootstrapping computations.
        shape : dict
            A dictionary initialized to store shape or dimensionality details of
            processed data.

        Methods
        -------
        __init__(verbose, paths, bootstrap_properties)
            Initializes the class instance with specified verbosity, file paths,
            and bootstrap properties, and prepares internal data structures.
        """
        self.verbose = verbose
        self.paths = paths
        self.load_data(paths)
        self.bootstrap_properties = bootstrap_properties

        self.lookup = {}
        self.lookup_X = {}
        self.lookup_Y = {}
        self.crit_value = {}
        self.shape = {}

        # todo: create the CreateBootstrappingData class

    def load_data(self, paths: list):
        """
        Load the data from a dataset created with CreateBootstrappingData
        Parameters
        ----------
        path: str
            The path to the dataset
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

            sorted_indices = np.lexsort(
                (sig_b, -sig_g)
            )  # Sort by sig_b ascending, then sig_g descending
            sigma_gauss.append(list(np.array(sig_g)[sorted_indices]))
            sigma_bessel.append(list(np.array(sig_b)[sorted_indices]))
            sigmas_want_get.append(sigma_want_get)

        self.sigma_gauss = np.concatenate(sigma_gauss)
        self.sigma_bessel = np.concatenate(sigma_bessel)
        self.sigma_want_get = dict(
            itertools.chain.from_iterable(d.items() for d in sigmas_want_get)
        )

    def plot_raw_data(self, figsize=(240 / 72, 180 / 72)):
        """
        Plots raw data depicting various sigma values and their relationships, visualized
        using color-coded lines with a thermal colormap. The function uses customizable
        figure size and includes a supplementary colorbar to represent sigma ratio labels.

        The plot presents the relationships among sigma values, specified as ratios,
        and encodes these through proper labeled lines and a visually aligned colorbar.
        A dashed reference line is included for comparative purposes, and axes are labeled
        to indicate their respective metrics. This visualization is primarily for analyzing
        and comparing sigma values and their derived ratios.

        :param figsize: Figure size specified as a tuple (width, height).
        :type figsize: tuple
        :return: Matplotlib figure and axis objects representing the constructed plot.
        :rtype: tuple
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=200)

        # extract N colors from the thermal colormap
        colors = thermal(np.linspace(0, 0.85, len(self.sigma_gauss)))

        for sg, si, c in zip(self.sigma_gauss, self.sigma_bessel, colors):
            if np.log10(si / sg) == -np.inf:
                label = "0"
            elif np.log10(si / sg) == np.inf:
                label = "$\infty$"
            elif int(np.log10(si / sg)) == 0 and np.log10(si / sg) < 0:
                label = f"{np.round(si / sg, 2)}"
            elif int(np.log10(si / sg)) == 0 and np.log10(si / sg) >= 0:
                label = f"{int(np.round(si / sg, 0))}"
            else:
                label = f"$10^{{{int(np.log10(si / sg))}}}$"
            ax.plot(
                self.sigma_want_get[(sg, si)][1],
                self.sigma_want_get[(sg, si)][0],
                # format the labels in scientific notation
                #    label=label,
                color=c,
            )

        ax.plot(
            self.sigma_want_get[(self.sigma_gauss[1], self.sigma_bessel[1])][0],
            self.sigma_want_get[(self.sigma_gauss[1], self.sigma_bessel[1])][0],
            color="lightgray",
            ls="--",
            alpha=1,
        )

        # in the bottom left of the plot, I want a small colorbar replacing the labels of the individual lines
        cbar_ax = fig.add_axes([0.55, 0.22, 0.3, 0.03])
        cbar = matplotlib.colorbar.ColorbarBase(
            ax=cbar_ax, cmap=thermal, orientation="horizontal"
        )

        cbar_ax.set_xlim(0, 0.85)

        # make the ticks of the colorbar appear on the top

        # cbar_ax.xaxis.set_ticks_position('top')

        pos_ticks = list(
            np.array([0, 7, 14]) / (len(self.sigma_gauss) - 1) * 0.85
        )
        ticks = ["0", "1", r"$\infty$"]

        # add a label of the colorbar to the right of the colorbar
        cbar.set_label(r"$\sigma_{\mathrm{IMB}} / \sigma_{\mathcal{N}}$")
        # make the label appear on top of the colorbar
        cbar_ax.xaxis.set_label_position("top")

        cbar.set_ticks(pos_ticks)
        cbar.set_ticklabels(ticks)

        # vertical line at 5
        # ax.axvline(5, color='k', ls='--', alpha=0.5)

        # plt.legend(fontsize=10, title=r'$\sigma_{\mathrm{IMB}} / \sigma_{\mathrm{Gauss}}$-ratio')

        ax.set_xlabel(r"$T_\alpha$")
        ax.set_ylabel(r"$T_\mathcal{N}$")

        # fig.savefig('sigma_lookup_samples.pdf', bbox_inches='tight')
        return fig, ax

    def fit_logistic(
        self, sig_actual, plot=False, guess=None, bounds=(-np.inf, np.inf)
    ):
        """
        Perform a logistic curve fitting based on given actual sigma values and pre-calculated
        sigma Bessel and Gauss ratios. This method utilizes interpolation coupled with
        `scipy.optimize.curve_fit` to calculate the logistic fit parameters, optionally displaying
        visualized fitting and residual plots.

        :param sig_actual: An array or list of actual sigma values to be used as the fitting reference.
        :param plot: A boolean indicating whether to display a plot of the fitting process and residuals.
        :param guess: Initial guess for the logistic parameters, provided as an array or list.
        :param bounds: Bounds for the parameters in the logistic fit, given as a tuple with shape
            (lower_bounds, upper_bounds).
        :return: A list containing the optimized parameters of the logistic function as obtained
            from the curve fitting process.
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
        update_guess=True,
    ):
        """
        Performs logistic fitting on the provided sigma values based on the specified
        shape range and fitting parameters. Each sigma value in the provided range
        is evaluated using a logistic fitting function. The fitting process can be
        either visualized or executed silently depending on the `make_n_plots` and
        `verbose` arguments. The initial guess and bounds can be updated dynamically
        through the fitting process.

        :param sigma_actual_shape: A tuple specifying the range (start, stop, step)
            for generating the sigma values to be evaluated.
        :param make_n_plots: An integer defining how many fitting processes should
            include plots for visualization during the iterations. Defaults to 10.
        :param initial_guess: Optional initial parameters for the logistic fitting.
            If not provided, defaults to None, and the algorithm employs the most
            recent values dynamically.
        :param bounds: Specifies the bounds for the logistic fitting parameters
            as a tuple. Defaults to (-inf, inf).
        :param update_guess: A boolean flag indicating whether to dynamically
            update the guessing parameters for the next iteration, based on the
            previous result. Defaults to True.
        :return: A tuple containing:
            - `sig_actual` (numpy.ndarray): The generated sigma values in the range
              specified by `sigma_actual_shape`.
            - `fit_values` (numpy.ndarray): An array containing the logistic fit
              parameters for each value in `sig_actual`.
        """
        sig_actual = np.linspace(*sigma_actual_shape)
        fit_values = []

        plot_every_n = len(sig_actual) // make_n_plots

        for i, sa in enumerate(sig_actual):
            if (i != 0) and update_guess and (fit_values[-1][0] > 0):
                guess = fit_values[-1]
            else:
                guess = initial_guess

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
        Extrapolates logistic distribution fitting parameters by performing a second-order polynomial
        fit on the parameters over a specified range. This method refines logistic fit parameters
        to allow extrapolation over the range of actual sigma values. Polynomial fits are performed
        for each parameter, and optional visualization of the fits is provided.

        :param sigma_actual_shape: Shape of the actual sigma values; determines the structure of
            the logistic fit input.
        :param start_fit_at_sigma: List of sigma values indicating the start of the range used for
            fitting each parameter.
        :param end_fit_at_sigma: List of sigma values indicating the end of the range used for fitting
            each parameter.
        :param polynomial_order: List of polynomial orders to be used for the fit of each parameter.
            Each order corresponds to the fitting performed for each parameter (e.g., for
            `center`, `steepness`, `min`, and `max`).
        :param make_n_plots: Optional number of plots to generate for visualizing the polynomial
            fit results for each parameter (default is 10).
        :return: Computes and stores the polynomial coefficients for each logistic parameter
            in `self.logistic_extrapolation`. Optionally generates a visualization if `self.verbose`
            is True.
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

    def plot_extrapolation(
        self, sigma_actual_shape: tuple, figsize: tuple = (3.55, 1.5)
    ):
        """
        Generates a plot for the extrapolated relation between measured sigmas
        and their corresponding analytical values using logistic extrapolation.
        The function also includes the ability to visualize the measured relations
        as well as a colorbar denoting the ratio of sigmas. This graphical
        representation is useful for understanding and analyzing the trends
        and relations within the dataset.

        :param sigma_actual_shape: A tuple specifying the start, stop, and
            number of points for generating sigma_actual values for extrapolation.
        :param figsize: A tuple defining the size of the figure. Defaults to
            (3.55, 1.5).
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

        fig, ax = plt.subplots(figsize=figsize, dpi=200)

        # extract N colors from the thermal colormap
        colors = thermal(np.linspace(0, 0.85, len(sigma_ratios)))

        sel = True
        for sg, si, c in zip(self.sigma_gauss, self.sigma_bessel, colors):
            if sel == False:
                label = None
            else:
                label = "Measured Relation"
                sel = False
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
                # label=f'${sigma_ratios[i]:.2e}$',
                color=c,
            )

        # --- COLORBAR ---
        cbar_ax = fig.add_axes([0.55, 0.22, 0.3, 0.03])
        cbar = matplotlib.colorbar.ColorbarBase(
            ax=cbar_ax, cmap=thermal, orientation="horizontal"
        )

        cbar_ax.set_xlim(0, 0.85)

        # make the ticks of the colorbar appear on the top

        # cbar_ax.xaxis.set_ticks_position('top')

        pos_ticks = list(
            np.array([0, 7, 14]) / (len(self.sigma_gauss) - 1) * 0.85
        )
        ticks = ["0", "1", r"$\infty$"]

        # add a label of the colorbar to the right of the colorbar
        cbar.set_label(r"$\sigma_{\mathrm{IMB}} / \sigma_{\mathrm{Gauss}}$")
        # make the label appear on top of the colorbar
        cbar_ax.xaxis.set_label_position("top")

        cbar.set_ticks(pos_ticks)
        cbar.set_ticklabels(ticks)

        # --- LEGEND ---

        # Create color lines for legend
        color_line = add_normalized_line_collection(
            ax, cmap=thermal, linewidth=4
        )
        gray_line = Line2D([0], [0], color="gray", lw=2)

        # Existing legend handles and labels
        handles = []
        labels = []

        handles.append(color_line)
        labels.append("Extrapolated Relation")

        handles.append(gray_line)
        labels.append("Measured Relation")

        ax.legend(
            handles,
            labels,
            handler_map={
                color_line: HandlerColorLineCollection(
                    cmap=thermal, numpoints=len(sigma_ratios)
                )
            },
            loc="upper left",
            frameon=True,
            fontsize=8,
        )

        ax.set_xlabel(r"$T_\alpha$")
        ax.set_ylabel(r"$T_\mathcal{N}$")
        # fig.savefig('sigma_lookup_extrapolation.pdf', bbox_inches='tight')
        plt.show()

    def interpolate_to_grid(
        self,
        points: np.ndarray,
        sigma_analytical_shape: tuple = (0, 8, 100),
        sigma_ratio_shape: tuple = (np.log10(6e-2), np.log10(5), 100),
        fill_nan: bool = True,
    ):
        """
        Interpolates the provided data points to a regularly spaced 2D grid using the given
        shape parameters for both axes. Optionally handles missing values (NaNs) in the
        output grid by filling them using linear approximation and copying valid values
        vertically to ensure data continuity.

        :param points: The input data points to interpolate. Expected to be a 2D array
            with shape (n, 3), where the first two columns represent the coordinates
            and the third column represents the corresponding data values.
        :type points: numpy.ndarray
        :param sigma_analytical_shape: A tuple containing the range for the first axis
            (start, stop, number of points).
        :type sigma_analytical_shape: tuple
        :param sigma_ratio_shape: A tuple containing the base-10 logarithmic range for the
            second axis (log_start, log_stop, number of points).
        :type sigma_ratio_shape: tuple
        :param fill_nan: A boolean flag indicating whether to handle and fill NaN values
            in the resultant grid by vertical propagation. Defaults to True.
        :type fill_nan: bool
        :return: Returns a tuple containing:

            - interp (numpy.ndarray): The interpolated 2D grid data.
            - X (numpy.ndarray): The meshgrid for the first axis.
            - Y (numpy.ndarray): The meshgrid for the second axis.
        :rtype: tuple
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
        table_name: str,
        crit_value: float,
        initial_guess: list = [0, 10, 0.1, 0.11],
        make_n_plots: int = 10,
        update_guess=True,
    ):
        """
        Generates a lookup table by performing logistic fitting on the actual
        and analytical shapes of a dataset, interpolates results to grid points,
        and plots various visualizations if verbose mode is enabled.

        :param sigma_actual_shape: Tuple specifying the shape of the actual
            sigma space for grid generation.
        :param sigma_analytical_shape: Tuple specifying the shape of the analytical
            sigma space for the grid.
        :param sigma_ratio_shape: Tuple defining the shape for the ratio
            of sigma spaces in the grid interpolation.
        :param n_analytical: Number of analytical points used in calculating
            sigma values.
        :param table_name: Name of the table being created for lookup and reference.
        :param crit_value: Critical value parameter for setting thresholds
            in the lookup table.
        :param initial_guess: List of initial guesses for parameters used
            in the logistic fitting process. Defaults to [0, 10, 0.1, 0.11].
        :param make_n_plots: Number of plots generated for visualization
            during the logistic fitting process. Defaults to 10.
        :param update_guess: Boolean flag indicating whether to update the
            initial_guess during the logistic fitting process. Defaults to True.
        :return: None.
        """
        self.crit_value[table_name] = crit_value

        sigma_actual, fit_values = self.do_logistic_fit(
            sigma_actual_shape=sigma_actual_shape,
            make_n_plots=make_n_plots,
            initial_guess=initial_guess,
            bounds=(
                [-np.inf, 0, 0, 0],
                [np.inf, 20, np.inf, np.inf],
            ),
            update_guess=update_guess,
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

        self.lookup[table_name + "_ratio"], X, Y = self.interpolate_to_grid(
            points=points_ratio,
            sigma_analytical_shape=sigma_analytical_shape,
            sigma_ratio_shape=sigma_ratio_shape,
            fill_nan=False,
        )

        self.shape[table_name + "_ta"] = sigma_actual_shape
        self.shape[table_name + "_sr"] = sigma_ratio_shape

        (
            self.lookup[table_name],
            self.lookup_X[table_name],
            self.lookup_Y[table_name],
        ) = self.interpolate_to_grid(
            points=points,
            sigma_analytical_shape=sigma_analytical_shape,
            sigma_ratio_shape=sigma_ratio_shape,
            fill_nan=False,
        )

        if self.verbose:
            fig, ax = plt.subplots()

            plt.imshow(
                self.lookup[table_name + "_ratio"],
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
                self.lookup[table_name + "_ratio"],
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
        Creates an extrapolated lookup table based on given sigma value shapes.

        This method computes and interpolates the sigma actual, sigma ratio, and
        sigma analytical values into lookup tables. It uses logistic functions and
        other calculations to build data grids for performing extrapolation and
        evaluation. The results are stored in the object's lookup dictionary, and
        visualization can optionally display the extrapolated data.

        :param sigma_actual_shape: Tuple specifying the shape and range of sigma_actual
            values, used for linspace generation.
        :param sigma_ratio_shape: Tuple specifying the shape and range of sigma_ratio
            values, used for logspace generation.
        :param sigma_analytical_shape: Tuple specifying the shape and range of
            sigma_analytical values for grid interpolation.
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
            self.lookup["extrapolation"],
            self.lookup_X["extrapolation"],
            self.lookup_Y["extrapolation"],
        ) = self.interpolate_to_grid(
            points=np.array((SAna, SR.flatten(), SAct.flatten())).T,
            sigma_analytical_shape=sigma_analytical_shape,
            sigma_ratio_shape=sigma_ratio_shape,
            fill_nan=False,
        )

        lookup_extrapolation_ratio, _, _ = self.interpolate_to_grid(
            points=np.array((SAna, SR.flatten(), SAct.flatten() / SAna)).T,
            sigma_analytical_shape=sigma_analytical_shape,
            sigma_ratio_shape=sigma_ratio_shape,
            fill_nan=False,
        )

        self.lookup["extrapolation_ratio"] = lookup_extrapolation_ratio

        if self.verbose:
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
            plt.colorbar()

            plt.contour(
                self.lookup_X["extrapolation"],
                np.log10(self.lookup_Y["extrapolation"]),
                lookup_extrapolation_ratio,
                levels=[0.8, 0.85, 0.9, 0.95, 1, 1.05],
                # levels=[5, 10, 15, 20, 25],
                colors="w",
                linewidths=1,
            )

            plt.xlabel(r"$\sigma_{\mathcal{N}, \mathrm{analytical}}$")
            plt.ylabel(r"$\log(\sigma_{\mathcal{B} / \sigma_{\mathcal{N}}})$")

            plt.show()

    def plot_combined_lookup(
        self,
        figsize=(3.55, 3),
    ):
        """
        Plots a combined lookup visualization using various pre-computed lookup tables and custom
        styling. Multiple lookup tables are aggregated into one cohesive plot with axes labels,
        colorbars, and tick formatting for intuitive presentation. The method utilizes deep copies
        of data arrays and selects subsets of the lookup tables based on critical values provided.

        :param figsize: Tuple specifying the figure size for the plot.
        :type figsize: tuple
        :return: None
        """
        fig, ax = plt.subplots(figsize=(3.55, 3), dpi=200)

        ax.imshow(
            self.lookup["small_ratio"],
            cmap=thermal_r,
            origin="lower",
            extent=(0.07, 0.25, -1, 0.75),
            aspect="auto",
            vmin=0.66,
            vmax=1.0,
        )
        im = ax.imshow(
            self.lookup["extrapolation_ratio"],
            cmap=thermal_r,
            origin="lower",
            extent=(5, 30, -1, 0.75),
            aspect="auto",
            vmin=0.66,
            vmax=1.0,
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(r"$T_\alpha / T_\mathcal{N}$")

        # ax.imshow(self.lookup['large_ratio'], cmap=thermal, origin='lower', extent=(0.2, 7, -1, 0.75), aspect='auto', vmin=0.66, vmax=1.0)

        large_ratio = deepcopy(self.lookup["large_ratio"])
        ts_alpha = np.linspace(
            self.shape["large_ta"][0],
            self.shape["large_ta"][1],
            large_ratio.shape[0],
        )
        idx = {
            "large": np.argmin(np.abs(ts_alpha - self.crit_value["large"])),
            "small": np.argmin(np.abs(ts_alpha - self.crit_value["small"])),
        }

        idx["large_co"] = ts_alpha[idx["large"]]
        idx["small_co"] = ts_alpha[idx["small"]]

        large_ratio = large_ratio[:, idx["small"] : idx["large"]]

        ax.imshow(
            large_ratio,
            cmap=thermal_r,
            origin="lower",
            extent=(idx["small_co"], idx["large_co"], -1, 0.75),
            aspect="auto",
            vmin=0.66,
            vmax=1.0,
        )

        ax.set_xlim(0.07, 30)

        yticks = np.log10(np.array([3e-1, 1, 3]))
        ytick_labels = [r"$3\cdot10^{-1}$", r"$10^{0}$", r"$3\cdot10^{0}$"]
        ax.set_yticks(yticks, ytick_labels)

        ax.set_ylabel(r"$\sigma_\mathrm{IMB} / \sigma_\mathrm{Gauss}$")
        ax.set_xlabel(r"$T_\mathcal{N}$")

        plt.show()

    def save_lookup(self, path: str):
        """
        Saves the lookup table data along with critical values, bootstrap properties,
        and logistic extrapolation parameters to the specified file path in binary
        format using Python's pickle module.

        The saved data includes the following:
        - Critical value used for analysis.
        - Paths related to the data.
        - Bootstrap properties for the system.
        - Lookup table attributes including its values and coordinate details.
        - Fit parameters derived from logistic extrapolation.

        :param path: The file path where the data will be saved in binary format.
        :type path: str
        :return: None
        """
        self.result = {
            "critical_value": self.crit_value,
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
