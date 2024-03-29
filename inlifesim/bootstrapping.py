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
    return (ymax - ymin) / (1 + np.exp(-k*(x - x0))) + ymin


def inverse_logistic(y, x0, k, ymin, ymax):
    return np.log((y - ymin) / (ymax - y) - 1) / k + x0


def get_interest_area(k, rt):
    return -np.log(1 / rt - 1)/k


class EvaluateBootstrapping():

    def __init__(self,
                 verbose,
                 paths,
                 crit_value,
                 bootstrap_properties):
        self.verbose = verbose
        self.paths = paths
        self.load_data(paths)
        self.crit_value = crit_value
        self.bootstrap_properties = bootstrap_properties

    def load_data(self,
                  paths: list):
        sigma_gauss = []
        sigma_bessel = []
        sigmas_want_get = []

        for path in paths:
            files = [f for f in listdir(path) if isfile(join(path, f))]

            sigmas = np.array([
                np.array(
                    f.replace('.npy', '').replace('-', '.').split('_')
                )[-2:] for f in files
            ]).astype(float)

            sig_g = sigmas[:, 0]
            sig_b = sigmas[:, 1]

            sigma_want_get = {}
            for sg, si in zip(sig_g, sig_b):
                sg_str = str(sg).replace('.', '-')
                si_str = str(si).replace('.', '-')

                res = np.load(path + f'sigma_lookup_{sg_str}_{si_str}.npy')
                sigma_want_get[(sg, si)] = np.array((res[:int(len(res) / 2)],
                                                     res[int(len(res) / 2):]))

            sigma_gauss.append(sig_g)
            sigma_bessel.append(sig_b)
            sigmas_want_get.append(sigma_want_get)

        self.sigma_gauss = np.concatenate(sigma_gauss)
        self.sigma_bessel = np.concatenate(sigma_bessel)
        self.sigma_want_get = dict(
            itertools.chain.from_iterable(d.items() for d in sigmas_want_get)
        )

    def fit_logistic(self,
                     sig_actual,
                     plot=False,
                     guess=None,
                     bounds=(-np.inf, np.inf)):
        sig_analytical = []
        rat_bes_gaus = []

        for sg, si in zip(self.sigma_gauss, self.sigma_bessel):
            sig_analytical.append(np.interp(sig_actual,
                                            self.sigma_want_get[(sg, si)][0],
                                            self.sigma_want_get[(sg, si)][1]))
            rat_bes_gaus.append(si / sg)

        ratio_sig = np.array((sig_analytical, rat_bes_gaus)).T

        # noinspection PyTupleAssignmentBalance
        popt, _ = curve_fit(f=logistic,
                            xdata=np.log10(ratio_sig[1:-1, 1]),
                            ydata=ratio_sig[1:-1, 0],
                            p0=guess,
                            bounds=bounds)

        if plot:
            x = np.log10(ratio_sig[1:-1, 1])
            y = ratio_sig[1:-1, 0]
            fig, ax = plt.subplots(nrows=2, height_ratios=[1, 0.3])
            # Plot the original data (with noise) and the fitted curve
            ax[0].scatter(x, y, label='Data', color='k')

            x_sample = np.linspace(np.min(x), np.max(x), 1000)
            ax[0].plot(x_sample, logistic(x_sample, *popt), '--',
                       c='tab:orange', label='Logistic fit')

            # print popt in the plot
            ax[0].text(0.5, 0.5,
                       f'x0={popt[0]:.2f}\nk={popt[1]:.2f}'
                       f'\nymin={popt[2]:.2f}\nymax={popt[3]:.2f}',
                       horizontalalignment='center',
                       verticalalignment='center',
                       transform=ax[0].transAxes)

            ax[0].set_xticks([])
            ax[0].legend()

            ax[1].set_xlabel(r'$\sigma_\mathcal{B} / \sigma_\mathcal{N}$')
            ax[0].set_ylabel(r'$\sigma_{\mathcal{N}, \mathrm{actual}}$')

            # draw horizontal line at y=0
            ax[1].axhline(0, color='gray', ls='--')
            # plot residual in lower axis
            ax[1].scatter(x, y - logistic(x, *popt),
                          color='k',
                          label='Data - Logistic')
            ax[1].set_ylim([-0.05, 0.05])
            ax[1].set_ylabel('Residuals')

            plt.show()

        return popt

    def do_logistic_fit(self,
                        sigma_actual_shape: tuple,
                        make_n_plots: int = 10,
                        initial_guess=None,
                        bounds=(-np.inf, np.inf)):
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
                        self.fit_logistic(sa,
                                          plot=True,
                                          guess=guess,
                                          bounds=bounds)
                    )
                else:
                    fit_values.append(
                        self.fit_logistic(sa,
                                          plot=False,
                                          guess=guess,
                                          bounds=bounds)
                    )
            except:
                print(f'Failed at sigma_actual={sa}')
                fit_values.append([np.nan, np.nan, np.nan, np.nan])

        return sig_actual, np.array(fit_values)

    def extrapolate_logistic_parameters(self,
                                        sigma_actual_shape: tuple,
                                        start_fit_at_sigma: list,
                                        end_fit_at_sigma: list,
                                        polynomial_order: list,
                                        make_n_plots: int = 10):

        sig_actual, fit_values = self.do_logistic_fit(
            sigma_actual_shape=sigma_actual_shape,
            make_n_plots=make_n_plots
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

            popt, _ = np.polyfit(sig_actual[idx_start:idx_end],
                                 fit_values[idx_start:idx_end, i],
                                 polynomial_order[i],
                                 cov=True)
            self.logistic_extrapolation.append(popt)

        if self.verbose:
            fig, ax = plt.subplots(1, 4, figsize=(16, 3), dpi=200)

            param_names = [
                '$x_0$ [$\sigma_{\mathcal{N}, \mathrm{actual}}$]',
                r'$k$ [$\frac{1}{\sigma_{\mathcal{N}, \mathrm{actual}}}$]',
                '$y_\mathrm{min}$ [$\sigma_{\mathcal{N},'
                ' \mathrm{analytical}}$]',
                '$y_\mathrm{max}$ [$\sigma_{\mathcal{N},'
                ' \mathrm{analytical}}$]'
            ]

            titles = ['Center', 'Steepness', 'Minimum', 'Maximum']

            for i in range(4):
                x = np.linspace(2, 8, 100)
                y = np.polyval(self.logistic_extrapolation[i], x)
                ax[i].plot(x, y, c='tab:orange', label='Polynomial fit')

                # differentiate between data used in fitting and not
                ax[i].scatter(sig_actual[:idx_starts[i]],
                              fit_values[:idx_starts[i], i],
                              c='gray', s=2, label='Data (not used in fit)')
                ax[i].scatter(sig_actual[idx_ends[i]:],
                              fit_values[idx_ends[i]:, i],
                              c='gray', s=2)
                ax[i].scatter(sig_actual[idx_starts[i]:idx_ends[i]],
                              fit_values[idx_starts[i]:idx_ends[i], i],
                              c='k', s=3, label='Data')
                ax[i].set_ylabel(param_names[i])
                ax[i].set_xlabel(r'$\sigma_{\mathcal{N}, \mathrm{actual}}$')
                ax[i].set_title(titles[i])
                ax[i].legend()

            plt.tight_layout()
            plt.show()

    def plot_extrapolation(self,
                           sigma_actual_shape: tuple):
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
            ax.plot(self.sigma_want_get[(sg, si)][1],
                    self.sigma_want_get[(sg, si)][0],
                    color='gray')

        for i, c in enumerate(colors):
            ax.plot(sigma_analytical[i], sigma_actual,
                    # format the labels in scientific notation
                    label=f'${sigma_ratios[i]:.2e}$',
                    color=c)

        ax.legend(fontsize=5)
        plt.show()

    def interpolate_to_grid(self,
                            points: np.ndarray,
                            sigma_analytical_shape: tuple = (0, 8, 100),
                            sigma_ratio_shape: tuple = (
                                    np.log10(6e-2), np.log10(5), 100
                            ),
                            fill_nan: bool = True):
        X, Y = np.meshgrid(np.linspace(*sigma_analytical_shape),
                           np.logspace(*sigma_ratio_shape))

        interp = griddata(points=points[:, :2],
                          values=points[:, 2],
                          xi=(X, Y),
                          method='nearest')

        if fill_nan:
            interp_ref = griddata(points=points[:, :2],
                                  values=points[:, 2],
                                  xi=(X, Y),
                                  method='linear')

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

    def create_lookup_table(self,
                            sigma_actual_shape: tuple,
                            sigma_analytical_shape: tuple,
                            sigma_ratio_shape: tuple,
                            n_analytical: int,
                            make_n_plots: int = 10, ):

        sigma_actual, fit_values = self.do_logistic_fit(
            sigma_actual_shape=sigma_actual_shape,
            make_n_plots=make_n_plots,
            initial_guess=[0, 10, 0.1, 0.11],
            bounds=([-np.inf, 0, 0, 0], [np.inf, 20, np.inf, np.inf])
        )

        if self.verbose:
            # make a figure showing the evolution of all fit parameters
            fig, ax = plt.subplots(1, 4, figsize=(16, 3), dpi=200)

            param_names = [
                '$x_0$ [$\sigma_{\mathcal{N}, \mathrm{actual}}$]',
                r'$k$ [$\frac{1}{\sigma_{\mathcal{N}, \mathrm{actual}}}$]',
                '$y_\mathrm{min}$ [$\sigma_{\mathcal{N},'
                ' \mathrm{analytical}}$]',
                '$y_\mathrm{max}$ [$\sigma_{\mathcal{N},'
                ' \mathrm{analytical}}$]'
            ]

            titles = ['Center', 'Steepness', 'Minimum', 'Maximum']

            for i in range(4):
                # differentiate between data used in fitting and not
                ax[i].scatter(sigma_actual, fit_values[:, i],
                              c='k', s=3, label='Data')
                ax[i].set_ylabel(param_names[i])
                ax[i].set_xlabel(r'$\sigma_{\mathcal{N}, \mathrm{actual}}$')
                ax[i].set_title(titles[i])
                ax[i].legend()

            plt.tight_layout()
            plt.show()

        crit_regions_size = np.array([
            get_interest_area(fv[1], 0.99) for fv in fit_values
        ])  # half the size of the critical region
        crit_region = np.array((
            fit_values[:, 0] - crit_regions_size,
            fit_values[:, 0] + crit_regions_size
        )).T

        sigma_analytical = []
        sigma_ratio = []

        for i in range(len(fit_values)):
            sig_r = np.linspace(crit_region[i, 0],
                                crit_region[i, 1],
                                n_analytical)
            sigma_ratio.append(sig_r)

            sigma_analytical.append(logistic(sig_r, *fit_values[i]))

        sigma_analytical = np.array(sigma_analytical)
        sigma_ratio = np.array(sigma_ratio)

        if self.verbose:
            fig, ax = plt.subplots()

            min_s = np.min(sigma_actual)
            max_s = np.max(sigma_actual)

            for i in range(len(sigma_actual)):
                ax.scatter(sigma_analytical[i,],
                           10 ** sigma_ratio[i,],
                           c=thermal((sigma_actual[i] - min_s) / (max_s - min_s)),
                           s=3)

            plt.xlabel(r'$\sigma_{\mathcal{N}, \mathrm{analytical}}$')
            plt.ylabel(r'$\sigma_{\mathcal{B} / \sigma_{\mathcal{N}}}$')

            plt.yscale('log')

        points_ratio = np.concatenate([
            np.array((
                sigma_analytical[i,],
                10 ** sigma_ratio[i,],
                sigma_actual[i]
                * np.ones(sigma_analytical.shape[1]) / sigma_analytical[i,]
            )).T for i in range(len(sigma_actual))
        ])
        points_ratio = points_ratio[~np.isnan(points_ratio).any(axis=1)]

        points = np.concatenate([
            np.array((
                sigma_analytical[i,],
                10 ** sigma_ratio[i,],
                sigma_actual[i] * np.ones(sigma_analytical.shape[1])
            )).T for i in range(len(sigma_actual))])
        points = points[~np.isnan(points).any(axis=1)]

        self.lookup_ratio, X, Y = self.interpolate_to_grid(
            points=points_ratio,
            sigma_analytical_shape=sigma_analytical_shape,
            sigma_ratio_shape=sigma_ratio_shape,
            fill_nan=False
        )
        self.lookup, self.lookup_X, self.lookup_Y = self.interpolate_to_grid(
            points=points,
            sigma_analytical_shape=sigma_analytical_shape,
            sigma_ratio_shape=sigma_ratio_shape,
            fill_nan=False
        )

        if self.verbose:
            fig, ax = plt.subplots()

            plt.imshow(self.lookup_ratio,
                       origin='lower',
                       cmap=thermal,
                       extent=(sigma_analytical_shape[0],
                               sigma_analytical_shape[1],
                               sigma_ratio_shape[0],
                               sigma_ratio_shape[1]),
                       aspect='auto')

            plt.colorbar()
            # add white contours
            plt.contour(X,
                        np.log10(Y),
                        self.lookup_ratio,
                        levels=[0.8, 0.85, 0.9, 0.95, 1, 1.05],
                        # levels=[1, 2, 3, 4, 5, 6, 7, 8],
                        colors='w',
                        linewidths=1)

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
            ana_limits = np.array((sigma_ratios,
                                   np.min(sigma_analytical, axis=1)))

            # plt.plot(ana_limits[1, ], np.log10(ana_limits[0, ]), color='r')

            plt.xlabel(r'$\sigma_{\mathcal{N}, \mathrm{analytical}}$')
            plt.ylabel(r'$\log(\sigma_{\mathcal{B} / \sigma_{\mathcal{N}}})$')

            # plt.ylim(-1.25, 0.75)

            plt.show()

    def create_extrapolated_lookup_table(self,
                                         sigma_actual_shape: tuple,
                                         sigma_ratio_shape: tuple,
                                         sigma_analytical_shape: tuple):

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

        (self.lookup_extrapolation,
         self.logistic_extrapolation_X,
         self.lookup_extrapolation_Y) = self.interpolate_to_grid(
            points=np.array((SAna, SR.flatten(), SAct.flatten())).T,
            sigma_analytical_shape=sigma_analytical_shape,
            sigma_ratio_shape=sigma_ratio_shape,
            fill_nan=False
        )

        if self.verbose:
            lookup_extrapolation_ratio, _, _ = self.interpolate_to_grid(
                points=np.array((SAna, SR.flatten(), SAct.flatten() / SAna)).T,
                sigma_analytical_shape=sigma_analytical_shape,
                sigma_ratio_shape=sigma_ratio_shape,
                fill_nan=False
            )

            fig, ax = plt.subplots()
            plt.imshow(lookup_extrapolation_ratio,
                       origin='lower',
                       cmap=thermal,
                       extent=(sigma_analytical_shape[0],
                               sigma_analytical_shape[1],
                               sigma_ratio_shape[0],
                               sigma_ratio_shape[1]),
                       aspect='auto')

            plt.contour(self.logistic_extrapolation_X,
                        np.log10(self.lookup_extrapolation_Y),
                        lookup_extrapolation_ratio,
                        levels=[0.8, 0.85, 0.9, 0.95, 1, 1.05],
                        # levels=[5, 10, 15, 20, 25],
                        colors='w',
                        linewidths=1)
            plt.colorbar()
            plt.show()

    def save_lookup(self,
                    path: str):

        self.result = {'critical_value': 6,
                       'data_path': self.paths,
                       'bootstrap_properties': self.bootstrap_properties,
                       'lookup_table': {'table': self.lookup,
                                        'X': self.lookup_X,
                                        'Y': self.lookup_Y,
                                        'coords': {'X': 'sigma_analytical',
                                                   'Y': 'sigma_ratio',
                                                   'Z': 'sigma_actual'}},
                       'lookup_table_extrapolation': {
                           'table': self.lookup_extrapolation,
                           'X': self.logistic_extrapolation_X,
                           'Y': self.lookup_extrapolation_Y,
                           'coords': {'X': 'sigma_analytical',
                                      'Y': 'sigma_ratio',
                                      'Z': 'sigma_actual'}
                       },
                       'fit_parameters': {
                           'x0': self.logistic_extrapolation[0],
                           'k': self.logistic_extrapolation[1],
                           'ymin': self.logistic_extrapolation[2],
                           'ymax': self.logistic_extrapolation[3]
                       }}
        with open(path, 'wb') as f:
            pickle.dump(self.result, f)


class InterpretBootstrapping():

    def __init__(self,
                 lookup_table_path: str):
        with open(lookup_table_path, 'rb') as f:
            self.lookup_table = pickle.load(f)

    def get_sigma_actual(self,
                         sigma_analytical: np.ndarray,
                         sigma_ratio: np.ndarray):
        idx = [np.unravel_index(
            np.argmin(
                (self.lookup_table['lookup_table']['X']['large']
                 - sigma_analytical[i]) ** 2 +
                (self.lookup_table['lookup_table']['Y']['large']
                 - sigma_ratio[i]) ** 2
            ),
            self.lookup_table['lookup_table']['table']['large'].shape
        ) for i in range(len(sigma_analytical))]

        sigma_actual_lookup = np.array([
            self.lookup_table['lookup_table']['table']['large'][idx[i]]
            for i in range(len(idx))
        ])

        idx = [np.unravel_index(
            np.argmin(
                (self.lookup_table['lookup_table']['X']['extrapolation']
                 - sigma_analytical[i]) ** 2 +
                (self.lookup_table['lookup_table']['Y']['extrapolation']
                 - sigma_ratio[i]) ** 2
            ),
            self.lookup_table['lookup_table']['table']['extrapolation'].shape
        ) for i in range(len(sigma_analytical))]

        sigma_actual_extrapolated = np.array([
            self.lookup_table['lookup_table']['table']['extrapolation'][idx[i]]
            for i in range(len(idx))
        ])

        sigma_actual_extrapolated[
            sigma_actual_extrapolated
            < self.lookup_table['critical_value']['large']
            ] = sigma_actual_lookup[
            sigma_actual_extrapolated
            < self.lookup_table['critical_value']['large']
            ]

        sigma_actual_extrapolated[
            sigma_actual_extrapolated <= 0.1
            ] = sigma_analytical[
            sigma_actual_extrapolated <= 0.1
            ]

        return sigma_actual_extrapolated


def run_bootstrapping(B: int,
                      N: int,
                      sigma_imb: np.ndarray,
                      sigma_gauss: np.ndarray,
                      path: str,
                      B_per:int =int(1e7),
                      n_cpu:int = 64,
                      n_sigma:int = 10000):

    i = 0

    for sg, si in zip(sigma_gauss, sigma_imb):
        print('Working on sigma_gauss =', sg,
              'sigma_imb =', si, '(', i, '/', len(sigma_gauss), ')')
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
            parallel=True
        )
        print('Saving the results...', end=' ', flush=True)
        results = np.concatenate((sigma_want, sigma_get))

        sg_str = str(sg).replace('.', '-')
        si_str = str(si).replace('.', '-')

        # save the results
        if path[-1] != '/':
            path += '/'
        np.save(path + f'sigma_lookup_{sg_str}_{si_str}.npy', results)

        print('[Done]')
