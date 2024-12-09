from typing import Union

import numpy as np
from scipy.special import kv, gamma
from scipy.stats import rv_continuous, norm, linregress
from scipy.stats import t as t_dist
from scipy.interpolate import UnivariateSpline, Akima1DInterpolator
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_config
from joblib_progress import joblib_progress
import time
from tqdm import tqdm

from inlifesim.util import freq2temp_fft, dict_sumover, remove_non_increasing


def draw_sample(params,
                return_variables=['xcorr']):

    if return_variables == 'all':
        ret_all = True
    else:
        ret_all = False

    size = (params['n_outputs'], params['n_draws'], params['n_sampling_rot'])

    rdict ={}
    # random noise
    rdict['pn_sgl_time'] = np.random.poisson(
        lam=params['pn_sgl'][0] ** 2 / params['n_sampling_rot'] / 2,
        size=size
    )

    rdict['pn_ez_time'] = np.random.poisson(
        lam=params['pn_ez'][0] ** 2 / params['n_sampling_rot'] / 2,
        size=size
    )

    rdict['pn_lz_time'] = np.random.poisson(
        lam=params['pn_lz'][0] ** 2 / params['n_sampling_rot'] / 2,
        size=size
    )

    # systematic noise
    d_a_time, d_a_ft = draw_fourier_noise(
        psd=params['d_a_psd'][0],
        n_sampling_rot=params['n_sampling_rot'],
        t_rot=params['t_rot'],
        n_draws=params['n_draws'],
        n_outputs=params['d_a_psd'].shape[0])

    d_phi_time, d_phi_ft = draw_fourier_noise(
        psd=params['d_phi_psd'][0],
        n_sampling_rot=params['n_sampling_rot'],
        t_rot=params['t_rot'],
        n_draws=params['n_draws'],
        n_outputs=params['d_a_psd'].shape[0])

    rdict['d_a_time'] = d_a_time
    rdict['d_phi_time'] = d_phi_time
    rdict['d_a_ft'] = d_a_ft
    rdict['d_phi_ft'] = d_phi_ft

    sys_nchop = calculate_systematic_response(gradient=params['gradient'],
                                              hessian=params['hessian'],
                                              d_a_time=d_a_time,
                                              d_phi_time=d_phi_time,
                                              chop=False)
    rdict.update(sys_nchop)

    sys_chop = calculate_systematic_response(gradient=params['gradient_chop'],
                                                hessian=params['hessian_chop'],
                                                d_a_time=d_a_time,
                                                d_phi_time=d_phi_time,
                                                chop=True)
    rdict.update(sys_chop)

    rdict['sys_timeseries'] = (rdict['sys_a']
                         + rdict['sys_phi']
                         + rdict['sys_aphi']
                         + rdict['sys_aa']
                         + rdict['sys_phiphi'])

    rdict['pn_timeseries'] = dict_sumover(d=rdict,
                                          keys=['pn_sgl_time',
                                                'pn_ez_time',
                                                'pn_lz_time'])

    rdict['sys_timeseries'] = dict_sumover(d=rdict,
                                           keys=['sys_a',
                                                 'sys_phi',
                                                 'sys_aphi',
                                                 'sys_aa',
                                                 'sys_phiphi']
                                           )

    rdict['sys_timeseries_chop'] = dict_sumover(d=rdict,
                                                keys=['sys_a_chop',
                                                      'sys_phi_chop',
                                                      'sys_aphi_chop',
                                                      'sys_aa_chop',
                                                      'sys_phiphi_chop']
                                                )

    rdict['noise_timeseries'] = ((rdict['pn_timeseries'][0]
                                  + rdict['sys_timeseries'])
                                 - (rdict['pn_timeseries'][1]
                                    + rdict['sys_timeseries_chop']))

    rdict['timeseries'] = (rdict['noise_timeseries']
                           + params['planet_signal'])

    rdict['noise_timeseries_singlebw'] = ((rdict['pn_timeseries'][0]
                                  + rdict['sys_timeseries']))

    rdict['timeseries_singlebw'] = (rdict['noise_timeseries_singlebw']
                           + params['planet_signal'])

    # do the cross correlation
    rdict['xcorr'] = np.sum(rdict['timeseries'] * params['planet_template'],
                            axis=-1)

    vartypes = ['', '_pn', '_sys', '_pn_sgl', '_pn_ez', '_pn_lz',
                '_a', '_phi', '_aphi', '_aa', '_phiphi']
    if (return_variables == 'all_xcorr') or (return_variables == 'all'):
        return_variables = ['xcorr' + vt for vt in vartypes]

    # filter all return variables that contain xcorr in their strings
    xcorr_requested = [k for k in return_variables if 'xcorr' in k]
    try:
        xcorr_requested.remove('xcorr')
    except ValueError:
        pass
    for xc_type in xcorr_requested:
        if xc_type == 'xcorr_pn':
            rdict[xc_type] = np.sum(
                (rdict['pn_timeseries'][0]
                 - rdict['pn_timeseries'][1])
                * params['planet_template'],
                axis=-1
            )
        elif xc_type == 'xcorr_sys':
            rdict[xc_type] = np.sum(
                (rdict['sys_timeseries']
                 - rdict['sys_timeseries_chop'])
                * params['planet_template'],
                axis=-1
            )
        elif 'pn' in xc_type:
            rdict[xc_type] = np.sum(
                np.diff(rdict[xc_type.split(sep='_', maxsplit=1)[1] + '_time'],
                        axis=0)[0]
                * params['planet_template'],
                axis=-1
            )
        else:
            rdict[xc_type] = np.sum(
                (rdict['sys_' + xc_type.split(sep='_')[1]]
                 - rdict['sys_' + xc_type.split(sep='_')[1] + '_chop'])
                * params['planet_template'],
                axis=-1
            )

    if ret_all:
        return rdict
    else:
        return {k: rdict[k] for k in return_variables}


def calculate_systematic_response(gradient: np.ndarray,
                                  hessian: np.ndarray,
                                  d_a_time: np.ndarray,
                                  d_phi_time: np.ndarray,
                                  chop: bool):
    if chop:
        ext = '_chop'
    else:
        ext = ''

    rdict = {}
    rdict['sys_a' + ext] = np.sum(
        gradient['a'][:, np.newaxis, np.newaxis] * d_a_time,
        axis=0)
    rdict['sys_phi' + ext] = np.sum(
        gradient['phi'][:, np.newaxis, np.newaxis] * d_phi_time,
        axis=0)

    rdict['sys_aphi' + ext] = np.sum(
        hessian['aphi'][:, :, np.newaxis, np.newaxis]
        * d_a_time[:, np.newaxis, :, :] * d_phi_time[np.newaxis, :, :, :],
        axis=(0, 1))

    rdict['sys_aa' + ext] = np.sum(
        hessian['aa'][:, :, np.newaxis, np.newaxis]
        * d_a_time[:, np.newaxis, :, :] * d_a_time[np.newaxis, :, :, :],
        axis=(0, 1))

    rdict['sys_phiphi' + ext] = np.sum(
        hessian['phiphi'][:, :, np.newaxis, np.newaxis]
        * d_phi_time[:, np.newaxis, :, :] * d_phi_time[np.newaxis, :, :, :],
        axis=(0, 1))

    return rdict

def draw_fourier_noise(psd: np.ndarray,
                       n_sampling_rot: int,
                       t_rot: float,
                       n_draws: int,
                       n_outputs: int = 1):
    '''
    Draw Fourier series from a given power spectral density (PSD). The output
    shape will be (n_draws, n_sampling_rot). Can be used to generate a
    realisation of the perturbation random variables.

    Parameters
    ----------
    psd
        Power spectral density
    n_sampling_rot
        Number of samples (exposures) per rotation of the array
    t_rot
        Array rotation period in [s]
    n_draws
        Number of times the experiment is drawn

    Returns
    -------
    x
        The temporal series.
    x_ft
        The Fourier series in frequency space
    '''

    if n_outputs == 1:
        size = (n_draws, int((psd.shape[-1] + 1) / 2))
        scale = n_sampling_rot * np.sqrt(
                                 psd[np.newaxis,
                                 int((psd.shape[-1] - 1) / 2)::]
                                 / 2 / t_rot
                             )
    else:
        size = (n_outputs, n_draws, int((psd.shape[-1] + 1) / 2))
        scale = n_sampling_rot * np.sqrt(
            psd[np.newaxis, np.newaxis,
            int((psd.shape[-1] - 1) / 2)::]
            / 2 / t_rot
        )

    x_ft = (np.random.normal(loc=0,
                             scale=scale,
                             size=size)
            + 1j * np.random.normal(loc=0,
                                    scale=scale,
                                    size=size)
            )

    if n_outputs == 1:
        x_ft = np.concatenate((np.flip(x_ft[:, 1:], axis=-1), x_ft),
                              axis=-1)
    else:
        x_ft = np.concatenate((np.flip(x_ft[:, :, 1:], axis=-1), x_ft),
                              axis=-1)

    x = freq2temp_fft(fourier_series=x_ft,
                      total_time=t_rot)

    return x, x_ft


class imb_gen(rv_continuous):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ppf_spline = None

    def _pdf(self, x, *args):
        n = args[0]
        pdf = ((2 ** (0.5 * (1 - n)) * np.abs(x * np.sqrt(n)) ** (0.5 * (n-1))
                * np.sqrt(n)
                * kv(0.5 * (n - 1), np.abs(x * np.sqrt(n))))
               / (np.sqrt(np.pi) * gamma(n / 2)))
        return pdf

    def _ppf(self, q, *args):
        if self.ppf_spline is None:
            # Generate some points
            # The PDF is symmetric, so we only evaluate the negative points
            # this leads to higher precision, as the PDF is small for
            # negative values.
            # Evaluate slightly over the 0 point to avoid fitting artifacts
            # close to 0
            x = np.linspace(-30, 1, 2000)
            if len(np.array(args).shape) != 1:
                args = args[0][0]
            y = self.cdf(x, args)

            # select only the values from y that are strictly increasing
            y, x = remove_non_increasing(y, x)

            # Fit a polynomial to the function
            self.ppf_spline = Akima1DInterpolator(y, x)

        # Evaluate the inverse of the polynomial at the given points
        ppf_res = np.zeros_like(q)
        ppf_res[q<0.5] = self.ppf_spline(q[q<0.5])
        ppf_res[q>0.5] = -self.ppf_spline(1 - q[q>0.5])
        return ppf_res


imb = imb_gen(name='imb', shapes='n')


def get_qq(data: np.ndarray,
           loc: float,
           scale: float,
           mode: str,
           nconv: Union[float, None] = None,
           siglim: float = 7,
           n_eval: int = 100):
    """
    A function to create quantile-quantile values that are evenly spaces along
    the theoretical quantiles. Implemented for a normal and an iterative
    modified bessel distribution.

    Args:
        data: np.ndarray
            The data set for which the Q-Q plot is to be calculated.
        loc: float
            The location parameter of the distribution.
        scale: float
            The scale parameter of the distribution.
        mode: str
            The mode of the distribution. Either 'normal' for a Gaussian
            distribution or 'imb' for an iterative modified bessel
            distribution.
        nconv: float, optional
            The number of convolution steps for the iterative modified bessel
            distribution. If the mode is 'normal', this parameter is ignored.
        siglim: float, optional
            The limit of up to how many standard deviations away from 0 the
            PDFs are evaluated. The default is 7.
        n_eval: int, optional
            The number of evaluation points for the Q-Q plot. The default is
            100.

    Returns:
        q: dict
            The theoretical and sample quantiles.
        q_good: dict
            The theoretical and sample quantiles for a purely data based
            calculation.
        q_fail: dict
            The theoretical and sample quantiles for the faulty extrapolation.
            Here, the numerical evaluation of the ppf of the distribution
            failed and is replaced by the native theoretical quantiles. Due to
            the discrete nature of the data, this can lead to some imprecision
            in the $q_sample$ values for this regime.
    """
    if mode == 'imb' and nconv is None:
        raise ValueError('nconv must be specified for mode "imb"')

    # data needs to be normalized
    # data /= np.std(data)

    # start with an equally spaced grid of quantiles
    q_theo_native = np.linspace(-siglim, siglim, n_eval)[1:-1]

    # evaluate the p_values of the theoretical quantiles
    if mode == 'normal':
        dist = norm(loc=loc, scale=scale)
    elif mode == 'imb':
        dist = imb(loc=loc, scale=scale, n=nconv)
    else:
        raise ValueError('Mode not recognized')

    p_theo_native = dist.cdf(q_theo_native)

    # snap the theoretical p_values to values that actually appear in the
    # sample. This is done to avoid discretization errors in the sample
    p_sample = np.round(p_theo_native * len(data)).astype(int)
    p_theo = p_sample / len(data)

    # remove the p_values that are 0 or 1, as the ppf is not defined for these
    # values.
    mask = np.logical_or(p_theo == 0, p_theo == 1)
    p_theo = p_theo[~mask]
    p_sample = p_sample[~mask]

    # get the data q-values by sorting the data and extracting at the selected
    # p_value positions
    data_sorted = np.sort(data)
    q_sample = data_sorted[p_sample]

    # get the theoretical q-values by evaluating the ppf of the distribution
    q_theo = dist.ppf(p_theo)

    # in some cases the ppf fails to evaluate. It then returns 0, for which
    # it is replaced by the non-gridded theoretical quantiles.
    mask_q_theo_fail = np.logical_or(q_theo == 0.,
                                     np.invert(np.isfinite(q_theo)))
    q_good = {'theo': q_theo[~mask_q_theo_fail],
              'sample': q_sample[~mask_q_theo_fail]}

    q_fail = {'theo': q_theo_native[~mask][mask_q_theo_fail],
              'sample': data_sorted[p_sample[mask_q_theo_fail]]}

    q = {'theo': np.append(q_good['theo'], q_fail['theo']),
         'sample': np.append(q_good['sample'], q_fail['sample'])}

    ss_res = np.nansum((q['sample'] - q['theo']) ** 2)
    ss_tot = np.nansum((q['sample'] - np.nanmean(q['sample'])) ** 2)
    r_2 = 1 - ss_res / ss_tot

    return q, q_good, q_fail, r_2


def test_dist(data: np.ndarray,
              mode: str,
              loc: float,
              scale: float,
              nconv: Union[float, None] = None,
              fit: bool = True,
              plot: bool = True,
              siglim: float = 7,
              n_eval_pdf: int = 1000,
              n_eval_qq: int = 100,
              y_log: bool = True,
              pdf_only: bool = False,
              color: str = 'tab:orange'):
    """
    A function to test the distribution of data against either a normal or an
    iterative modified bessel distribution. It can be used to check the fit of
    the PDF to the data histogram, as well as the Q-Q plot and the R^2 value.

    Args:
        data: np.ndarray
            The data set. Needs to be 1-D.
        mode: str
            The mode of the distribution. Either 'normal' for a Gaussian
            distribution or 'imb' for an iterative modified bessel
            distribution.
        loc: float
            The location parameter of the distribution.
        scale: float
            The scale parameter of the distribution.
        nconv: float, optional
            The number of convolution steps for the iterative modified bessel
            distribution. If the mode is 'normal', this parameter is ignored.
        fit: bool, optional
            If True, the function will fit the slope of the Q-Q plot. The
            default is True.
        plot: bool, optional
            If True, the function will create a plot of the Q-Q plot and the
            PDF. The default is True.
        siglim: float, optional
            The limit of up to how many standard deviations away from 0 the
            PDFs are evaluated. The default is 7.
        n_eval_pdf: int, optional
            The number of evaluation points for the PDF. The default is 1000.
        n_eval_qq: int, optional
            The number of evaluation points for the Q-Q plot. The default is
            100.
        y_log: bool, optional
            If True, the y-axis of the PDF will be logarithmic. The default is
            True.
        pdf_only: bool, optional
            If True, the function will only create the PDF plot. The default is
            False.
        color: str, optional
            The color of the Q-Q plot fit line. The default is 'tab:orange'.

    Returns:
        slope: float
            The slope of the Q-Q plot fit line.
        r_2: float
            The R^2 value of the Q-Q plot.
    """

    if nconv is None and mode == 'imb':
        raise ValueError('nconv must be specified for mode "imb"')

    if plot:
        fig, ax = plt.subplots(ncols=2, figsize=(8, 1.05 * 8 / 2), dpi=200)

        q_theo = np.linspace(-siglim, siglim, n_eval_pdf)

        hist = ax[0].hist(data, # / np.std(data),
                          bins=200,
                          density=True,
                          color='k',
                          histtype=u'step',
                          label='Data')

        if mode == 'normal':
            pdf_guess = norm.pdf(x=q_theo, loc=loc, scale=scale)
        elif mode == 'imb':
            dist = imb(loc=loc, scale=scale, n=nconv)
            pdf_guess = dist.pdf(x=q_theo)
        else:
            raise ValueError('Mode not recognized')

        if not fit:
            color_sec = color
        else:
            color_sec = 'k'

        ax[0].plot(q_theo, pdf_guess,
                   color=color_sec, ls='--', lw=0.75, alpha=0.5,
                   label='Distribution guess (hist)')

        if y_log:
            ax[0].set_yscale('log')
        ax[0].set_ylim(np.sort(np.unique(hist[0]))[1] * 0.5,
                       np.max(np.concatenate((hist[0], pdf_guess))) * 1.5)

    if not pdf_only:
        q, q_good, q_fail, r_2 = get_qq(data=data,
                                        loc=loc,
                                        scale=scale,
                                        mode=mode,
                                        nconv=nconv,
                                        siglim=siglim,
                                        n_eval=n_eval_qq)

        if fit:
            slope, _, _, _, _ = linregress(x=q['theo'], y=q['sample'])
            if not np.isfinite(slope):
                raise ValueError('Fitting failed, slope is not finite.')

        if plot:
            if fit:
                q, q_good, q_fail, r_2 = get_qq(data=data,
                                                loc=loc,
                                                scale=scale * slope,
                                                mode=mode,
                                                nconv=nconv,
                                                siglim=siglim,
                                                n_eval=n_eval_qq)

            lim = np.max(np.concatenate((np.abs(q['theo']),
                                         np.abs(q['sample']))))
            plt.scatter(q_good['theo'], q_good['sample'],
                        color='k', alpha=1, label='pure data')
            ax[1].scatter(q_fail['theo'], q_fail['sample'],
                          alpha=1, marker='o', fc='w',
                          label='potential faulty \nextrapolation',
                          color='gray')

            if fit:
                ax[1].plot([-lim, lim], [-lim, lim], color=color, ls='-')
                ax[1].plot([-lim, lim], [-lim / slope, lim / slope],
                           color=color_sec, alpha=0.5, lw=0.75, ls='--')
            else:
                ax[1].plot([-lim, lim], [-lim, lim],
                           color=color_sec, lw=0.75, ls='--')

            ax[1].text(0.95, 0.05, f'$R^2 = {r_2:.5f}$',
                       transform=ax[1].transAxes, va='bottom', ha='right')

            ax[1].set_xlabel('Theoretical quantiles')
            ax[1].set_ylabel('Sample quantiles')
            if np.isfinite(lim):
                ax[1].set_xlim(-lim * 1.1, lim * 1.1)
                ax[1].set_ylim(-lim * 1.1, lim * 1.1)
            ax[1].legend()

            if fit:
                if mode == 'normal':
                    pdf_fit = norm.pdf(x=q_theo,
                                             loc=loc,
                                             scale=scale * slope)
                elif mode == 'imb':
                    dist = imb(loc=loc, scale=scale * slope, n=nconv)
                    pdf_fit = dist.pdf(x=q_theo)
                else:
                    raise ValueError('Mode not recognized')
                ax[0].plot(q_theo, pdf_fit,
                           color=color, lw=0.75,
                           label='Distribution fit (R^2)')

            plt.tight_layout()
            plt.show()
        if plot:
            ax[0].legend(loc='lower center')
        if fit:
            return slope, r_2
        else:
            return r_2

def get_samples_lookup(scale_gauss: float,
                       scale_imb: float,
                       B: int,
                       N: int,
                       nconv: int = 11):
    """
    Generates samples of a test statistic for creating the sigma lookup
    table.

    This function generates samples by adding Gaussian random variables and
    iterative modified Bessel (imb) random variables. The test statistic

    .. math::

        T_X = \\frac{s_x}{\\sqrt{1 + \\frac{1}{N - 1}} \\sigma_{X_n}}

    is then calculated and returned.

    Parameters:
    scale_gauss (float): The scale parameter for the Gaussian random
    variables.
    scale_imb (float): The scale parameter for the iterative modified
    Bessel (imb) random variables.
    B (int): The number of Bootstap-samples to generate.
    N (int): The sample size for each set of Bootstrap-samples.
    nconv (int, optional): The number of convolutions for the iterative
    modified Bessel (imb) random variables. Defaults to 11.

    Returns:
    T_X (numpy.ndarray): The test statistic for each set of
    Bootstrap-samples.
    """

    s_x = (norm.rvs(loc=0,
                    scale=scale_gauss,
                    size=B)
           + imb.rvs(loc=0,
                     scale=scale_imb,
                     n=nconv,
                     size=B))
    X_n = (norm.rvs(loc=0,
                    scale=scale_gauss,
                    size=(B, N - 1))
           + imb.rvs(loc=0,
                     scale=scale_imb,
                     n=nconv,
                     size=(B, N - 1)))
    # T_X = s_x / np.std(X_n, axis=1)
    T_X = (np.mean(X_n, axis=1) - s_x) / np.std(X_n, axis=1) / np.sqrt(N / (N-2))

    return T_X

def get_sigma_lookup(sigma_gauss,
                     sigma_imb,
                     B,
                     N,
                     B_per,
                     n_sigma=1000,
                     nconv=11,
                     n_cpu=1,
                     verbose=False,
                     parallel=False):

    """
    Generates a lookup table for the sigma values of the combined Gaussian
    and iterative modified Bessel (IMB) distribution test statistic.

    This function generates a lookup table by drawing samples of a test
    statistic and calculating the desired and actual sigma values. The test
    statistic is generated by adding Gaussian random variables and
    iterative modified Bessel (imb) random variables.

    Parameters:
    sigma_gauss (float): The scale parameter for the Gaussian random
    variables.
    sigma_imb (float): The scale parameter for the iterative modified
    Bessel (imb) random variables.
    B (int): The total number of Bootstrap-samples to generate.
    N (int): The sample size for each set of Bootstrap-samples.
    B_per (int): The number of Bootstrap-samples to generate per job in
    multi-processing.
    n_sigma (int, optional): The number of sigma values to generate for the
    lookup table. Defaults to 1000.
    n_cpu (int, optional): The number of CPUs to use for parallel
    processing. Defaults to 1.

    Returns:
    sigma_want (numpy.ndarray): The desired sigma values for the lookup
    table.
    sigma_get (numpy.ndarray): The actual sigma values for the lookup
    table.
    """

    if verbose:
        print('Drawing samples ...')

    if n_cpu == 1:
        T_X = get_samples_lookup(scale_gauss=sigma_gauss,
                                 scale_imb=sigma_imb,
                                 B=B,
                                 N=N,
                                 nconv=nconv)
    else:
        with parallel_config(
                backend="loky",
                inner_max_num_threads=1
        ), joblib_progress(
            description="Drawing samples ...",
            total=int(B / B_per)
        ):
            results = Parallel(
                n_jobs=n_cpu
            )(delayed(get_samples_lookup)(
                scale_gauss=sigma_gauss,
                scale_imb=sigma_imb,
                B=B_per,
                nconv=nconv,
                N=N
            ) for _ in range(int(B / B_per)))

        T_X = np.concatenate(results)

    if verbose:
        print('[Done]')
        print('Sorting the test statistic ...', end=' ', flush=True)
        t = time.time()

    # sort the T_X values and note at which overall percentage each T_X value
    # appears
    if parallel:
        from parallel_sort import parallel_sort
        T_X_sort = parallel_sort(T_X)
    else:
        T_X_sort = np.sort(T_X)

    if verbose:
        print(f'[Done] ({(time.time() - t):.2f}s)')
        print('Calculating the sigma values ...', end=' ', flush=True)
        t = time.time()

    perc = np.linspace(start=1 / len(T_X_sort),
                       stop=1,
                       num=len(T_X_sort),
                       endpoint=True)

    sigma_want = np.linspace(start=0,
                             stop=t_dist(df=N - 1).ppf(1 - 1 / (B - 1)),
                             num=n_sigma)

    p_want = t_dist(df=N - 1).cdf(sigma_want)

    # Interpolate to find values in perc corresponding to p_want
    perc_interp = np.interp(p_want, perc, np.arange(len(perc)))

    # Convert interpolated indices to integers
    perc_indices = np.round(perc_interp).astype(int)

    # Get corresponding values of T_X
    sigma_get = T_X_sort[perc_indices]

    # indices = []
    # i = 0
    # for value in tqdm(p_want, disable=np.invert(verbose)):
    #     while i < len(perc) - 1 and perc[i + 1] < value:
    #         i += 1
    #     if (i < len(perc) - 1
    #             and abs(perc[i + 1] - value) < abs(perc[i] - value)):
    #         i += 1
    #     indices.append(i)
    # sigma_get = T_X_sort[indices]

    # the actual sigma is the value at the test statistic where the p-value
    # is equal to the desired sigma in a T-distribution
    # sigma_get = []
    # for sw in tqdm(sigma_want, disable=np.invert(verbose)):
    #     sigma_get.append(
    #         T_X_sort[np.where(perc > t_dist(df=N - 1).cdf(sw))[0][0]]
    #     )

    if verbose:
        print(f'[Done] ({(time.time() - t):.2f}s)')

    return sigma_want, np.array(sigma_get)
