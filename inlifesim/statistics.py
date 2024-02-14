from typing import Union

import numpy as np
from scipy.special import kv, gamma
from scipy.stats import rv_continuous, norm, linregress
from scipy.interpolate import UnivariateSpline, Akima1DInterpolator
import matplotlib.pyplot as plt

from inlifesim.util import freq2temp_fft, dict_sumover


def draw_sample(params,
                return_variables=['xcorr']):

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

    if return_variables == 'all':
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

        # params['n_sampling_rot'] * np.sqrt(d_phi_psd[:, np.newaxis, int((d_phi_psd.shape[-1]-1)/2)::] / 2 /params['t_rot'])
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

    # d_phi_ft = (np.random.normal(loc=0,
    #                              scale=params['n_sampling_rot'] * np.sqrt(
    #                                  d_phi_psd[:, np.newaxis, int((d_phi_psd.shape[-1] - 1) / 2)::] / 2 / params[
    #                                      't_rot']),
    #                              size=(4, ndraws, int((d_phi_psd.shape[-1] + 1) / 2)))
    #             + 1j * np.random.normal(loc=0,
    #                                     scale=params['n_sampling_rot'] * np.sqrt(
    #                                         d_phi_psd[:, np.newaxis, int((d_phi_psd.shape[-1] - 1) / 2)::] / 2 / params[
    #                                             't_rot']),
    #                                     size=(4, ndraws, int((d_phi_psd.shape[-1] + 1) / 2)))
    #             )
    #
    # d_phi_ft = np.concatenate((np.flip(d_phi_ft[:, :, 1:], axis=-1), d_phi_ft), axis=-1)
    #
    # d_phi_time = freq2temp_fft(fourier_series=d_phi_ft,
    #                            total_time=params['t_rot'])

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
        pdf = ((2 ** (0.5 * (1 - n)) * np.abs(x) ** (0.5 * (n-1)) * kv(0.5 * (n - 1), np.abs(x)))
               / (np.sqrt(np.pi) * gamma(n / 2)))
        return pdf

    def _ppf(self, q, *args):
        if self.ppf_spline is None:
            # Generate some points
            x = np.linspace(-30, 30, 1000)
            y = self._cdf(x, *args)

            # select only the values from y that are strictly increasing
            y, x = make_strictly_increasing(y, x)

            # Fit a polynomial to the function
            self.ppf_spline = Akima1DInterpolator(y, x)

        # Evaluate the inverse of the polynomial at the given points
        return self.ppf_spline(q)


imb = imb_gen(name='imb', shapes='n')

def make_strictly_increasing(arr1, arr2):
    epsilon = 1e-10
    for i in range(1, len(arr1)):
        if arr1[i] <= arr1[i-1]:
            increment = arr1[i-1] + epsilon - arr1[i]
            arr1[i] += increment
            arr2[i] += increment
    return arr1, arr2

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
    data /= np.std(data)

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
    mask_q_theo_fail = (q_theo == 0.)
    q_good = {'theo': q_theo[~mask_q_theo_fail],
              'sample': q_sample[~mask_q_theo_fail]}

    q_fail = {'theo': q_theo_native[~mask][mask_q_theo_fail],
              'sample': data_sorted[p_sample[mask_q_theo_fail]]}

    q = {'theo': np.append(q_good['theo'], q_fail['theo']),
         'sample': np.append(q_good['sample'], q_fail['sample'])}

    ss_res = np.sum((q['sample'] - q['theo']) ** 2)
    ss_tot = np.sum((q['sample'] - np.mean(q['sample'])) ** 2)
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

        hist = ax[0].hist(data / np.std(data),
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

        ax[0].plot(q_theo, pdf_guess,
                   color='k', ls='--', lw=0.75, alpha=0.5,
                   label='Distribution guess (hist)')

        if y_log:
            ax[0].set_yscale('log')
        ax[0].legend(loc='lower center')
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
                           color='k', alpha=0.5, lw=0.75, ls='--')
            else:
                ax[1].plot([-lim, lim], [-lim, lim],
                           color='k', lw=0.75, ls='--')

            ax[1].text(0.95, 0.05, f'$R^2 = {r_2:.5f}$',
                       transform=ax[1].transAxes, va='bottom', ha='right')

            ax[1].set_xlabel('Theoretical quantiles')
            ax[1].set_ylabel('Sample quantiles')
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

        if fit:
            return slope, r_2
        else:
            return r_2
