import numpy as np
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