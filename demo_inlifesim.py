import numpy as np

import inlifesim as ils

def compare_to_lay():
    integration_time = 50000
    wl_bins = np.array((10e-6, ))
    wl_bin_widths = np.array((0.5e-6, ))

    inst = ils.Instrument(wl_bins=wl_bins,
                          wl_bin_widths=wl_bin_widths,
                          image_size=500,
                          diameter_ap=4.,
                          flux_division=np.array((0.25, 0.25, 0.25, 0.25)),
                          throughput=0.1,
                          dist_star=15.,
                          radius_star=1.,
                          col_pos=np.array(((-30, 0), (-10, 0), (10, 0), (30, 0))),
                          phase_response=np.array((0, np.pi / 2, np.pi, 3 * np.pi / 2)),
                          phase_response_chop=-np.array((0, np.pi / 2, np.pi, 3 * np.pi / 2)),
                          t_rot=integration_time,
                          pix_per_wl=2.2,
                          n_sampling_rot=1000,
                          pink_noise_co=10000,
                          temp_star=5770.,
                          temp_planet=265.,
                          radius_planet=1.,
                          lat_star=0.79,
                          l_sun=1,
                          z=1,
                          separation_planet=1.,
                          detector_dark_current='manual',
                          dark_current_pix=0.0,
                          detector_thermal='MIRI',
                          det_temp=0.,
                          rms_mode='lay',
                          n_cpu=1,
                          chopping='both',
                          magnification=15.73,
                          f_number=20.21,
                          secondary_primary_ratio=0.114,
                          primary_temp=0.,
                          primary_emissivity=0.00,
                          integration_time=integration_time)

    inst.run()

    inst.photon_rates['lay_nchop'] = [0.097,
                                      0.070,
                                      1e-5,
                                      0.058,
                                      0.022,
                                      0.04,
                                      np.nan,
                                      np.nan,
                                      np.nan,
                                      0.004,
                                      0.006,
                                      np.nan,
                                      np.nan,
                                      np.nan,
                                      0.07,
                                      0.026,
                                      0.024,
                                      0.01,
                                      np.nan,
                                      0.037,
                                      0.025,
                                      0.025,
                                      0.042,
                                      0.025,
                                      0.061,
                                      0.071,
                                      0.074,
                                      0.071,
                                      0.971]

    inst.photon_rates['lay_nchop'][:-1] *= integration_time

    inst.photon_rates['dev_nchop'] = np.round(((inst.photon_rates['nchop'] - inst.photon_rates['lay_nchop'])
                                              / inst.photon_rates['lay_nchop'] * 100).values.astype(float))

    inst.photon_rates['lay_chop'] = [0.097,
                                     0.085,
                                     1e-5,
                                     0.058,
                                     0.022,
                                     0.04,
                                     np.nan,
                                     np.nan,
                                     np.nan,
                                     0.004,
                                     0.006,
                                     np.nan,
                                     np.nan,
                                     np.nan,
                                     0.07,
                                     np.nan,
                                     0.025,
                                     np.nan,
                                     np.nan,
                                     0.025,
                                     np.nan,
                                     np.nan,
                                     0.042,
                                     np.nan,
                                     0.042,
                                     0.049,
                                     0.074,
                                     0.050,
                                     1.132]

    inst.photon_rates['lay_chop'][:-1] *= integration_time

    inst.photon_rates['dev_chop'] = np.round(((inst.photon_rates['chop'] - inst.photon_rates['lay_chop'])
                                              / inst.photon_rates['lay_chop'] * 100).values.astype(float))

    a=1


if __name__ == '__main__':
    # create wavelength bins
    # wl_bins = np.arange(4, 6, 0.5)  # to mimic Lay2004 in first bin
    # wl_bin_widths = np.ones_like(wl_bins) * (wl_bins[1] - wl_bins[0])
    #
    # wl_bins *= 1e-6
    # wl_bin_widths *= 1e-6
    #
    # inst = itn.Instrument(wl_bins=wl_bins,
    #                       wl_bin_widths=wl_bin_widths,
    #                       image_size=500,
    #                       diameter_ap=4.,
    #                       flux_division=np.array((0.25, 0.25, 0.25, 0.25)),
    #                       throughput=0.1,
    #                       dist_star=15.,
    #                       radius_star=1.,
    #                       col_pos=np.array(((-30, 0), (-10, 0), (10, 0), (30, 0))),
    #                       phase_response=np.array((0, np.pi / 2, np.pi, 3 * np.pi / 2)),
    #                       phase_response_chop=-np.array((0, np.pi / 2, np.pi, 3 * np.pi / 2)),
    #                       t_rot=50000,
    #                       pix_per_wl=2.2,
    #                       n_sampling_rot=360,
    #                       pink_noise_co=10000)
    #
    # inst.run_multiwav(temp_star=5770.,
    #                   temp_planet=265.,
    #                   radius_planet=1.,
    #                   lat_star=0.79,
    #                   l_sun=1,
    #                   z=1,
    #                   separation_planet=1.,
    #                   dark_current_pix=0.0001,
    #                   det_temp=11.,
    #                   rms_mode='lay',
    #                   n_cpu=4)
    compare_to_lay()



