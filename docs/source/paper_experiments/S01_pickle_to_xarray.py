import pickle
import xarray as xr
import numpy as np

file = open('/home/ipa/quanz/user_accounts/fdannert/spie_22/40_experiments/S01_Lay/S01_base_noise_pickle.pickle', 'rb')
noise_catalog = pickle.load(file)
file.close()

noise_catalog = xr.Dataset(noise_catalog).to_array()
noise_catalog = noise_catalog.rename({'dim_0': 'wl_bins',
                                      'dim_1': 'params',
                                      'variable': 'ids'})

wl_edge = 4.
wl_bins = []
wl_bin_widths = []
wl_bin_edges = [wl_edge]

while wl_edge < 18.5:

    # set the wavelength bin width according to the spectral resolution
    wl_bin_width = wl_edge / 20. / \
                   (1 - 1 / 20. / 2)

    # make the last bin shorter when it hits the wavelength limit
    if wl_edge + wl_bin_width > 18.5:
        wl_bin_width = 18.5 - wl_edge

    # calculate the center and edges of the bins
    wl_center = wl_edge + wl_bin_width / 2
    wl_edge += wl_bin_width

    wl_bins.append(wl_center)
    wl_bin_widths.append(wl_bin_width)
    wl_bin_edges.append(wl_edge)

# convert everything to [m]
wl_bins = np.array(wl_bins) * 1e-6  # in m

noise_catalog = noise_catalog.assign_coords(
    wl_bins=wl_bins,
    params=noise_catalog.coords['params'].values.astype(str),
    ids=noise_catalog.coords['ids'].values.astype(int)
)

noise_catalog.to_netcdf(path='/home/ipa/quanz/user_accounts/fdannert/spie_22/40_experiments/S01_Lay/S01_base_noise.nc',
                        mode='w',
                        engine='h5netcdf')
