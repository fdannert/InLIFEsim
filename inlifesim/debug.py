import matplotlib.pyplot as plt
import numpy as np

global DEBUG
DEBUG = False

global DPI
DPI = 200

def debug_planet_signal(n_planet_nchop,
                        planet_template_nchop,
                        n_planet_chop,
                        planet_template_chop,
                        wl_bins,
                        nbin=-1):
    if DEBUG:
        fig, ax = plt.subplots(nrows=2, figsize=(6,8), dpi=DPI)
        ax[0].plot(n_planet_nchop[nbin, ] / np.max(n_planet_nchop[nbin, ]), label='n_planet_nchop')
        ax[0].plot(planet_template_nchop[nbin, ] / np.max(planet_template_nchop[nbin, ]), label='planet_template_nchop', ls='--')
        ax[0].set_title('Planet signal, no chop,  ' + str(np.round(wl_bins[nbin]*1e6, 1)) + 'µm, normalized')
        ax[0].legend()

        ax[1].plot(n_planet_chop[nbin, ] / np.max(n_planet_chop[nbin, ]), label='n_planet_chop')
        ax[1].plot(planet_template_chop[nbin, ] / np.max(planet_template_chop[nbin, ]), label='planet_template_chop', ls='--')
        ax[1].legend()

        plt.show()


def debug_sys_noise_chop(d_phi_b_2,
                         planet_template_c_fft,
                         d_phi_j_hat_2_chop):
    if DEBUG:

        delta = 40
        fig, ax = plt.subplots(dpi=DPI)
        handle1 = ax.plot(d_phi_b_2, color='tab:blue', label='d_phi_b_2')
        ax1 = ax.twinx()
        handle2 = ax1.plot(planet_template_c_fft, color='tab:orange', label='planet_template_c_fft')
        handle = handle1 + handle2
        labs = [l.get_label() for l in handle]
        plt.legend(handle, labs)
        w = len(planet_template_c_fft)
        plt.xlim(w/2-delta,
                 w/2+delta)
        ax.text(s='d_phi_j_hat_2_chop[0]: \n' + '{:.2e}'.format(d_phi_j_hat_2_chop[0]),
                 x=w/2-delta+5,
                    y=80)
        plt.show()


