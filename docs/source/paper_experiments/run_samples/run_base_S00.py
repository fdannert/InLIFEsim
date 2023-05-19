import os
#must set these before loading numpy:
os.environ["OMP_NUM_THREADS"] = '1'  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = '1'  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = '1'  # export MKL_NUM_THREADS=6

import lifesim

if __name__ == '__main__':

    # create bus
    bus = lifesim.Bus()

    # setting the options
    bus.data.options.set_scenario('baseline')
    bus.data.options.set_manual(n_cpu=1)
    bus.data.options.set_manual(
        output_path='/net/ipa-gate/export/ipa/quanz/user_accounts/fdannert/spie_22/40_experiments/S00_lifesim/')
    bus.data.options.set_manual(output_filename='S00_base_SAG13_April23')

    # ---------- Loading the Catalog ----------
    bus.data.catalog_from_ppop('/net/ipa-gate/export/ipa/quanz/user_accounts/fdannert/spie_22/10_data/'
                               'planet_populations_feb23/SAG13_500uni_PlanetPopulation.txt')
    bus.data.catalog_remove_distance(stype='A', mode='larger', dist=0.)  # remove all A stars
    bus.data.catalog_remove_distance(stype='M', mode='larger', dist=10.)  # remove M stars > 10pc to

    # ---------- Creating the Instrument ----------

    # create modules and add to bus
    instrument = lifesim.Instrument(name='inst')
    bus.add_module(instrument)

    transm = lifesim.TransmissionMap(name='transm')
    bus.add_module(transm)

    exo = lifesim.PhotonNoiseExozodi(name='exo')
    bus.add_module(exo)
    local = lifesim.PhotonNoiseLocalzodi(name='local')
    bus.add_module(local)
    star = lifesim.PhotonNoiseStar(name='star')
    bus.add_module(star)

    # connect all modules
    bus.connect(('inst', 'transm'))
    bus.connect(('inst', 'exo'))
    bus.connect(('inst', 'local'))
    bus.connect(('inst', 'star'))

    bus.connect(('star', 'transm'))

    # ---------- Running the Simulation ----------

    instrument.get_snr(save_mode=True)
    bus.save()
