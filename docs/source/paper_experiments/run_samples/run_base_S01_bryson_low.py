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
    bus.data.options.set_manual(n_cpu=20)
    bus.data.options.set_manual(
        output_path='/net/ipa-gate/export/ipa/quanz/user_accounts/fdannert/spie_22/40_experiments/S01_Lay/')
    bus.data.options.set_manual(output_filename='S01_base_bryson_hab2low')
    bus.data.options.set_noise_scenario(case='lay')

    # ---------- Loading the Catalog ----------
    bus.data.catalog_from_ppop('/net/ipa-gate/export/ipa/quanz/user_accounts/fdannert/spie_22/10_data/planet_populations_feb23/Brysonhab2low_500uni_PlanetPopulation.txt')
    bus.data.catalog_remove_distance(stype='A', mode='larger', dist=0.)  # remove all A stars
    bus.data.catalog_remove_distance(stype='M', mode='larger', dist=0.)  # remove M stars > 10pc to

    # ---------- Creating the Instrument ----------

    # create modules and add to bus
    instrument = lifesim.Instrument(name='inst')
    bus.add_module(instrument)

    instrument_prt = lifesim.InstrumentPrt(name='inst_prt')
    bus.add_module(instrument_prt)

    bus.connect(('inst', 'inst_prt'))

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

    instrument_prt.get_snr(safe_mode=True)
    bus.save()
