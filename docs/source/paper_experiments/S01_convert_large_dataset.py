import lifesim

# create bus
bus = lifesim.Bus()

# setting the options
bus.data.options.set_scenario('baseline')
bus.build_from_config(filename='/home/ipa/quanz/user_accounts/fdannert/spie_22/40_experiments/S01_Lay/S01_base.yaml')
bus.data.options.other['large_file'] = False
bus.data.options.other['pickle_mode'] = False

bus.data.import_catalog(input_path='/home/ipa/quanz/user_accounts/fdannert/spie_22/40_experiments/S01_Lay/S01_base.hdf5',
                        noise_catalog=True)

bus.data.options.set_manual(pickle_mode=True)
bus.save()
