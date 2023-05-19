import lifesim

# create bus
bus = lifesim.Bus()

# setting the options
bus.data.options.set_scenario('baseline')
bus.build_from_config(filename='/home/ipa/quanz/user_accounts/fdannert/spie_22/40_experiments/S01_Lay/S01_base.yaml')

bus.data.import_catalog(input_path='/home/ipa/quanz/user_accounts/fdannert/spie_22/40_experiments/S01_Lay/'
                                   'S01_base.hdf5',
                        noise_catalog=True)

ana = lifesim.SampleAnalysisModule(name='ana')
bus.add_module(ana)

ana.get_fundamental_snr()

# optimizing the result
opt = lifesim.Optimizer(name='opt')
bus.add_module(opt)
ahgs = lifesim.AhgsModule(name='ahgs')
bus.add_module(ahgs)

bus.connect(('opt', 'ahgs'))

opt.ahgs()

a=1