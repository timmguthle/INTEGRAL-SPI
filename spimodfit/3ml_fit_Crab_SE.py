from threeML import *
import matplotlib.pyplot as plt

###########
# DATASET #
###########
crab = OGIPLike("crab",
                observation='/afs/ipp-garching.mpg.de/mpe/gamma/instruments/integral/shared_analysis/cookbook/examples/Crab/fit_Crab_SE_02/spectra_Crab.fits',
                response='/afs/ipp-garching.mpg.de/mpe/gamma/instruments/integral/shared_analysis/cookbook/examples/Crab/fit_Crab_SE_02/spectral_response.rmf.fits')

###################
# ACTIVE CHANNELS #
###################
crab.set_active_measurements('20 - 600')

##################
# SPECTRAL MODEL #
##################
spec = Broken_powerlaw()

##############################
# DEFINITION OF POINT SOURCE #
##############################
ps = PointSource('crab',l=0,b=0,spectral_shape=spec)

####################
# MODEL DEFINITION #
####################
ps_model = Model(ps)

####################
# FIXED PARAMETERS #
####################
ps_model.crab.spectrum.main.Broken_powerlaw.xb = 100
ps_model.crab.spectrum.main.Broken_powerlaw.xb.fix = True

#################
# DISPLAY MODEL #
#################
ps_model.display(complete=True)

###################
# DATA DEFINITION #
###################
ps_data = DataList(crab)

#####################
# LIKELIHOOD OBJECT #
#####################
ps_jl = JointLikelihood(ps_model, ps_data)

#######
# FIT #
#######
best_fit_parameters_ps, likelihood_values_ps = ps_jl.fit()

####################
# RESTORE BEST FIT #
####################
ps_jl.restore_best_fit()

#######################
# PLOT DATA AND MODEL #
#######################
plt.figure()
display_spectrum_model_counts(ps_jl,step=True)
plt.savefig('Crab_spectrum_SE.pdf')
