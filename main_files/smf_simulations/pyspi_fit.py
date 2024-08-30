import sys, os
sys.path.insert(0, os.path.abspath('./main_files'))

import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
from astromodels import Powerlaw, PointSource, SpectralComponent
import astropy.time as at
from datetime import datetime
from pyspi.utils.function_utils import find_response_version
from pyspi.utils.response.spi_response_data import ResponseDataRMF
from pyspi.utils.response.spi_response import ResponseRMFGenerator
from pyspi.utils.response.spi_drm import SPIDRM
from pyspi.utils.livedets import get_live_dets
import os
from MultinestClusterFit import powerlaw_binned_spectrum, MultinestClusterFit
from RebinningFunctions import spimodfit_binning_SE, log_binning_function_for_x_number_of_bins, no_rebinning #, rebin_data_exp_50
from PointingClusters import PointingClusters, save_clusters, load_clusters
from ModelSources import *
import pickle
from datetime import datetime

revolution = "0374"

#spi_data_path = f"./spiselect_SPI_Data/{revolution}"
#afs_data_path = f"./afs_SPI_Data/{revolution}"


def pyspi_fit_0374_pre_ppc(
        data_path=None,
        fit_path_extension="pre_ppc", 
        new_pointings=True,
        Energy_range=(20, 600),
        piv=100,
        **kwargs):
    
    rev = "0374"
    ra, dec = 10, -40
    #data_path = f"./main_files/spimodfit_comparison_sim_source/pyspi_const_bkg_Timm13/{rev}"
    fit_path_old = f"{data_path}/pre_ppc"
    fit_path = f"{data_path}/{fit_path_extension}"
    
    if not os.path.exists(f"{fit_path}"):
        os.makedirs(f"{fit_path}")

    if new_pointings:
        pointings = PointingClusters(
                (data_path,),
                min_angle_dif=1.5,
                max_angle_dif=10., # different from ps
                max_time_dif=0.2,
                radius_around_source=10.,
                min_time_elapsed=300.,
                cluster_size_range=(2,2),
                center_ra=ra,
                center_dec=dec,
            ).pointings
        save_clusters(pointings, fit_path)
    else:  
        pointings = load_clusters(fit_path_old)
    
    if rev=="0374":
        s = simulated_pl_0374
    elif rev=="1380":
        s = simulated_pl_1380
    
    source_model = define_sources((
        (s, (piv,)),
    ))
    
    multinest_fit = MultinestClusterFit(
        pointings,
        source_model,
        Energy_range,
        np.arange(20, 600.5, 0.5),
        #np.geomspace(18, 2000, 200),
        no_rebinning,
        #log_binning_function_for_x_number_of_bins(70),
        #spimodfit_binning_SE,
        # true_values=true_values(),
        folder=fit_path,
    )
    
    multinest_fit.parameter_fit_distribution()
    multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
    #multinest_fit.ppc()
    
    p = ["Simulated Source 0374 K", "Simulated Source 0374 index"]
    val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()]) #type: ignore (dont know why vs code thinks this is an error, it is not)
    cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

    with open(f"{fit_path}/source_parameters.pickle", "wb") as f:
        pickle.dump((val, cov), f)

    with open(f"{fit_path}/pyspi_summary.txt", "w") as f:
        f.write(f"Energy range: {Energy_range}\n")
        f.write(f"Data path: {data_path}\n")
        f.write(f"Fit path: {fit_path}\n")
        f.write(f"Result: {val}\n")
        f.write(f"Covariance: {cov}\n")
        f.write(f"compleated at {datetime.now()}")



if __name__ == "__main__":
    # for i,c in enumerate(config_small_bins_2):
    #     #gen_and_fit(c) #fit takes to long with 1000+ bins so only generate data for now
    #     if i == 4:
    #         pyspi_fit_0374_pre_ppc(**c)

    #################################
    #Test

    # pyspi_fit_0374_pre_ppc(
    #     data_path="/home/tguethle/Documents/spi/Master_Thesis/main_files/smf_simulations/test_data_normal_new/",
    #     fit_path_extension="test_fit_better",
    #     new_pointings=True,
    # )

    # pyspi_fit_0374_pre_ppc(
    #     data_path="/home/tguethle/Documents/spi/Master_Thesis/main_files/smf_simulations/test_data_normal_new_big_K/",
    #     fit_path_extension="test_fit",
    #     new_pointings=True,
    # )


    # pyspi_fit_0374_pre_ppc(
    #     data_path="/home/tguethle/Documents/spi/Master_Thesis/main_files/smf_simulations/test_data_normal_new_small_K/",
    #     fit_path_extension="test_fit",
    #     new_pointings=True,
    # )


    # pyspi_fit_0374_pre_ppc(
    #     data_path="/home/tguethle/Documents/spi/Master_Thesis/main_files/smf_simulations/test_data_normal_new/",
    #     fit_path_extension="test_fit_wo_low_energy",
    #     new_pointings=True,
    #     Energy_range=(40, 600),
    # )


    # pyspi_fit_0374_pre_ppc(
    #     data_path="/home/tguethle/Documents/spi/Master_Thesis/main_files/smf_simulations/test_data_normal_new_small_K/",
    #     fit_path_extension="test_fit_wo_low_energy",
    #     new_pointings=True,
    #     Energy_range=(40, 600),
    # )

    ############################################## K_1 #########################################################
    pyspi_fit_0374_pre_ppc(
        data_path="/home/tguethle/Documents/spi/Master_Thesis/main_files/smf_simulations/K_01/",
        fit_path_extension="test_fit",
        new_pointings=True,
    )


