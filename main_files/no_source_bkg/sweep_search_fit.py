import sys, os
sys.path.insert(0, os.path.abspath('./main_files'))

import numpy as np
import matplotlib.pyplot as plt
import ligo.skymap.plot

from MultinestClusterFit import MultinestClusterFit
from RebinningFunctions import spimodfit_binning_SE, log_binning_function_for_x_number_of_bins, no_rebinning #, rebin_data_exp_50
from PointingClusters import PointingClusters, save_clusters, load_clusters
from ModelSources import *
import pickle
import datetime

from plot_search_result import create_positions, plot_positions


positions = [[10, -40], [10, -39], [10, -41], [11, -40], [11, -41], [11, -39], [9, -40], [9, -41], [9, -39]]
single_positions = [[10, -40]]



def pyspi_search_0374(
        data_path="/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/0374",
        fit_base_path="/home/tguethle/Documents/spi/Master_Thesis/main_files/no_source_bkg/sweep_search_1",
        positions=positions,
        piv=100):
    
    total_positions = len(positions)
    
    if not os.path.exists(fit_base_path):
        os.makedirs(fit_base_path)

    with open(f"{fit_base_path}/fit_summary.txt", "w") as f:
        f.write(f"Fit summary for Potential Source 0374\n")
        f.write(f"Data path: {data_path}\n")
        f.write(f"Fit base path: {fit_base_path}\n")
        f.write(f"Energy pivot: {piv}\n")

    for n,(ra,dec) in enumerate(positions):
        start_time = datetime.datetime.now()
        fit_path = f"{fit_base_path}/{ra}_{dec}"

        if not os.path.exists(fit_path):
            os.makedirs(fit_path)

        pointings = PointingClusters(
            (data_path,),
            min_angle_dif=1.5,
            max_angle_dif=10., # different from ps
            max_time_dif=0.2,
            radius_around_source=15.,
            min_time_elapsed=300.,
            cluster_size_range=(2,2),
            center_ra=ra,
            center_dec=dec,
        ).pointings
        #save_clusters(pointings, fit_path)
    
        
        
        source_model = define_sources((
            (sweep_search_pl_0374, (piv,ra,dec)),
        ))
        
        multinest_fit = MultinestClusterFit(
            pointings,
            source_model,
            (20, 600,),
            np.geomspace(18, 600, 50),
            no_rebinning,
            #log_binning_function_for_x_number_of_bins(70),
            #spimodfit_binning_SE,
            # true_values=true_values(),
            folder=fit_path,
        )
        multinest_fit.parameter_fit_distribution()
        multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
        #multinest_fit.ppc()
        
        p = ["Potential Source 0374 K", "Potential Source 0374 index"]
        val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
        cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

        with open(f"{fit_path}/source_parameters.pickle", "wb") as f:
            pickle.dump((val, cov), f)

        delta_time = datetime.datetime.now() - start_time

        with open(f"{fit_base_path}/fit_summary.txt", "a") as f:
            f.write(f"Position: {ra}, {dec}; {n+1}/{total_positions}\n")
            f.write(f"K: {val[0]} +/- {np.sqrt(cov[0,0])}\n")
            f.write(f"Index: {val[1]} +/- {np.sqrt(cov[1,1])}\n")
            f.write(f"Fit duration: {delta_time.seconds}s\n")
            f.write("\n")


def sweep_search_2():
    wide_positions_15 = create_positions(15, 0.75)
    wide_positions_9 = create_positions(9, 0.75)
    for pos in wide_positions_9:
        wide_positions_15.remove(pos)
    plot_positions(positions=wide_positions_15[50:])
    
    pyspi_search_0374(positions=wide_positions_15[50:], fit_base_path="/home/tguethle/Documents/spi/Master_Thesis/main_files/no_source_bkg/sweep_search_2")


def sweep_search_3():
    positions_big_diff = create_positions(15, 2.0)
    #plot_positions(positions=positions, fit_base_path="/home/tguethle/Documents/spi/Master_Thesis/main_files/no_source_bkg/sweep_search_3")
    pyspi_search_0374(positions=positions_big_diff, fit_base_path="/home/tguethle/Documents/spi/Master_Thesis/main_files/no_source_bkg/sweep_search_3")

if __name__ == "__main__":
    sweep_search_3()