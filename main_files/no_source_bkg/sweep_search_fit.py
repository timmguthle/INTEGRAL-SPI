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


positions = [[10, -40], [10, -39], [10, -41], [11, -40], [11, -41], [11, -39], [9, -40], [9, -41], [9, -39]]
single_positions = [[10, -40]]

def create_positions(n, delta=0.5):
    """
    Create a list of positions in a grid around (10, -40). Delta is the vertical and horizontal distance between the positions.
    the grid is n x n. To center the grid n should be odd.
    """
    positions = []
    for i in range(n//2, -n//2, -1):
        for j in range(n//2, -n//2, -1):
            positions.append([10+(i*delta), -40+(j*delta)])
    return positions




def pyspi_search_0374(
        data_path="/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/0374",
        fit_base_path="/home/tguethle/Documents/spi/Master_Thesis/main_files/no_source_bkg/sweep_search_1",
        positions=positions,
        piv=100):
    
    total_positions = len(positions)
    
    if not os.path.exists(fit_base_path):
        os.makedirs(fit_base_path)

    with open(f"{fit_base_path}/fit_summary.txt", "a") as f:
        f.write(f"Fit summary for Potential Source 0374\n")
        f.write(f"Data path: {data_path}\n")
        f.write(f"Fit base path: {fit_base_path}\n")
        f.write(f"Energy pivot: {piv}\n")

    for n,(ra,dec) in enumerate(positions):
        fit_path = f"{fit_base_path}/{ra}_{dec}"

        if not os.path.exists(fit_path):
            os.makedirs(fit_path)

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

        with open(f"{fit_base_path}/fit_summary.txt", "a") as f:
            f.write(f"Position: {ra}, {dec}; {n+1}/{total_positions}\n")
            f.write(f"K: {val[0]} +/- {np.sqrt(cov[0,0])}\n")
            f.write(f"Index: {val[1]} +/- {np.sqrt(cov[1,1])}\n")
            f.write("\n")

def plot_positions(positions=positions, fit_base_path="/home/tguethle/Documents/spi/Master_Thesis/main_files/no_source_bkg/sweep_search_2"):

    if not os.path.exists(fit_base_path):
        os.makedirs(fit_base_path)

    fig, ax = plt.subplots(figsize=(5,4), subplot_kw={'projection': 'astro degrees zoom', 'center': '10deg -40deg', 'radius': '7deg'})

    ax.grid()
    ax.set_title('Positions of Potential sources')

    for ra, dec in positions:
        ax.scatter(ra, dec, transform=ax.get_transform('fk5'), c='tab:blue', s=40)

    fig.savefig(f"{fit_base_path}/positions.png")


if __name__ == "__main__":
    wide_positions_15 = create_positions(15, 0.75)
    wide_positions_9 = create_positions(9, 0.75)
    for pos in wide_positions_9:
        wide_positions_15.remove(pos)
    plot_positions(positions=wide_positions_15[50:])
    pyspi_search_0374(positions=wide_positions_15[50:], fit_base_path="/home/tguethle/Documents/spi/Master_Thesis/main_files/no_source_bkg/sweep_search_2")