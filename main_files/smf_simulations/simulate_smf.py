import sys, os
sys.path.insert(0, os.path.abspath('/home/tguethle/Documents/spi/Master_Thesis/main_files'))


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
# from MultinestClusterFit import powerlaw_binned_spectrum, MultinestClusterFit
from RebinningFunctions import spimodfit_binning_SE, log_binning_function_for_x_number_of_bins, no_rebinning #, rebin_data_exp_50
# from PointingClusters import PointingClusters, save_clusters, load_clusters
from ModelSources import *
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib

def powerlaw(x, K=7e-4,index=-2,piv=100):
    return K * (x/piv)**index

def gen_smf_data(
        smf_name,
        data_path,
        K=7e-4,
        index=-2,
        orig_data_path='/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/0374_center/',
        use_time_elapsed=True,
        use_weighting=True
):
    """
    Simulate counts with spimodfit using the convolved sky image. safe data to new folder in pyspi format.

    Parameters
    ----------
    smf_name: str
        name of the spimodfit run. get the conv sky output and the pointing info from this name. 
    data_path: str
        path to the folder, where the data should be saved.
    orig_data_path: str
        path to the folder, where the original data is saved. Maybe redundant. should be consistent with name.
    
    """

    smf_path = f'/home/tguethle/cookbook/SPI_cookbook/examples/automated_Crab/fit_Crab_{smf_name}/'
    spiselect_path = f'/home/tguethle/cookbook/SPI_cookbook/examples/automated_Crab/dataset_{smf_name}/spi/'
    dets = np.arange(19)

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Energy Bins
    with fits.open(f"{orig_data_path}/energy_boundaries.fits") as file:

        t = Table.read(file[1])
        energy_bins = np.append(t["E_MIN"], t["E_MAX"][-1])
        print(f'Number of energy bins: {len(energy_bins)-1}')
        t.write(f"{data_path}energy_boundaries.fits", overwrite=True)
    e_bounds_width = np.array(energy_bins[1:]) - np.array(energy_bins[:-1])
    bins_mid = (energy_bins[1:] + energy_bins[:-1])/2
    # Pointings and Start Times
    with fits.open(f"{orig_data_path}/pointing.fits") as file:
        t = Table.read(file[1])
        
        pointings = np.array(t["PTID_SPI"])
        
        time_start = np.array(t["TSTART"]) + 2451544.5
        time_start = [at.Time(f"{i}", format="jd").datetime for i in time_start]
        time_start = np.array([datetime.strftime(i,'%y%m%d %H%M%S') for i in time_start])
        t.write(f"{data_path}pointing.fits", overwrite=True)

    with fits.open(f"{orig_data_path}/dead_time.fits") as file:
    
        t = Table.read(file[1])
        time_elapsed = np.array(t["LIVETIME"])
        t.write(f"{data_path}dead_time.fits", overwrite=True)

    # read in the conv sky output
    with fits.open(f"{smf_path}convolved_sky_images.fits") as hdul:
        t2 = Table(hdul[2].data)
        header = hdul[2].header
        t22 = Table(hdul[1].data)
    conv_counts = np.array(t2["COUNTS"])

    # generate the source counts
    source_counts = np.zeros((len(pointings)*85, len(energy_bins)-1), dtype=np.uint32) # not sure if int is the best...

    # Im not at all sure how to use the pl for weighting. I'll try it like that, but K loses its meaning.
    if use_time_elapsed:
        for i, p_i in enumerate(pointings):
            for j, det in enumerate(dets):
                source_counts[i*85 + j] = np.random.poisson(powerlaw(bins_mid, K=K, index=index, piv=100) * time_elapsed[i*85 + j] * conv_counts[i*19 + j])

    else:
        if use_weighting:
            for i, p_i in enumerate(pointings):
                for j, det in enumerate(dets):
                    source_counts[i*85 + j] = np.random.poisson(powerlaw(bins_mid, K=K, index=index, piv=100) * conv_counts[i*19 + j])
        else:
            print("Warning: Not using time elapsed or weighting.")
            for i, p_i in enumerate(pointings):
                for j, det in enumerate(dets):
                    source_counts[i*85 + j] = np.random.poisson(conv_counts[i*19 + j])

    # save the data
    with fits.open(f"{orig_data_path}evts_det_spec_orig.fits") as file:
        t = Table.read(file[1])
        bkg_counts = np.array(t["COUNTS"])

        assert bkg_counts.shape == source_counts.shape, f"bkg_counts.shape = {bkg_counts.shape}, source_counts.shape = {source_counts.shape}. They should be equal."
        new_counts = bkg_counts + source_counts

        t["COUNTS"] = new_counts
        t.write(f"{data_path}evts_det_spec.fits", overwrite=True)


test_config_1 = {
    "smf_name": "normal_new",
    "data_path": "/home/tguethle/Documents/spi/Master_Thesis/main_files/smf_simulations/test_data_normal_new/",
    # leave default for orig_data_path
}

test_config_2 = {
    "smf_name": "normal_new",
    "K": 7e-2,
    "data_path": "/home/tguethle/Documents/spi/Master_Thesis/main_files/smf_simulations/test_data_normal_new_big_K/",
}

test_config_3 = {
    "smf_name": "normal_new",
    "K": 7e-5,
    "data_path": "/home/tguethle/Documents/spi/Master_Thesis/main_files/smf_simulations/test_data_normal_new_small_K/",
}

config_K_1 = {
    "smf_name": "normal_new",
    "data_path": "/home/tguethle/Documents/spi/Master_Thesis/main_files/smf_simulations/K_1/",
    "K": 1,
    "use_time_elapsed": False,
}

config_K_2 = {
    "smf_name": "normal_new",
    "data_path": "/home/tguethle/Documents/spi/Master_Thesis/main_files/smf_simulations/K_2/",
    "K": 2,
    "use_time_elapsed": False,
}

config_K_01 = {
    "smf_name": "normal_new",
    "data_path": "/home/tguethle/Documents/spi/Master_Thesis/main_files/smf_simulations/K_01/",
    "K": 0.1,
    "use_time_elapsed": False,
}


debugging_config_2 = {
    "smf_name": "normal_new",
    "data_path": "/home/tguethle/Documents/spi/Master_Thesis/main_files/smf_simulations/debugging_2/",
    "K": 1,
    "use_time_elapsed": False,
}

test_pl_config = {
    # can not be used like this anymore because the convolved sky output was overwritten. If you want to use it, rerun spimodfit with changed source model parameters.
    "smf_name": "374_center_test",
    "data_path": "/home/tguethle/Documents/spi/Master_Thesis/main_files/smf_simulations/test_pl/",
    "K": 0.1,
    "use_time_elapsed": False,
    "use_weighting": False
}

test_pl_config_K_10 = {
    "smf_name": "374_center_test",
    "data_path": "/home/tguethle/Documents/spi/Master_Thesis/main_files/smf_simulations/test_pl_K_10/",
    "K": 10,
    "use_time_elapsed": False,
    "use_weighting": False
}

test_pl_config_K_1 = {
    "smf_name": "374_center_test_2",
    "data_path": "/home/tguethle/Documents/spi/Master_Thesis/main_files/smf_simulations/test_pl_K_1/",
    "K": 1,
    "use_time_elapsed": False,
    "use_weighting": False
}


if __name__ == "__main__":
    gen_smf_data(**test_pl_config_K_1)

    print("Data generated.")