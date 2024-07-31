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


def pyspi_real_bkg(
        data_path=None,
        orig_data_path='/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/0374',
        piv=100,
        scale_background=None,
        K=7e-4,
        **kwargs
        ):
    #destination_path = f"./main_files/spimodfit_comparison_sim_source/pyspi_real_bkg_para2/{revolution}"
    #print(f'creating data for {destination_path} with K={K}, piv={piv}, index={index_pl}, background scale={scale_background}')
    
    # defining the source
    ra, dec = 10, -40
    index_pl = -2
    piv = 100
    # ra, dec = 155., 75.
    # K, piv, index = 3e-3, 40, -1

    # Define  Spectrum
    pl = Powerlaw()
    pl.piv = piv
    pl.K = K
    pl.index = index_pl
    component1 = SpectralComponent("pl", shape=pl)
    source = PointSource("Test", ra=ra, dec=dec, components=[component1])

    #emod = np.geomspace(18, 2000, 200)
    emod = np.arange(20, 600.5, 0.5)
    spec = source(emod)
    spec_binned = powerlaw_binned_spectrum(emod, spec)
    print(f'creating data for {data_path} with K={K}, piv={piv}, index={index_pl}, background scale={scale_background}')
    

    if not os.path.exists(f"{data_path}"):
        os.makedirs(f"{data_path}")


    # read in Energy Bins
    with fits.open(f"{orig_data_path}/energy_boundaries.fits") as file:

        t = Table.read(file[1])
        energy_bins = np.append(t["E_MIN"], t["E_MAX"][-1])
        print(f'Number of energy bins: {len(energy_bins)-1}')
    # read in Pointings and Times for the Response generation
    with fits.open(f"{orig_data_path}/pointing.fits") as file:
        t = Table.read(file[1])
        
        pointings = np.array(t["PTID_SPI"])
        
        time_start = np.array(t["TSTART"]) + 2451544.5
        time_start = [at.Time(f"{i}", format="jd").datetime for i in time_start]
        time_start = np.array([datetime.strftime(i,'%y%m%d %H%M%S') for i in time_start])
    
    
    # Time Elapsed
    # det=i, pointing_index=j : index = j*85 + i
    with fits.open(f"{orig_data_path}/dead_time.fits") as file:
        
        t = Table.read(file[1])
        time_elapsed = np.array(t["LIVETIME"])

    # Generate Source Counts

    assert find_response_version(time_start[0]) == find_response_version(time_start[-1]), "Versions not constant"
    version = find_response_version(time_start[0])
    rsp_base = ResponseDataRMF.from_version(version)

    source_counts = np.zeros((len(pointings)*85, len(energy_bins)-1), dtype=np.uint32)

    for p_i, pointing in enumerate(pointings):
        
        time = time_start[p_i]
        dets = get_live_dets(time=time, event_types=["single"])
        
        rmfs = []
        for d in dets:
            rmfs.append(ResponseRMFGenerator.from_time(time, d, energy_bins, emod, rsp_base))
            
        sds = np.empty(0)
        for d in range(len(dets)):
            sd = SPIDRM(rmfs[d], ra, dec)
            sds = np.append(sds, sd.matrix.T)
        resp_mat = sds.reshape((len(dets), len(emod)-1, len(energy_bins)-1))
        
        count_rates = np.dot(spec_binned, resp_mat)
        
        for d_i, d in enumerate(dets):
            index = p_i * 85 + d
            source_counts[index,:] = np.random.poisson(count_rates[d_i,:] * time_elapsed[index])

    # Save Data for PySpi

    with fits.open(f"{orig_data_path}/evts_det_spec_orig.fits") as file:
        t = Table.read(file[1])
        
        counts = t
        
    updated_counts = counts.copy()
    if scale_background is not None:
        updated_counts["COUNTS"] = updated_counts['COUNTS'] // int(1/scale_background)
    updated_counts["COUNTS"] += source_counts

    hdu = fits.BinTableHDU(data=updated_counts, name="SPI.-OBS.-DSP")
    hdu.writeto(f"{data_path}/evts_det_spec.fits")
    
    os.popen(f"cp {orig_data_path}/energy_boundaries.fits {data_path}/energy_boundaries.fits")
    os.popen(f"cp {orig_data_path}/pointing.fits {data_path}/pointing.fits")
    os.popen(f"cp {orig_data_path}/dead_time.fits {data_path}/dead_time.fits")

    with open(f"{data_path}/data_summary.txt", "w") as f:
        f.write(f"K: {K}\n")
        f.write(f"piv: {piv}\n")
        f.write(f"index: {index_pl}\n")
        f.write(f"Scale Background: {scale_background}\n")
        f.write(f"Original Data Path: {orig_data_path}\n")
        f.write(f"Data Path: {data_path}\n")
        f.write(f"compleated at {datetime.now()}")
        
    print(f"Data saved to {data_path}")

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
    val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
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

def gen_and_fit(config_args: dict):
    pyspi_real_bkg(**config_args)
    pyspi_fit_0374_pre_ppc(**config_args)


# defines configs
old_config = [
    {
        'name': 'bright_100',
        "data_path": '/home/tguethle/Documents/spi/Master_Thesis/main_files/sim_source_real_bkg/bright_100/0374',
        "orig_data_path": '/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/0374',
        'rev': [374],
        "piv": 100,
        "scale_background": None,
        "K": 7e-2,
        "fit_path_extension": "pre_ppc",
        "new_pointings": True,
        "Energy_range": (20, 600)
    },
    {
        'name': 'bright_10',
        "data_path": '/home/tguethle/Documents/spi/Master_Thesis/main_files/sim_source_real_bkg/bright_10/0374',
        "orig_data_path": '/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/0374',
        'rev': [374],
        "piv": 100,
        "scale_background": None,
        "K": 7e-3,
        "fit_path_extension": "pre_ppc",
        "new_pointings": True,
        "Energy_range": (20, 600)
    },
    {
        'name': 'bright_100_reduced_bkg_10',
        "data_path": '/home/tguethle/Documents/spi/Master_Thesis/main_files/sim_source_real_bkg/bright_100_reduced_bkg_10/0374',
        "orig_data_path": '/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/0374',
        'rev': [374],
        "piv": 100,
        "scale_background": 0.1,
        "K": 7e-2,
        "fit_path_extension": "pre_ppc",
        "new_pointings": True,
        "Energy_range": (20, 600)
    },
    {
        'name': 'bright_10_reduced_bkg_10',
        "data_path": '/home/tguethle/Documents/spi/Master_Thesis/main_files/sim_source_real_bkg/bright_10_reduced_bkg_10/0374',
        "orig_data_path": '/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/0374',
        'rev': [374],
        "piv": 100,
        "scale_background": 0.1,
        "K": 7e-3,
        "fit_path_extension": "pre_ppc",
        "new_pointings": True,
        "Energy_range": (20, 600)
    },
]

config = [
    {
        'name': 'bright_100_new',
        "data_path": '/home/tguethle/Documents/spi/Master_Thesis/main_files/sim_source_real_bkg/bright_100_new/0374',
        "orig_data_path": '/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/0374_center',
        'rev': [374],
        "piv": 100,
        "scale_background": None,
        "K": 7e-2,
        "fit_path_extension": "pre_ppc",
        "new_pointings": True,
        "Energy_range": (20, 600),
        'center': [-48, -76]
    },
    {
        'name': 'bright_10_new',
        "data_path": '/home/tguethle/Documents/spi/Master_Thesis/main_files/sim_source_real_bkg/bright_10_new/0374',
        "orig_data_path": '/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/0374_center',
        'rev': [374],
        "piv": 100,
        "scale_background": None,
        "K": 7e-3,
        "fit_path_extension": "pre_ppc",
        "new_pointings": True,
        "Energy_range": (20, 600),
        'center': [-48, -76]
    },
    {
        'name': 'bright_100_reduced_bkg_10_new',
        "data_path": '/home/tguethle/Documents/spi/Master_Thesis/main_files/sim_source_real_bkg/bright_100_reduced_bkg_10_new/0374',
        "orig_data_path": '/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/0374_center',
        'rev': [374],
        "piv": 100,
        "scale_background": 0.1,
        "K": 7e-2,
        "fit_path_extension": "pre_ppc",
        "new_pointings": True,
        "Energy_range": (20, 600),
        'center': [-48, -76]
    },
    {
        'name': 'bright_10_reduced_bkg_10_new',
        "data_path": '/home/tguethle/Documents/spi/Master_Thesis/main_files/sim_source_real_bkg/bright_10_reduced_bkg_10_new/0374',
        "orig_data_path": '/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/0374_center',
        'rev': [374],
        "piv": 100,
        "scale_background": 0.1,
        "K": 7e-3,
        "fit_path_extension": "pre_ppc",
        "new_pointings": True,
        "Energy_range": (20, 600),
        'center': [-48, -76]
    },
    {
        'name': 'normal_new',
        "data_path": '/home/tguethle/Documents/spi/Master_Thesis/main_files/sim_source_real_bkg/normal_new/0374',
        "orig_data_path": '/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/0374_center',
        'rev': [374],
        "piv": 100,
        "scale_background": 0.1,
        "K": 7e-4,
        "fit_path_extension": "pre_ppc",
        "new_pointings": True,
        "Energy_range": (20, 600),
        'center': [-48, -76]
    },
    {
        'name': 'normal_reduced_bkg_10_new',
        "data_path": '/home/tguethle/Documents/spi/Master_Thesis/main_files/sim_source_real_bkg/normal_reduced_bkg_10_new/0374',
        "orig_data_path": '/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/0374_center',
        'rev': [374],
        "piv": 100,
        "scale_background": 0.1,
        "K": 7e-4,
        "fit_path_extension": "pre_ppc",
        "new_pointings": True,
        "Energy_range": (20, 600),
        'center': [-48, -76]
    },

]

config_small_bins = [
    {
        'name': 'bright_100_small_bins',
        "data_path": '/home/tguethle/Documents/spi/Master_Thesis/main_files/sim_source_real_bkg/bright_100_small_bins/0374',
        "orig_data_path": '/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/0374_center_small_bins',
        'rev': [374],
        "piv": 100,
        "scale_background": None,
        "K": 7e-2,
        "fit_path_extension": "pre_ppc",
        "new_pointings": True,
        "Energy_range": (20, 600),
        'center': [-48, -76]
    },
    {
        'name': 'bright_10_small_bins',
        "data_path": '/home/tguethle/Documents/spi/Master_Thesis/main_files/sim_source_real_bkg/bright_10_small_bins/0374',
        "orig_data_path": '/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/0374_center_small_bins',
        'rev': [374],
        "piv": 100,
        "scale_background": None,
        "K": 7e-3,
        "fit_path_extension": "pre_ppc",
        "new_pointings": True,
        "Energy_range": (20, 600),
        'center': [-48, -76]
    },
    {
        'name': 'bright_100_reduced_bkg_10_small_bins',
        "data_path": '/home/tguethle/Documents/spi/Master_Thesis/main_files/sim_source_real_bkg/bright_100_reduced_bkg_10_small_bins/0374',
        "orig_data_path": '/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/0374_center_small_bins',
        'rev': [374],
        "piv": 100,
        "scale_background": 0.1,
        "K": 7e-2,
        "fit_path_extension": "pre_ppc",
        "new_pointings": True,
        "Energy_range": (20, 600),
        'center': [-48, -76]
    },
    {
        'name': 'bright_10_reduced_bkg_10_small_bins',
        "data_path": '/home/tguethle/Documents/spi/Master_Thesis/main_files/sim_source_real_bkg/bright_10_reduced_bkg_10_small_bins/0374',
        "orig_data_path": '/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/0374_center_small_bins',
        'rev': [374],
        "piv": 100,
        "scale_background": 0.1,
        "K": 7e-3,
        "fit_path_extension": "pre_ppc",
        "new_pointings": True,
        "Energy_range": (20, 600),
        'center': [-48, -76]
    },
    {
        'name': 'normal_small_bins',
        "data_path": '/home/tguethle/Documents/spi/Master_Thesis/main_files/sim_source_real_bkg/normal_small_bins/0374',
        "orig_data_path": '/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/0374_center_small_bins',
        'rev': [374],
        "piv": 100,
        "scale_background": 0.1,
        "K": 7e-4,
        "fit_path_extension": "pre_ppc",
        "new_pointings": True,
        "Energy_range": (20, 600),
        'center': [-48, -76]
    },
    {
        'name': 'normal_reduced_bkg_10_small_bins',
        "data_path": '/home/tguethle/Documents/spi/Master_Thesis/main_files/sim_source_real_bkg/normal_reduced_bkg_10_small_bins/0374',
        "orig_data_path": '/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/0374_center_small_bins',
        'rev': [374],
        "piv": 100,
        "scale_background": 0.1,
        "K": 7e-4,
        "fit_path_extension": "pre_ppc",
        "new_pointings": True,
        "Energy_range": (20, 600),
        'center': [-48, -76]
    },

]

config_small_bins_2 = [
    {
        'name': 'bright_100_small_bins_2',
        "data_path": '/home/tguethle/Documents/spi/Master_Thesis/main_files/sim_source_real_bkg/bright_100_small_bins_2/0374',
        "orig_data_path": '/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/0374_center_small_bins',
        'rev': [374],
        "piv": 100,
        "scale_background": None,
        "K": 7e-2,
        "fit_path_extension": "pre_ppc",
        "new_pointings": True,
        "Energy_range": (20, 600),
        'center': [-48, -76]
    },
    {
        'name': 'bright_10_small_bins_2',
        "data_path": '/home/tguethle/Documents/spi/Master_Thesis/main_files/sim_source_real_bkg/bright_10_small_bins_2/0374',
        "orig_data_path": '/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/0374_center_small_bins',
        'rev': [374],
        "piv": 100,
        "scale_background": None,
        "K": 7e-3,
        "fit_path_extension": "pre_ppc",
        "new_pointings": True,
        "Energy_range": (20, 600),
        'center': [-48, -76]
    },
    {
        'name': 'bright_100_reduced_bkg_10_small_bins_2',
        "data_path": '/home/tguethle/Documents/spi/Master_Thesis/main_files/sim_source_real_bkg/bright_100_reduced_bkg_10_small_bins_2/0374',
        "orig_data_path": '/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/0374_center_small_bins',
        'rev': [374],
        "piv": 100,
        "scale_background": 0.1,
        "K": 7e-2,
        "fit_path_extension": "pre_ppc",
        "new_pointings": True,
        "Energy_range": (20, 600),
        'center': [-48, -76]
    },
    {
        'name': 'bright_10_reduced_bkg_10_small_bins_2',
        "data_path": '/home/tguethle/Documents/spi/Master_Thesis/main_files/sim_source_real_bkg/bright_10_reduced_bkg_10_small_bins_2/0374',
        "orig_data_path": '/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/0374_center_small_bins',
        'rev': [374],
        "piv": 100,
        "scale_background": 0.1,
        "K": 7e-3,
        "fit_path_extension": "pre_ppc",
        "new_pointings": True,
        "Energy_range": (20, 600),
        'center': [-48, -76]
    },
    {
        'name': 'normal_small_bins_2',
        "data_path": '/home/tguethle/Documents/spi/Master_Thesis/main_files/sim_source_real_bkg/normal_small_bins_2/0374',
        "orig_data_path": '/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/0374_center_small_bins',
        'rev': [374],
        "piv": 100,
        "scale_background": 0.1,
        "K": 7e-4,
        "fit_path_extension": "pre_ppc",
        "new_pointings": True,
        "Energy_range": (20, 600),
        'center': [-48, -76]
    },
    {
        'name': 'normal_reduced_bkg_10_small_bins_2',
        "data_path": '/home/tguethle/Documents/spi/Master_Thesis/main_files/sim_source_real_bkg/normal_reduced_bkg_10_small_bins_2/0374',
        "orig_data_path": '/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/0374_center_small_bins',
        'rev': [374],
        "piv": 100,
        "scale_background": 0.1,
        "K": 7e-4,
        "fit_path_extension": "pre_ppc",
        "new_pointings": True,
        "Energy_range": (20, 600),
        'center': [-48, -76]
    },

]


if __name__ == "__main__":
    for i,c in enumerate(config_small_bins_2):
        #gen_and_fit(c) #fit takes to long with 1000+ bins so only generate data for now
        if i == 4:
            pyspi_fit_0374_pre_ppc(**c)