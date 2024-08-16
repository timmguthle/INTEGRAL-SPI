import sys, os
sys.path.insert(0, os.path.abspath('./main_files'))

import numpy as np
from MultinestClusterFit import MultinestClusterFit
from RebinningFunctions import spimodfit_binning_SE, log_binning_function_for_x_number_of_bins, no_rebinning #, rebin_data_exp_50
from PointingClusters import *
from ModelSources import *
import pickle
from typing import Union
from chainconsumer import ChainConsumer


def crab_band_fit_wide_energy(
        data_path: Union[str, None] = None,
        fit_path: Union[str, None] = None,
        energy_range: tuple[int, int]= (20,1000),
        new_pointing_clustering: bool = True,
        binning_func = no_rebinning,
        just_crab: bool = False,
        crab_model = crab_band,
        parameters = None,
        p = ["Crab K", "Crab alpha", "Crab beta", "A 0535 262 K", "A 0535 262 index"],
        **kwargs,
):
    """
    crab with band function. break energy fixed at 500keV. Beta is fixed at -2.25
    Pulsar as pl included. 

    with this function I will only fit K and alpha(first index)
    parameters:
        data_path: path to the data file (path to .fits files)
        fit_path: path to the output file (generated if needed)

    """
    if just_crab:
        source_model = define_sources((
            (crab_model, (100,)),
        ))

    else:
        source_model = define_sources((
                (crab_model, (100,)),
                (s_1A_0535_262_pl, (100,)),
        ))

    assert data_path is not None, "data_path must be given"
    assert fit_path is not None, "fit_path must be given"

    if not os.path.exists(fit_path):
        os.makedirs(fit_path)

    if new_pointing_clustering:
        # maybe play around with these parameters, but they should be fine
        print("generateing pointing cluster...")
        Cluster = PointingClusters(
                (data_path,),
                min_angle_dif=1.5,
                max_angle_dif=10.,
                max_time_dif=0.2,
                radius_around_source=10.,
                min_time_elapsed=300.,
                cluster_size_range=(2,2),
            )
        pointings = Cluster.pointings
        save_clusters(pointings, data_path)
    else:
        pointings = load_clusters(data_path)

    fit = MultinestClusterFit(
        pointings,
        source_model,
        energy_range=energy_range,
        emod=np.geomspace(energy_range[0], energy_range[1], 500),
        binning_func=binning_func,
        folder=fit_path,
        parameter_names=parameters,
    )
    
    fit.parameter_fit_distribution()
    fit.text_summaries(reference_values=False)

    # chainconsumer
    
    val = np.array([i[1] for i in fit._cc.analysis.get_summary(parameters=p).values()])
    cov = fit._cc.analysis.get_covariance(parameters=p)[1]
    
    np.savetxt(f"{fit_path}/fit_val.txt", val, header=" ".join(p))
    np.savetxt(f"{fit_path}/fit_cov.txt", cov, header="cov matrix")

    with open(f"{fit_path}/pyspi_summary.txt", "w") as f:
        f.write(f"Energy range: {energy_range}\n")
        f.write(f"Data path: {data_path}\n")
        f.write(f"Fit path: {fit_path}\n")
        f.write(f"Result: {val}\n")
        f.write(f"Covariance: {cov}\n")
        f.write(f"compleated at {datetime.now()}")


def crab_band_fit(
        data_path: Union[str, None] = None,
        fit_path: Union[str, None] = None,
        energy_range: tuple[int, int]= (20,600),
        new_pointing_clustering: bool = True,
        binning_func = no_rebinning,
        **kwargs,
):
    """
    crab with band function. break energy fixed at 500keV. Beta is fixed at -2.25
    Pulsar as pl included. 

    with this function I will only fit K and alpha(first index)
    parameters:
        data_path: path to the data file (path to .fits files)
        fit_path: path to the output file (generated if needed)

    """

    source_model = define_sources((
            (crab_lower_band, (100,)),
            (s_1A_0535_262_pl, (100,)),
    ))

    assert data_path is not None, "data_path must be given"
    assert fit_path is not None, "fit_path must be given"

    if not os.path.exists(fit_path):
        os.makedirs(fit_path)

    if new_pointing_clustering:
        # maybe play around with these parameters, but they should be fine
        Cluster = PointingClusters(
                (data_path,),
                min_angle_dif=1.5,
                max_angle_dif=10.,
                max_time_dif=0.2,
                radius_around_source=10.,
                min_time_elapsed=300.,
                cluster_size_range=(2,2),
            )
        pointings = Cluster.pointings
        save_clusters(pointings, data_path)
    else:
        pointings = load_clusters(data_path)

    fit = MultinestClusterFit(
        pointings,
        source_model,
        energy_range=energy_range,
        emod=np.geomspace(energy_range[0], energy_range[1], 200),
        binning_func=binning_func,
        folder=fit_path,
        **kwargs,
    )
    
    fit.parameter_fit_distribution()
    fit.text_summaries(reference_values=False)

    # chainconsumer
    p = ["Crab K", "Crab alpha", "A 0535 262 K", "A 0535 262 index"]
    val = np.array([i[1] for i in fit._cc.analysis.get_summary(parameters=p).values()])
    cov = fit._cc.analysis.get_covariance(parameters=p)[1]
    
    np.savetxt(f"{fit_path}/fit_val.txt", val, header=" ".join(p))
    np.savetxt(f"{fit_path}/fit_cov.txt", cov, header="cov matrix")

    with open(f"{fit_path}/pyspi_summary.txt", "w") as f:
        f.write(f"Energy range: {energy_range}\n")
        f.write(f"Data path: {data_path}\n")
        f.write(f"Fit path: {fit_path}\n")
        f.write(f"Result: {val}\n")
        f.write(f"Covariance: {cov}\n")
        f.write(f"compleated at {datetime.now()}")

config_2003 = [
    {
        "data_path": "./main_files/crab_19/data_2003",
        "fit_path": "./main_files/crab_19/fit_2003/crab_band_fit",
        "energy_range": (20, 600),
    }, 
    {
        "data_path": "./main_files/crab_19/data_2003",
        "fit_path": "./main_files/crab_19/fit_2003/crab_band_fit_30_400",
        'new_pointing_clustering': False,
        "energy_range": (30, 400),
    }, 
    {
        "data_path": "./main_files/crab_19/data_2003",
        "fit_path": "./main_files/crab_19/fit_2003/crab_band_fit_35_81",
        'new_pointing_clustering': False,
        "energy_range": (35, 81),
    }, 
    {
        "data_path": "./main_files/crab_19/data_2003",
        "fit_path": "./main_files/crab_19/fit_2003/crab_band_fit_bins_70",
        "energy_range": (20, 600),
        "binning_func": log_binning_function_for_x_number_of_bins(70),
        "new_pointing_clustering": False,
    }, 
    {
        "data_path": "./main_files/crab_19/data_2003",
        "fit_path": "./main_files/crab_19/fit_2003/crab_band_fit_30_100_bins_70",
        'new_pointing_clustering': False,
        "energy_range": (30, 100),
        "binning_func": log_binning_function_for_x_number_of_bins(70),
    }, 
    {
        "data_path": "./main_files/crab_19/data_2003",
        "fit_path": "./main_files/crab_19/fit_2003/crab_band_fit_35_81_bins_70",
        'new_pointing_clustering': False,
        "energy_range": (35, 81),
        "binning_func": log_binning_function_for_x_number_of_bins(70),
    }, 
]

config_2017 = [
    {
        "data_path": "./main_files/crab_19/data_2017",
        "fit_path": "./main_files/crab_19/fit_2017/crab_band_fit",
        "energy_range": (20, 600),
    }, 
    {
        "data_path": "./main_files/crab_19/data_2017",
        "fit_path": "./main_files/crab_19/fit_2017/crab_band_fit_30_400",
        'new_pointing_clustering': False,
        "energy_range": (30, 400),
    }, 
    {
        "data_path": "./main_files/crab_19/data_2017",
        "fit_path": "./main_files/crab_19/fit_2017/crab_band_fit_35_81",
        'new_pointing_clustering': False,
        "energy_range": (35, 81),
    }, 
    {
        "data_path": "./main_files/crab_19/data_2017",
        "fit_path": "./main_files/crab_19/fit_2017/crab_band_fit_bins_70",
        "energy_range": (20, 600),
        "binning_func": log_binning_function_for_x_number_of_bins(70),
        "new_pointing_clustering": False,
    }, 
    {
        "data_path": "./main_files/crab_19/data_2017",
        "fit_path": "./main_files/crab_19/fit_2017/crab_band_fit_30_400_bins_70",
        'new_pointing_clustering': False,
        "energy_range": (30, 400),
        "binning_func": log_binning_function_for_x_number_of_bins(70),
    }, 
    {
        "data_path": "./main_files/crab_19/data_2017",
        "fit_path": "./main_files/crab_19/fit_2017/crab_band_fit_35_81_bins_70",
        'new_pointing_clustering': False,
        "energy_range": (35, 81),
        "binning_func": log_binning_function_for_x_number_of_bins(70),
    }, 
]

config_new = [
    {
        "data_path": "./main_files/crab_19/data_2003",
        "fit_path": "./main_files/crab_19/fit_2003/crab_band_fit_20_400",
        'new_pointing_clustering': False,
        "energy_range": (20, 400),
    },
    {
        "data_path": "./main_files/crab_19/data_2017",
        "fit_path": "./main_files/crab_19/fit_2017/crab_band_fit_20_400",
        'new_pointing_clustering': False,
        "energy_range": (20, 400),
    },
    {
        "data_path": "./main_files/crab_19/data_2003",
        "fit_path": "./main_files/crab_19/fit_2003/crab_band_fit_20_500",
        'new_pointing_clustering': False,
        "energy_range": (20, 500),
    },
    {
        "data_path": "./main_files/crab_19/data_2017",
        "fit_path": "./main_files/crab_19/fit_2017/crab_band_fit_20_500",
        'new_pointing_clustering': False,
        "energy_range": (20, 500),
    },
]

config_combined = [
    {
        "data_path": "./main_files/crab_19/data_2003_combined",
        "fit_path": "./main_files/crab_19/fit_2003_combined/crab_band_fit",
        'new_pointing_clustering': True,
        "energy_range": (20,600),
    }, 
    {
        "data_path": "./main_files/crab_19/data_2003_combined",
        "fit_path": "./main_files/crab_19/fit_2003_combined/crab_band_fit_20_400",
        'new_pointing_clustering': False,
        "energy_range": (20,400),
    }, 
    {
        "data_path": "./main_files/crab_19/data_2003_combined",
        "fit_path": "./main_files/crab_19/fit_2003_combined/crab_band_fit_35_600",
        'new_pointing_clustering': False,
        "energy_range": (35,600),
    }, 
    # 2017 data
    {
        "data_path": "./main_files/crab_19/data_2017_combined",
        "fit_path": "./main_files/crab_19/fit_2017_combined/crab_band_fit",
        'new_pointing_clustering': True,
        "energy_range": (20,600),
    }, 
    {
        "data_path": "./main_files/crab_19/data_2017_combined",
        "fit_path": "./main_files/crab_19/fit_2017_combined/crab_band_fit_20_400",
        'new_pointing_clustering': False,
        "energy_range": (20,400),
    }, 
    {
        "data_path": "./main_files/crab_19/data_2017_combined",
        "fit_path": "./main_files/crab_19/fit_2017_combined/crab_band_fit_35_600",
        'new_pointing_clustering': False,
        "energy_range": (35,600),
    }, 

]

config_combined_2 = [
    # data 2 comb is the combi with the eff from the paper 
    {
        "data_path": "./main_files/crab_19/data_2_2003_comb",
        "fit_path": "./main_files/crab_19/fit_2_2003_combined/crab_band_fit_200_1000",
        'new_pointing_clustering': False,
        "energy_range": (200,1000),
    }, 
    {
        "data_path": "./main_files/crab_19/data_2_2017_comb",
        "fit_path": "./main_files/crab_19/fit_2_2017_combined/crab_band_fit_200_1000",
        'new_pointing_clustering': False,
        "energy_range": (200,1000),
    }, 
    {
        "data_path": "./main_files/crab_19/data_2_2003_comb",
        "fit_path": "./main_files/crab_19/fit_2_2003_combined/crab_band_fit_20_1000",
        'new_pointing_clustering': False,
        "energy_range": (20,1000),
    }, 
    {
        "data_path": "./main_files/crab_19/data_2_2003_comb",
        "fit_path": "./main_files/crab_19/fit_2_2003_combined/crab_band_fit_100_1000",
        'new_pointing_clustering': False,
        "energy_range": (100,1000),
    }, 
    {
        "data_path": "./main_files/crab_19/data_2_2003_comb",
        "fit_path": "./main_files/crab_19/fit_2_2003_combined/crab_band_fit_40_1000",
        'new_pointing_clustering': False,
        "energy_range": (40,1000),
    }, 
    # 2017 data
    {
        "data_path": "./main_files/crab_19/data_2_2017_comb",
        "fit_path": "./main_files/crab_19/fit_2_2017_combined/crab_band_fit_20_1000",
        'new_pointing_clustering': False,
        "energy_range": (20,1000),
    }, 
    {
        "data_path": "./main_files/crab_19/data_2_2017_comb",
        "fit_path": "./main_files/crab_19/fit_2_2017_combined/crab_band_fit_100_1000",
        'new_pointing_clustering': False,
        "energy_range": (100,1000),
    }, 
    {
        "data_path": "./main_files/crab_19/data_2_2017_comb",
        "fit_path": "./main_files/crab_19/fit_2_2017_combined/crab_band_fit_27_1000",
        'new_pointing_clustering': False,
        "energy_range": (27,1000),
    }, 
]

config_beuermann_with_data_2 = [
    {
        "data_path": "./main_files/crab_19/data_2_2003_comb",
        "fit_path": "./main_files/crab_19/fit_2_2003_combined/crab_beuermann_fit_20_1000",
        'new_pointing_clustering': False,
        "energy_range": (20,1000),
        "crab_model": crab_beuermann,
        "just_crab": True,
        "p": ["Crab K", "Crab alpha", "Crab beta", "Crab n", "Crab E1", "Crab E2"]
    },
    {
        "data_path": "./main_files/crab_19/data_2_2017_comb",
        "fit_path": "./main_files/crab_19/fit_2_2017_combined/crab_beuermann_fit_20_1000",
        'new_pointing_clustering': False,
        "energy_range": (20,1000),
        "crab_model": crab_beuermann,
        "just_crab": True,
        "p": ["Crab K", "Crab alpha", "Crab beta", "Crab n", "Crab E1", "Crab E2"]
    },
]

config_band_free_Ec_data_2 = [
    {
        "data_path": "./main_files/crab_19/data_2_2003_comb",
        "fit_path": "./main_files/crab_19/fit_2_2003_combined/crab_band_free_E_c_fit_20_1000",
        'new_pointing_clustering': False,
        "energy_range": (20,1000),
        "crab_model": crab_band_free_E_c,
        "p": ["Crab K", "Crab alpha","Crab xp", "Crab beta", "A 0535 262 K", "A 0535 262 index"]
    },
    {
        "data_path": "./main_files/crab_19/data_2_2017_comb",
        "fit_path": "./main_files/crab_19/fit_2_2017_combined/crab_band_free_E_c_fit_20_1000",
        'new_pointing_clustering': False,
        "energy_range": (20,1000),
        "crab_model": crab_band_free_E_c,
        "p": ["Crab K", "Crab alpha","Crab xp", "Crab beta", "A 0535 262 K", "A 0535 262 index"]
    },
]

config_2_variable_E_c = [
    {
        "data_path": "./main_files/crab_19/data_2_2003_comb",
        "fit_path": "./main_files/crab_19/fit_2_2003_combined/crab_band_fit_20_1000_fit_E_c",
        'new_pointing_clustering': False,
        "energy_range": (20,1000),
        "binning_func": log_binning_function_for_x_number_of_bins(70),
        "crab_model": crab_band_free_E_c,
    }, 
    {
        "data_path": "./main_files/crab_19/data_2_2003_comb",
        "fit_path": "./main_files/crab_19/fit_2_2003_combined/crab_band_fit_20_700_fit_E_c",
        'new_pointing_clustering': False,
        "energy_range": (20,700),
        "binning_func": log_binning_function_for_x_number_of_bins(70),
        "crab_model": crab_band_free_E_c,
    }, 
    {
        "data_path": "./main_files/crab_19/data_2_2003_comb",
        "fit_path": "./main_files/crab_19/fit_2_2003_combined/crab_band_fit_25_1000_fit_E_c",
        'new_pointing_clustering': False,
        "energy_range": (25,1000),
        "binning_func": log_binning_function_for_x_number_of_bins(70),
        "crab_model": crab_band_free_E_c,
    }, 
    {
        "data_path": "./main_files/crab_19/data_2_2017_comb",
        "fit_path": "./main_files/crab_19/fit_2_2017_combined/crab_band_fit_20_1000_fit_E_c",
        'new_pointing_clustering': False,
        "energy_range": (20,1000),
        "binning_func": log_binning_function_for_x_number_of_bins(70),
        "crab_model": crab_band_free_E_c,
    }, 
    {
        "data_path": "./main_files/crab_19/data_2_2017_comb",
        "fit_path": "./main_files/crab_19/fit_2_2017_combined/crab_band_fit_20_700_fit_E_c",
        'new_pointing_clustering': False,
        "energy_range": (20,700),
        "binning_func": log_binning_function_for_x_number_of_bins(70),
        "crab_model": crab_band_free_E_c,
    }, 
    {
        "data_path": "./main_files/crab_19/data_2_2017_comb",
        "fit_path": "./main_files/crab_19/fit_2_2017_combined/crab_band_fit_25_1000_fit_E_c",
        'new_pointing_clustering': False,
        "energy_range": (25,1000),
        "binning_func": log_binning_function_for_x_number_of_bins(70),
        "crab_model": crab_band_free_E_c,
    }, 
]

config_combined_3 = [
    # here in data 3 sind die lower counts 
    {
        "data_path": "./main_files/crab_19/data_3_2003_comb",
        "fit_path": "./main_files/crab_19/fit_3_2003_combined/crab_band_fit",
        'new_pointing_clustering': True,
        "energy_range": (20,1000),
    }, 
    # {
    #     "data_path": "./main_files/crab_19/data_3_2003_comb",
    #     "fit_path": "./main_files/crab_19/fit_3_2003_combined/crab_band_fit_20_600",
    #     'new_pointing_clustering': False,
    #     "energy_range": (20,600),
    # }, 
    # {
    #     "data_path": "./main_files/crab_19/data_3_2003_comb",
    #     "fit_path": "./main_files/crab_19/fit_3_2003_combined/crab_band_fit_35_1000",
    #     'new_pointing_clustering': False,
    #     "energy_range": (35,1000),
    # }, 
    # 2017 data
    {
        "data_path": "./main_files/crab_19/data_3_2017_comb",
        "fit_path": "./main_files/crab_19/fit_3_2017_combined/crab_band_fit",
        'new_pointing_clustering': True,
        "energy_range": (20,1000),
    }, 
    # {
    #     "data_path": "./main_files/crab_19/data_3_2017_comb",
    #     "fit_path": "./main_files/crab_19/fit_3_2017_combined/crab_band_fit_20_600",
    #     'new_pointing_clustering': False,
    #     "energy_range": (20,600),
    # }, 
    # {
    #     "data_path": "./main_files/crab_19/data_3_2017_comb",
    #     "fit_path": "./main_files/crab_19/fit_3_2017_combined/crab_band_fit_35_1000",
    #     'new_pointing_clustering': False,
    #     "energy_range": (35,1000),
    # }, 

]

combined_just_crab = [
    {
        "data_path": "./main_files/crab_19/data_2_2003_comb",
        "fit_path": "./main_files/crab_19/fit_2_2003_combined/crab_band_fit_just_crab",
        'new_pointing_clustering': False,
        "energy_range": (20,1000),
        "just_crab": True,
    }, 
    {
        "data_path": "./main_files/crab_19/data_2_2017_comb",
        "fit_path": "./main_files/crab_19/fit_2_2017_combined/crab_band_fit_just_crab",
        'new_pointing_clustering': False,
        "energy_range": (20,1000),
        "just_crab": True,
    }, 
]

config_fit_psd_eff = [
    {
        "data_path": "./main_files/crab_19/data_3_2003_center", # make sure that the PE data is copied to the data folder
        "fit_path": "./main_files/crab_19/fit_2003_psd_eff/crab_band_",
        'new_pointing_clustering': False,
        "energy_range": (20,1000),
        "parameters": ["Crab K", "Crab alpha", "Crab beta", "A 0535 262 K", "A 0535 262 index", "PSD eff"],
    }, 
    {
        "data_path": "./main_files/crab_19/data_3_2003_center", # make sure that the PE data is copied to the data folder
        "fit_path": "./main_files/crab_19/fit_2003_psd_eff/crab_band_fit_20_600",
        'new_pointing_clustering': False,
        "energy_range": (20,600),
    },
    {
        "data_path": "./main_files/crab_19/data_3_2003_center", # make sure that the PE data is copied to the data folder
        "fit_path": "./main_files/crab_19/fit_2003_psd_eff/crab_band_fit_100_1000",
        'new_pointing_clustering': False,
        "energy_range": (100,1000),
    },
    # with 2017 data
    {
        "data_path": "./main_files/crab_19/data_3_2017_center", # make sure that the PE data is copied to the data folder
        "fit_path": "./main_files/crab_19/fit_2017_psd_eff/crab_band_",
        'new_pointing_clustering': True,
        "energy_range": (20,1000),
    }, 
    {
        "data_path": "./main_files/crab_19/data_3_2017_center", # make sure that the PE data is copied to the data folder
        "fit_path": "./main_files/crab_19/fit_2017_psd_eff/crab_band_fit_20_600",
        'new_pointing_clustering': False,
        "energy_range": (20,600),
    },
    {
        "data_path": "./main_files/crab_19/data_3_2017_center", # make sure that the PE data is copied to the data folder
        "fit_path": "./main_files/crab_19/fit_2017_psd_eff/crab_band_fit_100_1000",
        'new_pointing_clustering': False,
        "energy_range": (100,1000),
    },

]


broken_pl_low_energy = [
    {
        "data_path": "main_files/crab_19/data_2_2003_center",
        "fit_path": "main_files/crab_19/fit_2003_broken_pl/crab_fit_20_150",
        'new_pointing_clustering': True,
        "energy_range": (20,150),
        "just_crab": True,
        "crab_model": crab_broken_powerlaw,
        "p": ["Crab K", "Crab xb", "Crab alpha", "Crab beta"],
    }, 
    {
        "data_path": "main_files/crab_19/data_2_2017_center",
        "fit_path": "main_files/crab_19/fit_2017_broken_pl/crab_fit_20_150",
        'new_pointing_clustering': True,
        "energy_range": (20,150),
        "just_crab": True,
        "crab_model": crab_broken_powerlaw,
        "p": ["Crab K", "Crab xb", "Crab alpha", "Crab beta"],
    }, 
]

config_crab_pl_high_energy = [
    {
        "data_path": "./main_files/crab_19/data_HE_2003",
        "fit_path": "./main_files/crab_19/fit_2003_high_e/crab_fit_1_8",
        'new_pointing_clustering': True,
        "energy_range": (1000,8000),
        "just_crab": False,
        "crab_model": crab_pl_high_energy,
        "p": ["Crab K", "Crab index"],
    }, 
    {
        "data_path": "./main_files/crab_19/data_HE_2017",
        "fit_path": "./main_files/crab_19/fit_2017_high_e/crab_fit_1_8",
        'new_pointing_clustering': True,
        "energy_range": (1000,8000),
        "just_crab": False,
        "crab_model": crab_pl_high_energy,
        "p": ["Crab K", "Crab index"],
    }, 
]

if __name__ == "__main__":
    # for conf in config_2003[1:]:
    #     crab_band_fit(**conf)
    #     print(conf['fit_path'] + " done")
    # for conf in config_2017:
    #     crab_band_fit(**conf)
    #     print(conf['fit_path'] + " done")
    for conf in config_crab_pl_high_energy[1:]:
        crab_band_fit_wide_energy(**conf)
        print(conf['fit_path'] + " done")
    # for conf in config_band_free_Ec_data_2:
    #     crab_band_fit_wide_energy(**conf)
    #     print(conf['fit_path'] + " done")
    # for conf in broken_pl_low_energy:
    #     crab_band_fit_wide_energy(**conf)
    #     print(conf['fit_path'] + " done")
    # for conf in config_fit_psd_eff:
    #     crab_band_fit_wide_energy(**conf)
    #     print(conf['fit_path'] + " done")

    # parameter_names = config_fit_psd_eff[0]['parameters']
    # cc = ChainConsumer()
    # chain = np.loadtxt('./chains/1-post_equal_weights.dat')
    # cc.add_chain(chain, parameters=parameter_names, name='fit')

    # fig = cc.plotter.plot(
    #     parameters=parameter_names[:-1],
    #     figsize=1.5,
    # )
        
    # fig.savefig(f"parameter_fit_distributions.pdf")

    # summary = cc.analysis.get_summary(parameters=parameter_names)
    # print(summary)
