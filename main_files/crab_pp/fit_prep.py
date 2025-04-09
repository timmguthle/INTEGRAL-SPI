import sys, os
sys.path.insert(0, os.path.abspath('./main_files'))
sys.path.insert(0, os.path.abspath('./'))

from spimodfit.spimodfit_utils import SpimodfitWrapper, SpimselectDownloader, download_and_copy_to_pyspi

import numpy as np

from RebinningFunctions import spimodfit_binning_SE, log_binning_function_for_x_number_of_bins, no_rebinning #, rebin_data_exp_50
from PointingClusters import *
from ModelSources import *

import json

BASE_DATA_PATH = "/home/tguethle/data1/pyspi_data"
BASE_FIT_PATH = "./main_files/crab_pp"

CONFIG_DIRECTORY = "./main_files/crab_pp/config"

def extract_meta_data(data_path):
    with fits.open(f"{data_path}/pointing.fits") as file:
        t = Table.read(file[1])
        
        pointings = np.array(t["PTID_SPI"])
        
        time_start = np.array(t["TSTART"]) + 2451544.5
        time_start = [at.Time(f"{i}", format="jd").datetime for i in time_start]
        time_start = np.array([datetime.strftime(i,'%y%m%d %H%M%S') for i in time_start])
        
    with fits.open(f"{data_path}/energy_boundaries.fits") as file:
        t = Table.read(file[1])
        energy_bins = np.append(t["E_MIN"], t["E_MAX"][-1])
        
    with fits.open(f"{data_path}/dead_time.fits") as file:
        t = Table.read(file[1])
        time_elapsed = np.array(t["LIVETIME"])

    return pointings, time_start, energy_bins, time_elapsed


def combine_datasets(path_SE: str, path_PE:str, new_path:str, psd_eff:float=0.85, break_energy:int=400):
    """
    Combines a SE and PE dataset into one by adding the counts of the SE dataset to the PE dataset and correcting the
    livetime of the PE dataset.
    """
    data_SE = extract_meta_data(path_SE)
    data_PE = extract_meta_data(path_PE)

    if not os.path.exists(new_path):
        os.makedirs(new_path)

    assert len(data_SE[0]) == len(data_PE[0]), "Pointings do not match"
    assert len(data_SE[2]) == len(data_PE[2]), "Energy bins do not match"
    for i in range(len(data_SE[3])):
        assert round(data_SE[3][i], 1) == round(data_PE[3][i], 1), f"live times do not match at index {i}"

    # copy the meta data
    with fits.open(f'{path_SE}/pointing.fits') as hdul:
        hdul.writeto(f"{new_path}/pointing.fits", overwrite=True)

    with fits.open(f'{path_SE}/dead_time.fits') as hdul:
        hdul.writeto(f"{new_path}/dead_time.fits", overwrite=True)

    with fits.open(f'{path_SE}/energy_boundaries.fits') as hdul:
        hdul.writeto(f"{new_path}/energy_boundaries.fits", overwrite=True)

    # combine the counts
    with fits.open(f"{path_SE}/evts_det_spec.fits") as hdul_1:
        with fits.open(f"{path_PE}/evts_det_spec.fits") as hdul_2:
            counts_PE = hdul_2[1].data["COUNTS"]
            counts_SE = hdul_1[1].data["COUNTS"]

            counts_comb = counts_PE / psd_eff
            # find the first bin, where the energy is bigger than the break energy
            break_bin = np.where(data_PE[2] > break_energy)[0][0]

            # replace counts under the break energy with the counts from the SE dataset
            for i in range(break_bin):
                counts_comb[:,i] = counts_SE[:, i] 

            hdul_1[1].data["COUNTS"] = counts_comb

            hdul_1.writeto(f"{new_path}/evts_det_spec.fits", overwrite=True)


def get_data(config_values):
    """
    if the data_name exists, do nothing, if not get the data from afs via spiselect and safe it in data1...
    """
    data_path = BASE_DATA_PATH + f"/{config_values['data']['data_name']}/"
    if os.path.exists(data_path):
        return
    
    energies = np.geomspace(config_values["energy_range"][0] * 2, config_values["energy_range"][1] * 2, config_values["nr_ebins"], dtype=np.uint64) / 2
    E_Bins = list(energies)
    
    if config_values['data']['dataset'] in ("SE", "PE"):
            downloader = SpimselectDownloader(config_values['data']['data_name'],
                                               config_values['data']['revolutions'],
                                                 center=config_values['data']['center'],
                                                   E_Bins=E_Bins, dataset=config_values['data']['dataset'])
            # change base_dir for work on necromancer
            downloader.base_dir = "/home/tguethle/data1/cookbook/cookbook/examples/automated_Crab/"
            downloader.generate_and_run()
            downloader.adjust_for_pyspi()
            downloader.copy_to(data_path)
    elif config_values['data']['dataset'] == "combined":
        downloader_SE = SpimselectDownloader(config_values['data']['data_name']+"_SE",
                                               config_values['data']['revolutions'],
                                                 center=config_values['data']['center'],
                                                   E_Bins=E_Bins, dataset="SE")
        # change base_dir for work on necromancer
        downloader_SE.base_dir = "/home/tguethle/data1/cookbook/cookbook/examples/automated_Crab/"
        downloader_SE.generate_and_run()
        downloader_SE.adjust_for_pyspi()

        downloader_PE = SpimselectDownloader(config_values['data']['data_name']+"_PE",
                                               config_values['data']['revolutions'],
                                                 center=config_values['data']['center'],
                                                   E_Bins=E_Bins, dataset="PE")
        # change base_dir for work on necromancer
        downloader_PE.base_dir = "/home/tguethle/data1/cookbook/cookbook/examples/automated_Crab/"
        downloader_PE.generate_and_run()
        downloader_PE.adjust_for_pyspi()

        combine_datasets(f"/home/tguethle/data1/cookbook/cookbook/examples/automated_Crab/dataset_{config_values['data']['data_name']}_SE/spi2",
                                    f"/home/tguethle/data1/cookbook/cookbook/examples/automated_Crab/dataset_{config_values['data']['data_name']}_SE/spi2",
                                      data_path, config_values['data'].get("psd_eff", 0.85))

def generate_clusters(config_values:dict):
    """
    Run the pyspi fit with the given parameters.
    """

    data_path = BASE_DATA_PATH + f"/{config_values['data']['data_name']}"

    if not os.path.isfile(f"{data_path}/pointings.pickle"):
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
        print("pointing cluster saved")
    else:
        print("pointing cluster already exists, skipping...")
        
def main():
    # read in the config file
    if len(sys.argv) < 2:
        print("No config file given! \nUsage: python get_data.py <config_file>")
        sys.exit(1)

    config_file = CONFIG_DIRECTORY + "/" + sys.argv[1]

    if not os.path.isfile(config_file):
        print(f"Error: {config_file} does not exist!")
        sys.exit(1)

    with open(config_file, 'r') as f:
        config = json.load(f)

    for name, config_values in config.items():
        get_data(config_values)
        print(f"Data for {name} loaded.")

        generate_clusters(config_values)
        print(f"Clusters for {name} generated.")

        fit_path = BASE_FIT_PATH + f"/{config_values['data']['data_name']}/{name}"
        # create fit path if neccesary
        if not os.path.exists(fit_path):
            os.makedirs(fit_path)


if __name__ == "__main__":
    main()