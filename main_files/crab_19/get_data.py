import os, sys
sys.path.insert(0, os.path.abspath('./'))

from spimodfit.spimodfit_utils import SpimodfitWrapper, SpimselectDownloader, download_and_copy_to_pyspi
import numpy as np
from astropy.io import fits
from astropy.table import Table
import astropy.time as at
from datetime import datetime

"""
Run this file to download the data from the afs with spiselect and copy it to the correct location in the pyspi directory.
"""

energies = np.geomspace(40, 1200, 51, dtype=np.uint64) / 2
E_bins_50 = list(energies)

energies2 = np.geomspace(40, 2000, 101, dtype=np.uint64) / 2
E_bins_100 = list(energies2)

print(E_bins_100)

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

def combine_datasets(path_1:str, path_2:str, new_path:str):
    """
    Combines a SE and PE dataset into one by simply adding the counts of the SE dataset to the PE dataset.
    """
    data_1 = extract_meta_data(path_1)
    data_2 = extract_meta_data(path_2)

    if not os.path.exists(new_path):
        os.makedirs(new_path)

    assert len(data_1[0]) == len(data_2[0]), "Pointings do not match"
    assert len(data_1[2]) == len(data_2[2]), "Energy bins do not match"
    for i in range(len(data_1[3])):
        assert data_1[3][i] == data_2[3][i], f"live times do not match at index {i}"

    with fits.open(f"{path_1}/evts_det_spec.fits") as hdul_1:
        with fits.open(f"{path_2}/evts_det_spec.fits") as hdul_2:
            counts_1 = hdul_1[1].data
            hdul_2[1].data["COUNTS"] += counts_1["COUNTS"]
            hdul_2.writeto(f"{new_path}/evts_det_spec.fits")

    with fits.open(f'{path_1}/pointing.fits') as hdul:
        hdul.writeto(f"{new_path}/pointing.fits")

    with fits.open(f'{path_1}/dead_time.fits') as hdul:
        hdul.writeto(f"{new_path}/dead_time.fits")

    with fits.open(f'{path_1}/energy_boundaries.fits') as hdul:
        hdul.writeto(f"{new_path}/energy_boundaries.fits")

def combine_datasets_corrected(path_SE: str, path_PE:str, new_path:str, psd_eff:float, break_energy:int=400):
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


def get_data_and_combine_20_1000():
    download_and_copy_to_pyspi(
        'data_3_2003_center',
        rev=[43, 44, 45],
        E_Bins=E_bins_100,
        center='crab',
        use_rev_name=False,
        path='./main_files/crab_19/data_3_2003_center',
    )


    download_and_copy_to_pyspi(
        'data_3_2017_center',
        rev=[1856, 1857, 1927, 1928],
        E_Bins=E_bins_100,
        center='crab',
        use_rev_name=False,
        path='./main_files/crab_19/data_3_2017_center',
    )

    download_and_copy_to_pyspi(
        'data_3_2017_PE',
        rev=[1856, 1857, 1927, 1928],
        E_Bins=E_bins_100,
        center='crab',
        dataset='PE',
        use_rev_name=False,
        path='./main_files/crab_19/data_3_2017_PE',
    )

    download_and_copy_to_pyspi(
        'data_3_2003_PE',
        rev=[43, 44, 45],
        E_Bins=E_bins_100,
        center='crab',
        dataset='PE',
        use_rev_name=False,
        path='./main_files/crab_19/data_3_2003_PE',
    )

    combine_datasets_corrected(
        './main_files/crab_19/data_3_2003_center',
        './main_files/crab_19/data_3_2003_PE',
        './main_files/crab_19/data_3_2003_comb', 0.88)
    
    combine_datasets_corrected(
        './main_files/crab_19/data_3_2017_center',
        './main_files/crab_19/data_3_2017_PE',
        './main_files/crab_19/data_3_2017_comb', 0.85)
    


if __name__ == '__main__':


    get_data_and_combine_20_1000()



    # combine_datasets_corrected('./main_files/crab_19/data_2003_center',
    #                  './main_files/crab_19/data_2003_PE',
    #                  './main_files/crab_19/data_2003_test', 0.88)


# download_and_copy_to_pyspi(
#     'data_2003_center',
#     rev=[43, 44, 45],
#     E_Bins=E_bins_50,
#     center='crab',
#     use_rev_name=False,
#     path='./main_files/crab_19/data_2003_center',
# )


# download_and_copy_to_pyspi(
#     'data_2017_center',
#     rev=[1856, 1857, 1927, 1928],
#     E_Bins=E_bins_50,
#     center='crab',
#     use_rev_name=False,
#     path='./main_files/crab_19/data_2017_center',
# )

# download_and_copy_to_pyspi(
#     'data_2017_PE',
#     rev=[1856, 1857, 1927, 1928],
#     E_Bins=E_bins_50,
#     center='crab',
#     dataset='PE',
#     use_rev_name=False,
#     path='./main_files/crab_19/data_2017_PE',
# )

# download_and_copy_to_pyspi(
#     'data_2003_PE',
#     rev=[43, 44, 45],
#     E_Bins=E_bins_50,
#     center='crab',
#     dataset='PE',
#     use_rev_name=False,
#     path='./main_files/crab_19/data_2003_PE',
# )
