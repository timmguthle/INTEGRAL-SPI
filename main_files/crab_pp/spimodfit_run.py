import sys, os
sys.path.insert(0, os.path.abspath('./'))
sys.path.insert(0, os.path.abspath('./main_files'))

import numpy as np
from spimodfit.spimodfit_utils import SpimodfitWrapper
import sim_source_real_bkg.gen_data_and_pyspi_fit as gf
import spimodfit.threeml_spimodfit_fit as tsf
import json

DEBUG = False

CONFIG_DIRECTORY = "./main_files/crab_pp/config"
BASE_FIT_PATH = "/home/tguethle/Documents/spi/Master_Thesis/main_files/crab_pp"

energies = np.geomspace(40, 1200, 41, dtype=np.uint64) / 2
E_bins_SE = list(energies)

energies2 = np.geomspace(1000, 2000, 21, dtype=np.uint64) / 2
E_bins_PE = list(energies2)
base_path = '/home/tguethle/cookbook/SPI_cookbook/examples/automated_Crab/'

def generate_Ebins(number_bins: int, energy_range: list):
    """
    Generate the energy bins for the spimodfit run.
    """
    base_bins = np.geomspace(200, 1200, number_bins, dtype=np.uint64) / 2

    low_index = np.argmin(abs(base_bins - energy_range[0]))
    high_index = np.argmin(abs(base_bins - energy_range[1]))

    return list(base_bins[low_index:high_index + 1])

def run_spimodfit_SE(name:str, config_values:dict):

    name = name + "_smf_SE"

    # from old version, dont know if I want to use this
    # energies = np.geomspace(config_values["energy_range"][0] * 2, config_values["energy_range"][1] * 2, config_values["nr_ebins"], dtype=np.uint64) / 2
    # E_Bins = list(energies)

    if config_values["use_range_test_bins"]:
        E_bins = generate_Ebins(config_values["nr_ebins"], config_values["energy_range"])
    else:
        E_bins = E_bins_SE

    w = SpimodfitWrapper(
        name=name,
        revolutions=config_values['data']['revolutions'],
        source='cat_crab',
        E_Bins=E_bins,
        convsky_output=False,
        dataset='SE',
        center='crab',
    )

    w.generate_scripts()
    w.runscripts()

def run_spimodfit_PE(name:str, config_values:dict):

    name = name + "_smf_PE"
    energies = np.geomspace(config_values["energy_range"][0] * 2, config_values["energy_range"][1] * 2, config_values["nr_ebins"], dtype=np.uint64) / 2
    E_Bins = list(energies)

    w = SpimodfitWrapper(
        name=name,
        revolutions=config_values['data']['revolutions'],
        source='cat_crab',
        E_Bins=E_bins_PE,
        convsky_output=False,
        dataset='PE',
        center='crab',
    )

    w.generate_scripts()
    w.runscripts()

def apply_model_to_spimodfit(name:str, config_values:dict):
    """
    if the config file matches to a known function then execute this function.
    Right now only the crab model for combined data is implemented.
    """
    if config_values['data']['dataset'] == "combined":

        fit_path = BASE_FIT_PATH + f"/{config_values['data']['data_name']}/{name}_smf_combined"
        if config_values["crab_model"] == "crab_band":
            
            try:
                (val, cov, err, logL) = tsf.run_fit_band( # type: ignore (with return_objects=False the output is fine)
                    SE_path=f"{base_path}fit_Crab_{name}_smf_SE",
                    PE_path=f"{base_path}fit_Crab_{name}_smf_PE",
                    fit_path=fit_path,
                    psd_eff=config_values['data'].get("psd_eff", 0.85),
                    retrun_objects=False,
                    save_figure=True,
                    print_distance=False
                )
                tsf.save_fit(val, cov, fit_path)
                p = ["Crab K", "Crab alpha", "Crab beta"]
                np.savetxt(f"{fit_path}/fit_val.txt", val, header=" ".join(p))
                np.savetxt(f"{fit_path}/fit_cov.txt", cov, header="cov matrix") # type: ignore (with return_objects=False cov can be saved)
                return
            except RuntimeError:
                print(f"Fit failed for {name}")

        elif config_values["crab_model"] == "crab_cutoff_powerlaw":
            try:
                (val, cov, err, logL) = tsf.run_fit_cutoff_powerlaw( # type: ignore (with return_objects=False the output is fine)
                    SE_path=f"{base_path}fit_Crab_{name}_smf_SE",
                    PE_path=f"{base_path}fit_Crab_{name}_smf_PE",
                    fit_path=fit_path,
                    psd_eff=config_values['data'].get("psd_eff", 0.85),
                    save_figure=True,
    
                )
                tsf.save_fit(val, cov, fit_path)
                p = ["Crab K", "Crab alpha", "Crab beta"]
                np.savetxt(f"{fit_path}/fit_val.txt", val, header=" ".join(p))
                np.savetxt(f"{fit_path}/fit_cov.txt", cov, header="cov matrix") # type: ignore (with return_objects=False cov can be saved)
                return
            except RuntimeError:
                print(f"Fit failed for {name}")

    elif config_values['data']['dataset'] == "SE":
        fit_path = BASE_FIT_PATH + f"/{config_values['data']['data_name']}/{name}_smf_SE"

        if config_values["crab_model"] == "crab_powerlaw":
            try:
                (val, cov, err, logL) = tsf.run_fit_pl( 
                    path=f"{base_path}fit_Crab_{name}_smf_SE",
                    fit_path=fit_path,
                    save_figure=True,
    
                )
                tsf.save_fit(val, cov, fit_path)
                p = ["Crab K", "Crab index"]
                np.savetxt(f"{fit_path}/fit_val.txt", val, header=" ".join(p))
                np.savetxt(f"{fit_path}/fit_cov.txt", cov, header="cov matrix") # type: ignore (with return_objects=False cov can be saved)
                return
            except RuntimeError:
                print(f"Fit failed for {name}")
        

    print(f"No model avaliable for {name}. three ml fit must be done manually.")



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
        # skip first


        if DEBUG:
            print(name)
            print(config_values)
            print(generate_Ebins(config_values["nr_ebins"], config_values["energy_range"]))
            print("\n\n")
        else:
            #run the spimodfit scripts
            if config_values['data']['dataset'] == "SE":
                run_spimodfit_SE(name, config_values)
            elif config_values['data']['dataset'] == "PE":
                run_spimodfit_PE(name, config_values)
            elif config_values['data']['dataset'] == "combined":
                run_spimodfit_SE(name, config_values)
                run_spimodfit_PE(name, config_values)

            # try to use the three ml fit
            apply_model_to_spimodfit(name, config_values)
            


        

        

if __name__ == "__main__":
    main()