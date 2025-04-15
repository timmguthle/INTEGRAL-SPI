import sys, os
sys.path.insert(0, os.path.abspath('./main_files'))
sys.path.insert(0, os.path.abspath('./'))

import numpy as np
from MultinestClusterFit import MultinestClusterFit
from RebinningFunctions import spimodfit_binning_SE, log_binning_function_for_x_number_of_bins, no_rebinning #, rebin_data_exp_50
from PointingClusters import *
from ModelSources import *

import json

BASE_DATA_PATH = "/home/tguethle/data1/pyspi_data"
BASE_FIT_PATH = "./main_files/crab_pp"

CONFIG_DIRECTORY = "./main_files/crab_pp/config"



def run_pyspi_fit(name: str, config_values: dict, binning_func = no_rebinning,parameters = None):
    """
    Run the pyspi fit with the given parameters.
    """
    start_time = datetime.now()

    # convert the string of the functionname form the config file to the function
    try:
        crab_model = globals()[config_values["crab_model"]]
    except KeyError:
        raise ValueError(f"Model {config_values['crab_model']} not found in globals(). Make sure the model function exists.")

    # construct data and fit path
    data_path = BASE_DATA_PATH + f"/{config_values['data']['data_name']}" # check later but should be correct
    fit_path = BASE_FIT_PATH + f"/{config_values['data']['data_name']}/{name}"


    if config_values["just_crab"]:
        source_model = define_sources((
            (crab_model, (100,)),
        ))

    else:
        source_model = define_sources((
                (crab_model, (100,)),
                (s_1A_0535_262_pl, (100,)),
        ))


    if os.path.isfile(f"{data_path}/pointings.pickle"):
        pointings = load_clusters(data_path)
    else:
        raise ValueError(f"Pointings file {data_path}/pointings.pickle does not exist. Please run the data preparation script first.")

    energy_range = config_values["energy_range"]

    fit = MultinestClusterFit(
        pointings,
        source_model,
        energy_range=energy_range,
        emod=np.geomspace(energy_range[0], energy_range[1], 500),
        binning_func=binning_func,
        folder=fit_path,
        parameter_names=parameters,
    )
    try:
        fit.parameter_fit_distribution()
    except SyntaxError:
        print('saving plot failed')
    fit.text_summaries(reference_values=False)

    fit.save_chain()

    # chainconsumer
    p = config_values["p"]
    val = np.array([i[1] for i in fit._cc.analysis.get_summary(parameters=p).values()])
    cov = fit._cc.analysis.get_covariance(parameters=p)[1]
    
    np.savetxt(f"{fit_path}/fit_val.txt", val, header=" ".join(p))
    np.savetxt(f"{fit_path}/fit_cov.txt", cov, header="cov matrix")

    with open(f"{fit_path}/pyspi_summary.txt", "w") as f:
        f.write(f"Fit name: {name}\n")
        f.write(f"Energy range: {energy_range}\n")
        f.write(f"Data path: {data_path}\n")
        f.write(f"Fit path: {fit_path}\n")
        f.write(f"Result: {val}\n")
        f.write(f"Covariance: {cov}\n")
        f.write(f"compleated at {datetime.now()} in the total time of {datetime.now()-start_time}\n\n")
        f.write(json.dumps(config_values, indent=4))


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
        
        if name == "band_fit_727":

            run_pyspi_fit(name, config_values)
            print(f"Fit for {name} completed.")


if __name__ == "__main__":
    main()
