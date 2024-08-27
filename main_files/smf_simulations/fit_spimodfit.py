import sys, os
sys.path.insert(0, os.path.abspath('/home/tguethle/Documents/spi/Master_Thesis'))
sys.path.insert(0, os.path.abspath('/home/tguethle/Documents/spi/Master_Thesis/main_files'))
import spimodfit.spimodfit_utils as su 
# import sim_source_real_bkg.gen_data_and_pyspi_fit as gf
import spimodfit.threeml_spimodfit_fit as tsf

def run_spimodfit_and_threeml_fit(config):
    for i,c in enumerate(config):
        su.fit_with_data_from_pyspi(E_Bins=su.normal_E_Bins, **c)
        print(f"Finished spimodfit {c['name']} start threeML fit")
        dataset = [f"{c['data_path']}/spimodfit", f"fit_Crab_{c['name']}",[c['K'], -2]]
        # make sure the programm does not crash if the fit fails, which is possible for bad spimodfit results
        try:
            (val, cov, err, logL) = tsf.run_fit(["all"], dataset, retrun_objects=False) # type: ignore
            tsf.save_fit(val, cov, dataset[0])
        except RuntimeError:
            print(f"Fit failed for {c['name']}")


def only_good_channels(c):

    dataset = [f"{c['data_path']}/spimodfit_wo_low_energy", f"fit_Crab_{c['name']}",[c['K'], -2]]
    (val, cov, err, logL) = tsf.run_fit(['40-600'], dataset, retrun_objects=False, save_figure=True) # type: ignore
    tsf.save_fit(val, cov, dataset[0])



if __name__ == "__main__":
    test_configs = [
        {
        "name": "smf_sim_test_1",
        "data_path": "/home/tguethle/Documents/spi/Master_Thesis/main_files/smf_simulations/test_data_normal_new/",
        "rev": [374],
        "K": 7e-4,
        'center': [-48, -76]
        },
        # {
        # "name": "smf_sim_test_2",
        # "data_path": "/home/tguethle/Documents/spi/Master_Thesis/main_files/smf_simulations/test_data_normal_new_big_K/",
        # "rev": [374],
        # "K": 7e-2,
        # 'center': [-48, -76]
        # },
        {
        "name": "smf_sim_test_3",
        "data_path": "/home/tguethle/Documents/spi/Master_Thesis/main_files/smf_simulations/test_data_normal_new_small_K/",
        "rev": [374],
        "K": 7e-5,
        'center': [-48, -76]
        }
    ]

    # run_spimodfit_and_threeml_fit(test_configs[1:2])
    only_good_channels(test_configs[0])
    only_good_channels(test_configs[1])
     # first tree done only do the others

    