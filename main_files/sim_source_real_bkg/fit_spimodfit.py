import sys, os
sys.path.insert(0, os.path.abspath('/home/tguethle/Documents/spi/Master_Thesis'))
sys.path.insert(0, os.path.abspath('/home/tguethle/Documents/spi/Master_Thesis/main_files'))
import spimodfit.spimodfit_utils as su 
import sim_source_real_bkg.gen_data_and_pyspi_fit as gf
import spimodfit.threeml_spimodfit_fit as tsf

def run_spimodfit_and_threeml_fit(config):
    for i,c in enumerate(config):
        su.fit_with_data_from_pyspi(E_Bins="all", **c)
        print(f"Finished spimodfit {c['name']} start threeML fit")
        dataset = [f"{c['data_path']}/spimodfit", f"fit_Crab_{c['name']}",[c['K'], -2]]
        (val, cov, err, logL) = tsf.run_fit(['c1-c1160'], dataset, retrun_objects=False) # type: ignore
        tsf.save_fit(val, cov, dataset[0])

def only_good_channels():
    c = gf.config[0]
    dataset = [f"{c['data_path']}/spimodfit_tests", f"fit_Crab_{c['name']}",[c['K'], -2]]
    (val, cov, err, logL) = tsf.run_fit(['c11-c38'], dataset, retrun_objects=False, save_figure=True) # type: ignore
    tsf.save_fit(val, cov, dataset[0])


def retry_with_response():
    c = gf.config[0]
    dataset = [f"{c['data_path']}/spimodfit_tests", f"fit_Crab_{c['name']}",[c['K'], -2]]

if __name__ == "__main__":
    run_spimodfit_and_threeml_fit(gf.config_small_bins)

    