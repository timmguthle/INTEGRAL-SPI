import sys, os
sys.path.insert(0, os.path.abspath('/home/tguethle/Documents/spi/Master_Thesis'))
sys.path.insert(0, os.path.abspath('/home/tguethle/Documents/spi/Master_Thesis/main_files'))
import spimodfit.spimodfit_utils as su 
import sim_source_real_bkg.gen_data_and_pyspi_fit as gf
import spimodfit.threeml_spimodfit_fit as tsf

def run_spimodfit_and_threeml_fit():
    for i,c in enumerate(gf.config[1:]):
        su.fit_with_data_from_pyspi(**c)
        print(f"Finished spimodfit {c['name']} start threeML fit")
        dataset = [f"{c['data_path']}/spimodfit", f"fit_Crab_{c['name']}",[c['K'], -2]]
        (val, cov, err, logL) = tsf.run_fit(['c1-c41'], dataset, retrun_objects=False) # type: ignore
        tsf.save_fit(val, cov, dataset[0])

def only_good_channels():
    dataset = [f"{c['data_path']}/spimodfit_good_channels", f"fit_Crab_{c['name']}",[c['K'], -2]]
        (val, cov, err, logL) = tsf.run_fit(['c1-c41'], dataset, retrun_objects=False) # type: ignore
        tsf.save_fit(val, cov, dataset[0])
