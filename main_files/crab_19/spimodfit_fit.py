import sys, os
sys.path.insert(0, os.path.abspath('./'))
sys.path.insert(0, os.path.abspath('./main_files'))

import numpy as np
from spimodfit.spimodfit_utils import SpimodfitWrapper
import sim_source_real_bkg.gen_data_and_pyspi_fit as gf
import spimodfit.threeml_spimodfit_fit as tsf

energies = np.geomspace(40, 1200, 41, dtype=np.uint64) / 2
E_bins_SE = list(energies)

energies2 = np.geomspace(1000, 2000, 21, dtype=np.uint64) / 2
E_bins_PE = list(energies2)
base_path = '/home/tguethle/cookbook/SPI_cookbook/examples/automated_Crab/'

config_1 = [
    {
        "data_path": "./main_files/crab_19/data_2003",
        "fit_path": "./main_files/crab_19/fit_2003/crab_band_fit",
        "energy_range": (20, 600),
        "revolutions": [43, 44, 45],
        "dataset": 'SE',
        "E_Bins": E_bins_SE,
        "psd_eff": 0.88
    }, 
    {
        "data_path": "./main_files/crab_19/data_2017",
        "fit_path": "./main_files/crab_19/fit_2017/crab_band_fit",
        "energy_range": (20, 600),
        "revolutions": [1856, 1857, 1927, 1928],
        "dataset": 'SE',
        "E_Bins": E_bins_SE,
        "psd_eff": 0.85
    }, 
        {
        "data_path": "./main_files/crab_19/data_2003_PE",
        "fit_path": "./main_files/crab_19/fit_2003/crab_band_fit_PE",
        "energy_range": (500, 1000),
        "revolutions": [43, 44, 45],
        "dataset": 'PE',
        "E_Bins": E_bins_PE,
        "psd_eff": 0.88
    }, 
    {
        "data_path": "./main_files/crab_19/data_2017_PE",
        "fit_path": "./main_files/crab_19/fit_2017/crab_band_fit_PE",
        "energy_range": (500, 1000),
        "revolutions": [1856, 1857, 1927, 1928],
        "dataset": 'PE',
        "E_Bins": E_bins_PE,
        "psd_eff": 0.85
    }, 
]


def run_spimodfit(config):
    for i,c in enumerate(config):
        c['source'] = 'cat_crab'
        c['center'] = 'crab'
        fit_path = c['fit_path']
        name = fit_path.split('/')[-2] + '_' + fit_path.split('/')[-1]
        c['name'] = name
        w = SpimodfitWrapper(**c)
        print(c)
        w.generate_scripts()
        w.runscripts()

def run_three_ml_combined(config):
    l = len(config)
    assert l % 2 == 0, "The number of datasets must be even"

    for i in range(l//2):
        c = config[i]
        c_PE = config[l//2 + i]
        fit_path = c['fit_path']
        name = fit_path.split('/')[-2] + '_' + fit_path.split('/')[-1]
        c['name'] = name
        path_SE = f"{base_path}fit_Crab_{name}"
        path_PE = f"{base_path}fit_Crab_{name}_PE"
        
        # make sure the programm does not crash if the fit fails, which is possible for bad spimodfit results
        try:
            (val, cov, err, logL) = tsf.run_fit_band(path_SE, path_PE, c['fit_path'] + '_spimodfit', c['psd_eff'], print_distance=False, save_figure=True) # type: ignore
            tsf.save_fit(val, cov, c['fit_path'] + '_spimodfit')
            p = ["Crab K", "Crab alpha", "Crab beta"]
            np.savetxt(f"{c['fit_path'] + '_spimodfit'}/fit_val.txt", val, header=" ".join(p))
            np.savetxt(f"{c['fit_path'] + '_spimodfit'}/fit_cov.txt", cov, header="cov matrix")
        except RuntimeError:
            print(f"Fit failed for {c['name']}")



def run_three_ml_combined_free_break(config):
    l = len(config)
    assert l % 2 == 0, "The number of datasets must be even"

    for i in range(l//2):
        c = config[i]
        c_PE = config[l//2 + i]
        fit_path = c['fit_path']
        name = fit_path.split('/')[-2] + '_' + fit_path.split('/')[-1]
        c['name'] = name
        path_SE = f"{base_path}fit_Crab_{name}"
        path_PE = f"{base_path}fit_Crab_{name}_PE"
        
        # make sure the programm does not crash if the fit fails, which is possible for bad spimodfit results
        try:
            (val, cov, err, logL) = tsf.run_fit_band(path_SE, path_PE, c['fit_path'] + '_spimodfit_free_break', c['psd_eff'], print_distance=False, save_figure=True, fixed_break=False) # type: ignore
            tsf.save_fit(val, cov, c['fit_path'] + '_spimodfit_free_break')
            p = ["Crab K", "Crab alpha","Crab xb", "Crab beta"]
            np.savetxt(f"{c['fit_path'] + '_spimodfit_free_break'}/fit_val.txt", val, header=" ".join(p))
            np.savetxt(f"{c['fit_path'] + '_spimodfit_free_break'}/fit_cov.txt", cov, header="cov matrix")
        except RuntimeError:
            print(f"Fit failed for {c['name']}")

if __name__ == "__main__":
    # run_spimodfit(config_1)
    run_three_ml_combined_free_break(config_1)


    