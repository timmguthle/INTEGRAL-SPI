from crab_fits import *

if __name__ == '__main__':
    # Define the data
    for conf in broken_pl_low_energy:
        crab_band_fit_wide_energy(**conf)
        print(conf['fit_path'] + " done")