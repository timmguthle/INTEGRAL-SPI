# Crab pyspi puplication 

The purpose of this folder is to use pyspi for a few chosen datasets. the files a similar to 
the ones in crab_19. this folder will be designed to be used on necromancer. 

crab_fits should use spiselect directly on necromancer and safe the data in the `data1` directory.
pyspi can then directly use the data in the `data1` directory. This should make the process of using new data much easier
as there is no need to copy the data manually to necromancer.

if the data folder already exists, it will be used. if not crab_fits.py will run spiselect and create the data folder.

## New version

first run fit_prep.py with a single process. this will create the data folder, cluster the pointings and create the fit folder.
Only after that run fit_run.py with multiple processes. this will run the fit on the data and safe it in the fit folder.

# Spimodfit 
I think its still best to use spimodfit on ga76pc. There is no need to change much in the code. only adjust the paths.
maybe even use spimodfit in crab 19 as the data is already there.

## with config files

My current plan: run spimodfit similar to fit_run.py with the same config file. to do this run spimodfit_run.py. do this on ga76pc.
the spectral fitting after that is different. maybe do this in a jupyter notebook.


# config files
**Example config file for crab pulsar**
```json
{
    "test_config_fit": { //name

        "data": {
            "data_name": "test_data_2008", //unique name of the data folder
            "revolutions": [665, 666],
            "center": "crab", // point at which the data is centered. should stay crab
            "dataset": "combined", // SE, PE or combined. if combined, the script gets the SE and PE data and combines them.
            "psd_eff": 0.88, // efficiency of the psd.if not given 0.85 will be used as default.
        },
        "nr_ebins": 100,
        "energy_range": [20,1000],
        "just_crab": false, // if true only crab. if false crab and pulsar.

        "crab_model": "crab_band", // name of the model function. must exist in the corrosponding model file.
        "p": ["Crab K", "Crab alpha", "Crab beta", "A 0535 262 K", "A 0535 262 index"]
        // parameters to expect form the model.

    }
}
```

# Notes 

- fit_run.py can recive a second argument. if you only want to run one fit from the config file, give the name of this fit as the second argument. If no argument is given, all fits will be run.
- Test run with test_config.json is working fine now.

- It might be useful to split the file into two, on as a "Preparation" file and one as a "Analysis" file.
    - The preparation file handles the data and the clustering, it should be run on a single process to avoid problems with the memory and safe resources. the main analysis file can be run on multiple processes via mpiexec.
