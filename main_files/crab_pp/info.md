# Crab pyspi puplication 

The purpose of this folder is to use pyspi for a few chosen datasets. the files a similar to 
the ones in crab_19. this folder will be designed to be used on necromancer. 

get_data should use spiselect directly on necromancer and safe the data in the `data1` directory.
pyspi can then directly use the data in the `data1` directory. This should make the process of using new data much easier
as there is no need to copy the data manually to necromancer.

## Spimodfit 
I think its still best to use spimodfit on ga76pc. There is no need to change much in the code. only adjust the paths.
maybe even use spimodfit in crab 19 as the data is already there.


# config files
- name: name of the fit run
- data:
    -  data_name: name of the data folder. if it already exists on necromancer, it will be used. if not, it will be created.
    - revolutions: list of revolutions. 

- just_crab: if true only crab. if false crab and pulsar.