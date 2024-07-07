import os
import subprocess
from pyspi.utils.function_utils import find_response_version
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from colorama import Fore, Style
import healpy as hp
import ligo.skymap.plot

# Astropy utilities
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord


normal_E_Bins = [20.0, 21.5, 23.5, 25.5, 27.5, 30.0, 32.5, 35.5, 38.5, 42.0, 45.5, 49.5, 54.0, 58.5, 63.5, 69.0, 75.0, 81.5, 89.0, 96.5, 105.0, 114.0, 124.0, 134.5, 146.0, 159.0, 172.5, 187.5, 204.0, 221.5, 240.5, 261.5, 284.0, 308.5, 335.5, 364.5, 396.0, 430.0, 467.5, 508.0, 514, 600]
wide_E_Bins = [20, 29, 43, 62, 91, 132, 193, 282, 411, 600]
center_simulation = '312deg -76deg' # entspricht -48deg -


def read_summary_file(fit_base_path):
    with open(f"{fit_base_path}/fit_summary.txt", "r") as f:
        lines = f.readlines()
        ra, dec, K = [], [], []
        for line in lines:
            if line.startswith("Position"):
                ra.append(float(line.split()[1].strip(",")))
                dec.append(float(line.split()[2].strip(";")))
            elif line.startswith("K"):
                K.append(float(line.split()[1]))
    return ra, dec, K

class SpimselectDownloader():
    """
    # Parameters
    name: str, name of the dataset
    revolutions: list, list of revolutions to be included in the dataset
    E_Bins: list, list of energy bins to be used in the analysis. default is wide_E_Bins
    center: str or tuple or False, center of the data selection. Either 'crab', False or a tuple of two floats (chi, psi) in degrees GALACTIC coordinates.
    """
    def __init__(self, name: str, revolutions: list[int], E_Bins=wide_E_Bins, center=False) -> None:
        self.name = name
        self.revolutions = revolutions
        self.base_dir = "/home/tguethle/cookbook/SPI_cookbook/examples/automated_Crab/"
        self.spiselect_name = f'spiselectscw.dataset_{self.name}.par'
        self.spiselect_template = "spiselectscw.dataset_skymap43.par"
        self.center = center
        self.E_Bins = E_Bins
        self.Bins = [f"{E_Bins[i]}-{E_Bins[i+1]}" for i in range(len(E_Bins)- 1)]
        self.Bins_diff = [f"{E_Bins[i+1]-E_Bins[i]}" for i in range(len(E_Bins) - 1)]
        self.nr_E_bins = len(E_Bins) - 1

        assert center == 'crab' or center == False or len(center) == 2, "center must be either 'crab', False or a tuple of two floats (chi, psi) in degrees GALACTIC coordinates."

        os.chdir(self.base_dir)

    def generate_and_run(self):
        """
        generate the spiselect script and run it

        **Important**: get a afs ticket via klog first!!!
        
        **Warning**: spiselect will not ovewrite existing files. If you want to rerun the script, you have to delete the existing files first!
        """
        self._generatespiselect()

        print('running spiselect...')
        subprocess.run(f"./submit-spiselectscw_ga05us.sh dataset_{self.name}", shell=True)
        print(f'{Fore.GREEN}spiselect done{Style.RESET_ALL}')
        with open(f"{self.base_dir}dataset_{self.name}/spiselectscw.log", 'r') as f:
            lines = f.readlines()
        print(''.join(lines[-5:-1]))
        

    def _generatespiselect(self):
        # generate the spiselect script

        with open(self.base_dir + self.spiselect_template, 'r') as f:
            lines = f.readlines()

        lines[15] = lines[15].replace("43", f"{self.revolutions}"[1:-1])
        lines[16] = lines[16].replace("43", f"{self.revolutions}"[1:-1])

        lines[112] = f'energy_bins,s,h,"{", ".join(self.Bins)} keV",,,"Energy bins selection"\n'
        lines[113] = f'energy_rebin,s,h,"{", ".join(self.Bins_diff)} keV",,,"Energy rebinning (must match bins)"\n'

        # set the center of the data selection 
        if self.center:
            if self.center != 'crab':
                lines[89] = f'select_PtgX_masks_chi_list_1,r,h,{self.center[0]:.2f},-180,180,"chi center of the mask 1 for X pointing"\n'
                lines[90] = f'select_PtgX_masks_psi_list_1,r,h,{self.center[1]:.2f},-90,90,"psi center of the mask 1 for X pointing"\n'
        else:
            for i in range(86, 96):
                lines[i] = "# " + lines[i]

        with open(self.spiselect_name, "w") as f:
            f.writelines(lines)

        print(f'{self.spiselect_name} file generated')

        return lines

    def adjust_for_pyspi(self):
        """
        change dead_time.fits, evts_det_spec.fits
        
        """
        # generate a new directory for the dataset and copy pointings and energy_boundries
        os.chdir(self.base_dir + f"dataset_{self.name}/")
        if not os.path.exists("spi2"):
            os.mkdir("spi2")

        with fits.open("spi/pointing.fits.gz") as hdul:
            nr_pointings = hdul[1].data.shape[0]
            hdul.writeto("spi2/pointing.fits")

        with fits.open("spi/energy_boundaries.fits.gz") as hdul:
            hdul.writeto("spi2/energy_boundaries.fits")


        # change the format of the dead_time.fits and evts_det_spec.fits file
        with fits.open("spi/dead_time.fits.gz") as hdul:
            t = Table.read(hdul[1])

            for p in range(nr_pointings, 0, -1):
                for _ in range(66):
                    t.insert_row((19*p), (1, 0))
            hdul[1].data = Table.as_array(t)
            assert hdul[1].data.shape[0] == 85*nr_pointings, "wrong shape of dead_time.fits process failed"
            hdul.writeto("spi2/dead_time.fits")


        all_zeros = (np.zeros(self.nr_E_bins, dtype=np.uint32), np.zeros(self.nr_E_bins, dtype=np.float32))
        with fits.open("spi/evts_det_spec.fits.gz") as hdul:
            t = Table.read(hdul[1])

            for p in range(nr_pointings, 0, -1):
                for _ in range(66):
                    t.insert_row((19*p), all_zeros)
            hdul[1].data = Table.as_array(t)
            assert hdul[1].data.shape[0] == 85*nr_pointings, "wrong shape of evts_det_spec.fits process failed"
            hdul.writeto("spi2/evts_det_spec_orig.fits")

        # change back directory
        os.chdir(self.base_dir)

        print(f'{Fore.GREEN}adjusted files for pyspi{Style.RESET_ALL}')


    def copy_to_pyspi(self, path="/home/tguethle/Documents/spi/Master_Thesis/spiselect_SPI_Data/"):
        """
        copy the dataset to the pyspi directory
        """
        compleate_path = f'{path}{self.revolutions[0]:04}/'
        if not os.path.exists(compleate_path):
            os.mkdir(compleate_path)
        if len(self.revolutions) > 1:
            print(f'{Fore.RED}more than one revolution in the dataset. Nameing after first revolution{Style.RESET_ALL}')

        subprocess.run(f"cp {self.base_dir}dataset_{self.name}/spi2/*.fits {compleate_path}", shell=True)
        print(f'{Fore.GREEN}copied to pyspi{Style.RESET_ALL}')


    def adjust_for_spimodfit(self,
                              source_path="/home/tguethle/Documents/spi/Master_Thesis/main_files/spimodfit_comparison_sim_source/pyspi_real_bkg_Timm2/",
                              overwrite=True):
        """
        adjust the dataset for spimodfit and replace the original files with the newly generated ones. The new files 
        should have the simulated source included. The source simulation is done somewhere else.
        """
        # for now only with one revolution
        assert len(self.revolutions) == 1, "only one revolution is supported for now"

        subprocess.run(f"rm {self.base_dir}dataset_{self.name}/spi/evts_det_spec.fits.gz", shell=True)
        
        # dead_times, pointings and energy_boundaries are the same so already exist. only the evts_det_spec.fits file needs to be replaced
        with fits.open(f"{source_path}{self.revolutions[0]:04}/evts_det_spec.fits") as hdul:

            t = Table.read(hdul[1])
            nr_pointings = len(t) // 85
            for p in range(nr_pointings, 0, -1):
                for _ in range(66):
                    t.remove_row(85*(p-1) + 19)
            hdul[1].data = Table.as_array(t)
            assert hdul[1].data.shape[0] == 19*nr_pointings, "wrong shape of evts_det_spec.fits process failed"

            hdul.writeto(f"{self.base_dir}dataset_{self.name}/spi/evts_det_spec.fits.gz", overwrite=overwrite)
            print(f'{Fore.GREEN}adjusted files for spimodfit and copied to automated crab{Style.RESET_ALL}')


class SpimodfitWrapper():
    """
    source should either be False or catalog name (file must be in the cat directory)

    """
    def __init__(self, name: str, revolutions: list, source=False, source_name="Crab", E_Bins=wide_E_Bins) -> None:
        self.name = name # name of the parameter files and the generated directories. replaces "skymap43" in the template files
        self.spiselect_name = f'spiselectscw.dataset_{self.name}.par'
        self.background_name = f'background_model_{self.name}.pro'
        self.spimodfit_name = f'spimodfit.fit_Crab_{self.name}.par'
        self.revolutions = revolutions
        self.source = source
        self.source_name = source_name
        self.base_dir = "/home/tguethle/cookbook/SPI_cookbook/examples/automated_Crab/"
        # any parameter file should do. Here I chose one for rev 43
        self.spiselect_template = "spiselectscw.dataset_skymap43.par"
        self.background_template = "background_model_skymap43.pro"
        self.spimodfit_template = "spimodfit.fit_Crab_skymap43_noSource.par"
        self.threeML_template = "adjust4threeML_template.pro"
        self.E_Bins = E_Bins
        self.Bins = [f"{E_Bins[i]}-{E_Bins[i+1]}" for i in range(len(E_Bins)- 1)]
        self.Bins_diff = [f"{E_Bins[i+1]-E_Bins[i]}" for i in range(len(E_Bins) - 1)]
        self.IRF_versions = self._generate_IRF_list()
        os.chdir(self.base_dir)

    def generate_scripts(self):
        """
        wrapper function to generate all parameter files
        """
        self._generatespiselect()
        self._generatebackground()
        self._generatespimodfit()
        self._generate_adjust4threeML()
        print(f'{Fore.GREEN}{Style.BRIGHT}Parameter file generation done{Style.RESET_ALL}')
        

    def _generatespiselect(self):
        # generate the spiselect script

        with open(self.base_dir + self.spiselect_template, 'r') as f:
            lines = f.readlines()

        lines[15] = lines[15].replace("43", f"{self.revolutions}"[1:-1])
        lines[16] = lines[16].replace("43", f"{self.revolutions}"[1:-1])

        lines[112] = f'energy_bins,s,h,"{", ".join(self.Bins)} keV",,,"Energy bins selection"\n'
        lines[113] = f'energy_rebin,s,h,"{", ".join(self.Bins_diff)} keV",,,"Energy rebinning (must match bins)"\n'

        with open(self.spiselect_name, "w") as f:
            f.writelines(lines)

        print(f'{self.spiselect_name} file generated')

        return lines
    
    def _generatebackground(self):
        '''
        generate the background script

        assuming the spiselct dictionary is already generated
        '''
        with open(self.base_dir + self.background_template, 'r') as f:
            lines = f.readlines()

        lines[9] = f"spidir = '{self.base_dir}dataset_{self.name}/spi/'\n"
        lines[10] = f"scw_file = '{self.base_dir}dataset_{self.name}/scw.fits.gz'\n"
        lines[16] = f"bgdir = '{self.base_dir}dataset_{self.name}/spi/bg'\n"

        lines[22] = f"emin = {self.E_Bins[0]:.0f}.\n"
        lines[23] = f"emax = {self.E_Bins[-1]:.0f}.\n"

        with open(self.background_name, "w") as f:
            f.writelines(lines)

        print(f'{self.background_name} file generated')

        return lines
    
    def _generatespimodfit(self):
        '''
        generate the spimodfit script

        assuming the spiselect dictionary and the background is already generated
        '''
        with open(self.base_dir + self.spimodfit_template, 'r') as f:
            lines = f.readlines()

        for i in range(17, 23):
            lines[i] = lines[i].replace("skymap43", self.name).replace("/home/tguethle/cookbook/SPI_cookbook/examples/Crab/", self.base_dir)

        lines[22] = lines[22].replace("bg-e0020-0600", f"bg-e{self.E_Bins[0]:04.0f}-{self.E_Bins[-1]:04.0f}")

        # update the number of bins
        lines[31] = lines[31].replace("9", f"{len(self.Bins)}")
        lines[35] = lines[35].replace("9", f"{len(self.Bins)}")


        # chose the source or no source version if source, set the catalog name
        if self.source:
            assert type(self.source) == str, "source must be a string of False"
            for i in (45, 48, 49, 50):
                lines[i] = lines[i].replace("#", "")

            for i in (53, 55, 56, 57):
                lines[i] = "#" + lines[i]

            lines[45] = lines[45].replace("cat_crab", self.source)

        # modify the IRF versions
        lines[241] = lines[241].replace("1", f"{len(self.IRF_versions)}")

        # the IRF versions must be numbered from 1 to 5 but can only contain those versions that are acctually used

        for i in range(len(self.IRF_versions)):
            if self.IRF_versions[i] != 4:
                lines[242 + i] = f'irf_input_file_0{i+1},s,h,"/afs/ipp-garching.mpg.de/mpe/gamma/instruments/integral/data/ic/current/ic/spi/rsp/spi_irf_grp_002{self.IRF_versions[i]+1}.fits[GROUPING]",,,"input IRF file"\n'
            else:
                lines[242 + i] = f'irf_input_file_0{i+1},s,h,"/afs/ipp-garching.mpg.de/mpe/gamma/instruments/integral/data/ic/current/ic/spi/rsp/spi_irf_grp_002{self.IRF_versions[i]+2}.fits[GROUPING]",,,"input IRF file"\n'

        with open(self.spimodfit_name, "w") as f:
            f.writelines(lines)

        print(f'{self.spimodfit_name} file generated')
            
        return lines

    def _generate_adjust4threeML(self):

        with open(self.base_dir + self.threeML_template, 'r') as f:
            lines = f.readlines()

        lines[0] = f"response_file = '{self.base_dir}fit_Crab_{self.name}/spectral_response.rmf.fits'\n"
        lines[2] = f"spectrum_file_01 = '{self.base_dir}fit_Crab_{self.name}/spectra_{self.source_name}.fits'\n"

        with open(f"adjust4threeML_{self.name}.pro", "w") as f:
            f.writelines(lines)


    def _get_IRF_from_revs(self, rev):
        '''
        get the IRFs for the given revolution
        '''
        if rev < 140:
            return 0
        elif rev < 214:
            return 1
        elif rev < 776:
            return 2
        elif rev < 930:
            return 3
        else:  
            return 4
        
    def _generate_IRF_list(self):
        '''
        generate the list of IRFs for the given revolutions
        '''
        IRF_list = [self._get_IRF_from_revs(rev) for rev in self.revolutions]
        IRF_list = list(set(IRF_list)) # remove duplicates
        return IRF_list
        

    def runscripts(self):
        """        
        run the spiselect, background and spimodfit scripts

        **Important**: get a afs ticket via klog first!!!
        
        **Warning**: spiselect will not ovewrite existing files. If you want to rerun the script, you have to delete the existing files first!
        """
        self.run_spiselect()
        self.run_background()
        self.run_spimodfit()


    def run_spiselect(self):
        print('running spiselect...')
        subprocess.run(f"./submit-spiselectscw_ga05us.sh dataset_{self.name}", shell=True)
        print(f'{Fore.GREEN}spiselect done{Style.RESET_ALL}')
        with open(f"{self.base_dir}dataset_{self.name}/spiselectscw.log", 'r') as f:
            lines = f.readlines()
        print(''.join(lines[-5:-1]))
    
    def run_background(self):
        print('running background generation...')
        subprocess.run(f"idl idl-startup.pro background_model_{self.name}.pro", shell=True)
        print('background generation done')

    def run_spimodfit(self):
        print(f'{Style.BRIGHT}running spimodfit...{Style.RESET_ALL}')
        subprocess.run(f"./submit-spimodfit_v3.2_ga05us.sh fit_Crab_{self.name} clobber", shell=True)

        # print the last lines of the log file
        with open(f"{self.base_dir}fit_Crab_{self.name}/spimodfit.log", 'r') as f:
            lines = f.readlines()
        print(''.join(lines[-5:-1]))
        print(f'{Fore.GREEN}spimodfit done{Style.RESET_ALL}')

    def run_adjust4threeML(self):
        """
        ## Important:
        run the init_ga05us.sh script in the terminal before running this script. You need the environment variables set by this script.
        """
        # first run the auto generated script by spimodfit
        print(f'{Style.BRIGHT}running adjust4threeML...{Style.RESET_ALL}')
        os.chdir(self.base_dir)

        # subprocess.run("./init_ga05us.sh", shell=True) # run this command to set the correct environment vaiables
        # this does not work in the script, so I have to run it manually in the terminal

        os.chdir(f"{self.base_dir}fit_Crab_{self.name}/")
        subprocess.run("./spimodfit_rmfgen.csh", shell=True)
        os.chdir(self.base_dir)

        # then run the script to adjust the files for threeML
        subprocess.run(f"idl idl-startup.pro adjust4threeML_{self.name}.pro", shell=True)
        print(f'{Fore.GREEN}adjust4threeML done{Style.RESET_ALL}')

    def plot_skymap_plain(self):
        """
        plot the skymap
        """
        filepath = f"{self.base_dir}fit_Crab_{self.name}/residuals.fits"
        with fits.open(filepath) as hdul:
            # Access the data and header
            image_list = [hdul[bin].data for bin in range(3, len(self.Bins)+3)] # type: ignore
            wcs = WCS(hdul[3].header) # type: ignore

        fig, ax = plt.subplots(len(self.Bins)+1, 1, figsize=(50, 100), subplot_kw={'projection': wcs})
        for i in range(len(self.Bins)):
            #data = np.pad(image_list[i], ((0,0), (0,90)), mode="wrap")[:, 90:]
            data = image_list[i]
            ax[i].set_title(f"Energy bin {i+1} - lon shifted by 90 degrees")
            norm = TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())
            im = ax[i].imshow(data, cmap="PiYG", norm=norm, origin="lower")
            fig.colorbar(im)
            ax[i].scatter(181.44-90, -2.64, transform=ax[i].get_transform("galactic"), s=100, facecolors='none', edgecolors='r')
            ax[i].scatter(184.558-90, -5.784, transform=ax[i].get_transform("galactic"), s=100, facecolors='none', edgecolors='b')

        ax[-1].set_title("Sum of all energy bins - lon shifted by 90 degrees")
        combined_data = np.pad(np.sum(image_list, axis=0), ((0,0), (0,90)), mode="wrap")[:, 90:]
        norm = TwoSlopeNorm(vmin=combined_data.min(), vcenter=0, vmax=combined_data.max())
        im = ax[-1].imshow(combined_data, cmap="PiYG", norm=norm, origin="lower")
        fig.colorbar(im)
        ax[-1].scatter(181.44-90, -2.64, transform=ax[-1].get_transform("galactic"), s=100, facecolors='none', edgecolors='r', label='Pulsar 1A0535+262')
        ax[-1].scatter(184.558-90, -5.784, transform=ax[-1].get_transform("galactic"), s=100, facecolors='none', edgecolors='b', label="Crab")
        ax[-1].legend()

        plt.show()
        fig.savefig(f'fig_{self.name}.png')
        print(f'{Fore.GREEN}skymap plotted and saved{Style.RESET_ALL}')


    def plot_skymap_aitoff(self, center="180deg 0deg", radius='25deg', center_skymap='180deg 0deg', nside=64, sweep_search_path=None):
        """
        generate a skymap in aitoff projection. With a zoom panel to the side
        """
        os.makedirs(f"{self.base_dir}{self.name}_figures/", exist_ok=True)
        crab_center = SkyCoord.from_name("Crab")
        if sweep_search_path:
            ra, dec, K = read_summary_file(sweep_search_path)

        filepath = f"{self.base_dir}fit_Crab_{self.name}/residuals.fits"
        with fits.open(filepath) as hdul:
            # Access the data and transform it to the right format
            image_list = []
            for bin in range(3, len(self.Bins)+3):
                array = np.pad(hdul[bin].data, ((0,0), (0,180)), mode="wrap")[:, 180:]
                image_list.append(np.flip(array, axis=0).flatten())

        npix = hp.nside2npix(nside)
        hpx_maps = [np.zeros(npix) for _ in range(len(image_list))]
        theta, phi = np.mgrid[0:np.pi:180j, 0:2*np.pi:360j]

        pix_indices = hp.ang2pix(nside, theta, phi)

        for i, image in enumerate(image_list):
            hpx_maps[i][pix_indices.flatten()] = image

            fig = plt.figure(figsize=(16,8))

            ax = plt.axes((0.05,0,0.6,0.6),projection="galactic degrees aitoff", center=center_skymap)
            ax2 = plt.axes((0.6,0.1,0.4,0.4),projection="galactic degrees zoom", center=center, radius=radius)

            norm = TwoSlopeNorm(vmin=hpx_maps[i].min(), vcenter=0, vmax=hpx_maps[i].max())
            #norm2 = TwoSlopeNorm(vmin=hpx_map.min()/10, vcenter=0, vmax=hpx_map.max())

            ax.grid()
            ax2.grid()
            im = ax.imshow_hpx(hpx_maps[i], cmap='PiYG', norm=norm)
            im2 = ax2.imshow_hpx(hpx_maps[i], cmap="PiYG",norm=norm)

            fig.colorbar(im2)
            fig.colorbar(im)
            ax2.plot(
                10, -40,
                transform=ax2.get_transform('fk5'),
                marker=ligo.skymap.plot.reticle(),
                markersize=30,
                markeredgewidth=3)
            ax2.plot(
                crab_center.ra.deg, crab_center.dec.deg,
                transform=ax2.get_transform('fk5'),
                marker=ligo.skymap.plot.reticle(),
                markersize=30,
                markeredgewidth=3)

            ax.plot(
                10, -40,
                transform=ax.get_transform('fk5'),
                marker=ligo.skymap.plot.reticle(),
                markersize=30,
                markeredgewidth=3)
            
            ax.plot(
                crab_center.ra.deg, crab_center.dec.deg,
                transform=ax.get_transform('fk5'),
                marker=ligo.skymap.plot.reticle(),
                markersize=30,
                markeredgewidth=3)
            
            if sweep_search_path:
                pos = ax2.scatter(ra, dec, c=K, transform=ax2.get_transform('fk5'), s=10, alpha=0.4)    
                fig.colorbar(pos, ax=ax2, label="K")


        
            fig.savefig(f"{self.base_dir}{self.name}_figures/fig_{self.name}_{i}.png")
            print(f'{Fore.GREEN}skymap nr.{i} plotted and saved{Style.RESET_ALL}')
  
def get_data_from_pyspi(name, rev, source_path="/home/tguethle/Documents/spi/Master_Thesis/main_files/spimodfit_comparison_sim_source/pyspi_real_bkg_Timm2_para2/"):
    """
    get the ready simulated data from the pyspi 
    workflow. 
    remember to authenticate and initialize the environment variables
    """
    downloader = SpimselectDownloader(name, rev, center=False, E_Bins=normal_E_Bins)
    wrapper = SpimodfitWrapper(name, rev, source="cat_sim_source", source_name="sim_sourc", E_Bins=normal_E_Bins)
    wrapper.generate_scripts()

    downloader.generate_and_run()
    downloader.adjust_for_spimodfit(source_path=source_path)

    wrapper.run_background()
    wrapper.run_spimodfit()
    wrapper.run_adjust4threeML()



if __name__ == '__main__':
    get_data_from_pyspi("374_reduced_counts_bright_source", [374], source_path="/home/tguethle/Documents/spi/Master_Thesis/main_files/spimodfit_comparison_sim_source/reduced_counts_bright_source/")

    #gen = SpimodfitWrapper('skymap374-2', [374])
    # gen.generate_scripts()
    # gen.runscripts()
    #gen.plot_skymap_aitoff(radius='25deg', center=center_simulation, center_skymap=center_simulation, sweep_search_path="/home/tguethle/Documents/spi/Master_Thesis/main_files/no_source_bkg/sweep_search_2")
    #downloader = SpimselectDownloader('374_real_bkg_para2', [374], center=False, E_Bins=normal_E_Bins)
    #downloader.generate_and_run()
    #downloader.adjust_for_spimodfit(source_path="/home/tguethle/Documents/spi/Master_Thesis/main_files/spimodfit_comparison_sim_source/pyspi_real_bkg_Timm2_para2/")
    #downloader.adjust_for_spimodfit(source_path="/home/tguethle/Documents/spi/Master_Thesis/main_files/spimodfit_comparison_sim_source/pyspi_const_bkg_Timm2/")
    
    #wrapper = SpimodfitWrapper('374_real_bkg_para2', [374], source="cat_sim_source", source_name="sim_sourc", E_Bins=normal_E_Bins)
    #wrapper.generate_scripts()


    #wrapper.run_background()
    #wrapper.run_spimodfit()
    #wrapper.run_adjust4threeML()
    # #
    
    #wrapper.plot_skymap_aitoff(radius='30deg', center=center_simulation, center_skymap=center_simulation)

    # downloader.generate_and_run()
    # downloader.adjust_for_pyspi()
    #downloader.copy_to_pyspi()
    #downloader.adjust_for_spimodfit()