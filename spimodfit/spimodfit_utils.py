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


normal_E_Bins = []
wide_E_Bins = [20, 29, 43, 62, 91, 132, 193, 282, 411, 600]
center_simulation = '312deg -76deg' # entspricht -48deg -76deg

class SpimodfitWrapper():
    def __init__(self, name: str, revolutions: list, source=False, E_Bins=wide_E_Bins) -> None:
        self.name = name # name of the parameter files and the generated directories. replaces "skymap43" in the template files
        self.spiselect_name = f'spiselectscw.dataset_{self.name}.par'
        self.background_name = f'background_model_{self.name}.pro'
        self.spimodfit_name = f'spimodfit.fit_Crab_{self.name}.par'
        self.revolutions = revolutions
        self.source = source
        self.base_dir = "/home/tguethle/cookbook/SPI_cookbook/examples/automated_Crab/"
        # any parameter file should do. Here I chose one for rev 43
        self.spiselect_template = "spiselectscw.dataset_skymap43.par"
        self.background_template = "background_model_skymap43.pro"
        self.spimodfit_template = "spimodfit.fit_Crab_skymap43_noSource.par"
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


        # chose the source or no source version
        if self.source:
            for i in (45, 48, 49, 50):
                lines[i] = lines[i].replace("#", "")

            for i in (53, 55, 56, 57):
                lines[i] = "#" + lines[i]

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
        print('running spiselect...')
        subprocess.run(f"./submit-spiselectscw_ga05us.sh dataset_{self.name}", shell=True)
        print(f'{Fore.GREEN}spiselect done{Style.RESET_ALL}')
        with open(f"{self.base_dir}dataset_{self.name}/spiselectscw.log", 'r') as f:
            lines = f.readlines()
        print(''.join(lines[-5:-1]))

        print('running background generation...')
        subprocess.run(f"idl idl-startup.pro background_model_{self.name}.pro", shell=True)
        print('background generation done')
        print(f'{Style.BRIGHT}running spimodfit...{Style.RESET_ALL}')
        subprocess.run(f"./submit-spimodfit_v3.2_ga05us.sh fit_Crab_{self.name} clobber", shell=True)

        # print the last lines of the log file
        with open(f"{self.base_dir}fit_Crab_{self.name}/spimodfit.log", 'r') as f:
            lines = f.readlines()
        print(''.join(lines[-5:-1]))
        print(f'{Fore.GREEN}spimodfit done{Style.RESET_ALL}')

    
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


    def plot_skymap_aitoff(self, center="180deg 0deg", radius='25deg', center_skymap='180deg 0deg', nside=64):
        """
        generate a skymap in aitoff projection. With a zoom panel to the side
        """
        os.makedirs(f"{self.base_dir}{self.name}_figures/", exist_ok=True)
        crab_center = SkyCoord.from_name("Crab")

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
        
            fig.savefig(f"{self.base_dir}{self.name}_figures/fig_{self.name}_{i}.png")
            print(f'{Fore.GREEN}skymap nr.{i} plotted and saved{Style.RESET_ALL}')
  


if __name__ == '__main__':
    gen = SpimodfitWrapper('skymap0044', [44])
    gen.generate_scripts()
    gen.runscripts()
    gen.plot_skymap_aitoff(radius='30deg')