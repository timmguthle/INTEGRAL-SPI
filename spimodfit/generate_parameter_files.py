import os
import subprocess


normal_E_Bins = []
wide_E_Bins = []


class SpimodfitScriptGenerator():
    def __init__(self, name: str, revolutions: list, source=True, E_Bins=wide_E_Bins) -> None:
        self.name = name # name of the parameter files and the generated directories. replaces "skymap43" in the template files
        self.spiselect_name = f'spiselect.dataset_{self.name}.par'
        self.background_name = f'background_model_{self.name}.pro'
        self.spimodfit_name = f'spimodfit.fit_Crab_{self.name}.par'
        self.revolutions = revolutions
        self.source = source
        self.base_dir = "/home/tguethle/cookbook/SPI_cookbook/examples/Crab/"
        # any parameter file should do. Here I chose one for rev 43
        self.spiselect_template = "spiselect.dataset_skymap43.par"
        self.background_template = "background_model_skymap43.pro"
        self.spimodfit_template = "spimodfit.fit_Crab_skymap43_noSource.par"
        self.E_Bins = E_Bins
        self.Bins = [f"{E_Bins[i]}-{E_Bins[i+1]}" for i in range(len(E_Bins)- 1)]
        self.Bins_diff = [E_Bins[i+1]-E_Bins[i] for i in range(len(E_Bins) - 1)]

    def _generatespiselect(self):
        # generate the spiselect script

        with open(self.base_dir + self.spiselect_template, 'r') as f:
            lines = f.readlines()

        lines[15] = lines[15].replace("43", f"{self.revolutions}"[1:-1])
        lines[16] = lines[15].replace("43", f"{self.revolutions}"[1:-1])

        lines[112] = f'energy_bins,s,h,"{", ".join(self.Bins)} keV",,,"Energy bins selection"'
        lines[113] = f'energy_rebin,s,h,"{", ".join(self.Bins_diff)} keV",,,"Energy rebinning (must match bins)"'

        with open(self.spiselect_name, "w") as f:
            f.writelines(lines)

        return lines
    
    def _generatebackground(self):
        '''
        generate the background script

        assuming the spiselct dictionary is already generated
        '''
        with open(self.base_dir + self.background_template, 'r') as f:
            lines = f.readlines()

        lines[9] = lines[9].replace("skymap43", self.name)
        lines[10] = lines[10].replace("skymap43", self.name)
        lines[16] = lines[16].replace("skymap43", self.name)

        lines[22] = f"emin = {self.E_Bins[0]:.0f}."
        lines[23] = f"emax = {self.E_Bins[-1]:.0f}."

        with open(self.background_name, "w") as f:
            f.writelines(lines)

        return lines
    
    def _generatespimodfit(self):
        '''
        generate the spimodfit script

        assuming the spiselect dictionary and the background is already generated
        '''
        with open(self.base_dir + self.spimodfit_template, 'r') as f:
            lines = f.readlines()

        for i in range(17, 23):
            lines[i] = lines[i].replace("skymap43", self.name)


        # chose the source or no source version
        if self.source:
            for i in (45, 48, 49, 50):
                lines[i] = lines[i].replace("#", "")

            for i in (53, 55, 56, 57):
                lines[i] = "#" + lines[i]


        with open(self.spimodfit_name, "w") as f:
            f.writelines(lines)
            
        return lines
    

    def runscripts(self):
        """        
        run the spiselect, background and spimodfit scripts
        """
        pass


