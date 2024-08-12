import sys, os

sys.path.insert(0, os.path.abspath('./main_files'))

from threeML import *
from threeML.plugins.OGIPLike import OGIPLike
import matplotlib.pyplot as plt
import numpy as np
from CustomAstromodels import C_Band
import pickle
from threeML.minimizer.minimization import FitFailed



def save_fit(val, cov, fit_path):
    with open(f"{fit_path}/source_parameters.pickle", "wb") as f:
        pickle.dump((val, cov),f)

def mahalanobis_dist(vals, cov, real_vals):
    dif = (vals - real_vals)
    return np.sqrt(np.linalg.multi_dot([dif, np.linalg.inv(cov), dif]))

def select_good_channels(data_path, max_chi2=1.1, min_chi2=0.0):
    """
    only chooes channels with a good fit. The goodnes is characterized by the chi2 value. 
    """
    chi2_list = []
    mod_chi2_list = []
    with open(f'{data_path}/spimodfit.log') as f:
        lines = f.readlines()
        for line in lines:
            if "Corresponding Pearson's chi2 stat / dof" in line:
                chi2 = float(line.split()[-3][:-4])
                chi2_list.append(chi2)
            if "Reduced mod Chi square at opt" in line:
                mod_chi2 = float(line.split()[-1])
                mod_chi2_list.append(mod_chi2)

    chi2_list = np.array(mod_chi2_list)
    print(chi2_list)
    max_good_channels = chi2_list < max_chi2
    min_good_channels = chi2_list > min_chi2
    good_channels = max_good_channels & min_good_channels
    channels = []
    for i, good in enumerate(good_channels):
        if good:
            channels.append(f'c{i+1}')
    return channels

def generate_channel_options(nr_channles=41, min_size=6):
    """
    generate options with which the fit can be performed. 
    only gives options where a certain number of low/high energys are cut of.
    """
    out = []
    out2 = []
    for i in range(nr_channles):
        for j in range(i+min_size, nr_channles):
            out2.append([f'c{i+1}-c{j+1}'])
            out.append([f'c{n}' for n in range(i+1, j+2)])

    return out

def run_fit_band(
            SE_path: str,
            PE_path: str,
            fit_path: str,
            psd_eff: float,
            save_figure=False, 
            test_goodness=False, 
            retrun_objects=False,
            print_distance=True
    ):
    """
    run the fit with the given channels and channel option. 
    """
    if not os.path.exists(fit_path):
        os.makedirs(fit_path)

    crab_SE = OGIPLike("crab_SE", observation=f'{SE_path}/spectra_Crab.fits', response=f'{SE_path}/spectral_response.rmf.fits')
    crab_SE.set_active_measurements('35 - 514')

    crab_PE = OGIPLike("crab_PE", observation=f'{PE_path}/spectra_Crab.fits', response=f'{PE_path}/spectral_response.rmf.fits')
    crab_PE.set_active_measurements('514 - 1000')
    crab_PE.fix_effective_area_correction(psd_eff)

    ps_data = DataList(crab_SE, crab_PE)

    spec = C_Band()

    ps = PointSource('crab',l=0,b=0,spectral_shape=spec)

    ps_model = Model(ps)

    ps_model.crab.spectrum.main.C_Band.alpha.min_value = -2.1
    ps_model.crab.spectrum.main.C_Band.alpha = -2.0

    ps_model.crab.spectrum.main.C_Band.beta.max_value = -2.1
    ps_model.crab.spectrum.main.C_Band.beta = -2.2

    ps_model.crab.spectrum.main.C_Band.piv = 100
    ps_model.crab.spectrum.main.C_Band.xp = 500
    ps_model.crab.spectrum.main.C_Band.xp.fix = True

    
    ps_jl = JointLikelihood(ps_model, ps_data)

    best_fit_parameters_ps, likelihood_values_ps = ps_jl.fit()

    
    ps_jl.restore_best_fit()

    val = np.array(best_fit_parameters_ps["value"])
    err = np.array(best_fit_parameters_ps["error"])
    cor = ps_jl.correlation_matrix
    cov = cor * err[:, np.newaxis] * err[np.newaxis, :]
    logL = float(likelihood_values_ps.values[1])

    if save_figure:
        fig = display_spectrum_model_counts(ps_jl, step=True)
        fig.savefig(f'{fit_path}/sim_spource.pdf')
        print(f'fit saved at {fit_path}/sim_spource.pdf')

    if print_distance:
        print(mahalanobis_dist(val, cov, dataset[2]))

    if test_goodness:
        test_goodness_of_fit(ps_jl)

    if retrun_objects:
        return s_1A, ps_jl

    return val, cov, err, logL

def run_fit(channels: list[str],
            dataset, 
            save_figure=False, 
            test_goodness=False, 
            retrun_objects=False,
            print_distance=True
    ):
    """
    run the fit with the given channels and channel option. 
    """
    fit_path = dataset[0]
    if not os.path.exists(fit_path):
        os.makedirs(fit_path)
    data_path = f'/home/tguethle/cookbook/SPI_cookbook/examples/automated_Crab/{dataset[1]}'
    s_1A = OGIPLike("sim_source", observation=f'{data_path}/spectra_sim_sourc.fits', response=f'{data_path}/spectral_response.rmf.fits')
    s_1A.set_active_measurements(*channels)

    spec = Powerlaw()

    ps = PointSource('crab',l=0,b=0,spectral_shape=spec)

    ps_model = Model(ps)

    ps_model.crab.spectrum.main.Powerlaw.piv = 100

    ps_data = DataList(s_1A)

    ps_jl = JointLikelihood(ps_model, ps_data)

    best_fit_parameters_ps, likelihood_values_ps = ps_jl.fit()

    

    ps_jl.restore_best_fit()

    val = np.array(best_fit_parameters_ps["value"])
    err = np.array(best_fit_parameters_ps["error"])
    cor = ps_jl.correlation_matrix
    cov = cor * err[:, np.newaxis] * err[np.newaxis, :]
    logL = float(likelihood_values_ps.values[1])

    if save_figure:
        fig = display_spectrum_model_counts(ps_jl, step=True)
        fig.savefig(f'{fit_path}/sim_spource.pdf')
        print(f'fit saved at {fit_path}/sim_spource.pdf')

    if print_distance:
        print(mahalanobis_dist(val, cov, dataset[2]))

    if test_goodness:
        test_goodness_of_fit(ps_jl)

    if retrun_objects:
        return s_1A, ps_jl

    return val, cov, err, logL



def test_goodness_of_fit(joint_likelihood: JointLikelihood):
    """
    test the goodness of fit with the threeML build in function. 


    Ich glaube die Idee ist folgende: Wenn man mit vielen monte carlo Datensätzen simuliert und für fast alle eine 
    höhere -logL als bei dem tatsächlichen fit raukommt, kann man davon ausgehen, dass der fit gut ist.
    """
    gof_obj = GoodnessOfFit(joint_likelihood)
    gof, data_frame, like_data_frame = gof_obj.by_mc(
        n_iterations=300, continue_on_failure=True
    )
    print(gof)


def run_fit_bayes(channels: list[str], source_name='sim_sourc'):
    """
    run the fit with the given channels and channel option. 
    """
    s_1A = OGIPLike("sim_source", observation=f'{data_path}/spectra_{source_name}.fits', response=f'{data_path}/spectral_response.rmf.fits')
    s_1A.set_active_measurements(*channels)

    spec = Powerlaw()

    ps = PointSource('crab',l=0,b=0,spectral_shape=spec)

    ps_model = Model(ps)

    ps_model.crab.spectrum.main.Powerlaw.piv = 100
    ps_model.crab.spectrum.main.Powerlaw.K.prior = Log_uniform_prior(lower_bound=1e-12, upper_bound=1e0)
    ps_model.crab.spectrum.main.Powerlaw.index.prior = Uniform_prior(lower_bound=-5, upper_bound=0)

    ps_data = DataList(s_1A)

    bayes = BayesianAnalysis(ps_model, ps_data)
    bayes.set_sampler("multinest")
    bayes.sampler.setup(n_live_points=800, resume=False, verbose=True, auto_clean=True)
    bayes.sample()

    val = bayes.results._values


    cov = bayes.results.estimate_covariance_matrix()
    err = np.sqrt(np.diag(cov))
    logL = bayes.results.get_statistic_frame()

    print(mahalanobis_dist(val, cov, dataset[2]))

    return val, cov, err, logL

def run_multiple_fits(dataset, metric='m_distance', remove_bad_channels=True, max_chi2=1.2, min_chi2=0.0):
    assert metric in ['m_distance', 'logL'], 'metric not implemented'
    channels = generate_channel_options(41)
    good_channels = select_good_channels(data_path, max_chi2=max_chi2, min_chi2=min_chi2)
    real_vals = dataset[2]

    if remove_bad_channels:
        for i,co in enumerate(channels):
            for c in co.copy():
                if c not in good_channels:
                    co.remove(c)
            channels[i] = tuple(co)
        # remove duplicates
        channels = list(set(channels))

    with open(f'{fit_path}/fit_results.txt', 'w') as f:
        f.write('fit summary\n')
        f.write(f'total nr of combinations: {len(channels)}, metric: {metric}\n')

    best_metric_value = 100000
    for channel in channels:
        if len(channel) == 0:
            continue
        try:
            val, cov, err, logL = run_fit(channel, dataset)
        except FitFailed:
            continue

        dist = mahalanobis_dist(val, cov, real_vals)

        if metric == 'm_distance':
            metric_value = dist
        elif metric == 'logL':
            metric_value = logL

        with open(f'{fit_path}/fit_results.txt', 'a') as f:
            f.write(f'distance: {dist:.3f}, -LogL: {logL:.3f}, channels: {channel}, values: {val}, err: {err}\n')

        if metric_value < best_metric_value and err[0] < 0.5e-3 and err[1] < 0.05:
            best_metric_value = metric_value
            best_channel = channel
            save_fit(val, cov, dataset[0])

    with open(f'{fit_path}/fit_results.txt', 'a') as f:
        f.write(f'best fit: {best_channel}, {metric}: {best_metric_value}\n')

# possible datasets
reduced_counts_Timm2 = ['/home/tguethle/Documents/spi/Master_Thesis/main_files/spimodfit_comparison_sim_source/reduced_counts_Timm2', 'fit_Crab_374_reduced_bkg', [7e-4, -2]]
reduced_counts_Timm2_all = ['/home/tguethle/Documents/spi/Master_Thesis/main_files/spimodfit_comparison_sim_source/reduced_counts_Timm2/all', 'fit_Crab_374_reduced_bkg', [7e-4, -2]]

reduced_counts_bright_source = ['/home/tguethle/Documents/spi/Master_Thesis/main_files/spimodfit_comparison_sim_source/reduced_counts_bright_source/bayes', 'fit_Crab_374_reduced_counts_bright_source', [7e-3, -2]]

bright_source_Timm2 = ['/home/tguethle/Documents/spi/Master_Thesis/main_files/spimodfit_comparison_sim_source/pyspi_real_bkg_very_bright/0374/spimodfit', 'fit_Crab_374_very_bright', [7e-2, -2]]

real_bkg_Timm2 = ['/home/tguethle/Documents/spi/Master_Thesis/main_files/spimodfit_comparison_sim_source/spimodfit_fits/0374_real_bkg_Timm2_para2', 'fit_Crab_374_real_bkg_para2', [7e-4, -2]]
real_bkg_Timm2_all = ['/home/tguethle/Documents/spi/Master_Thesis/main_files/spimodfit_comparison_sim_source/spimodfit_fits/0374_real_bkg_Timm2_para2/all', 'fit_Crab_374_real_bkg_para2', [7e-4, -2]]

real_bkg_100_bins = ["/home/tguethle/Documents/spi/Master_Thesis/main_files/spimodfit_comparison_sim_source/pyspi_real_bkg_100_bins/0374", "fit_Crab_374_100_bins_source", [7e-3, -2]]



if __name__ == '__main__':
    c = ['c21', 'c22', 'c23', 'c24', 'c25', 'c26', 'c27', 'c28', 'c29', 'c30', 'c31', 'c32', 'c33', 'c34', 'c35', 'c36', 'c37', 'c38', 'c40', 'c41']
    #.remove('c21')
    c_for_reduced = ['c22', 'c23', 'c24', 'c25', 'c26', 'c27', 'c28', 'c29', 'c30', 'c31', 'c32', 'c33', 'c34', 'c35', 'c36', 'c37', 'c38', 'c39', 'c40', 'c41']
    c_reduced_bright = ['c20', 'c21', 'c22', 'c23', 'c24', 'c25', 'c26', 'c27', 'c28', 'c29', 'c30', 'c31', 'c32', 'c33', 'c34', 'c35', 'c36', 'c37', 'c38', 'c39', 'c40', 'c41']

    # only modify this line to change the dataset
    dataset = real_bkg_Timm2

    # define the paths
    data_path = f'/home/tguethle/cookbook/SPI_cookbook/examples/automated_Crab/{dataset[1]}'
    fit_path = dataset[0]
    
    good_channles = select_good_channels(data_path, max_chi2=1.2)
    val, cov, err, logL = run_fit(good_channles, dataset, test_goodness=False)
    save_fit(val, cov, fit_path)
    #logL = -float(np.array(logL)[0]) # only for bayesian fit
    nr_channles = len(good_channles)
    print(f"- log(L) at minimum: {logL}")
    reduced_chi2 = 2 * logL / (nr_channles- 2)
    print(f"reduced chi2: {reduced_chi2}")

