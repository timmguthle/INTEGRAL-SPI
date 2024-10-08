import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
from numba import njit
from pyspi.utils.function_utils import find_response_version
from pyspi.utils.response.spi_response_data import ResponseDataRMF
from pyspi.utils.response.spi_response import ResponseRMFGenerator
from pyspi.utils.response.spi_drm import SPIDRM
from pyspi.utils.livedets import get_live_dets
from chainconsumer import ChainConsumer
import pymultinest
import os
import astropy.time as at
from astromodels import Uniform_prior
from scipy.stats import poisson, norm

# just load the required version 
# generate a list of all versions would only be better if there are several pointing that need different irf versions


rsp_bases = tuple([ResponseDataRMF.from_version(i) for i in range(5)])

# solution for now: hardcode the solution. should be fixed in the future!!!
# this only works for simulations. For fitting Crab we need all the irfs again
#rsp_bases = {2:ResponseDataRMF.from_version(2)}



@njit
def b_maxL_2(m, t, C):
    first = C[0]+C[1]-(m[0]+m[1])*(t[0]+t[1])
    root = (C[0]+C[1]+(m[0]-m[1])*(t[0]+t[1]))**2-4*C[0]*(m[0]-m[1])*(t[0]+t[1])
    res = (first+np.sqrt(root))/(2*(t[0]+t[1]))
    # if res < 0:
    #     print()
    #     print("WARNING: Maximum Likelihood Background is less than 0!")
    #     print()
    return res

@njit
def b_maxL_3(m, t, C):
    mt = m[0] + m[1] + m[2]
    tt = t[0] + t[1] + t[2]
    Ct = C[0] + C[1] + C[2]
    a = -tt
    b = -tt*mt + Ct
    c = Ct*mt - C[0]*m[0] - C[1]*m[1] - C[2]*m[2] -tt*(m[0]*m[1] + m[1]*m[2] + m[2]*m[0])
    d = C[0]*m[1]*m[2] + C[1]*m[2]*m[0] + C[2]*m[0]*m[1] - tt*m[0]*m[1]*m[2]
    D0 = b**2 - 3*a*c
    D1 = 2*b**3 - 9*a*b*c + 27*(a**2)*d
        
    if D0 == 0. and D1 == 0.:
        return -b/(3*a)
    
    C0 = ((D1 + np.sqrt(D1**2 - 4*D0**3 + 0j)) / 2)**(1/3)
    
    if C0 == 0:
        C0 = ((D1 - np.sqrt(D1**2 - 4*D0**3 + 0j)) / 2)**(1/3)
        
    x0 = -1/(3*a) * (b + C0 + D0/C0)
    
    # if x0.real < 0:
    #     return 0.
    
    return x0.real

@njit
def b_maxL_4(m, t, C): ###################### doesnt work?
    t_t = np.sum(t)
    C_ = np.zeros(4)
    m_plus = np.zeros(4)
    m_cross = np.zeros(4)
    m_dot = np.zeros(4)
    
    for i in range(4):
        C_[i] = C[i] - t[i] * m[i]
        m_plus[i] = m[(i+1) % 4] + m[(i+2) % 4] + m[(i+3) % 4]
        m_cross[i] = (m[(i+1) % 4] * m[(i+2) % 4]
                      + m[(i+2) % 4] * m[(i+3) % 4]
                      + m[(i+3) % 4] * m[(i+1) % 4])
        m_dot[i] = m[(i+1) % 4] * m[(i+2) % 4] * m[(i+3) % 4]
        
    A = -t_t.item()
    B = np.sum(C_ - t * m_plus).item()
    C = np.sum(C_ * m_plus - t * m_cross).item()
    D = np.sum(C_ * m_cross - t * m_dot).item()
    E = np.sum(C_ * m_dot).item()
    
    # print(A,B,C,D,E)
    
    alpha = -3 * B**2 / (8 * A**2) + C/A
    beta = B**3 / (8 * A**3) - B * C / (2 * A**2) + D/A
    gamma = -3 * B**4 / (256 * A**4) + C * B**2 / (16 * A**3) - B * D / (4 * A**2) + E / A
    
    if beta == 0.:
        s1 = (alpha**2 - 4*gamma)**0.5
        s2 = ((-alpha + s1) / 2)**0.5
        x = -B/(4*A) + s2
        return x
    
    P = -(alpha**2)/12 - gamma
    Q = -(alpha**3)/108 + alpha*gamma/3 - (beta**2) / 8
    R = -Q/2 + (Q**2 / 4 + P**3 / 27)**0.5
    U = R**(1/3)
    if U == 0.:
        y = -5/6*alpha - (Q**(1/3))
    else:
        y = -5/6*alpha + U - P/(3*U)
    W = (alpha + 2*y)**0.5
    
    s1 = 2*beta/W
    s2 = (-(3*alpha + 2*y + s1))**0.5
    s3 = (W + s2) / 2
    x = -B / (4*A) + s3
    return x.real


@njit
def logLcore(
    spec_binned,
    pointings,
    dets,
    resp_mats,
    num_sources,
    t_elapsed,
    counts
):
    logL=0
    for p_i in range(len(pointings)):
        for d_i in range(len(dets[p_i])):
            n_p = len(pointings[p_i])
            m = np.zeros((n_p, len(resp_mats[p_i][0][0][0,0,:])))
            
            t_b = np.zeros(n_p)
            for t_i in range(n_p):
                t_b[t_i] = t_elapsed[p_i][t_i][d_i]
            C_b = np.zeros(n_p)
            
            for s_i in range(num_sources):
                for m_i in range(n_p):
                    m[m_i,:] += np.dot(spec_binned[s_i,:], resp_mats[p_i][s_i][m_i][d_i])
            for e_i in range(len(m[0])):
                m_b = m[:,e_i]
                for C_i in range(n_p):
                    C_b[C_i] = counts[p_i][C_i][d_i, e_i]
                    
                if n_p == 2:
                    b = b_maxL_2(m_b, t_b, C_b)
                elif n_p == 3:
                    b = b_maxL_3(m_b, t_b, C_b)
                elif n_p == 4:
                    b = b_maxL_4(m_b, t_b, C_b)
                else:
                    print()
                    print("b_maxL is not defined")
                    print()
                    return 0.
                for m_i in range(n_p):
                    logL += (counts[p_i][m_i][d_i, e_i]*math.log(t_elapsed[p_i][m_i][d_i]*(m[m_i,e_i]+b))
                            -t_elapsed[p_i][m_i][d_i]*(m[m_i,e_i]+b))
    return logL

@njit
def sample_count_rates(
    c_i,
    source_rate,
    background_rate,
    posterior_samples,
    dets,
    ebs,
    t_elapsed,
    variance_matrix,
    dimension_values,
    b_int_funcs,
    c_int_funcs,
    s_int_funcs
):
    expected_counts_combination = np.zeros(source_rate[c_i].shape)
    for d_i in range(len(dets[c_i])):
        for e_i in range(len(ebs[c_i])-1):
            for p_i in range(len(posterior_samples)):
                matrix_pos = np.array([
                    background_rate[c_i][d_i, e_i, p_i],
                    source_rate[c_i][0,d_i,e_i,p_i],
                    t_elapsed[c_i][0][d_i],
                    source_rate[c_i][1,d_i,e_i,p_i],
                    t_elapsed[c_i][1][d_i]
                ])
                
                b_vars = interpolate_matrix_5_dim(
                    matrix_pos,
                    variance_matrix[:,:,:,:,:,:,0,0],
                    dimension_values,
                    b_int_funcs
                )
                
                co_vars = interpolate_matrix_5_dim(
                    matrix_pos,
                    variance_matrix[:,:,:,:,:,:,0,1],
                    dimension_values,
                    c_int_funcs
                )
                
                s_vars = interpolate_matrix_5_dim(
                    matrix_pos,
                    variance_matrix[:,:,:,:,:,:,1,1],
                    dimension_values,
                    s_int_funcs
                )
                
                co_var_matrix1 = np.array([
                    [b_vars[0], co_vars[0]],
                    [co_vars[0], s_vars[0]]
                ])
                
                # matrix_pos = np.array([
                #     background_rate[c_i][d_i, e_i, p_i],
                #     source_rate[c_i][1,d_i,e_i,p_i],
                #     t_elapsed[c_i][1][d_i],
                #     source_rate[c_i][0,d_i,e_i,p_i],
                #     t_elapsed[c_i][0][d_i]
                # ])
                
                # b_vars = interpolate_matrix_5_dim(
                #     matrix_pos,
                #     variance_matrix[:,:,:,:,:,:,0,0],
                #     dimension_values,
                #     b_int_funcs
                # )
                
                # co_vars = interpolate_matrix_5_dim(
                #     matrix_pos,
                #     variance_matrix[:,:,:,:,:,:,0,1],
                #     dimension_values,
                #     c_int_funcs
                # )
                
                # s_vars = interpolate_matrix_5_dim(
                #     matrix_pos,
                #     variance_matrix[:,:,:,:,:,:,1,1],
                #     dimension_values,
                #     s_int_funcs
                # )
                
                # co_var_matrix2 = np.array([
                #     [b_vars[0], co_vars[0]],
                #     [co_vars[0], s_vars[0]]
                # ])
                
                co_var_matrix2 = np.array([
                    [b_vars[1], co_vars[1]],
                    [co_vars[1], s_vars[1]]
                ])
                
                expected_counts1 = t_elapsed[c_i][0][d_i] * np.array([background_rate[c_i][d_i, e_i, p_i], source_rate[c_i][0,d_i,e_i,p_i]])
                expected_counts2 = t_elapsed[c_i][1][d_i] * np.array([background_rate[c_i][d_i, e_i, p_i], source_rate[c_i][1,d_i,e_i,p_i]])
                
                expected_counts_combination[0, d_i, e_i, p_i] = np.sum(
                    multivariate_normal_numba(expected_counts1, co_var_matrix1)
                )
                expected_counts_combination[1, d_i, e_i, p_i] = np.sum(
                    multivariate_normal_numba(expected_counts2, co_var_matrix2)
                )
    return expected_counts_combination

def extract_pointing_info(path, p_id):
    num_dets = 19 

    with fits.open(f"{path}/pointing.fits") as file:
        t = Table.read(file[1])
        index = np.argwhere(t["PTID_ISOC"]==p_id[:8])
        
        if len(index) < 1:
            raise Exception(f"{p_id} not found")

        pointing_info = t[index[-1][0]]
        
        t1 = at.Time(f'{pointing_info["TSTART"]+2451544.5}', format='jd').datetime
        time_start = datetime.strftime(t1,'%y%m%d %H%M%S')
            
    with fits.open(f"{path}/dead_time.fits") as file:
        t = Table.read(file[1])
        
        time_elapsed = np.zeros(num_dets)
        
        for i in range(num_dets):
            for j in index:
                time_elapsed[i] += t["LIVETIME"][j[0]*85 + i]
        
    with fits.open(f"{path}/energy_boundaries.fits") as file:
        t = Table.read(file[1])
        
        energy_bins = np.append(t["E_MIN"], t["E_MAX"][-1])
    
    # this gets the SE data
    with fits.open(f"{path}/evts_det_spec.fits") as file:
        t = Table.read(file[1])
        
        counts = np.zeros((num_dets, len(energy_bins)-1))
        for i in range(num_dets):
            for j in index:
                counts[i, : ] += t["COUNTS"][j[0]*85 + i]
    # get the PE data
    with fits.open(f"{path}/evts_det_spec_PE.fits") as file:
        t = Table.read(file[1])
        
        counts_PE = np.zeros((num_dets, len(energy_bins)-1))
        for i in range(num_dets):
            for j in index:
                counts_PE[i, : ] += t["COUNTS"][j[0]*85 + i]
    # counts in format (det, energy_bin)
    return time_start, time_elapsed, energy_bins, counts, counts_PE

def generate_resp_mat(
    rmfs,
    len_dets,
    len_ebs,
    len_emod,
    ra,
    dec,
):
    sds = np.empty(0)
    for d in range(len_dets):
        sd = SPIDRM(rmfs[d], ra, dec)
        sds = np.append(sds, sd.matrix.T)
    return sds.reshape((len_dets, len_emod-1, len_ebs-1))

def calc_mahalanobis_dist(summary, cov, true_vals):
    fit_val = np.array([i[1] for i in summary.values()])
    fit_cov = cov[1]
    rel_distance = []
    
    for i in range(len(true_vals)):
        dif = fit_val - true_vals[i]
        
        rel_distance.append(np.sqrt(
            np.linalg.multi_dot([dif, np.linalg.inv(fit_cov), dif])
        ))
        
    return np.array(rel_distance)

@njit
def powerlaw_binned_spectrum(energy_bins, spectrum):
    assert np.amin(energy_bins) > 0, "All energy bin values must be greater 0"
    assert np.amin(spectrum) > 0, "All spectrum values must be greater 0"
    
    B = np.log(spectrum[1:] / spectrum[:-1]) / np.log(energy_bins[1:] / energy_bins[:-1])
    A = spectrum[:-1] / (energy_bins[:-1] ** B)
    
    C = B + 1.
    # prevent rounding errors
    C[np.abs(C) < 1e-7] = 0
    
    regular = np.nonzero(C)[0]
    non_regular = []
    for i in range(len(energy_bins) - 1):
        if not i in regular:
            non_regular.append(i)
    non_regular = np.array(non_regular)
    
    binned_spectrum = np.zeros(len(energy_bins) - 1)
    binned_spectrum[regular] = A[regular] / (C[regular]) * (energy_bins[regular+1]**(C[regular]) - energy_bins[regular]**(C[regular]))
    binned_spectrum[non_regular] = A[non_regular] * (np.log(energy_bins[non_regular+1]) - np.log(energy_bins[non_regular]))
    
    return binned_spectrum

@njit
def interpolate_constant(x1, x2, y1, y2, x):
    return (y1 + y2) / 2 + 0*x

@njit
def interpolate_linear(x1, x2, y1, y2, x):
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

@njit
def interpolate_logarithmic(x1, x2, y1, y2, x):
    A = (y2 - y1) / np.log(x2 / x1)
    B = y1 - A * np.log(x1)
    return A * np.log(x) + B

@njit
def interpolate_powerlaw(x1, x2, y1, y2, x):
    B = np.log(y2 / y1) / np.log(x2 / x1)
    # A = y1 * x1**(-B)
    # if np.isnan(y1 * (x / x1) ** B).any():
    #     print(x1, x2, y1, y2, x)
    #     print(y1 * (x / x1) ** B)
    #     print()
    # return A * x**B
    return y1 * (x / x1) ** B

@njit
def interpolate_matrix_5_dim(position, matrix, dimension_values, interpolation_functions):
    # this is a really ugly implementation, but I couldn't figure out a better way using numba
    
    indices = np.zeros(len(position), dtype=np.int32)
    for i in range(len(position)):
        indices[i] = np.searchsorted(dimension_values[i], position[i], side="left")
        
    surrounding_matrix = matrix[
        indices[0]-1 : indices[0]+1,
        indices[1]-1 : indices[1]+1,
        indices[2]-1 : indices[2]+1,
        indices[3]-1 : indices[3]+1,
        indices[4]-1 : indices[4]+1,
    ]
    
    interpolation1 = interpolation_functions[0](
        dimension_values[0][indices[0]-1],
        dimension_values[0][indices[0]],
        surrounding_matrix[0],
        surrounding_matrix[1],
        position[0]
    )
    
    interpolation2 = interpolation_functions[1](
        dimension_values[1][indices[1]-1],
        dimension_values[1][indices[1]],
        interpolation1[0],
        interpolation1[1],
        position[1]
    )
    
    interpolation3 = interpolation_functions[2](
        dimension_values[2][indices[2]-1],
        dimension_values[2][indices[2]],
        interpolation2[0],
        interpolation2[1],
        position[2]
    )
    
    interpolation4 = interpolation_functions[3](
        dimension_values[3][indices[3]-1],
        dimension_values[3][indices[3]],
        interpolation3[0],
        interpolation3[1],
        position[3]
    )
    
    interpolation5 = interpolation_functions[4](
        dimension_values[4][indices[4]-1],
        dimension_values[4][indices[4]],
        interpolation4[0],
        interpolation4[1],
        position[4]
    )
        
    return interpolation5

@njit
def cholesky_numba(A):
    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i+1):
            s = 0
            for k in range(j):
                s += L[i][k] * L[j][k]

            if (i == j):
                L[i][j] = (A[i][i] - s) ** 0.5
            else:
                L[i][j] = (1.0 / L[j][j] * (A[i][j] - s))
    return L

@njit
def multivariate_normal_numba(mean, covariance):
    L = cholesky_numba(covariance)
    X = np.array([np.random.normal(), np.random.normal()])
    return L.dot(X) + mean



@njit
def calc_bmaxL_variance_matrix(b_array, s1_array, t1_array, s2_array, t2_array, num_samples=10000):
    variance_matrix = np.zeros((len(b_array), len(s1_array), len(t1_array), len(s2_array), len(t2_array), 2, 2, 2))
    for b_i, b in enumerate(b_array):
        for s1_i, s1 in enumerate(s1_array):
            for t1_i, t1 in enumerate(t1_array):
                for s2_i, s2 in enumerate(s2_array):
                    for t2_i, t2 in enumerate(t2_array):
                        bd1 = np.zeros(num_samples)
                        bd2 = np.zeros(num_samples)
                        sd1 = np.zeros(num_samples)
                        sd2 = np.zeros(num_samples)
                        for n_i in range(num_samples):
                            s1m = np.random.poisson(t1 * s1)
                            s2m = np.random.poisson(t2 * s2)
                            b1m = np.random.poisson(t1 * b)
                            b2m = np.random.poisson(t2 * b)
                            
                            c1m = s1m + b1m
                            c2m = s2m + b2m
                            b_max_L = b_maxL_2(np.array([s1, s2]), np.array([t1, t2]), np.array([c1m, c2m]))
                            
                            bd1[n_i] = b1m - b_max_L*t1
                            bd2[n_i] = b2m - b_max_L*t2
                            sd1[n_i] = s1m - s1*t1
                            sd2[n_i] = s2m - s2*t2
                            
                        variance_matrix[b_i, s1_i, t1_i, s2_i, t2_i, 0, :, :] = np.cov(bd1, sd1)
                        variance_matrix[b_i, s1_i, t1_i, s2_i, t2_i, 1, :, :] = np.cov(bd2, sd2)
                        
    return variance_matrix




class MultinestClusterFit:
    def __init__(
        self,
        pointings,
        source_model,
        energy_range,
        emod,
        binning_func,
        true_values=None,
        folder=None,
        parameter_names=None,
        source_spectrum_powerlaw_binning=True,
    ):
        self._pointings = pointings
        self._source_model = source_model
        self._binning_func = binning_func
        self._energy_range = energy_range
        self._emod = emod
        self._source_spectrum_powerlaw_binning = source_spectrum_powerlaw_binning
        
        self._true_values = true_values
        self.set_folder(folder)
        
        self._prepare_fit_data()

        # self._load_required_rsp()
        
        self._find_updatable_sources()
        
        self._initialize_resp_mats()
        
        self._run_multinest()
        
        if parameter_names is None:
            self._extract_parameter_names_simple()
        else:
            self._parameter_names = parameter_names
        self._parameter_names.extend(["$z$"])
            
        self._cc = ChainConsumer()
        self._chain = np.loadtxt('./chains/1-post_equal_weights.dat')
        self._cc.add_chain(self._chain, parameters=self._parameter_names, name='fit')
        
        # import pickle
        # with open("temp_dump","wb") as f:
        #     pickle.dump(
        #         (
        #             self._pointings,
        #             self._source_model,
        #             self._energy_range,
        #             self._emod,
        #             self._folder,
        #             self._cc,
        #             self._chain,
        #             self._parameter_names,
        #             self._dets,
        #             self._ebs,
        #             self._t_elapsed,
        #             self._resp_mats,
        #             self._updatable_sources,
        #             self._counts
        #         ),
        #         f
        #     )
        
        # with open("temp_dump", "rb") as f:
        #     (
        #             self._pointings,
        #             self._source_model,
        #             self._energy_range,
        #             self._emod,
        #             self._folder,
        #             self._cc,
        #             self._chain,
        #             self._parameter_names,
        #             self._dets,
        #             self._ebs,
        #             self._t_elapsed,
        #             self._resp_mats,
        #             self._updatable_sources,
        #             self._counts
        #         ) = pickle.load(f)
        
        
    def _run_multinest(self):
        num_sources = len(self._source_model.sources)
        
        import pickle # why is this necessary for 1380?????????????????????????????????????????????
        with open("error_test","wb") as f:
            pickle.dump(
                (self._counts,),
                f
            )
        
        with open("error_test", "rb") as f:
            (self._counts,) = pickle.load(f)

        with open("error_test_PE","wb") as f:
            pickle.dump(
                (self._counts_PE,),
                f
            )
        
        with open("error_test_PE", "rb") as f:
            (self._counts_PE,) = pickle.load(f)

        # was bringt das hier?
        
        
        def logLba_mult(trial_values, ndim=None, params=None):
            spec_binned = np.zeros((num_sources, len(self._emod)-1))
            for i, parameter in enumerate(self._source_model.free_parameters.values()):
                parameter.value = trial_values[i]
            for i, source in enumerate(self._source_model.sources.values()):
                spec = source(self._emod)
                if self._source_spectrum_powerlaw_binning:
                    spec_binned[i,:] = powerlaw_binned_spectrum(self._emod, spec)
                else:
                    spec_binned[i,:] = (self._emod[1:]-self._emod[:-1])*(spec[:-1]+spec[1:])/2
            if 1 in self._updatable_sources:
                self._update_resp_mats()

            # modify the counts with psd_eff
            psd_eff_index = len(self._source_model.free_parameters.values())
            psd_eff = trial_values[psd_eff_index]
            combined_counts = []
            for c_i, combination in enumerate(self._pointings):
                combined_counts.append([self._counts_PE[c_i][p_i].copy() / psd_eff for p_i in range(len(combination))])
            
            # TODO: find out how ebins look like

            # what is the shape of counts?
            for c_i, combination in enumerate(self._pointings):
                for p_i, pointing in enumerate(combination):
                    combined_counts[c_i][p_i][:,:self._break_bin] = self._counts[c_i][p_i][:,:self._break_bin].copy()

            # transform the list to a tuple

            combined_counts_tuple = tuple([tuple(combined_counts[i]) for i in range(len(combined_counts))])


            return logLcore(
                spec_binned,
                self._pointings,
                self._dets,
                self._resp_mats,
                num_sources,
                self._t_elapsed,
                combined_counts_tuple,
            )
        
        def prior(params, ndim=None, nparams=None):
            for i, parameter in enumerate(self._source_model.free_parameters.values()):
                try:
                    params[i] = parameter.prior.from_unit_cube(params[i])

                except AttributeError:
                    raise RuntimeError(
                        "The prior you are trying to use for parameter %s is "
                        "not compatible with sampling from a unitcube"
                        % parameter.path
                    )
            # psd_eff (params[-1]) should be between 0 and 1. but no avoide division by 0, we set it to 0.2 + 0.8*params[-1] so it should be between 0.2 and 1
            psd_eff_index = len(self._source_model.free_parameters.values()) 
            params[psd_eff_index] = Uniform_prior(lower_bound=0.3, upper_bound=1).from_unit_cube(params[psd_eff_index])


        num_params = len(self._source_model.free_parameters) + 1

        if not os.path.exists("./chains"):
            os.mkdir("chains")
        sampler = pymultinest.run(
            logLba_mult, prior, num_params, num_params, n_live_points=800, resume=False, verbose=True, use_MPI=False
        )
    
    def _prepare_fit_data(self):
        ebs = []
        counts = []
        counts_PE = []
        dets = []
        t_elapsed = []
        t_start = []
            
        for combination in self._pointings:
            c_time_start, c_time_elapsed = [], []
            for p_i, pointing in enumerate(combination):
                time_start, time_elapsed, energy_bins, counts_f, counts_f_PE = extract_pointing_info(pointing[1], pointing[0])
                # pointing[0] is the pointing id, pointing[1] is the path to the pointing
                # counts_f in format (det, energy_bin)
                c_time_start.append(time_start)
                dets_temp = get_live_dets(time=time_start, event_types=["single"])
                c_time_elapsed.append(time_elapsed[dets_temp])
                
                if p_i == 0:
                    dets_0 = dets_temp
                    energy_bins_0 = energy_bins
                    c_counts_f = counts_f[dets_0]
                    c_counts_f_PE = counts_f_PE[dets_0]
                else:
                    c_counts_f = np.append(c_counts_f, counts_f[dets_0], axis=0)
                    c_counts_f_PE = np.append(c_counts_f_PE, counts_f_PE[dets_0], axis=0)
                    assert np.array_equal(dets_0, dets_temp), f"Active detectors are not the same for {combination[0][0]} and {combination[p_i][0]}"
                    assert np.array_equal(energy_bins_0, energy_bins), f"Energy bins are not the same for {combination[0][0]} and {combination[p_i][0]}"
                

            # binning function returns new bins in the given energy interval, returns ebins and the corrosponding counts
            eb, c = self._binning_func(
                energy_bins_0,
                c_counts_f,
                self._energy_range
            )
            _, c_PE = self._binning_func(
                energy_bins_0,
                c_counts_f_PE,
                self._energy_range
            )

            nd = len(dets_0)
            counts.append(tuple([c[i*nd : (i+1)*nd] for i in range(len(combination))]))
            counts_PE.append(tuple([c_PE[i*nd : (i+1)*nd] for i in range(len(combination))]))
            ebs.append(eb)
            
            t_start.append(tuple(c_time_start))
            dets.append(dets_0)
            t_elapsed.append(tuple(c_time_elapsed))
                
        self._ebs = tuple(ebs) 
        self._counts = tuple(counts)
        self._counts_PE = tuple(counts_PE)
        self._dets = tuple(dets)
        self._t_elapsed = tuple(t_elapsed)
        self._t_start = tuple(t_start)

        break_energy = 400
        # find the bin where the break energy is. not sure if this works. need to test first
        self._break_bin = np.where(self._ebs[0] > break_energy)[0][0]
        print(self._break_bin)

    def _load_required_rsp(self):
        """
        should load only the needed versions of the irf. Problem is, that this way it has to be loaded every time a MultinestClusterFit class 
        is created, which is not optimal either...
        """
        versions = []
        for count, comb in enumerate(self._pointings):
            for pointing_nr in range(len(comb)):
                time = self._t_start[count][pointing_nr]
                version = find_response_version(time)
                if version not in versions:
                    versions.append(version)
        print(f"loading the following irf versions {versions}")
        self._rsp_bases = {v:ResponseDataRMF.from_version(v) for v in versions}

    
    def _initialize_resp_mats(self):
        # index order: tuple(combination, source, pointing, np_array(dets, e_in, e_out))
        resp_mats = []
        rmfs = []
        
        for count, combination in enumerate(self._pointings):
            source_resp_mats = []
            
            dets = self._dets[count]
            ebs = self._ebs[count]
            
            for source in self._source_model.sources.values():
                combination_resp_mats = []
                combination_rmfs = []
                
                for pointing in range(len(combination)):
                    time = self._t_start[count][pointing]
                    version = find_response_version(time)
                    rsp_base = rsp_bases[version]
                    
                    pointing_rmfs = []
                    for d in dets:
                        pointing_rmfs.append(ResponseRMFGenerator.from_time(time, d, ebs, self._emod, rsp_base))
                    pointing_rmfs = tuple(pointing_rmfs)
                                        
                    combination_resp_mats.append(
                        generate_resp_mat(
                            pointing_rmfs,
                            len(dets),
                            len(ebs),
                            len(self._emod),
                            source.position.get_ra(),
                            source.position.get_dec(),
                        )
                    )
                    combination_rmfs.append(pointing_rmfs)
                    
                source_resp_mats.append(tuple(combination_resp_mats))
                    
            resp_mats.append(tuple(source_resp_mats))
            rmfs.append(tuple(combination_rmfs))
            
        self._resp_mats = tuple(resp_mats)
        if 1 in self._updatable_sources:
            self._updatable_rmfs = tuple(rmfs)

    def _update_resp_mats(self):
        for count, combination in enumerate(self._pointings):
            for source_num, source in enumerate(self._source_model.sources.values()):
                if self._updatable_sources[source_num] == 1:
                    for pointing in range(len(combination)):
                        self._resp_mats[count][source_num][pointing][:,:,:] = generate_resp_mat(
                            self._updatable_rmfs[count][pointing],
                            len(self._dets[count]),
                            len(self._ebs[count]),
                            len(self._emod),
                            source.position.get_ra(),
                            source.position.get_dec(),
                        )

    def _find_updatable_sources(self):
        keywords = ["position"]
        self._updatable_sources = np.zeros(len(self._source_model.sources), np.int8)
        for s_i, source in enumerate(self._source_model.sources.values()):
            for parameter in source.free_parameters.values():
                first_pos = parameter.path.find(".")
                second_pos = parameter.path.find(".", first_pos+1)
                if parameter.path[first_pos+1 : second_pos] in keywords:
                    self._updatable_sources[s_i] = 1

    def parameter_fit_distribution(self, true_values=[]):
        assert not self._folder is None, "folder is not set"
        
        fig = self._cc.plotter.plot(
            parameters=self._parameter_names[:-1],
            figsize=1.5,
            truth=true_values
        )
        
        plt.savefig(f"{self._folder}/parameter_fit_distributions.pdf")
        plt.close()
        
    def text_summaries(
        self,
        reference_values=True,
        pointing_combinations=True,
        parameter_fit_constraints=True
    ):
        assert not self._folder is None, "folder is not set"
        
        
        if reference_values:
            assert not self._true_values is None, "true_values not set"
            summary = self._cc.analysis.get_summary(parameters=self._true_values[0])
            cov = self._cc.analysis.get_covariance(parameters=self._true_values[0])
            rel_distances = calc_mahalanobis_dist(summary, cov, self._true_values[1])
            
            with open(f"{self._folder}/reference_values", "w") as f:
                f.write(f"{' : '.join(self._true_values[0])} : Rel. Dist.\n")
                for i in range(self._true_values[1].shape[0]):
                    f.write(f"{' : '.join([f'{j:.3}' for j in self._true_values[1][i,:]])} : {rel_distances[i]:.3}\n")
                
        if pointing_combinations:
            with open(f"{self._folder}/pointing_combinations", "w") as f:
                for combination in self._pointings:
                    f.write(f'{"  ".join(i[0] for i in combination)}\n')
        
        if parameter_fit_constraints:
            summary = self._cc.analysis.get_summary(parameters=self._parameter_names[:-1])
            with open(f"{self._folder}/parameter_fit_constraints", "w") as f:
                for param in self._parameter_names[:-1]:
                    f.write(f"{param}:\n")
                    try:
                        f.write(f"{summary[param][0]:.5}  {summary[param][1]:.5}  {summary[param][2]:.5}\n")
                    except:
                        f.write(f"None  {summary[param][1]:.5}  None\n")
    
    def ppc( ## add check for cluster size = 2
        self,
        max_posterior_samples=300,
        count_energy_plots=False,
        qq_plots=False,
        rel_qq_plots=False,
        cdf_hists=True,
        energy_residual_plot=True,
    ):
        assert self._folder is not None, "folder is not set"
        
        # expected_counts, source_rate, background_rate, times = self._calc_expected_counts(max_posterior_samples)
        expected_counts = self._calc_expected_counts(max_posterior_samples)
        
        if cdf_hists:
            # number of bins divisble by 4
            xs = np.linspace(0,1,13)
            cdf_counts = np.zeros(len(xs)-1)
            if not os.path.exists(f"{self._folder}/cdf"):
                os.mkdir(f"{self._folder}/cdf")
                
        if energy_residual_plot:
            self._energy_residual_plot(expected_counts)
                        
        for c_i, combination in enumerate(self._pointings):
            for p_i in range(len(combination)):
            
                if count_energy_plots:
                    if not os.path.exists(f"{self._folder}/count_energy"):
                        os.mkdir(f"{self._folder}/count_energy")
                    self._count_energy_plot(
                        expected_counts[c_i][p_i],
                        self._ebs[c_i],
                        self._counts[c_i][p_i],
                        self._dets[c_i],
                        combination[p_i][0]
                    )
                if qq_plots:
                    if not os.path.exists(f"{self._folder}/qq"):
                        os.mkdir(f"{self._folder}/qq")
                    self._qq_plot(
                        expected_counts[c_i][p_i],
                        self._counts[c_i][p_i],
                        self._dets[c_i],
                        combination[p_i][0]
                    )
                if rel_qq_plots:
                    if not os.path.exists(f"{self._folder}/rel_qq"):
                        os.mkdir(f"{self._folder}/rel_qq")
                    self._rel_qq_plot(
                        expected_counts[c_i][p_i],
                        self._counts[c_i][p_i],
                        self._dets[c_i],
                        combination[p_i][0]
                    )
            if cdf_hists:
                cdf_counts += self._cdf_hist(
                    expected_counts[c_i],
                    self._counts[c_i],
                    combination,
                    xs,
                )
            
                    
        if cdf_hists:
            left_ratio = np.sum(cdf_counts[:int(len(cdf_counts)/2)]) / np.sum(cdf_counts)
            center_ratio = np.sum(cdf_counts[int(len(cdf_counts)/4):int(3*len(cdf_counts)/4)]) / np.sum(cdf_counts)
            self._cdf_plot([cdf_counts], xs, ["Total"], [left_ratio], [center_ratio])
                    
            
    def _calc_expected_counts(self, max_posterior_samples):
        
        print("Calculating Count Rates")
        

        source_rate = []
        background_rate = []
        
        b_range = [None, None]
        s_range = [None, None]
        t_range = [None, None]
        
        if len(self._chain) < max_posterior_samples:
            print(f"Using all {len(self._chain)} equal-weight posterior values.")
            max_posterior_samples = len(self._chain)
        posterior_samples = self._chain[
            np.random.choice(len(self._chain), max_posterior_samples, replace=False)
        ]
        
        for c_i, combination in enumerate(self._pointings):
            source_rate.append(np.zeros((len(combination), len(self._dets[c_i]), len(self._ebs[c_i])-1, len(posterior_samples))))
            background_rate.append(np.zeros((len(self._dets[c_i]), len(self._ebs[c_i])-1, len(posterior_samples))))
        
        num_sources = len(self._source_model.sources)
        
        for p_i, params in enumerate(posterior_samples):
            spec_binned = np.zeros((num_sources, len(self._emod)-1))
            for fp_i, parameter in enumerate(self._source_model.free_parameters.values()):
                parameter.value = params[fp_i]
            for s_i, source in enumerate(self._source_model.sources.values()):
                spec = source(self._emod)
                if self._source_spectrum_powerlaw_binning:
                    spec_binned[s_i,:] = powerlaw_binned_spectrum(self._emod, spec)
                else:
                    spec_binned[s_i,:] = (self._emod[1:]-self._emod[:-1])*(spec[:-1]+spec[1:])/2
            if 1 in self._updatable_sources:
                self._update_resp_mats()
            
            for c_i, combination in enumerate(self._pointings):
                for d_i in range(len(self._dets[c_i])):
                    for s_i in range(num_sources):
                        for m_i in range(len(combination)):
                            source_rate[c_i][m_i,d_i,:,p_i] += np.dot(spec_binned[s_i,:], self._resp_mats[c_i][s_i][m_i][d_i])
                            
                            if t_range[0] is None:
                                t_range = [np.amin(self._t_elapsed[c_i][m_i][d_i]), np.amax(self._t_elapsed[c_i][m_i][d_i])]
                            else:
                                min = np.amin(self._t_elapsed[c_i][m_i][d_i])
                                max = np.amax(self._t_elapsed[c_i][m_i][d_i])
                                if min < t_range[0]:
                                    t_range[0] = min
                                elif max > t_range[1]:
                                    t_range[1] = max
                            
                    for e_i in range(len(self._ebs[c_i])-1):
                        s_b = np.array([source_rate[c_i][i,d_i,e_i,p_i] for i in range(len(combination))])
                        t_b = np.array([self._t_elapsed[c_i][i][d_i] for i in range(len(combination))])
                        C_b = np.array([self._counts[c_i][i][d_i, e_i] for i in range(len(combination))])
                        if len(combination) == 2:
                            background_rate[c_i][d_i,e_i,p_i] = b_maxL_2(s_b, t_b, C_b)
                        elif len(combination) == 3:
                            background_rate[c_i][d_i,e_i,p_i] = b_maxL_3(s_b, t_b, C_b)
                            
                        if b_range[0] is None:
                            b_range = [background_rate[c_i][d_i,e_i,p_i], background_rate[c_i][d_i,e_i,p_i]]
                        else:
                            if background_rate[c_i][d_i,e_i,p_i] < b_range[0]:
                                b_range[0] = background_rate[c_i][d_i,e_i,p_i]
                            elif background_rate[c_i][d_i,e_i,p_i] > b_range[1]:
                                b_range[1] = background_rate[c_i][d_i,e_i,p_i]
                                
                if s_range[0] is None:
                    s_range = [np.amin(source_rate[c_i][:,:,:,p_i]), np.amax(source_rate[c_i][:,:,:,p_i])]
                else:
                    min = np.amin(source_rate[c_i][:,:,:,p_i])
                    max = np.amax(source_rate[c_i][:,:,:,p_i])
                    if min < s_range[0]:
                        s_range[0] = min
                    elif max > s_range[1]:
                        s_range[1] = max
               
        ### correct ppc         
        background_rate = tuple(background_rate)
        source_rate = tuple(source_rate)
        
        
        # b_int_funcs = (interpolate_linear, interpolate_linear, interpolate_logarithmic, interpolate_linear, interpolate_logarithmic)
        # c_int_funcs = (interpolate_logarithmic, interpolate_linear, interpolate_powerlaw, interpolate_linear, interpolate_logarithmic)
        # s_int_funcs = (interpolate_constant, interpolate_linear, interpolate_powerlaw, interpolate_constant, interpolate_constant)
        
        b_int_funcs = (interpolate_linear, interpolate_linear, interpolate_linear, interpolate_linear, interpolate_linear)
        c_int_funcs = (interpolate_linear, interpolate_linear, interpolate_linear, interpolate_linear, interpolate_linear)
        s_int_funcs = (interpolate_constant, interpolate_linear, interpolate_linear, interpolate_constant, interpolate_constant)
        
        b_num = 9
        s_num = 12
        t_num = 5
        
        input_b = np.geomspace(b_range[0]*0.999, b_range[1]*1.001, b_num)
        if s_range[0] == 0.0:
            input_s = np.geomspace(s_range[1]*0.005, s_range[1]*1.001, s_num-1)
            input_s = np.insert(input_s, 0, 0.0)
        else:
            input_s = np.geomspace(s_range[0]*0.999, s_range[1]*1.001, s_num)
        input_t = np.geomspace(t_range[0]*0.999, t_range[1]*1.001, t_num)
        
        
        dimension_values = (input_b, input_s, input_t, input_s, input_t)
        print("Matrix - B:")
        print(input_b)
        print("Matrix - S:")
        print(input_s)
        print("Matrix - T:")
        print(input_t)
        
        print("Generating Variance Matrix")
        
        variance_matrix = calc_bmaxL_variance_matrix(input_b, input_s, input_t, input_s, input_t)
        
        expected_counts = []
        
        print("Sampling Count Rates")
        
        for c_i, combination in enumerate(self._pointings):   
            expected_counts.append(sample_count_rates(
                c_i,
                source_rate,
                background_rate,
                posterior_samples,
                self._dets,
                self._ebs,
                self._t_elapsed,
                variance_matrix,
                dimension_values,
                b_int_funcs,
                c_int_funcs,
                s_int_funcs
            ))
                        
        print("Sampling Finished")
        ### end correct ppc
        
        # #### simple ppc
        # total_num_pointings = self._number_of_total_pointings()
        # expected_counts = []
        # for c_i, combination in enumerate(self._pointings):
        #     source_counts = np.zeros(source_rate[c_i].shape)
        #     background_counts = np.zeros(source_rate[c_i].shape)
            
        #     len_comb = len(combination)
        #     for m_i in range(len_comb):
        #         source_counts[m_i,:,:,:] = source_rate[c_i][m_i,:,:,:] * self._t_elapsed[c_i][m_i][:,np.newaxis,np.newaxis]
        #         background_counts[m_i,:,:,:] = background_rate[c_i][:,:,:] * self._t_elapsed[c_i][m_i][:,np.newaxis,np.newaxis]
                
        #     variance = (len_comb-1) / len_comb * background_counts + (total_num_pointings-1) / total_num_pointings * source_counts
            
        #     total_counts = source_counts + background_counts
            
        #     sampled_counts = np.random.normal(total_counts, np.sqrt(variance))
            
        #     expected_counts.append(sampled_counts)
        # #### end simple ppc

        # times = []
        # for c_i, combination in enumerate(self._pointings):
        #     background_rate[c_i] = np.repeat(background_rate[c_i][np.newaxis,:,:,:], len(combination), axis=0)
        #     t = np.zeros(source_rate[c_i].shape)
        #     for m_i in range(len_comb):
        #         t[m_i,:,:,:] = np.repeat(
        #             np.repeat(
        #                 self._t_elapsed[c_i][m_i][:,np.newaxis,np.newaxis], len(source_rate[c_i][m_i,0,:,0]), axis=1
        #             ), len(source_rate[c_i][m_i,0,0,:]), axis=2
        #         )
        #     times.append(t)
            

        
        return expected_counts #, source_rate, background_rate, times

    def _count_energy_plot(
        self,
        expected_counts,
        eb,
        c,
        dets,
        name
    ):
        xs = eb[:-1]
        ys = (expected_counts, np.average(expected_counts, axis=2), c)
        styles = (
            {"label":"Posterior Predictive", "c":"#17becf","lw":0.1, "alpha":0.3},
            {"label":"Posterior Predictive Mean", "c":"#1f77b4"},
            {"label":"Measured", "c":"k"}
        )
        legend_elements = [Line2D([0],[0], c=i["c"], label=i["label"]) for i in styles]
        
        fig, axes = self._detector_plots(
            dets,
            xs,
            ys,
            styles,
            {"xlabel":"Detected Energy [keV]"},
            {"ylabel":"Counts per Energy Bin", "labelpad":12},
            ylog=True,
            step_plot=True
        )
        
        fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.9, 0.5), fontsize='x-large')
        
        fig.savefig(f"{self._folder}/count_energy/{name}_count_energy.pdf")
        plt.close()
    
    def _qq_plot(
        self,
        expected_counts,
        c,
        dets,
        name
    ):
        xs = np.cumsum(c, axis=1)
        ys = (
            np.cumsum(expected_counts, axis=1),
            xs,
            np.cumsum(np.average(expected_counts, axis=2), axis=1)
        )
        styles = (
            {"label":"Posterior Predictive", "c":"#17becf","lw":0.1, "alpha":0.3},
            {"c":"k", "ls":"--"},
            {"label":"Posterior Predictive Mean", "c":"#1f77b4"},
        )
        legend_elements = [Line2D([0],[0], c=i["c"], label=i["label"]) for i in (styles[0],styles[2])]
        
        fig, axes = self._detector_plots(
            dets,
            xs,
            ys,
            styles,
            {"xlabel":"Cumulative Measured Counts"},
            {"ylabel":"Cumulative Predicted Counts", "labelpad":25}
        )
        
        fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.9, 0.5), fontsize='x-large')
        
        fig.savefig(f"{self._folder}/qq/{name}_qq.pdf")
        plt.close()
        
    def _rel_qq_plot(
        self,
        expected_counts,
        c,
        dets,
        name
    ):
        xs = np.cumsum(c, axis=1)
        ys = (
            np.cumsum(expected_counts, axis=1) / xs[:,:,np.newaxis],
            np.ones(xs.shape),
            np.cumsum(np.average(expected_counts, axis=2), axis=1) / xs
        )

        styles = (
            {"label":"Posterior Predictive", "c":"#17becf","lw":0.1, "alpha":0.3},
            {"c":"k", "ls":"--"},
            {"label":"Posterior Predictive Mean", "c":"#1f77b4"},
        )
        legend_elements = [Line2D([0],[0], c=i["c"], label=i["label"]) for i in (styles[0],styles[2])]
        
        fig, axes = self._detector_plots(
            dets,
            xs,
            ys,
            styles,
            {"xlabel":"Cumulative Measured Counts"},
            {"ylabel":"Cumulative Predicted Counts / Cumulative Measured Counts", "labelpad":15}
        )
        
        fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.9, 0.5), fontsize='x-large')
        
        
        fig.savefig(f"{self._folder}/rel_qq/{name}_rel_qq.pdf")
        plt.close()
        
    def _detector_plots(# does step need to be post????????????????????????????????????????????????????
        self,
        dets,
        xs,
        ys,
        styles,
        xlabel=None,
        ylabel=None,
        ylog=False,
        step_plot=False
    ):
        fig, axes = plt.subplots(5,4, sharex=True, sharey=True, figsize=(10,10))
        axes = axes.flatten()
        i=0
        for d in range(19):
            axes[d].text(.5,.9,f"Det {d}",horizontalalignment='center',transform=axes[d].transAxes)
            if d in dets:
                if step_plot:
                    plotting_func = axes[d].step
                else:
                    plotting_func = axes[d].plot
                for y_i in range(len(ys)):
                    if len(xs.shape) != 1:
                        plotting_func(xs[i], ys[y_i][i], **styles[y_i])
                    else:
                        plotting_func(xs, ys[y_i][i], **styles[y_i])
                
                i += 1
            if ylog == True:
                axes[d].set_yscale("log")
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.subplots_adjust(hspace=0, top=0.96, bottom=0.1)
        
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        if not xlabel is None:
            plt.xlabel(**xlabel)
        if not ylabel is None:
            plt.ylabel(**ylabel)
            
        return fig, axes
    
        
    def _cdf_hist(
        self,
        expected_counts,
        counts,
        combination,
        xs,
    ):
        cdfs = []
        names = []
        left_ratios = []
        center_ratios = []
        for p_i in range(len(combination)):
            
        
            argsort = np.argsort(expected_counts[p_i], axis=2)
            sorted_expected_counts = np.take_along_axis(expected_counts[p_i], argsort, axis=2)
            
            num_samples = sorted_expected_counts.shape[2]
            
            cdf = np.zeros(sorted_expected_counts.shape[:2])
            for d_i in range(sorted_expected_counts.shape[0]):
                for e_i in range(sorted_expected_counts.shape[1]):
                    cdf[d_i, e_i] = ((np.searchsorted(sorted_expected_counts[d_i,e_i], counts[p_i][d_i,e_i], "left")
                                    + np.searchsorted(sorted_expected_counts[d_i,e_i], counts[p_i][d_i,e_i], "right"))
                                    / (2 * num_samples))
                    
            cdf_counts, _ = np.histogram(cdf.flatten(), bins=len(xs)-1, range=(0,1))
            
            left_ratio = np.sum(cdf_counts[:int(len(cdf_counts)/2)]) / np.sum(cdf_counts)
            center_ratio = np.sum(cdf_counts[int(len(cdf_counts)/4):int(3*len(cdf_counts)/4)]) / np.sum(cdf_counts)
            
            cdfs.append(cdf_counts)
            names.append(combination[p_i][0])
            left_ratios.append(left_ratio)
            center_ratios.append(center_ratio)
            
        self._cdf_plot(cdfs, xs, names, left_ratios, center_ratios)
        
        total_cdf = cdfs[0]
        for i in cdfs[1:]:
            total_cdf += i
        
        return total_cdf
    
    def _cdf_plot(
        self,
        cdfs,
        xs,
        names,
        left_ratios,
        center_ratios
    ):
        num_plots = len(cdfs)
        fig, axes = plt.subplots(nrows=num_plots, figsize=(6, 1+3*num_plots))
        
        if num_plots == 1:
            axes = [axes]
        
        for i in range(num_plots):
            axes[i].bar(xs[:-1], cdfs[i], 1/(len(xs)-1), align="edge", color="#1f77b4")
            axes[i].axhline(np.average(cdfs[i]), color="black")
            if i == num_plots-1:
                axes[i].set_xlabel("Cumulative Probabilty")
            axes[i].set_ylabel("Counts per Bin")
            axes[i].set_title(f"{names[i]}   Center: {center_ratios[i]:.3f}   Left: {left_ratios[i]:.3f}", fontsize=10)
        
        fig.tight_layout()  
        fig.savefig(f"{self._folder}/cdf/{'_'.join(names)}_cdf.pdf")
        plt.close()
        
    def _energy_residual_plot(
        self,
        expected_counts,
    ):
        
        mean_expected_counts = np.zeros(self._counts[0][0][0,:].shape)
        expected_totals = np.zeros((len(self._counts[0][0][0,:]), len(expected_counts[0][0][0,0,:])))
        measured_totals = np.zeros(len(self._counts[0][0][0,:]))
        for c_i, combination in enumerate(self._pointings):
            assert len(self._counts[c_i][0][0, :]) == len(self._counts[0][0][0, :]), "PPC Residual: Energy Bins not constant in fit"
            for p_i in range(len(combination)):
                assert np.array_equal(self._dets[c_i], self._dets[0]), "PPC Residual: Active Detectors not constant in fit"
                for d_i in range(len(self._dets[c_i])):
                    expected_totals += expected_counts[c_i][p_i][d_i, :, :]
                    # print(measured_totals.shape)
                    # print(self._counts[c_i][p_i][d_i, :].shape)
                    # print(self._counts[c_i][p_i][d_i, :])
                    measured_totals += self._counts[c_i][p_i][d_i, :]
                    mean_expected_counts += np.mean(expected_counts[c_i][p_i][d_i, :, :], axis=1)
                    
        variances = np.var(expected_totals, axis=1)
        residual_devation = (measured_totals - mean_expected_counts) / np.sqrt(variances)
        
        fig, axes = plt.subplots(nrows=3, figsize=(7,9))
        
        styles = (
            {"label":"Posterior Predictive", "color":"#17becf","lw":0.1, "alpha":0.3},
            {"label":"Posterior Predictive Mean", "color":"#1f77b4", "lw":3},
            {"label":"Measured Counts", "color":"k"},
        )
        
        for i in range(len(expected_counts[0][0][0,0,:])):
            axes[0].stairs(expected_totals[:,i], self._ebs[0], **styles[0])
            
        axes[0].stairs(mean_expected_counts[:], self._ebs[0], **styles[1])
        
        axes[0].stairs(measured_totals[:], self._ebs[0], **styles[2])
        
        axes[1].axhline(y=0, ls="--", color="k")
        for i in range(len(expected_counts[0][0][0,0,:])):
            axes[1].stairs(expected_totals[:,i]/measured_totals[:]-1, self._ebs[0], **styles[0])
            
        axes[1].stairs(mean_expected_counts[:]/measured_totals[:]-1, self._ebs[0], **styles[1])
        
        # axes[0].stairs(measured_totals[:], self._ebs[0], **styles[2])
        
        axes[2].axhline(y=0, ls="--", color="k")
        axes[2].stairs(residual_devation[:], self._ebs[0], color="#1f77b4")
        
        
        axes[2].set_xlabel("Energy [keV]")
        axes[0].set_ylabel("Total Counts")
        axes[1].set_ylabel("$\\frac{\\mathrm{Posterior\;Predictive\;Counts}}{\\mathrm{Measured\;Counts}}-1$")
        axes[2].set_ylabel("Residuals [$\sigma$]")
        
        axes[0].set_yscale("log")
        axes[0].set_xscale("log")
        axes[1].set_xscale("log")
        axes[2].set_xscale("log")
                    
        
        legend_elements = [Line2D([0],[0], c=i["color"], label=i["label"]) for i in (styles)]
        # axes[0].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.9, 0.5), fontsize='x-large')
        axes[0].legend(handles=legend_elements)
        
        fig.savefig(f"{self._folder}/energy_residual_plot.pdf", bbox_inches='tight')
        
        
        
                

    def _extract_parameter_names_simple(self):
        self._parameter_names = []
        for full_name in self._source_model.free_parameters.keys():
            source = full_name[ : full_name.find(".")]
            source = source[1:] if source[0]=="_" else source
            source = source.replace("__", "+").replace("_", " ")
            parameter = full_name[-1 * full_name[::-1].find(".") : ]
            self._parameter_names.extend([f"{source} {parameter}"])
    
    def set_folder(self, folder):
        if not folder is None:
            if not os.path.exists(f"{folder}"):
                os.mkdir(folder)
        self._folder = folder

    def _number_of_total_pointings(self):
        n = 0
        for combination in self._pointings:
            n += len(combination)
        return n