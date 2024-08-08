#Imports
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

sys.path.append("../../simulator/")
sys.path.append("../../fitting/")

from tissue import *
import tissplot as tplt
import matplotlib.path as mplPath
import analysis
import datasets as ds 
import tiss_optimize as topt
import sim_model
import os


'''
Run command:

parallel --max-procs 80 python Cluster_Fit_SD_model_Exp.py ::: {0..15}



To get all the fits together set i = -1: python3 Cluster_Fit_SD_model_Exp.py -1
    
'''


#If we pass in all parameters as a string
if len(sys.argv) == 2:
    sys.argv = [sys.argv[0]] + sys.argv[1].split()

#Varying
i = int(sys.argv[1])

if len(sys.argv) > 2:
    ncores = int(sys.argv[2])
else:
    ncores = 5

def death(t, tscale = 1, dshift = -1, tmin = -3, b1 = 3.2553e-2, b2 = 8.7969e-1, bx = 0.5240):
    t = t/tscale + dshift
    if type(t) == int or type(t) == float:
        t = np.array([t])
    y = np.zeros(len(np.array(t)))
    y[t >= 0] = np.exp(-bx*t[t >= 0])
    y[t  < 0] = np.exp(b1*(t[t < 0] - tmin))*(1 - np.exp(-b2*(t[t < 0] - tmin)))/np.exp(-tmin*b1)/(1 - np.exp(tmin*b2))
    y[t < tmin] = 0.0
    return y


s0 = topt.sim_anneal(sim_model.SD_mech_switch, 
                     {'deathscale': [-0.95238095], 'Ahigh': [1.7, 1.3], 'tadiv': [30, 15], 'r': [0.2, 0.3],
                      'hilln': [0.625, 10.0]}, 
                     params = {'tau': 500.0, 'L': 15, 'tscale': 65, 'rho': 0.9}, simtype = 'a',
                exp_file = '../../experiment/ablation.pkl',  exp_prefix = 'exp', tshift = -5.0,
                   metrics = [{'cost': 'cost_prd', 'dist_metric': 'ksd', 
                               'dist': ['basalCS'], 'exclude_times': {'basalCS': [0,1,2]},
                               'quantity': ['density', 'divrate'], 'qmetric': 'pls'}],
                   move_seed = i, use_linear = ['Ahigh', 'tadiv', 'r'],
                   move_sizes     = {'deathscale': 1.05, 'Ahigh': 0.05, 'tadiv': 5, 'r': 0.05, 'hilln': 2}, 
                   free_param_min = {'deathscale': -3.0,  'Ahigh': 1.0, 'tadiv': 5, 'r': 0.05, 'hilln': 0.1}, 
                   free_param_max = {'deathscale': -0.5,  'Ahigh': 2.2, 'tadiv': 150, 'r': 0.55, 'hilln': 10.0})


if i >= 0:
    s0.runSA(ncells = 225, tsteps = 1500, seeds = np.arange(5), savefolder = 'exp_SD0D/',
            tscalevals = 65, shedvals = None,
            use_saved_metrics = False, ncores = ncores, save_every = 5, mergefirst = False, 
            SAsteps = 100, T0 = 0.01, start_idx = i,
            sim_args = {'p0': 3.5}, norm_by_control = ['density', 'divrate'], 
            init_params = {'relaxsteps': 1500, 'homeosteps': 2000, 'ctrsteps': 2000, 
                            'kill_fn': death, 'kill_params': {'dshift': -6}})


if i < 0:
    s0.merge_seeded_searchspace(move_seeds = np.arange(16), ncells = 225, tsteps = 1500, savefolder = 'exp_SD0D/')