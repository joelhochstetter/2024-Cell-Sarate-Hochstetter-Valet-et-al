'''

This code runs a grid search over the parameters of the SD model to fit the control data.

Run command:
parallel --max-procs 120 python Cluster_Fit_SD_Ctrl.py ::: {0..240}

Then to combine results together, once we have finished running:
python Cluster_Fit_SD_Ctrl.py -1 5 1 &
    
'''



#Imports
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../../simulator/")
sys.path.append("../../fitting/")


from tissue import *
import tissplot as tplt
import matplotlib.path as mplPath
import datasets as ds 
import tiss_optimize as topt
import sim_model
import os




#If we pass in all parameters as a string
if len(sys.argv) == 2:
    sys.argv = [sys.argv[0]] + sys.argv[1].split()

#Varying
i = int(sys.argv[1])

if len(sys.argv) > 2:
    ncores = int(sys.argv[2])
else:
    ncores = 1
    
if len(sys.argv) > 3:
    mode = int(sys.argv[3])
else:
    mode = 1
    
    
print(i, ncores, mode)
    
if mode == 0:
    savefolder = 'ctr_SD0D/'
    s0 = topt.optimizer(sim_model.SD_0D, {'r': np.arange(0.05, 0.51, 0.05), 'rho': np.hstack([np.arange(0.1, 1.0, 0.05), np.arange(0.96,0.991,0.01)])},
                        params = {'tau': 500.0}, simtype = '',
                    exp_file = '../../experiment/ablation.pkl',  exp_prefix = 'ctr', tshift = -5.0,
                    metrics = [{'cost': 'cost_prd', 'dist_metric': 'ksd', 'dist': ['basalCS', 'totalCS']}])
    
elif mode == 1:
    savefolder = 'ctr_SD_mech_switch/'
    s0 = topt.optimizer(sim_model.SD_mech_switch, 
                        {'r': np.arange(0.05, 0.55, 0.05), 
                         'tau': [1000.0],
                         'hilln': [0.25, 2.0, 10.0], 
                         'rho': np.arange(0.3, 1.01, 0.1)},
                    params = {'L': 20, 'useTerm': True, 
                              'tadiv': 100, 'Ahigh': np.inf}, simtype = 'h',
                exp_file = '../../experiment/ablation.pkl',  exp_prefix = 'ctr', tshift = -5.0,
                   metrics = [{'cost': 'cost_sum', 'dist_metric': 'ksd', 'dist': ['basalCS', 'totalCS']},
                              {'cost': 'cost_prd', 'dist_metric': 'ksd', 'dist': ['basalCS', 'totalCS']}],
                   mergefirst = False)


if i >= 0:
    s0.optimise(ncells = 400, tsteps = 7500, seeds = np.arange(5), savefolder = savefolder,
            init_params = {'relaxsteps': 2000, 'homeosteps': 5000}, sim_args = {'p0': 3.5},
            tscalevals = -np.arange(2,10,0.2), shedvals = -np.arange(0.2, 3.1, 0.1),
            use_saved_metrics = False, ncores = ncores, merge_during = False, save_every = 1, i = i,
            end_save = True, save_sweeps = True,
            nboots = 100, save_to_runspace = False)
    
elif i == -1:
    s0.merge_indexed_searchspace(np.arange(241), ncells = 400, tsteps = 7500, savefolder = savefolder)