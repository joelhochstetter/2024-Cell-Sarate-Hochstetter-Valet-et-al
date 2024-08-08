'''

This code runs a grid search over the parameters of the SD model to fit the control data.

Run command:
parallel --max-procs 60 python3 Cluster_Fit_Smodel_Ctrl.py ::: {0..60}

Then to combine results together, once we have finished running:
python Cluster_Fit_S_Ctrl.py -1 5 1 &
    
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
    ncores = 5

if len(sys.argv) > 3:
    mode = int(sys.argv[3])
else:
    mode = 0
    

if mode == 0:
    s0 = topt.optimizer(sim_model.S_mech_switch, {'r': np.arange(0.05, 0.51, 0.05), 'tau': [1000.0],
                                                'Ahigh': [2.0, np.inf], 'hilln': [0.25, 2.0, 10.0]},
                        params = {'L': 15, 'useTerm': True, 
                                'tadiv': 100}, simtype = 'h',
                    exp_file = '../../experiment/ablation.pkl',  exp_prefix = 'ctr', tshift = -5.0,
                    metrics = [{'cost': 'cost_prd', 'dist_metric': 'ksd', 'dist': ['basalCS', 'totalCS']}])

    s0.optimise(ncells = 225, tsteps = 10000, seeds = np.arange(5), savefolder = 'ctr_S_mech_switch/',
            init_params = {'relaxsteps': 2000, 'homeosteps': 5000}, sim_args = {'p0': 3.5},
            tscalevals = -np.arange(3,15,0.2), shedvals = -np.arange(0.2, 4.1, 0.1),
            use_saved_metrics = True, ncores = ncores, merge_during = (i > 0), save_every = 1, i = i,
            end_save = (i < 0), save_to_runspace = (i < 0), save_sweeps = (i < 0),
            nboots = (1000)*(i < 0))
    
elif mode == 1:
    s0 = topt.optimizer(sim_model.S_0D, {'r': np.arange(0.05, 0.51, 0.025)}, 
                        params = {'tau': 500.0}, simtype = '',
                exp_file = '../../experiment/ablation.pkl',  exp_prefix = 'ctr', tshift = -5.0,
                   metrics = [{'cost': 'cost_prd', 'dist_metric': 'ksd', 'dist': ['basalCS', 'totalCS']}])

    s0.optimise(ncells = 250, tsteps = 5000, seeds = np.arange(5), savefolder = 'ctr_S0D/',
            tscalevals = -np.arange(1.6,12,0.2), shedvals = -np.arange(0.2, 3.1, 0.1),
            use_saved_metrics = False, ncores = 5, merge_during = True, save_every = 5, i = i,
            end_save = (i < 0), save_to_runspace = (i < 0), save_sweeps = True,
            nboots = 1000)