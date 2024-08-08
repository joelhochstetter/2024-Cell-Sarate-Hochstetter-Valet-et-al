'''

This code runs a grid run over the parameters of the S model to compute different quantities of interest.

Run command:
parallel --max-procs 100 python Sweep_T1s.py ::: {0..159} &

Then to combine results together, once we have finished running:
python Sweep_T1s.py -1 &

'''



#Imports
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle


sys.path.append("../../../simulator/")
sys.path.append("../../../fitting/")

from tissue import *
import tissplot as tplt
import matplotlib.path as mplPath
import datasets as ds 
import tiss_optimize as topt
import sim_model




#If we pass in all parameters as a string
if len(sys.argv) == 2:
    sys.argv = [sys.argv[0]] + sys.argv[1].split()

#Varying
i = int(sys.argv[1])


s0 = topt.simulator(sim_model.S_mech_switch, {'p0': [3.25, 3.5, 3.75, 4.0, 4.25], 
                                              'tau': [100, 200, 400, 800, 1600, 3200, 6400, 12800],
                                              'rho_c': [0.5, 0.75, 1.0, 1.25]},
                    params = {'Ahigh': np.inf, 'hilln': 2.0, 'L': 25}, simtype = 'h',
                    order_params = [('ncells', 'mean'), ('ncells', 'std'),
                                    ('T1s', 'sum'), ('T1s', 'av_rate_per_cell'),
                                    ('MSD', 'all'), ('MSD', 'slope'),
                                    ('shape', 'mean')])


if i >= 0:
    s0.run_sim(ncells = 625, tsteps = 5000, seeds = np.arange(1), savefolder = 'ctr_S_mech_switch/',
           init_params = {'relaxsteps': 2000, 'homeosteps': 5000}, sim_args = {'save_shape': True},
           tscalevals = -1, use_saved_metrics = False, ncores = 1, i = i)
else:
    s0.merge_indexed_searchspace(np.arange(100), ncells = 625, tsteps = 5000, savefolder = 'ctr_S_mech_switch/')    