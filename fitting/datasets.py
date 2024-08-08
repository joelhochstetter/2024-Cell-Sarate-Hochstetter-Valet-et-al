import os
import pickle
import numpy as np
import copy
import analysis
import tissue
from ds_plot import *
import clones
import shape_analysis as sa

#Preamble: defines the types of data outlined below

#Data types currently implemented
categories = ['basalCS', 'totalCS', 'supraCS', 'density', 'divrate', 'pers', 
              'lcd', 'death', 'T1s', 'MSD', 'shape']

shortname = {'basalCS': 'bas', 'totalCS': 'tot', 'supraCS': 'sup', 'density': 'dens', 
             'divrate': 'dr', 'pers': 'pers', 'lcd': 'lcd', 'death': 'dth'}


#Whether the quantities are skipped in simulation
skipped = {'basalCS': True, 'totalCS': True, 'supraCS': True, 
           'density': False, 'divrate': False, 'pers': True, 'lcd': True, 'death': True,
           'T1s': True, 'MSD': True, 'shape': True}

#Categories which affected by cells in the suprabasal layer
supra_cats = ['totalCS', 'supraCS']
    

class dataset():
    '''
        Dataset type stores:
            quantity: is a quantity e.g. average size
               dists: is a probability distribution, stored as a pdf form
             For distributions, we also store the number of samples 
                and dist_bins, if dist_bins = None then we 
                    assume the bins are 0,1,2,...
                To store continuous data len(dist_bins) = len(dist) + 1
                To store discrete data len(dist_bins) = len(dist)
             
        Data can be seeded or non-seeded, seeded data has the 0-th index corresponding to the seed
            or index of the biological replicate
    '''
    
    def __init__(self, toCalculate = [], max_pdf_bin = None, default_pdf_max = 100):
        '''
            Create base datasets, quantities, distributions and times associated with indices
            we also have seeds which allows the instantiation of simulations or biological 
            replicates, seeds
            
            Stores extra-dimensions and names which might be different for different variables
            Defaults are for tscales and suprabasal
            
        '''
        
        self.toCalculate = toCalculate
        self.check_valid_cats()
        self.quantities = {} #quantities at each time-point
        self.dists = {} #probability distributions,
        self.dist_bins = {}
        self.joint_dists = {} #joint probability distributions
        self.joint_bins  = {} #joint bins
        self.nsamples = {} #number of samples at each time-point
        self.max_pdf_bin = {}
        self.times = {}
        self.nseeds = 0
        self.added_seeds = set()
        self.type = 'ds'
        self.se_quants  = {}       
        
        if max_pdf_bin is None:
            max_pdf_bin = {}
        
        self._max_pdf_bin = max_pdf_bin
        self.default_pdf_max = default_pdf_max
         
        
        #Flags: important  for behaviour of how it operates
        self.collapsed = True #flag indicating whether we have a grid over additional dimensions
        self.seeded = False

    
    def check_valid_cats(self):
        for tc in self.toCalculate:
            assert(tc in categories), 'Invalid quantity specified ' + tc
            
    @property
    def max_pdf_bin(self):
        for d in self.dists.keys():
            if d not in self._max_pdf_bin:
                self._max_pdf_bin[d] = self.default_pdf_max
        return self._max_pdf_bin            
        
    def get_max_pdf_bin(self, d):
        if d not in self._max_pdf_bin:
            self._max_pdf_bin[d] = self.default_pdf_max
        return self._max_pdf_bin[d]
        
    @max_pdf_bin.setter
    def max_pdf_bin(self, value):
        self._max_pdf_bin = value
        
    def sample(self, d, t = 0, seed = None):
        '''
            Returns samples for a given distribution at a given time-point
                indexed by t for tscale-idx 0

            Defaults to: returning combining data across all seeds
            Alternatively can specify a seed
            
            Does not work with data with extra-dimensions
            
        '''
                            
        if self.seeded is True:
            if seed is None: #use combined over sims
                N = np.sum(self.nsamples[d][:,t],0)
                hist_counts = (np.average(self.dists[d][:, t, :], weights=self.nsamples[d][:, t], axis=0) * N).astype(int)
            else:
                N = self.nsamples[d][seed,t]
                hist_counts = (self.dists[d][seed, t, :]*N).astype(int)         
        else:
            N = self.nsamples[d][t]
            hist_counts = (self.dists[d][t, :]*N).astype(int)
            
        return np.repeat(np.arange(len(hist_counts)), hist_counts)
        
        
    
    def timeq(self, q):
        return self.times[q]
    
    def all_quantities(self):
        return {**self.quantities, **self.dists} #, **self.samples}
    
    def copy(self):
        return copy.deepcopy(self)       
    
    def load_data_file(self, filename, prefix = 'ctr', suffix = ''):
        self.filename   = filename.split('/')[-1].split('.pkl')[0]
        self.folder = os.path.dirname(filename)  
        if self.folder == '':
            self.folder = '.'
        data_full = load_object(self.folder + '/' + self.filename + '.pkl')
        return data_full


    def max_time(self):
        ''' 
            Returns maximum time
        '''
        
        mt = -np.inf
        for q in self.times.keys():
            mt1 = np.max(self.times[q])
            if mt1 > mt:
                mt = mt1
        return mt


    def min_inc(self):
        ''' 
            Returns the minimum time increment used in the data
        '''
        
        mt = np.inf
        for q in self.times.keys():
            if len(self.times[q]) > 1:
                mt1 = np.min(self.times[q][1:] - self.times[q][:-1])
                if mt1 < mt:
                    mt = mt1
        return mt


    def reshape_pdf(self, d, max_bin = None, zeropad = 2, update = True):
        '''
            Re-shapes an individual pdf, by following reshape_pdfs
        '''
        
        newdists = self.reshape_pdfs(zeropad, max_bin = {d: max_bin}, dists_to_reshape = [d], update = update)
        return newdists[d]
        

    def reshape_pdfs(self, zeropad = 2, max_bin = None, dists_to_reshape = None, update = True):
        '''
            Re-shapes pdfs to ensure they have consistent sizes to compare with simulation
            
            The default is re-shape so the bins go up to maximum size + zeropad 
            
            Then simulation clones larger than experimental clones can go in the largest bin
            
            Reshapes pdfs to zeropad, else can force to max_bin
            
            Assumes the dataset is not seeded
            
            If dists_to_reshape is provided then we re-shape specific dists, 
                otherwise we re-shape all
            
            If update = True, we save the data
            
        '''
        
        if dists_to_reshape is None:
            dists_to_reshape = self.dists.keys()
            
        s = int(self.seeded) + 1
        
        newdists = {}
        
        for d in dists_to_reshape:
            if max_bin is not None and d in max_bin.keys():
                max_pdf_bin  = max_bin[d]  
            else:
                max_pdf_bin = np.nonzero(np.sum(self.dists[d],0))[0][-1] + zeropad
            
            dist = self.dists[d]
            if np.shape(dist)[s] >= (max_pdf_bin + 1):
                #Put the tail into the last datapoint
                dist[..., max_pdf_bin] = np.sum(dist[..., max_pdf_bin:], -1)
                dist = dist[...,:(max_pdf_bin + 1)]           
                        
            else:
                if self.seeded:
                    newpdf = np.zeros((self.nseeds, self.ntimes_by_quantity(d), max_pdf_bin + 1))
                else:
                    newpdf = np.zeros((self.ntimes_by_quantity(d), max_pdf_bin + 1))                    
                newpdf[...,:np.shape(dist)[s]] = dist
                dist = newpdf
            
            newdists[d] = dist
            
            if update is True:
                self._max_pdf_bin[d] = max_pdf_bin
                self.dists[d] = dist
                            
        return newdists
 
                
           
    def add_seeded_dataset(self, data1):
        '''
            Merge data-file with self, replacing the seeds from self with those from data1
        '''
        assert(self.seeded == True)
        
        self.extinct = self.extinct or data1.extinct
        
        for s in data1.added_seeds:
            for q in self.quantities.keys():
                if data1.seeded is True:
                    self.quantities[q][s,...] = data1.quantities[q][s,...]
                else:
                    self.quantities[q][s,...] = data1.quantities[q]                   
                    
            for d in self.dists.keys():
                if data1.seeded is True:                
                    self.dists[d][s,...] = data1.dists[d][s,...] 
                    self.nsamples[d][s,...] = data1.nsamples[d][s,...]
                else:
                    self.dists[d][s,...] = data1.dists[d]
                    self.nsamples[d][s,...] = data1.nsamples[d]                   
                   
            if hasattr(self, 'pop_quantities'):
                for q in self.pop_quantities.keys():
                    if data1.seeded is True:                                
                        self.pop_quantities[q][s,...] = data1.pop_quantities[q][s,...]
                    else:
                        self.pop_quantities[q][s,...] = data1.pop_quantities[q]
            
            if hasattr(self, 'pop_dists'):
                for d in self.pop_dists.keys():
                    if data1.seeded is True:                                                
                        self.pop_dists[d][s,...] = data1.pop_dists[d][s,...]
                        self.pop_nsamples[d][s,...] = data1.nsamples[d][s,...]
                    else:
                        self.pop_dists[d][s,...] = data1.pop_dists[d]  
                        self.pop_nsamples[d][s,...] = data1.pop_nsamples[d]                  
                    
        return self
                
                
    def ntimes_by_quantity(self, quantity):
            '''
                Extracts the numbers of times that are saved
            '''
 
            return len(self.times[quantity])  
        
       
    def seed_dataset(self, nseeds = 1):
        ''' 
            Takes existing datasets and makes quantities and distributions seeded
                Useful for bootstrapping, etc.
                
            Note this will alter the current dataset
        '''
        
        if self.seeded is False:
            self.seeded = True
            self.nseeds = nseeds
            
            for q in self.quantities.keys():
                self.quantities[q] = np.tile(np.array(self.quantities[q]), (nseeds, *self.quantities[q].ndim *(1,)))
                
            for d in self.dists.keys():
                self.dists[d] = np.tile(np.array(self.dists[d]), (nseeds, *self.dists[d].ndim *(1,)))
                self.nsamples[d] = np.tile(np.array(self.nsamples[d]), (nseeds, *self.nsamples[d].ndim *(1,)))
                
            if hasattr(self, 'pop_quantities'):                
                for q in self.pop_quantities.keys():
                    self.pop_quantities[q] = np.tile(np.array(self.pop_quantities[q]), (nseeds, *self.pop_quantities[q].ndim *(1,)))
            
            if hasattr(self, 'pop_dists'):
                for d in self.pop_dists.keys():
                    self.pop_dists[d] = np.tile(np.array(self.pop_dists[d]), (nseeds, *self.pop_dists[d].ndim *(1,)))
                    self.pop_nsamples[d] = np.tile(np.array(self.pop_nsamples[d]), (nseeds, *self.pop_nsamples[d].ndim *(1,)))
     
     
    def bootstrap(self, nboots = 100, seed = None):
        '''
            Returns a bootstrapped dataset:
                Currently only works for basal, supra and total clonal size
        '''
        
        if seed is not None:
            np.random.seed(seed)
            print('Setting seed to ' + str(seed) + ' for bootstrapping')

        assert(self.seeded is False)
        assert(self.collapsed is True)
                
        boot_data = self.copy()
        boot_data.seed_dataset(nboots)
        
        if 'basalCS' in self.toCalculate:
            for t in range(len(self.times['basalCS'])):
                sample = np.random.choice(np.arange(self.get_max_pdf_bin('basalCS') + 1), size=(nboots, self.nsamples['basalCS'][t]), p = self.dists['basalCS'][t,:])
                boot_data.quantities['basalCS'][:, t] = np.mean(sample, 1)
                for b in range(nboots):
                    boot_data.dists['basalCS'][b, t, :] = getPdf(sample[b, :], self.max_pdf_bin['basalCS'])

        if 'totalCS' in self.toCalculate:
            for t in range(len(self.times['totalCS'])):
                sample = np.random.choice(np.arange(self.get_max_pdf_bin('totalCS') + 1), size=(nboots, self.nsamples['totalCS'][t]), p = self.dists['totalCS'][t,:])
                boot_data.quantities['totalCS'][:, t] = np.mean(sample, 1)
                for b in range(nboots):
                    boot_data.dists['totalCS'][b, t, :] = getPdf(sample[b,:], self.max_pdf_bin['totalCS'])
                    
        if 'supraCS' in self.toCalculate:
            for t in range(len(self.times['supraCS'])):
                sample = np.random.choice(np.arange(self.get_max_pdf_bin('supraCS') + 1), size=(nboots, self.nsamples['supraCS'][t]), p = self.dists['supraCS'][t,:])
                boot_data.quantities['supraCS'][:, t] = np.mean(sample, 1)
                for b in range(nboots):
                    boot_data.dists['supraCS'][b, t, :] = getPdf(sample[b,:], self.max_pdf_bin['supraCS'])

        return boot_data    
    
    
    
    def bootstrap_SEs(self, d, exp_nsamples, nboots = 100, seed = None, sedists = False):
        '''
            Bootstraps standard errors of distribution, 
                this only is valid for unseeded data
                
                saves to standard error of quantity 
                sedists = True: returns the standard error of the distribution
                    else returns the standard error of the quantity
                
            Inputs:
                d: distribution to bootstrap
                exp_nsamples: number of samples to use from distribution, use the experimental ons
                nboots: number of bootstrap samples to take
                seed: seed for reproducibility of bootstrapping
        '''
        
        if exp_nsamples is None:
            exp_nsamples = self.nsamples[d]
            print('No samples provided, using simulated')
        
        if seed is not None:
            np.random.seed(seed)
            print('Setting seed to ' + str(seed) + ' for bootstrapping')        
        
        assert(self.seeded is False)
        assert(self.collapsed is True)
        
        if d not in self.dists: #do not if not a distribution
            return
        
        se_dists = np.zeros((self.ntimes_by_quantity(d), self.max_pdf_bin[d] + 1))
        assert(np.shape(se_dists) == np.shape(self.dists[d])), 'dists and se_dists have same shapes'
        self.se_quants[d] = np.zeros(self.ntimes_by_quantity(d))
        assert(np.shape(self.se_quants[d]) == np.shape(self.quantities[d])), 'quantities and se_quants have same shapes'
                
        for t in range(self.ntimes_by_quantity(d)): 
            sample = np.random.choice(np.arange(self.max_pdf_bin[d] + 1), 
                                      size=(nboots, exp_nsamples[self.closest_exp_time(d, t)]), p = self.dists[d][t,:])
            if sedists is True:
                boot_dists = np.zeros((nboots, self.max_pdf_bin[d] + 1))
                for bi in range(nboots):
                    boot_dists[bi,:] = getPdf(sample[bi,:], self.max_pdf_bin[d])
                se_dists[t,:] = np.std(boot_dists, 0)
            self.se_quants[d][t] = np.std(np.mean(sample,1),0)
            
        if sedists is True:
            return se_dists
        else:
            return self.se_quants[d]        
    
    
    
    def subdataset(self, seed):
        '''
            Returns an unseeded dataset object, by taking quantities/dists from the s-th seed
        '''         
        
        new_data = self.copy()
        new_data.seeded = False
        new_data.seeds = None
    
        for q in new_data.quantities.keys():
            new_data.quantities[q] = new_data.quantities[q][seed,...]
            
        for d in new_data.dists.keys():
            new_data.dists[d] = new_data.dists[d][seed,...]
            new_data.nsamples[d] = new_data.nsamples[d][seed,...]
            if hasattr(new_data.se_quants, d):
                new_data.se_quants[d] = new_data.se_quants[d][seed,...]
        
        if hasattr(new_data, 'pop_quantities'):
            for q in new_data.pop_quantities.keys():
                new_data.pop_quantities[q] = new_data.pop_quantities[q][seed,...]
            
        if hasattr(new_data, 'pop_dists'):
            for d in new_data.pop_dists.keys():
                new_data.pop_dists[d] = new_data.pop_dists[d][seed,...]
                new_data.pop_nsamples[d] = new_data.pop_nsamples[d][seed,...]
        

        return new_data        
                   
    
    def closest_exp_time(self, quantity, t):
        '''  
            Default behaviour for closest experimental time is to return the same time
        '''
        return t      
    
    
    
class sim_dataset(dataset):
    '''
        Object for interacting with tissue simulations, 
            can have quantities which relate to suprabasal layer only
            
        Inputting negative tscalevals or shedvals re-scales by characteristic scale,
            so we can specify simulation scale times in physical days per characteristic 
            division to obtain times in sim units
        
        Encapsulated with this we have the ability to re-run simulations of the 
            suprabasal layer
            
            
    '''
    def __init__(self, toCalculate, exp_times, tsteps, seeds = None, skip = 10, npops = 1, tscalevals = 1, shedvals = None, 
                            max_pdf_bin = None, alltimes = False, save_results = False, tshift = 0,
                            sim_params = None, norm_by_control = [], ref_scale = 'tau', default_pdf_max = 100):
        #Set-up
        dataset.__init__(self, toCalculate, max_pdf_bin = max_pdf_bin, default_pdf_max = default_pdf_max)
        self.seeds = seeds
        self.nseeds = len(seeds)
        self.type = 'sim'
        self.times = exp_times #experimental times we use
        self.set_default_times()
        
        if sim_params is None:
            sim_params = {}
        self.sim_params  = sim_params
        
        #Set-up control values (must be scalar)
        self.norm_by_control = norm_by_control #specifies which list of quantities to normalise by average ctrl values
        self.ctrl_vals = {}
                
        #Flags: important  for behaviour of how we operate
        self.seeded = True #indicates we are saving data in seeded format, default
        self.collapsed = False #flag indicating whether we have a grid over additional dimensions
        self.alltimes = alltimes #flag to indicate whether we save data at all or just specific times
        self.extinct = False #flag for extinction
        
        #Simulation time-saving
        self.tsteps = tsteps
        self.skip   = skip
        self.tshift = tshift
        
       #Values if we allow variation of shedding parameter and tscales:
        #These are fixed
        if tscalevals is None: 
            tscalevals = [1]
        elif np.ndim(tscalevals) == 0:
            tscalevals = [tscalevals]                
        self.tscalevals = np.array(tscalevals)
        self.manage_tscale(ref_scale)
        
        extradims = (len(self.tscalevals),)
        if shedvals is None or len(shedvals) == 0:
            shedvals = [0]
        self.shedvals   = shedvals
        supradims = (len(self.shedvals),)        
        
        #Make pop quantities:
        self.npops = npops
        self.pop_quantities = {} #quantities at each time-point
        self.pop_dists = {} #probability distributions
        self.pop_nsamples = {} #number of samples at each time-point for each population
        self.se_pop_quants = {} #standard error in population quantities
        
        #Saving results data as well
        self.save_results = save_results
        if self.save_results is True:
            self.results = {}
         
        #Set-up simulation quantities
        if seeds is not None:
            self.init_seeded_quantities(seeds, npops, extradims, supradims)
            
            
    def set_default_times(self):
        '''
            Sets default times to be time = 0
        '''
        for q in self.toCalculate:
            if q not in self.times:
                self.times[q] = np.array([0])
            
            
    def manage_tscale(self, ref_scale = 'tau', round = 2):
        '''
            Allows ensuring that there are no invalid t-scale values provided.
            
            If any tscale values are negative, then we re-scale by the reference 
                scale which defaults to the average division time
                
            We also save the transformed indices of tscalevals to the old indices
        '''
        
        #Re-scaling negative values by reference scale
        if np.min(self.tscalevals) < 0:
            if ref_scale in self.sim_params:
                self.tscalevals[self.tscalevals < 0] = self.sim_params[ref_scale]/self.tscalevals[self.tscalevals < 0]
            else:
                print('No reference scale provided, defaulting to 1')
                
        self.tscalevals = np.abs(self.tscalevals)
        idxs = np.argsort(self.tscalevals)
        self.tscalevals = self.tscalevals[idxs]

        #Ensuring values fall within the valid range
        minInc = self.min_inc()
        T = self.max_time() - self.tshift
        tscalemin = self.skip/minInc
        tscalemax = self.tsteps/T
        
        #Set conversions to the old-tscales
        idxs = idxs[(self.tscalevals >= tscalemin) & (self.tscalevals <= tscalemax)]
        self.new_to_old_tscidx = idxs
        
        #Set the new tscalevals
        self.tscalevals = self.tscalevals[(self.tscalevals >= tscalemin) & (self.tscalevals <= tscalemax)]
        self.tscalevals = np.round(self.tscalevals, round)
        
        
        return self.tscalevals
        
    


    def set_quantity(self, q, values, s = 0, ti = 0, sbsi = 0):
        '''
            Setting quantity, handling indices safely depending on whether seeded or suprabasal
                for a certain value of tscale and suprabasal shed (if required)
            
            We normalise by control if set to do so
        '''
                
        if self.seeded is True:
            assert(np.shape(self.quantities[q])[1] == len(values)) 
            if q in supra_cats:
                self.quantities[q][s, :, ti, sbsi] = values
                #Normalise by control
                if q in self.norm_by_control:
                    self.quantities[q][s, :, ti, sbsi] /= self.ctrl_vals[q]                  
            else:
                self.quantities[q][s, :, ti] = values
                #Normalise by control
                if q in self.norm_by_control:
                    self.quantities[q][s, :, ti] /= self.ctrl_vals[q]                  
                         
        else:
            assert(np.shape(self.quantities[q])[0] == len(values)) 
            if q in supra_cats:
                self.quantities[q][:, ti, sbsi] = values
                #Normalise by control
                if q in self.norm_by_control:
                    self.quantities[q][:, ti, sbsi] /= self.ctrl_vals[q]                
            else:
                self.quantities[q][:, ti] = values
                #Normalise by control
                if q in self.norm_by_control:
                    self.quantities[q][:, ti] /= self.ctrl_vals[q]                
            
        
        return values
    
    
    def set_dist(self, d, values, s = 0, ti = 0, sbsi = 0):
        '''
            Setting distribution, handling indices safely depending on whether seeded or suprabasal
                for a certain value of tscale and suprabasal shed (if required)
        '''
        
        if self.seeded is True:
            assert(np.shape(self.dists[d])[1] == len(values)) 
            if d in supra_cats:
                self.dists[d][s, :, :, ti, sbsi] = values
            else:
                self.dists[d][s, :, :, ti] = values
        else:
            assert(np.shape(self.dists[d])[0] == len(values)) 
            if d in supra_cats:
                self.dists[d][:, :, ti, sbsi] = values
            else:
                self.dists[d][:, :, ti] = values
        return values
    
    def set_nsamples(self, c, values, s = 0, ti = 0):
        ''' 
            Sets number of samples for a given category
        '''
        assert(np.sum(values) > 0)
        if self.seeded is True:
            assert(np.shape(self.nsamples[c])[1] == len(values)) 
            self.nsamples[c][s, :, ti] = values
        else:
            assert(np.shape(self.nsamples[c])[0] == len(values)) 
            self.nsamples[c][:, ti] = values    
    
    def set_pop_nsamples(self, d, values, s = 0, ti = 0):
        '''
            Sets population number of samples for a given distribution
        '''
        if self.seeded is True:
            assert(np.shape(self.pop_nsamples[d])[1] == np.shape(values, 0)) 
            assert(np.shape(self.pop_nsamples[d])[2] == np.shape(values, 1)) 
            self.pop_nsamples[d][s, :, :, ti] = values
        else:
            assert(np.shape(self.pop_nsamples[d])[0] == np.shape(values, 0)) 
            assert(np.shape(self.pop_nsamples[d])[1] == np.shape(values, 1)) 
            self.pop_nsamples[d][:, ti] = values
        
    
    def set_pop_quantity(self, q, values, s = 0, ti = 0, sbsi = 0):
        '''
            Setting population quantity, handling indices safely depending on whether seeded or suprabasal
                for a certain value of tscale and suprabasal shed (if required)
        '''
                
        if self.seeded is True:
            assert(np.shape(self.pop_quantities[q])[1:3] == np.shape(values)) 
            if q in supra_cats:
                self.pop_quantities[q][s, :, :, ti, sbsi] = values
            else:
                self.pop_quantities[q][s, :, :, ti] = values
        else:
            assert(np.shape(self.pop_quantities[q])[0:2] == np.shape(values)) 
            if q in supra_cats:
                self.pop_quantities[q][:, :, ti, sbsi] = values
            else:
                self.pop_quantities[q][:, :, ti] = values
        return values    
        
        
    def set_pop_dist(self, q, values, s = 0, ti = 0, sbsi = 0):
        '''
            Setting population distributions, handling indices safely depending on whether seeded or suprabasal
                for a certain value of tscale and suprabasal shed (if required)
        '''
                
        if self.seeded is True:
            assert(np.shape(self.pop_dists[q])[1] == np.shape(values)) 
            if q in supra_cats:
                self.pop_dists[q][s, :, :, :, ti, sbsi] = values
            else:
                self.pop_dists[q][s, :, :, :, ti] = values
        else:
            assert(np.shape(self.pop_dists[q])[0:2] == np.shape(values)) 
            if q in supra_cats:
                self.pop_dists[q][:, :, :, ti, sbsi] = values
            else:
                self.pop_dists[q][:, :, :, ti] = values
        return values   
                
    def all_quantities(self):
        return {**self.quantities, **self.dists, #**self.samples, 
                **self.pop_quantities, **self.pop_dists}# , **self.pop_samples}
    
            
    def ntimes_by_quantity(self, quantity):
        '''
            Extracts the numbers of times that are saved
        '''
        
        if self.alltimes is True:
            if skipped[quantity] is True:
                ntimes = int(np.ceil((self.tsteps + 1)/self.skip))
            else:
                ntimes = self.tsteps + 1                
        else:
            ntimes = len(self.times[quantity])
            
        return ntimes
    
    
    def timeslice_by_quantity(self, quantity, tscale, tshift, alltimes = None):
        '''
            Time-slice for selecting from time-vector of quantity
        '''
        
        if alltimes is None:
            alltimes = self.alltimes        
        
        if alltimes is False:
            if skipped[quantity] is True:
                tslice = exp2simSkippedTimes(self.times[quantity], tscale, -tshift, self.skip)
            else:
                tslice = exp2simTimes(self.times[quantity], tscale, -tshift)
        else:
            tslice = slice(None,None,None)
            
        return tslice        
    
    
    def tvals_by_quantity(self, quantity, tscale, tshift, alltimes = None):
        '''
            tvals returns the simulation times for alltimes is False
                else returns None to select alltimes
        '''
        
        if alltimes is None:
            alltimes = self.alltimes
        
        if alltimes is False:
            if skipped[quantity] is True:
                tvals = exp2simSkippedTimes(self.times[quantity], tscale, -tshift, self.skip)*self.skip
            else:
                tvals = exp2simTimes(self.times[quantity], tscale, -tshift)
        else:
            tvals = None
            
        return tvals    
    
    
    def tvec_by_quantity(self, quantity, tscale = None, tshift = None):
        '''
            Gets the time-vector to plot alongside the time-course data.
        '''
        
        if np.ndim(tscale) == 0 or len(tscale) == 1:
            tscale = self.tscalevals[0]
        else:
            tscale = 1.0
        
        if tshift is None:
            tshift = self.tshift
        
        if self.alltimes:
            if skipped[quantity] is True:
                skip = self.skip
            else:
                skip = 1
                
            tvals = np.arange(0, self.tsteps + 1, skip)
        else:
            tvals = self.tvals_by_quantity(quantity, tscale, tshift)
        
        return tvals/tscale + tshift
        
    
    def closest_exp_time(self, quantity, t, tscale = None, tshift = None):
        '''
            Obtains the index of the closest experimental time by simulation time
        '''
        
        if tshift is None:
            tshift = self.tshift
            
        if tscale is None:
            tscale = self.tscalevals[0]
            
        idxless = np.where(self.tvec_by_quantity(quantity, tscale, tshift)[t] <= self.times[quantity])[0]
        if len(idxless) == 0:
            idx = len(self.times[quantity]) - 1
        else:
            idx = idxless[0]
            
        return idx
        
        
    def discrete_pdf(self, d):
        '''
            Returns whether distribution is a discrete distribution
                To store continuous data len(dist_bins) = len(dist) + 1
                To store discrete data len(dist_bins) = len(dist)            
        '''
        if self.dist_bins[d] is None:
            return True
        else:
            if len(self.dist_bins[d]) == self.dists[d].shape[int(self.seeded) + 1]:
                return True
            elif len(self.dist_bins[d]) == self.dists[d].shape[int(self.seeded) + 1] + 1:
                return False
            else:
                assert(0), 'distribution' + d + 'has invalid length'
            
            
    def init_seeded_quantities(self, seeds, npops = 1, extradims = (1,), supradims = (1,)):
        '''
            Sets up data structures to compare simulation quantities across seeds
                
            Extra-dimensions allows you to also save data by t-scale value, but also other
                variables you might want to loop over 
            
            Supradims allows you to store
                suprabasal shedding value 
                (for total clonal dynamics)
                
            fixntimes: specifies whether to use experimental times (ntimes = None) 
                or use simulation times (e.g. use ntimes = tsteps) for which we
                can decimate later. This is useful when optimising tscale/shed at
                the end
                
            save_results, which is needed for example if we re-run the suprabasal layer
        '''
        
        #convert to struct
        if np.ndim(extradims) == 0:
            extradims = (extradims,)
        
        nseeds = len(seeds)
        
        #Making more concise / general
        '''
        for q in self.toCalculate:
            if fixntimes is None:
                ntimes = len(self.times[q])
                
            if q in supra_cats:
                self.quantity[]
            else:
                
            self.max_pdf_bin = max_pdf_bin[q]
        '''

        if 'basalCS' in self.toCalculate:
            ntimes = self.ntimes_by_quantity('basalCS')
            self.quantities['basalCS'] = np.zeros((nseeds, ntimes, *extradims))
            self.dists['basalCS'] = np.zeros((nseeds, ntimes, self.get_max_pdf_bin('basalCS') + 1, *extradims))
            self.nsamples['basalCS'] = np.zeros((nseeds, ntimes, *extradims), dtype = int)
            self.dist_bins['basalCS'] = None
            if npops > 1 and 'popCS' in self.toCalculate:
                self.pop_quantities['basalCS'] = np.zeros((nseeds, npops, ntimes, *extradims))
                self.pop_dists['basalCS'] = np.zeros((nseeds, npops, ntimes, self.max_pdf_bin['basalCS'] + 1, *extradims))
                self.pop_nsamples['basalCS'] = np.zeros((nseeds, npops, ntimes, *extradims), dtype = int)
                        
        if 'totalCS' in self.toCalculate:  
            ntimes = self.ntimes_by_quantity('totalCS')
            self.quantities['totalCS'] = np.zeros((nseeds, ntimes, *extradims, *supradims))
            self.dists['totalCS'] = np.zeros((nseeds, ntimes, self.get_max_pdf_bin('totalCS') + 1, *extradims, *supradims))
            self.nsamples['totalCS'] = np.zeros((nseeds, ntimes, *extradims), dtype = int)
            self.dist_bins['totalCS'] = None            
            if npops > 1 and 'popCS' in self.toCalculate:
                self.pop_quantities['totalCS'] = np.zeros((nseeds, npops, ntimes, *extradims, *supradims))
                self.pop_dists['totalCS'] = np.zeros((nseeds, npops, ntimes, self.max_pdf_bin['totalCS'] + 1, *extradims, *supradims))
                self.pop_nsamples['totalCS'] = np.zeros((nseeds, npops, ntimes, *extradims), dtype = int)
                        
        if 'supraCS' in self.toCalculate:
            ntimes = self.ntimes_by_quantity('supraCS')
            self.quantities['supraCS'] = np.zeros((nseeds, ntimes, *extradims, *supradims), dtype = int)
            self.dists['supraCS'] = np.zeros((nseeds, ntimes, self.get_max_pdf_bin('supraCS') + 1, *extradims, *supradims))
            self.nsamples['supraCS'] = np.zeros((nseeds, ntimes, *extradims), dtype = int)
            self.dist_bins['supraCS'] = None  
            if npops > 1 and 'popCS' in self.toCalculate:
                self.quantities['supraCS'] = np.zeros((nseeds, npops, ntimes, *extradims, *supradims))
                self.pop_dists['supraCS'] = np.zeros((nseeds, npops, ntimes, self.max_pdf_bin['supraCS'] + 1, *extradims, *supradims))
                self.pop_nsamples['supraCS'] = np.zeros((nseeds, npops, ntimes, *extradims), dtype = int)
                
        if 'density' in self.toCalculate:
            ntimes = self.ntimes_by_quantity('density')
            self.quantities['density'] = np.zeros((nseeds, ntimes, *extradims))
            self.pop_quantities['density'] = np.zeros((nseeds, npops, ntimes, *extradims))           

        if 'divrate' in self.toCalculate:
            ntimes = self.ntimes_by_quantity('divrate')
            self.quantities['divrate'] = np.zeros((nseeds, ntimes, *extradims))

        if 'pers' in self.toCalculate:
            ntimes = self.ntimes_by_quantity('pers')
            self.quantities['pers'] = np.zeros((nseeds, ntimes, *extradims))

        if 'lcd' in self.toCalculate:
            ntimes = self.ntimes_by_quantity('lcd')
            self.quantities['lcd'] = np.zeros((nseeds, ntimes, *extradims))

        if 'death' in self.toCalculate:
            ntimes = self.ntimes_by_quantity('death')
            self.quantities['death'] = np.zeros((nseeds, ntimes, *extradims))
            
        if 'T1s' in self.toCalculate:
            ntimes = self.ntimes_by_quantity('T1s')
            self.quantities['T1s'] = np.zeros((nseeds, ntimes, *extradims))
            
        if 'MSD' in self.toCalculate:
            ntimes = self.ntimes_by_quantity('MSD')
            self.quantities['MSD'] = np.zeros((nseeds, ntimes, *extradims))
            
        if 'shape' in self.toCalculate:
            ntimes = self.ntimes_by_quantity('shape')
            self.quantities['shape'] = np.zeros((nseeds, ntimes, *extradims))
            
    
        
    def quantities_by_seed(self, tiss, results, seed):
        '''

        For a given simulation, seed, and array of self.tscalevals
        Uses experimental time values for specified self.tscalevals and tshift
        And adds them to data structures by seed

        Indicates whether the re-running of the suprabasal layer should occur here

        Can store non-decimated data for optimization of self.tscalevals.
        '''
        
        #Add to tracked seeds
        self.added_seeds.add(seed)

        if self.shedvals is not None and self.shedvals is not [0]:
            rerunSupra = True

        if self.save_results:
            self.results[seed] = results
            
        #Only consider simulations which haven't reach extinction
        if np.sum(results['ncells'][-1,:]) <= 5:
            print('Simulation went extinct, cannot analyse')
            self.extinct = True
            return self
        elif np.sum(results['ncells'][-1,-1]) == 0:
            print('Stem population went extinct, cannot analyse')
            self.extinct = True
            return self
        
        #Set-up control values
        for q in self.norm_by_control:
            assert('control' in results.keys())
            assert(q in results['control'].keys())            
            self.ctrl_vals[q] = results['control'][q]
              
        #Calculate clonal dynamics
        if 'basalCS' in self.toCalculate or 'totalCS' in self.toCalculate or 'supraCS' in self.toCalculate or 'pers' in self.toCalculate:
            clone_anal = clones.sanalysis(results, skip = self.skip) #analysis
            
        
        #Save quantities
        if 'basalCS' in self.toCalculate:
            '''
            # Can be sped up for a single self.tscalevals value
            if len(self.tscalevals) == 1:
                tval1 = exp2simTimes(self.times['basalCS'], self.tscalevals[0], tshift)
            else:
                tval1 = None
            '''
            #self.set_pop_nsamples('basalCS', np.array(list(basalResults['nsur'].values())).transpose(), seed)
            for ti in range(len(self.tscalevals)):
                tslice = self.timeslice_by_quantity('basalCS', self.tscalevals[ti], self.tshift)
                #tvals = self.tvals_by_quantity('basalCS', self.tscalevals[ti], self.tshift)
                self.set_quantity('basalCS', clone_anal.av_bas_cs[tslice], seed, ti=ti)
                self.set_nsamples('basalCS', clone_anal.nsur[tslice], seed, ti)
                self.set_dist('basalCS', clone_anal.bas_pdf(max_cs = self.max_pdf_bin['basalCS'])[tslice,:], seed, ti = ti)
                

        # Fit quantities to suprabasal layer, otherwise leave as zeros
        if rerunSupra:
            if 'totalCS' in self.toCalculate or 'supraCS' in self.toCalculate:
                for si in range(len(self.shedvals)):
                    sb = self.shedvals[si]
                    self.rerun_suprabasal(results, tiss, sb, seed, si, self.tscalevals, clone_anal)

        if 'density' in self.toCalculate:
            for ti in range(len(self.tscalevals)):
                tslice = self.timeslice_by_quantity('density', self.tscalevals[ti], self.tshift)
                self.set_quantity('density', np.sum(results['ncells'], 1)[tslice], seed, ti = ti)
                self.set_pop_quantity('density', results['ncells'][tslice, :].transpose(), seed, ti = ti)
                
        if 'divrate' in self.toCalculate:
            for ti in range(len(self.tscalevals)):
                tvals = self.tvals_by_quantity('divrate', self.tscalevals[ti], self.tshift)
                divrate_values = simDivrate(tiss, results, times=tvals)
                self.set_quantity('divrate', divrate_values, seed, ti=ti)

        if 'pers' in self.toCalculate:
            for ti in range(len(self.tscalevals)):
                tslice = self.timeslice_by_quantity('pers', self.tscalevals[ti], self.tshift)
                self.set_quantity('pers', clone_anal.pers[tslice], seed, ti=ti)
                
        if 'lcd' in self.toCalculate:
            for ti in range(len(self.tscalevals)):
                tvals = self.tvals_by_quantity('lcd', self.tscalevals[ti], self.tshift)
                lcd_values = clone_anal.lcd(tiss, times=tvals)
                self.set_quantity('lcd', lcd_values, seed, ti=ti)
                
        if 'death' in self.toCalculate:
            for ti in range(len(self.tscalevals)):
                tvals = self.tvals_by_quantity('death', self.tscalevals[ti], self.tshift)
                death_values = simDeaths(tiss, times=tvals)
                self.set_quantity('death', death_values, seed, ti=ti)
                
        #'T1s', 'MSD', 'shape'
        if 'T1s' in self.toCalculate:
            for ti in range(len(self.tscalevals)):
                tvals = self.tvals_by_quantity('T1s', self.tscalevals[ti], self.tshift)
                T1_values = sa.num_T1s(results['shapes']['jlengths'], results['cellid'])
                self.set_quantity('T1s', T1_values, seed, ti=ti)
                
        if 'MSD' in self.toCalculate:
            for ti in range(len(self.tscalevals)):
                tvals = self.tvals_by_quantity('MSD', self.tscalevals[ti], self.tshift)
                MSD_values = sa.msddiv(results['ncells'], results['anc'], results['cellid'], results['cellpos'], self.skip)
                self.set_quantity('MSD', MSD_values, seed, ti = ti)
                
        if 'shape' in self.toCalculate:
            for ti in range(len(self.tscalevals)):
                tvals = self.tvals_by_quantity('shape', self.tscalevals[ti], self.tshift)
                shape_values = sa.p0(results['shapes']['areas'], perims = results['shapes']['perims'])
                self.set_quantity('shape', np.array([np.mean(p0s) for p0s in shape_values]), seed, ti = ti)

        return self

 
    def rerun_suprabasal(self, results, tiss, sb, seed, sbsi = 0, tscalevals = None, clone_anal = None,
                         shed_profile = None):
        '''
            Runs suprabasal layer once and save's the necessary suprabasal quantities for an
                array or individual tscales
                
            Note this will overwrite data in these indices.
            
        '''
        
        if clone_anal is None:
            clone_anal = clones.sanalysis(results, skip = self.skip) 
        
        #Set tscalevals
        if tscalevals is None:
            tscalevals = self.tscalevals
        else:
            assert(len(tscalevals) <= len(self.tscalevals))
        
        if sb < 0:
            sb *= -get_division_scale(tiss)
            #self.shedvals[sbsi] = sb
             
        np.random.seed(supraseed(seed, get_division_scale(tiss), results['seed']))
        results = tiss.rerunsuprabasal(results, tiss.time, sb, skip = self.skip, shed_profile = shed_profile)
        results['tiss'] = tiss
        
        if self.save_results:
            self.results[seed] = results
        
        clone_anal.recalc_supra_livesize(results)
        
        
        if 'totalCS' in self.toCalculate:     
            for ti in range(len(tscalevals)):
                tslice = self.timeslice_by_quantity('totalCS', tscalevals[ti], self.tshift)
                #tvals  = self.tvals_by_quantity('totalCS', tscalevals[ti], self.tshift)
                self.set_quantity('totalCS', clone_anal.av_tot_cs[tslice], seed, ti = ti, sbsi = sbsi)
                self.set_nsamples('totalCS', clone_anal.nsur[tslice], seed, ti)                
                assert self.dists['totalCS'].flags.writeable, "broken" + str(ti) + ','+ str(sbsi)
                self.set_dist('totalCS', clone_anal.tot_pdf(max_cs = self.max_pdf_bin['totalCS'])[tslice,:], seed, ti=ti, sbsi=sbsi)

        if 'supraCS' in self.toCalculate:    
            for ti in range(len(tscalevals)):
                tslice = self.timeslice_by_quantity('supraCS', tscalevals[ti], self.tshift)
                #tvals  = self.tvals_by_quantity('supraCS', tscalevals[ti], self.tshift)
                self.set_nsamples('supraCS', clone_anal.nsur[tslice], seed, ti)                                
                self.set_quantity('supraCS', clone_anal.av_sup_cs[tslice], seed, ti=ti, sbsi=sbsi)
                self.set_dist('supraCS', clone_anal.sup_pdf(max_cs = self.max_pdf_bin['supraCS'])[tslice,:], seed, ti = ti, sbsi = sbsi)                

        return results


    
    def calculate_combined_quantities(self, use_wam = True):
        '''
            Returns an simulation dataset object.
            Takes averages of samples for pdfs and quantities
            
            Takes data from different simulation seeds and combines them into calculated
                quantities that we want from simulations
            
            We use a weighted average unless use_wam = False
        '''
        
        sim_data = self.copy()
        sim_data.seeded = False
        sim_data.seeds = None
    
        for q in sim_data.quantities.keys():
            #Calculate SE in simplest way
            if q not in sim_data.dists: #there is no weighting of std by samples
                sim_data.se_quants[q] = np.std(sim_data.quantities[q],0) 
                
            #Calculate quantity using weighted or unweighted average           
            if use_wam is True and q in sim_data.nsamples and np.sum(sim_data.nsamples[q]) > 0:
                if q in supra_cats and self.collapsed is False:
                    #Makes sure dimensions of weights are correct
                    weights = np.repeat(sim_data.nsamples[q].reshape(*np.shape(sim_data.nsamples[q]),1), 
                                        sim_data.quantities[q].shape[-1], axis = -1)
                else:
                    weights = sim_data.nsamples[q]

                assert(np.sum(weights) > 0), q + ' has no samples' +  str(sim_data.quantities[q]) +  str(weights)
                sim_data.quantities[q] = np.average(sim_data.quantities[q], 0, weights = weights)
            else:
                sim_data.quantities[q] = np.mean(sim_data.quantities[q],0)
            
        for d in sim_data.dists.keys():
            if use_wam is True and np.sum(sim_data.nsamples[d]) > 0:
                if d in supra_cats and self.collapsed is False:
                    #Makes sure dimensions of weights are correct   
                    weights = np.repeat(sim_data.nsamples[d].reshape(*sim_data.nsamples[d].shape[:-1],1,sim_data.nsamples[d].shape[-1]), 
                                        sim_data.dists[d].shape[-3], axis = -2)                                     
                    weights = np.repeat(weights.reshape(*np.shape(weights),1), 
                                        sim_data.dists[d].shape[-1], axis = -1)
                elif self.collapsed is False:
                    #Add dimension along size axis                    
                    weights = np.repeat(sim_data.nsamples[d].reshape(*sim_data.nsamples[d].shape[:-1],1,sim_data.nsamples[d].shape[-1]), 
                                        sim_data.dists[d].shape[-2], axis = -2)
                else: #collapsed is True
                    weights = np.repeat(np.expand_dims(sim_data.nsamples[d],axis=-1), sim_data.dists[d].shape[-1], axis=-1)
                    
                sim_data.dists[d] = np.average(sim_data.dists[d], 0, weights = weights)
            else:
                sim_data.dists[d] = np.mean(sim_data.dists[d],0) 
                
        #Calculate number of samples and updated SD quants
        for d in sim_data.dists.keys():
            sim_data.nsamples[d] = np.sum(self.nsamples[d], 0)
            sim_data.se_quants[d] = sim_data.sd_dist(d, True)


        for q in sim_data.pop_quantities.keys():
            if use_wam is True and q in sim_data.nsamples and np.sum(sim_data.nsamples[q]) > 0:
                sim_data.pop_quantities[q] = np.average(sim_data.pop_quantities[q], 0, weights = sim_data.pop_nsamples[q])
            else:
                sim_data.pop_quantities[q] = np.mean(sim_data.pop_quantities[q],0)
            
        for d in sim_data.pop_dists.keys():
            if use_wam is True and np.sum(sim_data.nsamples[d]) > 0:
                sim_data.pop_dists[d] = np.average(sim_data.pop_dists[d], 0, weights = self.pop_nsamples[d])
            else:
                sim_data.pop_dists[d] = np.mean(sim_data.pop_dists[d],0)  
            sim_data.pop_nsamples[d] = np.sum(self.pop_nsamples[d], 0)
            
            
        return sim_data    
    
    
    
    def sd_dist(self, d, useSE = False):
        '''
            Calculate the standard deviation of a probability distribution
                Assumes that the average quantity is the mean and is up to date
                Also that nsamples is up to date if we using useSE = True
                
            If useSE is True, then we use the standard error of the mean
        '''
        if self.dist_bins[d] is None:
            bins = np.arange(0, self.max_pdf_bin[d] + 1)
        elif self.discrete_pdf(d) is True:
            bins = self.dist_bins[d]
        else:
            #use-bin centres for numerical integration of continuous pdf
            bins = (self.dist_bins[d][:-1] + self.dist_bins[1:])/2
            
        #calculate the standard deviation, making the dimensions work
        if self.seeded:
            if self.collapsed:
               sd = np.sqrt(np.sum(self.dists[d]*(bins[None,None,:] - self.quantities[d][:,:,None])**2, axis = int(self.seeded) + 1))               
            else:
                if d in supra_cats:
                   sd = np.sqrt(np.sum(self.dists[d]*(bins[None,None,:,None,None] - self.quantities[d][:,:,None,:,:])**2, axis = int(self.seeded) + 1))               
                else:
                   sd = np.sqrt(np.sum(self.dists[d]*(bins[None,None,:,None] - self.quantities[d][:,:,None,:])**2, axis = int(self.seeded) + 1))                                                   
        else:
            if self.collapsed:
               sd = np.sqrt(np.sum(self.dists[d]*(bins[None,:] - self.quantities[d][:,None])**2, axis = int(self.seeded) + 1))               
            else:
                if d in supra_cats:
                   sd = np.sqrt(np.sum(self.dists[d]*(bins[None,:,None,None] - self.quantities[d][:,None,:,:])**2, axis = int(self.seeded) + 1))               
                else:
                   sd = np.sqrt(np.sum(self.dists[d]*(bins[None,:,None] - self.quantities[d][:,None,:])**2, axis = int(self.seeded) + 1))                                                   

        
        #convert to the standard error of the mean
        if useSE:
            if d in supra_cats and self.collapsed is False:
                sd /= np.sqrt(self.nsamples[d][...,None])
            else:
                sd /= np.sqrt(self.nsamples[d])

        return sd
    
    
    def data_time_crop(self, tscales = None):
        '''
            Takes each simulated data in self.toCalculate and crops time course to the appropriate 
                length and returns a copy
                            
        '''
        
        if tscales is None:
            tscales = self.tscalevals
        
        sim_data = self.copy()
        
        if np.ndim(tscales) == 0:
            tscales = [tscales]        
        
        if sim_data.alltimes is False:
            print('Already cropped')
            return sim_data
        
        sim_data.alltimes = False
        
        for qtype in ['quantities', 'dists', 'nsamples']:
            for quant in getattr(sim_data, qtype):
                for ti in range(len(tscales)):  
                    tslice = sim_data.timeslice_by_quantity(quant, tscales[ti], sim_data.tshift, alltimes = False)
                    if sim_data.seeded:
                        getattr(sim_data, qtype)[quant] = getattr(sim_data, qtype)[quant][:, tslice, ...]
                    else:
                        getattr(sim_data, qtype)[quant] = getattr(sim_data, qtype)[quant][tslice, ...]


                '''
                #Old implementation: does not have the flags of what type of data we store
                if np.shape(getattr(sim_data, qtype)[quant], axis) == tsteps + 1:
                    for ti in range(len(tscales)):
                        tscale = tscales[ti]    
                        tvals = exp2simTimes(self.times[quant], tscale, self.tshift)                
                        setattr(sim_data, qtype[quant], getattr(sim_data, qtype)[quant][:,tvals,...])
                elif np.shape(getattr(sim_data, qtype)[quant], axis) == np.ceil((tsteps + 1)/10):
                    for ti in range(len(tscales)):
                        tscale = tscales[ti]    
                        tvals = exp2simSkippedTimes(self.times[quant], tscale, self.tshift, self.skip)                
                        setattr(sim_data, qtype[quant], getattr(sim_data, qtype)[quant][:,tvals,...])
                elif np.shape(getattr(sim_data, qtype)[quant], axis) == len(self.times[quant]):
                    print('Already cropped')
                else:
                    print('Warning: length not changed') 
                '''                        

        for qtype in ['pop_quantities', 'pop_dists', 'pop_nsamples']:
            for quant in getattr(sim_data, qtype):
                for ti in range(len(tscales)):  
                    tslice = sim_data.timeslice_by_quantity(quant, tscales[ti], sim_data.tshift)
                    if sim_data.seeded:
                        getattr(sim_data, qtype)[quant] = getattr(sim_data, qtype)[quant][:, :, tslice, ...]
                    else:
                        getattr(sim_data, qtype)[quant] = getattr(sim_data, qtype)[quant][:, tslice, ...]

        return sim_data
    
    
    
    def get_instance(self, ti = 0, sbsi = 0):
        '''
            Get's a particular dataset for certain tscale-values and suprabasal shed values
        '''

        if self.seeded is True:
            sim_data = self.get_seeded_instance(ti, sbsi)
        else:
            sim_data = self.copy()
            sim_data.collapsed = True  
                        
            for q in sim_data.quantities.keys():
                if q in supra_cats:
                    sim_data.quantities[q] = sim_data.quantities[q][:,ti,sbsi]
                    if hasattr(sim_data.se_quants, q):
                        sim_data.se_quants[q]  = sim_data.se_quants[q][:,ti,sbsi]
                else:
                    sim_data.quantities[q] = sim_data.quantities[q][:,ti]
                    if hasattr(sim_data.se_quants, q):                    
                        sim_data.se_quants[q]  = sim_data.se_quants[q][:,ti]
                    
            for d in sim_data.dists.keys():
                if d in supra_cats:
                    sim_data.dists[d] = sim_data.dists[d][:,:,ti,sbsi]
                else:
                    sim_data.dists[d] = sim_data.dists[d][:,:,ti]  
                                        
            for q in sim_data.pop_quantities.keys():
                if q in supra_cats:
                    sim_data.pop_quantities[q] = sim_data.pop_quantities[q][:,:,ti,sbsi]
                else:
                    sim_data.pop_quantities[q] = sim_data.pop_quantities[q][:,:,ti]
                                
                
            for d in sim_data.pop_dists.keys():
                if d in supra_cats:
                    sim_data.pop_dists[d] = sim_data.pop_dists[d][:,:,:,ti,sbsi]
                else:
                    sim_data.pop_dists[d] = sim_data.pop_dists[d][:,:,:,ti]                
    
    
            for d in sim_data.dists.keys():                    
                sim_data.nsamples[d] = sim_data.nsamples[d][:,ti]
    
    
        return sim_data
    
    
    def get_seeded_instance(self, ti = 0, sbsi = 0):
        '''
            Get's a particular dataset for certain tscale-values and suprabasal shed values
                when data is in the seeded formal
        '''
        
        sim_data = self.copy()
        sim_data.collapsed = True 
        
        sim_data.tscalevals = [self.tscalevals[ti]]
        sim_data.shedvals = [self.shedvals[sbsi]] 

        for q in sim_data.quantities.keys():
            if q in supra_cats:
                sim_data.quantities[q] = sim_data.quantities[q][:,:,ti,sbsi]
                if hasattr(sim_data.se_quants, q):
                    sim_data.se_quants[q]  = sim_data.se_quants[q][:, :,ti,sbsi]                
            else:
                sim_data.quantities[q] = sim_data.quantities[q][:,:,ti]
                if hasattr(sim_data.se_quants, q):                    
                    sim_data.se_quants[q]  = sim_data.se_quants[q][:,:,ti]                
                
        for d in sim_data.dists.keys():
            if d in supra_cats:
                sim_data.dists[d] = sim_data.dists[d][:,:,:,ti,sbsi]
            else:
                sim_data.dists[d] = sim_data.dists[d][:,:,:,ti]  
            sim_data.nsamples[d] = sim_data.nsamples[d][:,:,ti]
                
        for q in sim_data.pop_quantities.keys():
            if q in supra_cats:
                sim_data.pop_quantities[q] = sim_data.pop_quantities[q][:,:,:,ti,sbsi]
            else:
                sim_data.pop_quantities[q] = sim_data.pop_quantities[q][:,:,:,ti]
                            
            
        for d in sim_data.pop_dists.keys():
            if d in supra_cats:
                sim_data.pop_dists[d] = sim_data.pop_dists[d][:,:,:,:,ti,sbsi]
            else:
                sim_data.pop_dists[d] = sim_data.pop_dists[d][:,:,:,:,ti]                
        
        
        return sim_data
    
    
    def get_tscale_idx(self, tscale):
        if tscale in self.tscalevals:
            return np.where(tscale == self.tscalevals)[0][0]
        else:
            return None        

    def get_shed_idx(self, shed):
        if shed in self.shedvals:
            return np.where(shed == self.shedvals)[0][0]
        else:
            return None        
        
        
    def sample(self, d, t = 0, seed = None, ti = 0, si = 0):
        '''
            Returns samples for a given distribution at a given time-point
                indexed by t for tscale-idx ti and suprabasal shed idx si

            Defaults to: returning combining data across all seeds
            Alternatively can specify a seed
            
            Does not work with data with extra-dimensions
            
        '''
        
        #for collapsed data
        if self.collapsed is True:
            return dataset.sample(self, d, t, seed, ti, si)
          
        #for data with extra dimensions
        if self.seeded is True:
            if seed is None: #use combined over sims
                if d in supra_cats:
                    N = np.sum(self.nsamples[d][:,t, ti, si], 0)
                    hist_counts = (np.average(self.dists[d][:, t, :], weights=self.nsamples[d][:, t, ti, si], axis=0) * N).astype(int)
                else:
                    N = np.sum(self.nsamples[d][:, t, ti], 0)
                    hist_counts = (np.average(self.dists[d][:, t, :], weights=self.nsamples[d][:, t, ti], axis=0) * N).astype(int)
            else:
                if d in supra_cats:
                    N = self.nsamples[d][seed,t, ti, si]
                    hist_counts = (self.dists[d][seed, t, :, ti, si]*N).astype(int)  
                else:
                    N = self.nsamples[d][seed,t, ti]
                    hist_counts = (self.dists[d][seed, t, :, ti]*N).astype(int)
        else:
            if d in supra_cats:
                N = self.nsamples[d][t, ti, si]
                hist_counts = (self.dists[d][t, :, ti, si]*N).astype(int)
            else:
                N = self.nsamples[d][t, ti]
                hist_counts = (self.dists[d][t, :, ti]*N).astype(int)
            
        return np.repeat(np.arange(len(hist_counts)), hist_counts)
    
        
class exp_dataset(dataset):
    '''
        Object for manipulating experimental data
    '''
    
    def __init__(self, filename, prefix, suffix, toCalculate, max_pdf_bin = None, default_pdf_max = 200):  
        self.samples = {}               
        dataset.__init__(self, toCalculate, max_pdf_bin = max_pdf_bin, default_pdf_max = default_pdf_max)        
        data_full = self.load_data_file(filename, prefix, suffix)
        self.extract_quantities(data_full, prefix)
        self.type = 'exp'        
        self.exp_prefix = prefix
        self.exp_suffix = suffix 
    
    def extract_quantities(self, data_full, prefix = '', suffix = ''):
        '''
            Extracts relevant quantities for analyis
                and saves alongside their resulting time-values (normally in days)
        '''
        
        if 'basalCS' in self.toCalculate:
            self.quantities['basalCS'] = np.array(data_full[prefix + 'Basalsize' + suffix])
            self.dists['basalCS'] = np.array(data_full[prefix + 'pdf'][:,0,:])
            self._max_pdf_bin['basalCS'] = np.shape(self.dists['basalCS'])[1]
            self.dist_bins['basalCS'] = None
            if (prefix + 'Nsamples' + suffix) in data_full.keys():
                self.nsamples['basalCS'] = np.array(data_full[prefix + 'Nsamples' + suffix])
            else:
                samples = data_full[prefix + 'FullBasalsize' + suffix]
                self.nsamples['basalCS'] = np.array([len(s) for s in samples])
                self.samples['basalCS'] = data_full[prefix + 'FullBasalsize' + suffix]
            self.times['basalCS'] = np.array(data_full[prefix + 'TimeClonesize'])
            self.se_quants['basalCS'] = np.array(data_full[prefix + 'SEBasalsize' + suffix])


        if 'totalCS' in self.toCalculate:
            self.quantities['totalCS'] = np.array(data_full[prefix + 'Totalsize' + suffix])
            self.dists['totalCS'] = np.array(data_full[prefix + 'pdf'][:,2,:])
            self._max_pdf_bin['totalCS'] = np.shape(self.dists['totalCS'])[1]   
            self.dist_bins['totalCS'] = None     
            if (prefix + 'Nsamples' + suffix) in data_full.keys():
                self.nsamples['totalCS'] = np.array(data_full[prefix + 'Nsamples' + suffix])
            else:
                samples = data_full[prefix + 'FullBasalsize' + suffix]
                self.nsamples['totalCS'] = np.array([len(s) for s in samples])
                self.samples['totalCS'] = data_full[prefix + 'FullTotalsize' + suffix]                      
            self.times['totalCS'] = np.array(data_full[prefix + 'TimeClonesize'])
            self.se_quants['totalCS'] = np.array(data_full[prefix + 'SETotalsize' + suffix])

        if 'supraCS' in self.toCalculate:
            self.quantities['supraCS'] = np.array(data_full[prefix + 'Suprasize' + suffix])
            self.dists['supraCS'] = np.array(data_full[prefix + 'pdf'][:,1,:])
            self._max_pdf_bin['supraCS'] = np.shape(self.dists['supraCS'])[1] 
            self.dist_bins['supraCS'] = None    
            if (prefix + 'Nsamples' + suffix) in data_full.keys():
                self.nsamples['supraCS'] = np.array(data_full[prefix + 'Nsamples' + suffix])
            else:
                samples = data_full[prefix + 'FullSuprasize' + suffix]
                self.nsamples['supraCS'] = np.array([len(s) for s in samples])                     
                self.samples['supraCS'] = data_full[prefix + 'FullSuprasize' + suffix]
            
            self.times['supraCS'] = np.array(data_full[prefix + 'TimeClonesize'])
            self.se_quants['supraCS'] = np.array(data_full[prefix + 'SESuprasize' + suffix])

        if 'density' in self.toCalculate:
            self.quantities['density'] = np.array(data_full[prefix + 'Density'])
            self.se_quants['density']  = np.array(data_full[prefix + 'SEDensity' + suffix])
            self.times['density'] = np.array(data_full[prefix + 'TimeDensity'])

        if 'divrate' in self.toCalculate:
            self.quantities['divrate'] = np.array(data_full[prefix + 'Divrate'])/np.mean(data_full['ctrDivrate'])
            self.se_quants['divrate']  = np.array(data_full[prefix + 'SEDivrate' + suffix])
            self.times['divrate'] = np.array(data_full[prefix + 'TimeDivrate'])

        if 'pers' in self.toCalculate:
            self.quantities['pers'] = np.array(data_full[prefix + 'Persistence'])
            self.times['pers'] = np.array(data_full[prefix + 'TimePers'])
            self.se_quants['pers'] = np.array(data_full[prefix + 'SEPersistence' + suffix])

        if 'lcd' in self.toCalculate:
            self.quantities['lcd'] = np.array(data_full[prefix + 'LCD'])
            self.times['lcd'] = np.array(data_full['time_LCD'])

        if 'death' in self.toCalculate:
            self.quantities['death'] = np.array(data_full[prefix + 'Cas3'])
            self.times['death'] = np.array(data_full['timeCas3'])
           
          
        
          
                     
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filename, deleteEOF = False):
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except EOFError:
        if deleteEOF:
            print(f"Deleting file due to EOFError: {filename}")
            os.remove(filename)
    except Exception as e:
        print(f"Error while processing file {filename}: {e}")
    
    try:    
        with open(filename, 'rb') as outp:  # Overwrites any existing file.
            return pickle.load(outp)
    except Exception as e:
        print(f"Error while processing file {filename}: {e}")
        return None
    
    
def exp2simTimes(texp, tscale, tshift = 0):
    if np.ndim(texp) == 0:
        return int((texp + tshift)*tscale)
    else:
        return ((texp + tshift)*tscale).astype('int32')

def exp2simSkippedTimes(texp, tscale, tshift = 0, skip = 10):
    if np.ndim(texp) == 0:
        return int((texp + tshift)*tscale/skip)
    else:
        return ((texp + tshift)*tscale/skip).astype('int32')


def join_datasets(dss):
    '''
        Takes a list of datasets from different seeds and then joins
            them into one dataset object
    '''
    data = dss[0]
    
    if len(dss) > 1:   
        for d in dss[1:]:
            if data.added_seeds.isdisjoint(d.added_seeds):
                data = data.add_seeded_dataset(d)
            else:
                print('Joining data with repeated seeds')
                assert(0)
            
    return data


def simDivrate(tiss, results, tsteps = None, times = None, dx = 50):
    '''
        Calculates simulation division rate
        https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
        This is shifted by dx = 50
    '''

    if tsteps is None:
        tsteps = tiss.time

    if times is None:
        times = np.arange(tsteps + 1) 
        
    ndivs = np.zeros(tsteps + 1)
    ndies = np.zeros(tsteps + 1)
    nkils = np.zeros(tsteps + 1)
    ndifs = np.zeros(tsteps + 1)

    for t in range(tsteps):
        for p in range(tiss.npops):
            if t in list(tiss.cellpops[p].dupTime.keys()):
                #print(len(tiss.cellpops[p].dupTime[t]))
                ndivs[t] += len(tiss.cellpops[p].dupTime[t])
                ndifs[t] += np.sum([len(ev) == 2 for ev in tiss.cellpops[p].dupTime[t]])

            if t in list(tiss.cellpops[p].dieTime.keys()):
                #print((tiss.cellpops[p].dieTime[t]))
                ndies[t] += len(tiss.cellpops[p].dieTime[t])

            if t in list(tiss.cellpops[p].kilTime.keys()):
                #print((tiss.cellpops[p].kilTime[t]))
                nkils[t] += len(tiss.cellpops[p].kilTime[t])

    nevnt = ndivs + ndies - nkils - ndifs
    times = times.astype('int')
    return np.convolve(nevnt/np.sum(results['ncells'],1),np.ones(dx)/dx, mode='same')[times]


def simDeaths(tiss, tsteps = None, times = None, dx = 100):
    '''
        Calculates simulation deaths
        https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
        This is shifted by dx = 50
    '''

    if tsteps is None:
        tsteps = tiss.time + 1

    if times is None:
        times = np.arange(tsteps + 1) 
        
    nkils = np.zeros(tsteps + 1)

    for t in range(tsteps):
        for p in range(tiss.npops):
            if t in list(tiss.cellpops[p].kilTime.keys()):
                #print((tiss.cellpops[p].kilTime[t]))
                nkils[t] += len(tiss.cellpops[p].kilTime[t])

    times = times.astype('int')
    return np.convolve(nkils,np.ones(dx)/dx, mode='same')[times]


def get_division_scale(tiss, popidx = None):
    '''
        Get the governing time-scale of division for a multiple population tissue,
            takes the value from the highest indexed population that has a non-zero 
            rate unless we specify population
            
    '''
    time = 0
    if popidx is None:
        for p in reversed(range(tiss.npops)):
            if tiss.cellpops[p].divtime > 0:
                time = tiss.cellpops[p].divtime
    return time

def supraseed(s, tau = 1000, maxseed = 1000):
    '''
        Characteristic way to generate suprabasal seeds for simulations
        
        Use results['seed'] as the maxseed if it exists
    '''
    return int(tau*maxseed + s)


def getCdf(spdf):
    '''
        Converts pdf to cdf, works provided the values are kept along last dimension,
        Assumes but does not check normalisation
    '''
    return np.cumsum(spdf[...,::-1], axis = -1)[...,::-1]   

def getPdf(sample, maxbin = None):
    '''
        Takes a sample and extracts the pdf
    '''
    if maxbin is None:
        maxbin = np.max(sample)
    
    return np.histogram(sample, bins = np.arange(0, maxbin + 2), density = True)[0]


