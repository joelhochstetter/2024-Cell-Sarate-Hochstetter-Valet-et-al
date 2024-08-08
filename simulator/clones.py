'''
    Analysis pipeline for clones from simulations and experiments,
        we can handle static or time-lapse data. 
        
        Allows quick calculations of different quantities of interest 
        and importantly we don't calculate more quantities than we need, 
        which speeds up fitting and analysis.
'''


import numpy as np
import matplotlib.pyplot as plt
import analysis
import pandas as pd
import copy
import scipy
import sparse
from numba import jit, njit



class lanalysis():
    def __init__(self, livesize, popnames = None, tvals = None, containsBasal = True, suprabasal = True,
                 miss_flag = -100, ctypes = []):
        '''
            Analysis of clones from simulations or experiments given a live-size is provided,
                or static clone sizes (currently not implemented)   
                
            Assumes no data is missing
            Missing data is denoted by clone-sizes smaller than 0 in both categories
                
            Flags:
                containsBasal, set to True to only include clones that contain basal cells
                suprabasal, set to True to include suprabasal cells in the analysis
                    Currently coded to only work with one supra-basal layer                      
        '''
        self.inducttime = 0
        
        #Attributes
        if livesize is not None:
            self.npops = 1
            
            self.nmarked = [livesize.shape[0]]
            self.nsb0 = np.sum(livesize[:,0,1]) #number of suprabasal cells at t =0
            self.nclones = np.sum(self.nmarked)
            self.ntimes = livesize.shape[1]
            self._popnames = popnames  

            #Set times
            if tvals is None:
                self.tvals = np.arange(self.ntimes)
            else:
                self.tvals = np.array(tvals)
              
            if np.min(livesize) < 0:
                print('Warning: Dataset contains missing data')
                #Ensure only data which has values in each compartment is included
                livesize[np.min(livesize, 2) < 0, :] = miss_flag
              
        #Used during time-cropping, for simulations
        self.full_tvals = None
        self.full_livesize = None
        self.timecropped = False
            
        #Flag for calculating quantities
        self.containsBasal = containsBasal
        self.suprabasal = suprabasal
            
        #Calculate the live-size of each clone, all other quantities are calculated on the fly
        self.livesize = livesize
        
        #Cell-types defined based on future lineage behaviour
        self.ctypes = ctypes
        
        #set properties to initial values
        self.reset_properties()
        
    
    def reset_properties(self):
        #Base quantities
        self._joint_pdf = None
        self._joint_pdf_sparse = None
        self._joint_bas_sb = None
        self._live_bas_sb = None
        self._bas_ls = None 
        self._tot_ls = None
        self._sup_ls = None
        
        self._nsur = None #Number of surviving clones
        self._pers = None #Fraction of surviving clones
        
        self._av_bas_cs = None #Average number of basal cells per clone
        self._av_tot_cs = None #Average number of total cells per clone
        self._av_sup_cs = None #Average number of suprabasal cells per clone
        
        self._se_bas_cs = None #Standard error of basal cells per clone
        self._se_tot_cs = None #Standard error of total cells per clone
        self._se_sup_cs = None #Standard error of suprabasal cells per clone
        
        self._ev_bas_cs = None #Even bias for basal clone sizes
        self._ev_tot_cs = None #Even bias for total clone sizes
        self._ev_sup_cs = None #Even bias for suprabasal clone sizes
        
        self._bas_pdf = None #PDF of basal cells per clone
        self._tot_pdf = None #PDF of total cells per clone
        self._sup_pdf = None #PDF of suprabasal cells per clone
        self._lcd = None #labelled cell density
        self._missing = None #Number of missing clones per time-point
        
        #Population quantities
        self._splits = None #Splits between different population indices
        self._popsur = None #Survival probability of each population
        self._pop_ave_bas_cs = None #Average number of basal cells per clone labelled by a certain population
        self._pop_ave_tot_cs = None #Average number of total cells per clone labelled by a certain population        
        self._pop_ave_sup_cs = None #Average number of suprabasal cells per clone labelled by a certain population
        self._pop_bas_pdf = None #PDF of basal cells per clone labelled by a certain population
        self._pop_tot_pdf = None #PDF of total cells per clone labelled by a certain population
        self._pop_sup_pdf = None #PDF of suprabasal cells per clone labelled by a certain population
        self._ave_cs_by_pop = None # Average cells of type per clone
        self._pop_ave_cs_by_pop = None # Average cells of type per clone labelled by a certain population

       
        
    def timecrop(self, skip = None):
        ''' 
            Re-calculates clonal dynamics for a new set of tvals
                based on skipping to have only every n elements
        '''
        
        if skip is None:
            if self.timecropped == True: #revert to uncroppd
                self.tvals = self.full_tvals
                self.livesize = self.full_livesize
            else:
                return
        else:
            if self.timecropped == False:
                self.full_tvals = self.tvals.copy()
                self.full_livesize = self.livesize.copy()
            self.timecropped = True                
            self.tvals = self.full_tvals[::skip]
            self.livesize = self.full_livesize[:,::skip,:]
            
        self.reset_properties()
        
        
    def ctype_ids(self, ctype, livesize = None):
        ''' 
            Extract the ids of clones of a given type
        '''
        
        if livesize is None:
            livesize = self.livesize
        
        maxtot = np.max(np.sum(livesize,2),1)
        
        if ctype == 'S':
            return np.where(maxtot >= 3)[0]
        elif ctype == 'T':
            return np.where(maxtot == 2)[0]
        elif ctype == 'D':
            return np.where(maxtot == 1)[0]
        elif ctype == 'S2':
            return np.where(maxtot >= 2)[0]
        else:
            return np.array([])
        
        
        
        
    @property
    def missing(self):
        ''' 
            Number of missing clones per time-point
        '''
        if self._missing is None:
            self._missing = np.sum(np.min(self.livesize, 2) < 0, 0)
            
        return self._missing

    @property
    def non_missing(self):
        ''' 
            Number of non-missing clones per time-point
        '''
        return self.nclones - self.missing
        
    @property
    def popnames(self):
        '''
            Population names, including suprabasal
        '''
        if self._popnames is None:
            self._popnames = list(np.arange(self.npops)) + ['sb']
        else:
            if len(self._popnames) == self.npops:
                self._popnames.append('sb')
                
        return self._popnames

    @property
    def joint_pdf(self):
        '''
            Returns the joint pdf of the clone sizes
        '''
        if self._joint_pdf is None:
            self._joint_pdf = analysis.joint_pdf(self.livesize, basal_cont = self.containsBasal)

        return self._joint_pdf
    
    @property
    def joint_pdf_sparse(self):
        '''
            Returns the joint pdf of the clone sizes
        '''
        if self._joint_pdf_sparse is None:
            self._joint_pdf_sparse = analysis.joint_pdf(self.livesize, basal_cont = self.containsBasal, sparse = True)

        return self._joint_pdf_sparse
        
    
    @property
    def bas_ls(self):
        '''
            Returns the live number of basal cells per clone
        '''
        if self._bas_ls is None:
            self._bas_ls = np.sum(self.livesize[:,:,:self.npops],2)
            
        return self._bas_ls
    
    @property
    def tot_ls(self):
        '''
            Returns the live number of total cells per clone
        '''
        if self._tot_ls is None:
            self._tot_ls = np.sum(self.livesize, 2)
            
        return self._tot_ls
    
    @property
    def sup_ls(self):
        '''
            Returns the live number of suprabasal cells per clone
        '''
        if self._sup_ls is None:
            self._sup_ls = np.sum(self.livesize[:,:,self.npops:],2)
            
        return self._sup_ls
    
    @property
    def live_bas_sb(self):
        '''
            Returns the number of basal and suprabasal cells per clone
        '''
        
        #There are no hidden populations stored
        if np.shape(self.livesize[2]) == 2:
            return self.livesize
        
        if self._live_bas_sb is None:
            self._live_bas_sb = np.zeros((self.nclones, self.ntimes, 2), dtype = 'int32') #basal and supra-basal livesize
            self._live_bas_sb[:,:,0] = self.bas_ls
            self._live_bas_sb[:,:,1] = self.sup_ls
        
        return self._live_bas_sb
    
    
    @property
    def joint_bas_sb(self):
        '''
            Returns the joint pdf of the clone sizes
        '''

        if self._joint_bas_sb is None:
            self._joint_bas_sb = analysis.joint_pdf(self.live_bas_sb, basal_cont = self.containsBasal)
            
        return self._joint_bas_sb
    
    
    @property
    def nsur(self):
        '''
            Returns the number of surviving clones
        '''
        if self._nsur is None:
            if np.max(self.missing) == 0:
                self._nsur = np.sum(self.bas_ls > 0, 0)
            else:
                self._nsur = np.sum((self.bas_ls > 0) & (self.sup_ls >= 0), 0)
            
        return self._nsur
    
    @property
    def pers(self):
        '''
            Returns the fraction of surviving clones
        '''
        if self._pers is None:
            self._pers = self.nsur/self.non_missing
            
        return self._pers
    
    @property
    def lcd(self):
        '''
            Returns the labelled cell density
        '''
        if self._lcd is None:
            if self.containsBasal:
                self._lcd = self.av_bas_cs*self.pers
            else:
                self._lcd = np.sum(self.bas_ls,0)/self.non_missing
                
        return self._lcd
    
    
    @property
    def av_bas_cs(self):
        '''
            Returns the average number of basal cells per clone
        '''
        if self._av_bas_cs is None:
            if self.containsBasal:
                self._av_bas_cs = np.sum(self.bas_ls*(self.bas_ls > 0),0)/self.nsur
            else:
                self._av_bas_cs = np.sum(self.bas_ls*(self.bas_ls >= 0),0)/self.non_missing
                #self._av_bas_cs = np.mean(self.bas_ls,0)
            
        return self._av_bas_cs
    
    @property
    def av_tot_cs(self):
        '''
            Returns the average number of total cells per clone
        '''
        if self._av_tot_cs is None:
            if self.containsBasal:
                bas_mask = self.bas_ls > 0
                self._av_tot_cs = np.sum(self.tot_ls*bas_mask*(self.tot_ls >= 0),0)/self.nsur
            else:
                self._av_tot_cs = np.sum(self.tot_ls*(self.tot_ls >= 0),0)/self.non_missing
                #self._av_tot_cs = np.mean(self.tot_ls,0)
            
        return self._av_tot_cs
    
    @property
    def av_sup_cs(self):
        '''
            Returns the average number of suprabasal cells per clone
        '''
        if self._av_sup_cs is None:
            if self.containsBasal:
                bas_mask = self.bas_ls > 0                
                self._av_sup_cs = np.sum(self.sup_ls*bas_mask*(self.sup_ls >= 0),0)/self.nsur
            else:
                self._av_sup_cs = np.sum(self.sup_ls*(self.sup_ls >= 0),0)/self.non_missing
                #self._av_sup_cs = np.mean(self.sup_ls,0)
            
        return self._av_sup_cs    
    
    
    @property
    def se_bas_cs(self):
        '''
            Returns the standard error of basal cells per clone
        '''
        if self._se_bas_cs is None:
            if self.containsBasal:
                self._se_bas_cs = np.sqrt(np.sum((self.bas_ls > 0)*(self.bas_ls - self.av_bas_cs[None,:])**2,0))/self.nsur
            else:
                self._se_bas_cs = np.sqrt(np.sum((self.bas_ls >= 0)*(self.bas_ls - self.av_bas_cs[None,:])**2,0))/self.non_missing
            
        return self._se_bas_cs
    
    
    @property
    def se_tot_cs(self):
        '''
            Returns the standard error of total cells per clone
        '''
        if self._se_tot_cs is None:
            if self.containsBasal:
                bas_mask = self.bas_ls > 0
                self._se_tot_cs = np.sqrt(np.sum(bas_mask*(self.tot_ls - self.av_tot_cs[None,:])**2,0))/self.nsur
            else:
                self._se_tot_cs = np.sqrt(np.sum((self.tot_ls >= 0)*(self.tot_ls - self.av_tot_cs[None,:])**2,0))/self.non_missing
            
        return self._se_tot_cs
    
    @property
    def se_sup_cs(self):
        '''
            Returns the standard error of suprabasal cells per clone
        '''
        if self._se_sup_cs is None:
            if self.containsBasal:
                bas_mask = self.bas_ls > 0
                self._se_sup_cs = np.sqrt(np.sum(bas_mask*(self.sup_ls - self.av_sup_cs[None,:])**2,0))/self.nsur
            else:
                self._se_sup_cs = np.sqrt(np.sum((self.sup_ls >= 0)*(self.sup_ls - self.av_sup_cs[None,:])**2,0))/self.non_missing
            
        return self._se_sup_cs
    
    @property 
    def ev_bas_cs(self):
        '''
            Returns the even bias for basal clone sizes
        '''
        if self._ev_bas_cs is None:
            self._ev_bas_cs = analysis.evenbias(self.bas_ls.T)
            
        return self._ev_bas_cs
    
    @property
    def ev_tot_cs(self):
        '''
            Returns the even bias for total clone sizes
        '''
        if self._ev_tot_cs is None:
            self._ev_tot_cs = analysis.evenbias(self.tot_ls.T)
            
        return self._ev_tot_cs
    
    @property
    def ev_sup_cs(self):
        '''
            Returns the even bias for suprabasal clone sizes
        '''
        if self._ev_sup_cs is None:
            self._ev_sup_cs = analysis.evenbias(self.sup_ls.T)
            
        return self._ev_sup_cs
    
    
    def bas_pdf(self, max_cs = None):
        '''
            Returns the pdf of basal cells per clone
        '''
        if self._bas_pdf is None or (max_cs is not None and self._bas_pdf.shape[1] != max_cs):
            if self.containsBasal:
                self._bas_pdf = analysis.basal_cont_pdf(self.bas_ls, self.bas_ls, self.nsur, max_cs)
            else:
                self._bas_pdf = analysis.joint_pdf(self.bas_ls, max_cs)
            
        return self._bas_pdf
    
    
    def tot_pdf(self, max_cs = None):
        '''
            Returns the pdf of total cells per clone
        '''
        if self._tot_pdf is None or (max_cs is not None and self._tot_pdf.shape[1] != max_cs):
            if self.containsBasal:
                self._tot_pdf = analysis.basal_cont_pdf(self.tot_ls, self.bas_ls, self.nsur, max_cs)
            else:
                self._tot_pdf = analysis.joint_pdf(self.tot_ls, max_cs)
            
        return self._tot_pdf
    
    
    def sup_pdf(self, max_cs = None):
        '''
            Returns the pdf of suprabasal cells per clone
        '''
        if self._sup_pdf is None or (max_cs is not None and self._sup_pdf.shape[1] != max_cs):
            if self.containsBasal:
                self._sup_pdf = analysis.basal_cont_pdf(self.sup_ls, self.bas_ls, self.nsur, max_cs)
            else:
                self._sup_pdf = analysis.joint_pdf(self.sup_ls, max_cs)
            
        return self._sup_pdf
        

        
    def surviving_at_time(self, t):
        '''
            Returns a mask of which clones are surviving at a given time
        '''
        return self.bas_ls[:,t] > 0        
        
    
    def boot(self, quantity, nboots = 1000, calc = 'se', seed = 0):
        ''' 
            Can boostrap the standard error in any function from clones,
                does this in a slow way that minimises code
        '''
        
        assert(hasattr(self, quantity))
        
        bclones = self.copy()
        bdata = []
        np.random.seed(seed)

        bclones.reset_properties()

        for i in range(nboots):
            sample = np.random.randint(0, bclones.nclones, size = bclones.nclones)
            bclones.livesize = self.livesize[sample,:,:]
            bclones.reset_properties()
            bdata.append(getattr(bclones, quantity))

        bdata = np.array(bdata)
        
        if calc == 'se':
            return np.std(bdata,0)
        elif calc == 'mean':
            return np.mean(bdata,0)
        else:
            return bdata
    
    
    def copy(self):
        return copy.deepcopy(self)   
    
    
    
    
    
         



class xanalysis(lanalysis):
    def __init__(self, livesize, popnames=None, tvals=None, containsBasal=True, suprabasal=True):
        '''
            Analysis of clones from experiments given a live-size is provided,
                handles missing data,
        '''
        
        lanalysis.__init__(self, livesize, popnames, tvals, containsBasal, suprabasal)
        
        


class sanalysis(lanalysis):
    def __init__(self, results = None, popnames = None, skip = 1, tvals = None, containsBasal = True, suprabasal = True, 
                 #If don't provide results can provide the following
                 cellids = None, ncells = None, ancs = None, nsbcells = None, livesize = None):
        '''
            Analysis of clones from simulations
            
            Initially we calculate the live-size of each clone
            
            Handles population quantities which are not included in the base class,
                as they require more complex handling of missing data
            Assums that there is no missing data
            
            Flags:
                containsBasal, set to True to only include clones that contain basal cells
                suprabasal, set to True to include suprabasal cells in the analysis
                    Currently coded to only work with one supra-basal layer
        '''
        
        lanalysis.__init__(self, livesize, popnames, tvals, containsBasal, suprabasal)
        
        if results is not None:
            cellids = results['cellid']
            ncells = results['ncells']
            nsbcells = results['nsbcells']
            ancs = results['anc']
            
            if 'tiss' in results:
                sb_born = results['tiss'].cellpops[-1].born #extract suprabasal
            else:
                sb_born = None
         
        #Set times
        self.skip = skip
        
        if livesize is None:
            if tvals is None:
                self.tvals = np.arange(self.inducttime, len(ncells), skip)
            else:
                self.tvals = np.array(tvals)
            
            #Attributes
            self.npops = ncells.shape[1]
            self.inducttime = 0
            self.nmarked = ncells[self.inducttime]
            self.nsb0 = nsbcells[self.inducttime] #number of suprabasal cells at t =0
            self.nclones = np.sum(self.nmarked)
            self.ntimes = len(cellids)
                
                
            #For calculating clone-size, use pandas series because vectorisation is allowed
            self.clone_inverseid = pd.Series(dict(zip(cellids[self.inducttime], np.arange(self.nclones))), dtype = 'int32')
            self.ancs = {p: pd.Series(ancs[p], dtype = 'int32') for p in range(self.npops + 1)}
            
            #Storing of when suprabasal cells are born, allows re-running suprabasal layer
            if sb_born is not None:
                self.sb_born = pd.Series(sb_born, dtype = 'int32')
            else:
                self.sb_born = None
                        
            #Calculate the live-size of each clone, all other quantities are calculated on the fly
            self.livesize = self.calc_livesize(cellids, ncells, ancs, nsbcells)
        
        
        #Enforce no missing data
        assert(np.min(self.livesize) >= 0)
        
        
        
    def reset_properties(self):
        #Reset base quantities
        lanalysis.reset_properties(self)
        
        #Population quantities
        self._splits = None #Splits between different population indices
        self._popsur = None #Survival probability of each population
        self._pop_ave_bas_cs = None #Average number of basal cells per clone labelled by a certain population
        self._pop_ave_tot_cs = None #Average number of total cells per clone labelled by a certain population        
        self._pop_ave_sup_cs = None #Average number of suprabasal cells per clone labelled by a certain population
        self._pop_bas_pdf = None #PDF of basal cells per clone labelled by a certain population
        self._pop_tot_pdf = None #PDF of total cells per clone labelled by a certain population
        self._pop_sup_pdf = None #PDF of suprabasal cells per clone labelled by a certain population
        self._ave_cs_by_pop = None # Average cells of type per clone
        self._pop_ave_cs_by_pop = None # Average cells of type per clone labelled by a certain population
               
               
    def rerun_suprabasal(self, shed, refract = 0):
        '''
            Reruns the suprabasal layer and calculates new live-size
        '''
        
        cloneidx = self.clone_inverseid[self.ancs[self.npops][self.sb_born.index.to_numpy()]].values        
        self.livesize = fast_rerun_supra(self.livesize, cloneidx, self.sb_born.values, shed, refract, self.skip)
        
        self.reset_properties()
        
        return self.livesize


    
    @property
    def splits(self):
        '''
            Returns the splits between different population indices
        '''
        if self._splits is None:
            self._splits = np.array([0] + list(np.cumsum(self.nmarked)) + [np.sum(self.nmarked) + self.nsb0])
            
        return self._splits
    
    @property
    def popsur(self):
        '''
            Returns the survival probability of each population
        '''
        if self._popsur is None:
            self._popsur = np.zeros((self.npops, self.ntimes)) # Survival probability of clones marked from specific populations
            for p in range(self.npops):
                self._popsur[p,:] = np.sum(self.bas_ls[self.splits[p]:self.splits[p+1],:] > 0,0)/np.sum(self.bas_ls[self.splits[p]:self.splits[p+1],0])

        return self._popsur
    
    @property
    def pop_ave_bas_cs(self):
        '''
            Returns the average number of basal cells per clone labelled by a certain population
        '''
        if self._pop_ave_bas_cs is None:
            self._pop_ave_bas_cs = np.zeros((self.npops, self.ntimes))
            
            for p in range(self.npops):
                if self.containsBasal:
                    self._pop_ave_bas_cs[p,:] = np.sum(self.bas_ls[self.splits[p]:self.splits[p+1],:],0)/self.popsur[p,:]/self.nmarked[p]
                else:
                    self._pop_ave_bas_cs[p,:] = np.mean(self.bas_ls[self.splits[p]:self.splits[p+1],:],0)

        return self._pop_ave_bas_cs
    
    @property
    def pop_ave_tot_cs(self):
        '''
            Returns the average number of total cells per clone labelled by a certain population
        '''
        if self._pop_ave_tot_cs is None:
            self._pop_ave_tot_cs = np.zeros((self.npops, self.ntimes))
            
            for p in range(self.npops):
                if self.containsBasal:
                    bas_mask = self.bas_ls[self.splits[p]:self.splits[p+1],:] > 0                    
                    self._pop_ave_tot_cs[p,:] = np.sum(self.tot_ls[self.splits[p]:self.splits[p+1],:]*bas_mask,0)/self.popsur[p,:]/self.nmarked[p]
                else:
                    self._pop_ave_tot_cs[p,:] = np.mean(self.tot_ls[self.splits[p]:self.splits[p+1],:],0)

        return self._pop_ave_tot_cs
    
    @property
    def pop_ave_sup_cs(self):
        '''
            Returns the average number of suprabasal cells per clone labelled by a certain population
        '''
        if self._pop_ave_sup_cs is None:
            self._pop_ave_sup_cs = np.zeros((self.npops, self.ntimes))
            
            for p in range(self.npops):
                if self.containsBasal:
                    bas_mask = self.bas_ls[self.splits[p]:self.splits[p+1],:] > 0                    
                    self._pop_ave_sup_cs[p,:] = np.sum(self.sup_ls[self.splits[p]:self.splits[p+1],:]*bas_mask,0)/self.popsur[p,:]/self.nmarked[p]
                else:
                    self._pop_ave_sup_cs[p,:] = np.mean(self.sup_ls[self.splits[p]:self.splits[p+1],:],0)

        return self._pop_ave_sup_cs
                    

    def pop_bas_pdf(self, max_cs = None):
        '''
            Returns the pdf of basal cells per clone labelled by a certain population
        '''
        if self._pop_bas_pdf is None or (max_cs is not None and (self._sup_pdf.shape[1] != max_cs).any()):
            if max_cs is None:
                max_cs = self.bas_ls.max()
                
            self._pop_bas_pdf = np.zeros((self.npops, self.ntimes, max_cs + 1))
            
            for p in range(self.npops):
                if self.containsBasal:
                    self._pop_bas_pdf[p,:,:] = analysis.basal_cont_pdf(self.bas_ls[self.splits[p]:self.splits[p+1],:], self.bas_ls[self.splits[p]:self.splits[p+1],:], None, max_cs)
                else:
                    self._pop_bas_pdf[p,:,:] = analysis.joint_pdf(self.bas_ls[self.splits[p]:self.splits[p+1],:], max_cs)

        return self._pop_bas_pdf
    

    def pop_tot_pdf(self, max_cs = None):
        '''
            Returns the pdf of total cells per clone labelled by a certain population
        '''
        if self._pop_tot_pdf is None or (max_cs is not None and (self._pop_tot_pdf.shape[1] != max_cs).any()):
            if max_cs is None:
                max_cs = self.tot_ls.max()
                
            self._pop_tot_pdf = np.zeros((self.npops, self.ntimes, max_cs + 1))
            
            for p in range(self.npops):
                if self.containsBasal:
                    self._pop_tot_pdf[p,:,:] = analysis.basal_cont_pdf(self.tot_ls[self.splits[p]:self.splits[p+1],:], self.bas_ls[self.splits[p]:self.splits[p+1],:], None, max_cs)
                else:
                    self._pop_tot_pdf[p,:,:] = analysis.joint_pdf(self.tot_ls[self.splits[p]:self.splits[p+1],:], max_cs)
                    
                    
        return self._pop_tot_pdf
    
    
    def pop_sup_pdf(self, max_cs = None):
        '''
            Returns the pdf of suprabasal cells per clone labelled by a certain population
        '''
        if self._pop_sup_pdf is None or (max_cs is not None and (self._pop_sup_pdf.shape[1] != max_cs).any()):
            if max_cs is None:
                max_cs = self.sup_ls.max()
                
            self._pop_sup_pdf = np.zeros((self.npops, self.ntimes, max_cs + 1))
            
            for p in range(self.npops):
                if self.containsBasal:
                    self._pop_sup_pdf[p,:,:] = analysis.basal_cont_pdf(self.sup_ls[self.splits[p]:self.splits[p+1],:], self.bas_ls[self.splits[p]:self.splits[p+1],:], None, max_cs)
                else:
                    self._pop_sup_pdf[p,:,:] = analysis.joint_pdf(self.sup_ls[self.splits[p]:self.splits[p+1],:], max_cs)
                    
                    
        return self._pop_sup_pdf
    
    @property
    def ave_cs_by_pop(self):
        '''
            Average cells of type per clone
        '''
        
        if self._ave_cs_by_pop is None:
            if self.containsBasal:
                self._ave_cs_by_pop  = np.sum(self.livesize,0)/self.nsur[:,None]
            else:
                self._ave_cs_by_pop  = np.mean(self.livesize,0)
                
        return self._ave_cs_by_pop
    
    @property
    def pop_ave_cs_by_pop(self):
        '''
            Average cells of type per clone, given clones marking a specific population
        '''
        
        if self._pop_ave_cs_by_pop is None:
            if self.containsBasal:
                self._pop_ave_cs_by_pop  = np.zeros((self.npops, self.ntimes, self.npops + 1))
                for p in range(self.npops):
                    bas_mask = self.bas_ls[self.splits[p]:self.splits[p+1],:] > 0
                    self._pop_ave_cs_by_pop[p,:,:] = np.sum(self.livesize[self.splits[p]:self.splits[p+1],:,:]*bas_mask[:,:,None],0)/self.popsur[p,:,None]/self.nmarked[p]
            else:
                self._pop_ave_cs_by_pop  = np.mean(self.livesize[self.splits[p]:self.splits[p+1],:,:],0)
                
        return self._pop_ave_cs_by_pop
        


    def calc_livesize(self, cellids = None, ncells = None, ancs = None, nsbcells = None, 
                      results = None, pops = None, inititialise = True):
        '''
            Calculate the live-size of each clone
        '''
        
        if inititialise:
            self.livesize = np.zeros((self.nclones, len(self.tvals), self.npops + 1), dtype = 'int32')
        
        if results is not None:
            cellids = results['cellid']
            ncells = results['ncells']
            nsbcells = results['nsbcells']
            ancs = results['anc']
            
        if pops is None:
            pops = np.arange(self.npops + 1)
        elif np.ndim(pops) == 0:
            pops = [pops]
            
        if self.ancs is None:
            anc1 = {p: pd.Series(ancs[p], dtype = 'int32') for p in pops}
            self.ancs = anc1
        else:
            anc1 = self.ancs
        
        for t in range(len(self.tvals)):
            tt = int(self.tvals[t]/self.skip)
            assert tt*self.skip <= len(ncells) , 'Time out of range at ' + str(tt)
            assert(len(cellids[int(tt)]) == np.sum(ncells[tt*self.skip]) + nsbcells[tt*self.skip])        
            self.livesize = analysis.update_live_sizes(self.livesize, self.npops, t, ncells[tt*self.skip], nsbcells[tt*self.skip], cellids[tt], anc1, self.clone_inverseid,
                                                       pops = pops)
        
        return self.livesize
        

    def recalculate_livesize(self, new_results):
        '''
            Recalculate livesize if we have new results
        '''
        
        self.livesize = self.calc_livesize(results = new_results, pops = None, inititialise = True)
        self.reset_properties()
        return self.livesize
        
        
    def recalc_supra_livesize(self, new_results):
        '''
            Re-calculate live-size for suprabasal cells only
        '''
        self.livesize[:,:,-1] = 0
        self.livesize = self.calc_livesize(results = new_results, pops = self.npops, inititialise = False)
        self.reset_properties()
        return self.livesize
    

    


@njit
def fast_rerun_supra(livesize, cloneidx, born, shed, refract, skip):
    ''' 
        Re-running the suprabasal layer with numbas

    '''
    
    n = len(cloneidx)
    assert(n == len(cloneidx))
    
    shed_times = np.random.exponential(shed, n) + refract + 1
    livesize[:,:,-1] = 0

    for i in range(len(cloneidx)):
        livesize[cloneidx[i], 1 + int(born[i]/skip):(1 + int(born[i]/skip + shed_times[i]/skip)),-1] += 1
        
    return livesize