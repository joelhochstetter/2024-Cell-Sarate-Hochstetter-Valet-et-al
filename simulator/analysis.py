'''

    Clonal analysis of tissue simulations


'''



import time
import collections
import scipy.signal 
import sys
sys.path.append("../")
from tissue import *
from utils import *
from cellulation import *
from collections import ChainMap
import os
import random
import scipy
from scipy.sparse.csgraph import connected_components
from numba import jit, njit
import pandas as pd
import sparse as sp


def calculateSkip(results = None, ncells = None, cellids = None):
    #problem has a well-posed solution if we have sufficiently many points
    # condition is m(m-1) >= t - 1
    if ncells is None:
        t = len(results['ncells'])
    else:
        t = len(ncells)
        
    if cellids is None:
        m  = len(results['cellid'])
    else:
        m = len(cellids)
        
    #print((t - 1)/(m-1), (t - 1)/m)
    if (t - 1)*(1/(m-1) - 1/m) > 1:
        return np.nan
    else:
        return int(np.floor((t-1)/(m-1)))
        

        
def livesize(cellids = None, ncells = None, ancs = None, nsbcells = None, skip = 1, tvals = None, results = None, 
             inverseid = None):
    '''
        Calculate the size of each clone at each time-point broken down by cell-types
    '''

    if results is not None:
        cellids = results['cellid']
        ncells = results['ncells']
        nsbcells = results['nsbcells']
        ancs = results['anc']

    inducttime = 0
    time = len(ncells)
    npops = ncells.shape[1]
    
    #ancs = dict(ChainMap(*ancs))
    #anc1 = np.zeros((1 + np.max(list(ancs.keys()))), dtype = 'int32')
    #anc1[np.array(list(ancs.keys()))] = np.array(list(ancs.values()))

    anc1 = {p: pd.Series(ancs[p], dtype = 'int32') for p in range(len(ancs))}

    if skip is None:
        skip = calculateSkip(ncells = ncells, cellids = cellids)
        if np.isnan(skip):
            print('Skip not defined')
            assert(0)

    if tvals is None:
        tvals = np.arange(inducttime, time + 1, skip)

    nummarked = ncells[inducttime]
    nclones = np.sum(nummarked) 
    ntimes = len(tvals)
    
    if inverseid is None:
        inverseid = pd.Series(dict(zip(cellids[inducttime], np.arange(nclones))), dtype = 'int32')

        
    ls = np.zeros((nclones, ntimes, npops + 1), dtype = 'int32')
    
    for t in range(len(tvals)):
        tt = int(tvals[t]/skip)
        assert tt*skip <= len(ncells) , 'Time out of range at ' + str(tt)
        assert(len(cellids[int(tt)]) == np.sum(ncells[tt*skip]) + nsbcells[tt*skip])        
        ls = update_live_sizes(ls,  npops, t, ncells[tt*skip], nsbcells[tt*skip], cellids[tt], anc1, inverseid)
    
    return ls


    
def update_live_sizes(ls, npops, t, ncells, nsbcells, cellid, ancs, inverseid, pops = None):
    '''
        Updates live_sizes at a single time-point
        
        Need to use inverseid and ancs as pandas series (or numpy arrays)
    '''
    
    splits = pop_splits(npops, ncells, nsbcells)
    
    if pops is None:
        pops = np.arange(npops + 1)
    
    for p in pops:
        clone_by_id = inverseid[ancs[p][cellid[splits[p]:splits[p+1]]].values].values
        for i in range(len(clone_by_id)):
            ls[clone_by_id[i], t, p] += 1
        
    return ls
            
        
    
"""
#@jit(nopython = True)
def update_live_sizes(ls, npops, t, cellid, ncells, nsbcells, ancs, inverseid):
    '''
        One implementation using inverseid and ancs as numpy arrays
    '''
    clone_by_id = inverseid[ancs[cellid]]
    pop_by_cell = pop_by_cellid(cellid, ncells, nsbcells, npops)
    for i in range(len(cellid)):
        ls[clone_by_id[i], t, pop_by_cell[i]] += 1 #check this works
    
    return ls
"""

@njit
def create_dict(items):
    return {k: v for k,v in items}

def pop_splits(npops, ncells, nsbcells):
    splits = np.zeros(npops + 2, dtype = 'int32')
    splits[1:-1] = np.cumsum(ncells)
    splits[-1] = np.sum(ncells) + nsbcells
    return splits

#@jit(nopython = True)
def pop_by_cellid(cellids, ncells, nsbcells, npops):
    splits = pop_splits(npops, ncells, nsbcells)
    pop_by_id = np.zeros(len(cellids), dtype = 'int32')
    
    for p in range(npops + 1):
        pop_by_id[splits[p]:splits[p+1]] = p
        
    return pop_by_id
    
    
#@njit
def joint_pdf(ls, max_pop_cs = None, basal_cont = False, sparse = False):
    '''
        Gives joint pdf if multiple populations are provided, 
            else gives pdf a single population
            
        max_pop_cs is maximum clone-size per population + 1
            includes for suprabasal layer
            
        Capping is only implemented on 1D distributions
        
        basal_cont = True, specifies only clones with at least one basal cell
    '''
    nmarked = ls.shape[0]
    ntimes  = ls.shape[1]
    
    if max_pop_cs is None:
        max_pop_cs  = np.max(ls, axis = (0,1)) + 1
    elif np.ndim(max_pop_cs) > 0:
        max_pop_cs = np.array(max_pop_cs) + 1
    else:
        max_pop_cs += 1
    
    if np.ndim(ls) >= 3: 
        if sparse is False:
            jpdf = np.zeros((ntimes, *max_pop_cs))
        else:
            counts = {}
            total_counts = np.zeros(ntimes, dtype=int)
        #jpdf = csr_matrix((ntimes, *max_pop_cs), dtype = np.float64)
                
        # Check values are not too large
        for p in range(ls.shape[2]):
            assert(np.max(ls[:,:,p]) <= max_pop_cs[p])
        
            
        for t in range(ntimes):
            for m in range(nmarked):  
                if np.min(ls[m,t,:]) >= 0: #e.g. not including missing data
                    if sparse is False:
                        #jpdf[t, *ls[m,t,:]] += 1  #works on new python but seemingly not old
                        jpdf[tuple([t] + list(ls[m,t,:]))] += 1
                        #jpdf[t, *tuple(ls[m, t, :])] += 1
                    else:
                        total_counts[t] += 1
                        if tuple([t] + list(ls[m,t,:])) in counts:
                            counts[tuple([t] + list(ls[m,t,:]))] += 1.0
                        else:
                            counts[tuple([t] + list(ls[m,t,:]))] = 1.0
                            
    elif np.ndim(ls) == 2: #only one population
        jpdf = np.zeros((ntimes, max_pop_cs))
        
        if np.max(ls) > max_pop_cs:
            capped_ls = ls.copy() #capped at max_pop_cs
            capped_ls[capped_ls > max_pop_cs] = max_pop_cs       
        else:
            capped_ls = ls

        for t in range(ntimes):
            for m in range(nmarked):
                if capped_ls[m, t] >= 0:
                    jpdf[t, capped_ls[m, t]] += 1
    else:
        print('Invalid shape for live sizes')
        assert(0)
     
    if sparse is False:
        if basal_cont:
            jpdf[:,0,...] = 0
            
        jpdf /= nmarked
        
        #Normalising
        norms = np.sum(jpdf, axis = tuple(np.arange(1, np.ndim(jpdf))), keepdims = True)
        if np.max(np.abs(norms - 1.0)) > 1e-10:
            print('Fixing normalisation, may be due to missing data')
            jpdf /= norms
    else:
        for k in counts.keys():
            counts[k] /= total_counts[k[0]]
        jpdf = sp.COO(np.array(list(counts.keys())).T, np.array(list(counts.values())), shape = (ntimes, *max_pop_cs))
            
    return jpdf


def basal_cont_pdf(reduced_ls, bas_ls, nsur = None, max_pop_cs = None):
    '''
        From the reduced_ls (either sup_ls or total_ls or a pop_ls)
            and the bas_ls, calculates the pdf of clones that contain
            basal cells
            
        nsur is the number of surviving clones, if not provided it is calculated
            
        Assumes reduced_ls is nclones x ntimes, and same shape as bas_ls
    '''

    assert(np.ndim(reduced_ls) == 2)
    assert(np.shape(bas_ls) == np.shape(reduced_ls))
    
    if nsur is None:
        nsur = np.sum(bas_ls > 0, axis = 0)    
    
    ntimes  = reduced_ls.shape[1]
    if max_pop_cs is None:
        max_pop_cs = np.max(reduced_ls)

    pdf = np.zeros((ntimes, max_pop_cs + 1))

    capped_ls = reduced_ls.copy() #capped at max_pop_cs
    capped_ls[capped_ls > max_pop_cs] = max_pop_cs
    
    for t in range(ntimes):
        for m in np.where(bas_ls[:,t] > 0)[0]:
            pdf[t, capped_ls[m, t]] += 1

    pdf[nsur > 0, :] /= nsur[nsur > 0,None]
    pdf[nsur == 0, 0] = 1 #retain normalisation for extinct population
    return pdf



def clonestats(results, tiss, inducttime = 0, tvals = None, skip = None, useBas = True, useSB = False, containsBasal = False):
    #Basal      clone sizes: useBas = True,  useSB = False
    #Suprabasal clone sizes: useBas = False, useSB = True
    #Total      clone sizes: useBas = True,  useSB = True
    #containsBasal = True: Only count clones with at least one basal cell, only matters for total clone sizes
    
    if useSB is True and useBas is False:
        print('Warning code is broken, for suprabasal only. Use new clone-sizes paradigm instead')
    
    if skip is None:
        skip = calculateSkip(results)
        if np.isnan(skip):
            print('Skip not defined')
            assert(0)
    
    if tvals is None:
        tvals = list(range(inducttime, tiss.time + 1, skip))
        # Must not be earlier than induction time
        
    
    nummarked = results['ncells'][inducttime]
    npts = np.sum(nummarked)    
    marked = results['cellid'][int(inducttime/skip)][:npts]
    #print(marked)
    falsemark = results['cellid'][int(inducttime/skip)][npts:] #clones originating in suprabasal
    #print(len(falsemark))
    
    if inducttime == 0:
        if tiss.npops > 1 or tiss.suprabasal:
            ancs  = dict(ChainMap(*results['anc']))
        else:
            ancs = results['anc']
    else:
        ancs = ancestorsMarked(results['dupTimes'], marked)
    
    cellid = results['cellid']

    popsizes  = {t: [[] for p in range(tiss.npops)] for t in tvals}    
    allsizes  = {t: [] for t in tvals}
    allclones = {t: {} for t in tvals}
    ncells = {t: [] for t in tvals}
    nsur   = {t: [] for t in tvals}
    
    #print(marked, falsemark)
    
    
    if useBas:
        lowerid = np.zeros(tiss.time + 1, dtype = np.int32)
    else:
        lowerid = np.sum(results['ncells'],1)
    
    if useSB:
        upperid = np.sum(results['ncells'],1) + results['nsbcells']
    else:
        upperid = np.sum(results['ncells'],1)
    
    
    for t in tvals:
        tt = int(t/skip)
        #print(lowerid[t], upperid[t], np.sum(results['ncells'],1)[t], results['nsbcells'][t])
        #print(len(cellid[int(t/skip)]))
        #print(np.sum(results['ncells'],1)[t] + results['nsbcells'][t])

        if tt*skip > len(results['ncells']) or tt > len(cellid):
            print(tt, tt*skip, len(results['ncells']), len(cellid))

        assert(tt*skip <= len(results['ncells']))
        if len(cellid[int(tt)]) != np.sum(results['ncells'][tt*skip]) + results['nsbcells'][tt*skip]:
            print(t, len(cellid[int(tt)]), np.sum(results['ncells'][tt*skip]), results['nsbcells'][tt*skip])
        #print(results['ncells'][tt*skip], results['nsbcells'][tt*skip], cellid[int(tt)][lowerid[tt*skip]:upperid[tt*skip]])
        assert(len(cellid[int(tt)]) == np.sum(results['ncells'][tt*skip]) + results['nsbcells'][tt*skip])
        cloneidx = [ancs[i] for i in cellid[int(tt)][lowerid[tt*skip]:upperid[tt*skip]]] #stores the clone-index per cell in tissue
        
        clones = {i: [] for i in marked} #stores which cell currently in tissue is which clone
        
        
        #print(marked)
        for i in range(len(cloneidx)):
            if cloneidx[i] in marked: #excludes clones originating in suprabasal layer,etc.
                clones[cloneidx[i]].append(i)

        allclones[t] = clones
        clonesizes = collections.Counter(cloneidx)
        #print(clonesizes)
        
        
        
        
        nobas = 0
        if containsBasal:
            for m in marked:
                if len(clones[m]) > 0 and np.min(clones[m]) >= np.sum(results['ncells'][tt*skip]):
                    #if t == 10:
                    #    print(clones[m], np.sum(results['ncells'][tt*skip]))
                    nobas += clonesizes[m]
                    #print(m, clonesizes[m])               
                    clones.pop(m)
                    clonesizes.pop(m)
            '''
            if not useBas:
                basclones = [ancs[i] for i in cellid[int(tt)][:lowerid[tt*skip]]]
                unqClones = len(list(set(basclones)))
                clonesizes = dict(clonesizes)
                for m in marked:
                    if m not in clonesizes.keys and m in unqClones:
                        clonesizes[m] = 0
            '''   

                
                       
        fmark = 0
        for m in falsemark:
            if m in clonesizes.keys():
                fmark += clonesizes[m]  
                #print(m, clonesizes[m])
                clones.pop(m)                         
                clonesizes.pop(m)

                if t == 0:
                    print('Falsely marked', m)
        
        

        
        allsizes[t] = np.sort(list(clonesizes.values()))
        #if containsBasal:
        #    print(np.mean(allsizes[t]), np.mean(allsizes[t][allsizes[t] > 0]))        
        ncells[t] = results['ncells'][tt*skip]
        #print(t)
        popsizes[t], nsur[t] = getPopSizes(clonesizes, marked, nummarked) 
        #print(t, np.sum(allsizes[t]), np.sum(ncells[t]), results['nsbcells'][tt*skip], nobas, fmark, len(cloneidx), upperid[tt*skip], (useBas*np.sum(ncells[t]) + useSB*(results['nsbcells'][tt*skip] - fmark - nobas)))
        assert(np.sum(allsizes[t]) == (useBas*np.sum(ncells[t]) + useSB*(results['nsbcells'][tt*skip] - fmark - nobas)))
        
    return clonedetails(allsizes, allclones, popsizes, nsur, marked, nummarked, ncells, tiss, tvals) 
    

    
def clonedetails(allsizes, allclones, popsizes, nsur, marked, nummarked, ncells, tiss, tvals):

    npts = np.sum(nummarked)
    popave = np.zeros((tiss.npops, len(tvals))) #average clone-size
    poperr = np.zeros((tiss.npops, len(tvals))) #sem in clone size
    popsur = np.zeros((tiss.npops, len(tvals))) #survival probability
    popser = np.zeros((tiss.npops, len(tvals))) #survival probability error
    poplcd = np.zeros((tiss.npops, len(tvals))) #labelled cell density
    poplce = np.zeros((tiss.npops, len(tvals))) #labelled cell density error
    popeve = np.zeros((tiss.npops, len(tvals))) #Fraction of even sized clones
    popeer = np.zeros((tiss.npops, len(tvals))) #Fraction of even sized clones error
    popeco = np.zeros((tiss.npops, len(tvals))) #Even bias error
    popnsr = np.zeros((tiss.npops, len(tvals))) #number of surviving clones

    allcave = np.zeros(len(tvals)) #average for all clones
    allcerr = np.zeros(len(tvals)) #sem in clone size for all clones
    allcsur = np.zeros(len(tvals)) #survival probability
    allclcd = np.ones( len(tvals)) #labelled cell density
    allclce = np.zeros(len(tvals)) #labelled cell density error
    allceve = np.zeros(len(tvals)) #fraction even
    allceco = np.zeros(len(tvals)) #even correlation / bias
    allcnsr = np.zeros(len(tvals)) #number of surviving clones
    
    
       
   
    avecell = np.zeros(len(tvals)) #average number of cells
    stdcell = np.zeros(len(tvals)) #std number of cells

    for t in range(len(tvals)):
        #all statistics
        allcave[t] = np.mean(allsizes[tvals[t]])
        allcerr[t] = np.std(allsizes[tvals[t]])/np.sqrt(len(allsizes[tvals[t]]))
        allcsur[t] = np.sum(nsur[tvals[t]])/npts
        allceve[t] = 1 - np.mean(np.array(allsizes[tvals[t]]) % 2)
        allclcd[t] = np.sum(ncells[tvals[t]])/npts
        avecell[t] = np.sum(ncells[tvals[t]])
        allcnsr[t] = np.sum(nsur[tvals[t]])
        #stdcells[p,t] = np.std(ncells[tvals[t]][p])
        
        
        
        
        #population statistics
        for p in range(tiss.npops):
            if nsur[tvals[t]][p] > 0:
                popave[p,t] = np.mean(popsizes[tvals[t]][p])
                poperr[p,t] = np.std(popsizes[tvals[t]][p])/np.sqrt(nsur[tvals[t]][p])
                popsur[p,t] = np.mean(np.array(nsur[tvals[t]][p])/(nummarked[p]))
                #popser[p,t] = np.std(np.array(nsur[tvals[t]][p])/(nummarked[p]))/np.sqrt(nsur[tvals[t]][p])
                popeve[p,t] = 1 - np.mean(np.array(popsizes[tvals[t]][p]) % 2)
                
                poplcd[p,t] = np.mean(np.array(ncells[tvals[t]])/nummarked[p]) #labelled cells
                #poplce[p,t] = np.std(np.array(ncells[tvals[t]])/nummarked[p])
                popnsr[p,t] = nsur[tvals[t]][p]
            
    
    allceco = evCorr(allceve, allcave)

    for p in range(tiss.npops):
        popeco[p,:] = evCorr(popeve[p,:], popave[p,:])
    
    names = ['popsizes', 'allsizes', 'ncells', 'nsur', 'nummarked', 'marked', 'popave', 
                'poperr', 'popsur', 'popser', 'poplcd', 'poplce', 'popeve', 'popeer', 
                'allcave', 'allcerr', 'allcsur', 'allclcd', 'allclce', 'avecell', 'stdcell',
                'allceve', 'tvals', 'allclones', 'allceco', 'popeco', 'allcnsr', 'popnsr']
                
    datum = [popsizes, allsizes, ncells, nsur, nummarked, marked, popave, poperr, popsur,
                 popser, poplcd, poplce, popeve, popeer, allcave, allcerr, allcsur, 
                 allclcd, allclce, avecell, stdcell, allceve, np.array(tvals), allclones,
                  allceco, popeco, allcnsr, popnsr]
    
    cloneResults = dict(zip(names, datum))

    return cloneResults
                

def cloneFragments(allclones, results, L,  marked = None, tvals = None, inducttime = 0, skip = 10):
    '''
        To do: 
            Get working for subpopulations. The trick is I need to determine the cell-type of 
            ancestor cells. i.e. associate each clone with an ancestor marked cell. Then select
            the appropriate nummarked to resolve the discrepancy. 

    '''
    
    if tvals is None:
        tvals = list(range(inducttime, len(results['ncells']), skip))
        # Must not be earlier than induction time
    
    npops = len(results['ncells'][0,:])

    #subpopsizes  = {t: [[] for p in range(tiss.npops)] for t in tvals}    
    suballsizes  = {t: [] for t in tvals}
    nsubclones   = {t: [] for t in tvals}
    ncells       = {t: [] for t in tvals}
    
    for t in tvals:
        tiss = tissue(results['cellpos'][int(t/skip)], L = L)
        adjmat = np.zeros((tiss.npts, tiss.npts), dtype = bool)
        pbcidx = tiss.pbcidx
        npts = tiss.npts
        ridge_points = tiss.ridge_points

        for p in ridge_points:
            if np.min(p) < npts:
                p = pbcidx[p] 
                adjmat[p[0], p[1]] = 1.0
                adjmat[p[1], p[0]] = 1.0

        clonemat = np.zeros((tiss.npts, tiss.npts), dtype = bool)
        
        clones =  list(allclones[t].values())
        
        for cl in clones:
            pairs = np.array([(a, b) for idx, a in enumerate(cl) for b in cl[idx + 1:]])
            for p in pairs:
                clonemat[p[0], p[1]] = True
                clonemat[p[1], p[0]] = True

        subclonemat = clonemat & adjmat

        nsubclones[t], subcloneidx = connected_components(subclonemat)

        subclones = {i: [] for i in range(nsubclones[t])}

        for i in range(len(subcloneidx)):
            subclones[subcloneidx[i]].append(i)
            
        subclonesizes = collections.Counter(subcloneidx)
        
        suballsizes[t] = np.sort(list(subclonesizes.values()))

        #subpopsizes[t], subnsur[t] = getPopSizes(subclonesizes, marked, nummarked)
        ncells[t] = results['ncells'][t]
        
    return suballsizes, nsubclones
    #clonedetails(allsizes, popsizes, nsur, marked, nmarked, ncells, tiss, tvals) 


        
      
def combineDicts(listofdicts):
    if len(listofdicts) == 1:
        return listofdicts[0]
    else:
        d1 = listofdicts[0]
        d2 = listofdicts[1]
        combined_keys = d1.keys() | d2.keys()
        d_comb = {key: d1.get(key, []) + d2.get(key, []) for key in combined_keys}

        for i in range(len(listofdicts) - 2):
            d = listofdicts[i + 2]
            combined_keys = d_comb.keys() | d.keys()
            d_comb = {key: d_comb.get(key, []) + d.get(key, []) for key in combined_keys}

        return d_comb
    
    
def ancestorsMarked(dupTimes, marked):
    allDupTimes = combineDicts(dupTimes)
    #print(allDupTimes.values())
    
    
    daughters = {}
    for dt in list(allDupTimes.keys()):
        for d in allDupTimes[dt]:
            daughters[d[0]] = list(d[1:])
    
    #Only works for divisions not differentiations
    #divisions = np.array(sum(list(allDupTimes.values()), []))
    #daughters = dict(zip(divisions[:,0], divisions[:,1:]))
    
    ancs = {}
    
    for m in marked:
         buildancs(ancs, daughters, m, m)
    
    return ancs

    
def buildancs(ancs, daughters, m, d):
    # Adds add descendents of a common ancestor m starting from d recursively
    ancs[d] = m    
    if d in daughters:
        for dd in daughters[d]:
            buildancs(ancs, daughters, m, dd)
        
    return ancs
               
def getPopSizes(clonesizes, marked, nummarked):
    
    popsizes = [[] for p in range(len(nummarked))]
    splits = [0] + list(np.cumsum(nummarked))
    nsur = []    
    
    for p in range(len(nummarked)):
        popsizes[p] = np.array([clonesizes[c] for c in marked[splits[p]:splits[p+1]]])
        #if p == 0 and np.max(popsizes[p]) > 1:
        #    print(np.array([c for c in marked[splits[p]:splits[p+1]]])[popsizes[p] > 1], popsizes[p][popsizes[p] > 1])
        popsizes[p] = popsizes[p][popsizes[p] > 0]
        nsur.append(len(popsizes[p]))

    return popsizes, nsur
    

#old even bias function
def evCorr(fracEven, avSize):
    fracexpect  = (1 - 1/np.array(avSize))/(2 - 1/np.array(avSize))
    return (np.array(fracEven) - fracexpect)/(1 - fracexpect)
    
    
#new even bias function
def evenbias(clonesize, singletime = False):
    #Mathematica:  p = 1/m; Simplify[Sum[p*(1 - p)^(2*k - 1), {k, 1, Infinity}]]
    #We need a way of assessing the error on even bias
    
    if singletime == True:
        clonesize = np.array(clonesize[np.array(clonesize) > 0])
        fracEven = 1 - np.mean(clonesize % 2) 
        avSize    = np.mean(clonesize)
        fracexpect  = (avSize - 1)/(2*avSize - 1)
        
        return (fracEven - fracexpect)/(1 - fracexpect)
    else:
        clonesize = [np.array(cst[np.array(cst) > 0]) for cst in clonesize]
        avSize = np.array([np.mean(cst) for cst in clonesize])
        fracEven = np.array([1 - np.mean(cst % 2) for cst in clonesize])
        
    fracexpect  = (avSize - 1)/(2*(avSize) - 1)
    return (fracEven - fracexpect)/(1 - fracexpect)
    
    
def popsizes(cloneResults, popidx):
    return {t: cloneResults['popsizes'][t][popidx] for t in cloneResults['popsizes'].keys()}

    
def simpdf(basalResults, totalResults, simtimes, maxbinS = 500, popidx = -1, verbose = True):
    '''
        maxbinS is the size of the largest bin (i.e. maxbinS + 1 bins)
            maxbinS should be chosen to be at least as large as the maximum experimental clone
            size + 1, to avoid errors from truncation of the distribution
    
    '''


        

    if popidx < 0:
        if basalResults is not None:    
            basalsizes = basalResults['allsizes'].copy()
        if totalResults is not None:
            totalsizes = totalResults['allsizes'].copy()
    else:
        if basalResults is not None:        
            basalsizes = {t: basalResults['popsizes'][t][popidx] for t in simtimes}
        if totalResults is not None:        
            totalsizes = {t: totalResults['popsizes'][t][popidx] for t in simtimes}        
    
    #Get simulation times
    if simtimes is None:
        if basalResults is not None:
            simtimes = basalResults['tvals']
        elif totalResults is not None:
            simtimes = totalResults['tvals']

    spdf = np.zeros((len(simtimes), 3, maxbinS + 1)) # index 1: basal, suprabasal, total        


    for i in range(len(simtimes)):
        t = simtimes[i]
    
        if len(basalsizes[t]) == 0:
            if verbose:
                print('No basal clones at time', t)
            continue
    
        if  (totalResults is not None and np.max(totalsizes[t]) > maxbinS):
            if verbose:
                print('Fixing total  clone sizes too big at time', t)
            totalsizes[t][totalsizes[t] >= maxbinS] = maxbinS
        
        if (np.max(basalsizes[t]) > maxbinS):
            if verbose:
                print('Fixing basal clone sizes too big at time', t)                      
            basalsizes[t][basalsizes[t] >= maxbinS] = maxbinS
        
        if basalResults is not None:
            spdf[i,0,:], bins = np.histogram(basalsizes[t], bins = np.arange(0, maxbinS + 2), density = True)
            
        if totalResults is not None:
            spdf[i,2,:], bins = np.histogram(totalsizes[t], bins = np.arange(0, maxbinS + 2), density = True)

    return spdf


def survivors(cloneResults, time  = -1):
    #get indices of surviving clones
    if time < 0:
        time = cloneResults['tvals'][time]

    return np.array([c for c, s in cloneResults['allclones'][time].items() if len(s)  > 0])


def diers(cloneResults, time  = -1):
    #get indices of extinct clones    
    if time < 0:
        time = cloneResults['tvals'][time]

    return np.array([c for c, s in cloneResults['allclones'][time].items() if len(s)  == 0])


def delam_by_time(tiss, results, showfor = 10, skip = 1, verbose = False):
    delambytime = [[] for t in range(tiss.time + 1)]
    positions = results['cellpos']

    p = 0
    for t in list(tiss.cellpops[p].dieTime.keys()):
        if t == 0:
            continue

        for c in tiss.cellpops[p].dieTime[t]:
            if not (t in list(tiss.cellpops[p].kilTime.keys()) and c in tiss.cellpops[p].kilTime[t]):
                if c in results['cellid'][int((t-1)/skip)]:
                    if verbose:
                        print(c, 'delaminates at', t, 'at location', positions[int((t-1)/skip)][np.argwhere(results['cellid'][int((t-1)/skip)] == c)[0][0]])
                    for tm in range(1,showfor):
                        if tm <= t:
                            delambytime[t - tm].append(c)
         
    for t in list(results['dupTimes'][0].keys()):
        for c in results['dupTimes'][0][t]:
            if c[0] in results['cellid'][int((t-1)/skip)]:
                if verbose:
                    print(c[0], 'delaminates at', t, 'at location', positions[int((t-1)/skip)][np.argwhere(results['cellid'][int((t-1)/skip)] == c[0])[0][0]])
                for tm in range(1,showfor):
                    if tm <= t:
                        delambytime[t - tm].append(c[0])
                            
    return delambytime
    
    
def die_by_time(tiss, results, showfor = 10, skip = 1, verbose = False):
    diebytime   = [[] for t in range(tiss.time + 1)]
    positions = results['cellpos']

    for p in range(tiss.npops):
        for t in list(tiss.cellpops[p].dieTime.keys()):
            if t == 0:
                continue
            for c in tiss.cellpops[p].dieTime[t]:
                if t in list(tiss.cellpops[p].kilTime.keys()) and c in tiss.cellpops[p].kilTime[t]:
                    if verbose:
                        print(c, 'is killed at', t, 'at location', positions[int((t-1)/skip)][np.argwhere(results['cellid'][int((t-1)/skip)] == c)[0][0]])
                    for tm in range(1,showfor):
                        if tm <= t:
                            diebytime[t - tm].append(c)

    return diebytime