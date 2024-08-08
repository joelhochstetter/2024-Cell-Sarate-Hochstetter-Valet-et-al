import numpy as np
import scipy
import copy

dist_metrics = ['lsq', 'kld', 'ksd', 'lLL']
costs = ['cost_prd', 'cost_sum']

qmetrics = ['lsq', 'pls', 'els']


shortname = {'basalCS': 'bas', 'totalCS': 'tot', 'supraCS': 'sup', 'density': 'dens', 
             'divrate': 'dr', 'pers': 'pers', 'lcd': 'lcd', 'death': 'dth'}


class metric():
    '''
        Define a custom metric, which could generically depend on different pdfs
            or time-varying quantities
            
        Specify a metric as the type of data averages
            metric = {cost = 'cost_prd' / 'cost_sum',
                    quantity = [] %choose from 'basalCS' / 'totalCS' / 'density' / 'divrate',
                    dist = [] % choose from 'basalCS' / 'supraCS' / 'totalCS',
                    dist_metric = 'lsq' / 'kld' / 'ksd' / 'cvM' (cramer von Mieses, least squares of cdf) / 'lL' (negative log-likelihood),
                    qmetric = 'lsq' / 'pls' / 'els' (for quantity use least squares or percentage least squares or 
                        least squares normalised by the standard error)
                    exclude_times = {}  (indices of the times to to exclude),
                    av_time = False (if True, then the metric is averaged over time, else use a sum),
                    qweights = {} (weights for each quantity in metric, defaults to 1)
                    dweights = {} (weights for each distribution in metric, defaults to 1)
                    }  
                                                           
    '''
    def __init__(self, cost = 'cost_sum', quantity = [], dist = [], 
                 qmetric = 'lsq', dist_metric = 'lsq', 
                 exclude_times = None, av_time = False):
        
        assert(cost in costs)
        self.cost = cost
        self.quantity = quantity
        self.dist = dist
        self.dist_metric = dist_metric
        self.qmetric = qmetric
        
        self.sub_metrics = {}
        
        self.av_time = av_time
        
        if exclude_times is None:
            self.exclude_times = {}
        else:
            self.exclude_times = exclude_times
        
        self.name = self.name()
        
        
    def allquantities(self):
        '''
            Returns names of all quantities and distributions
        '''
        return self.quantity + self.dist
        
    def copy(self):
        return copy.deepcopy(self)        
        
    def name(self, popnames = None):
        if len(self.dist) + len(self.quantity) > 1:
            mname = self.cost
        else:
            mname = 'cost'
        
        if len(self.dist) > 0:
            if self.dist_metric != 'lsq':
                mname += '_' + self.dist_metric
            else:
                mname += '_lsqd'
            for d in sorted(list(self.dist)):
                mname += '_' + self.qsavename(d)
            
        if len(self.quantity) > 0:
            mname += '_av'
            if self.qmetric != 'lsq':
                mname += '_' + self.qmetric
            for q in sorted(list(self.quantity)):
                mname += '_' + self.qsavename(q)
            
                             
        if len(self.exclude_times) > 0:
            mname += '_excld'
            for k in self.exclude_times:
                for t in self.exclude_times[k]:
                    mname += '_' + str(t)
                                         
        return mname
    
    def qsavename(self, q):
        '''
            Returns name of quantity
        '''
        if q in shortname:
            return shortname[q]
        else:
            return q
         
        
    def calc(self, quants1, quants2, dists1 = None, dists2 = None, 
                se_quants1 = None, se_quants2 = None, 
                nsamples1  = None, nsamples2  = None):
        '''
            Calculate the metric on two datasets
        '''
        
        simAvgs = [self.get_av_data(quants1,  q) for q in self.quantity]
        expAvgs = [self.get_av_data(quants2,  q) for q in self.quantity]
        simPdfs = [self.get_dist_data(dists1, d) for d in self.dist]
        expPdfs = [self.get_dist_data(dists2, d) for d in self.dist]  
        
        if self.qmetric == 'els':
            assert(se_quants2 is not None)
            assert(len(quants2) == len(se_quants2)) 
            simSEs = None
            #simSEs = [self.get_av_data(se_quants1, q) for q in self.quantity]
            expSEs = [self.get_av_data(se_quants2, q) for q in self.quantity]
        else:
            simSEs = None
            expSEs = None
        
        if   self.cost == 'cost_prd':
            return self.cost_prd(simAvgs, expAvgs, simPdfs, expPdfs, self.dist_metric,
                                 simSEs = simSEs, expSEs = expSEs)
        elif self.cost == 'cost_sum':
            return self.cost_sum(simAvgs, expAvgs, simPdfs, expPdfs, self.dist_metric,
                                 simSEs = simSEs, expSEs = expSEs)   
            
            
    def get_av_data(self, data, q):
        '''
            Processes average data to account for excluded times
        '''
        if q in self.exclude_times:
            included_mask = np.ones(len(data[q]), dtype=bool)
            included_mask[np.array(self.exclude_times[q])] = False
            return data[q][included_mask]
        else:        
            return data[q]
       
        
    def get_dist_data(self, data, d):
        '''
            Processes distribution data to account for excluded times
                and smoothing
        '''
        
        
        
        if d in self.exclude_times:
            included_mask = np.ones(data[d].shape[0], dtype=bool)
            included_mask[np.array(self.exclude_times[d])] = False
            return data[d][included_mask,:]
        else:        
            return data[d]
        
     
    def exclude_quants(self, excluded):
        '''
            Creates a copy of the same metric with certain quantities excluded
        '''
         
        if np.ndim(excluded) == 0:
            excluded = [excluded]
        
        newmet = self.copy()
        
        newmet.quantity = [x for x in newmet.quantity if x not in excluded]
        newmet.dist     = [x for x in newmet.dist     if x not in excluded]
        newmet.name = newmet.name()
        return newmet
        


    def cost_prd(self, simAvgs = [], expAvgs = [], simPdfs = [], expPdfs = [], dist_metric = 'lsq', 
                qmetric = 'lsq', mindiff = 1e-8, sum_dist = True, simSEs = None, expSEs = None):
        '''
            Calculates cost function as log(product of errors) (which individually are SSE)
            
            Input:
                simAvgs, expAvgs: lists (same length)
                simPdfs, expPdfs: should be lists of what can be 1D or 2D arrays
                Distance metric between distributions: 'lsq'/'kld'/'ksd'/'lLL'
                Distance metric between quantities: 'lsq'/'pls' 
                    (for quantity use least squares or percentage least squares)
                                
                mindiff: is the minimum difference between datapoints
        '''       
        
        cost = 0
        assert(len(simAvgs) == len(expAvgs))
        assert(dist_metric in dist_metrics)
        
        for i in range(len(simAvgs)):
            #Replace equalities with minimum difference
            sse = quant_dist(simAvgs[i], expAvgs[i], qmetric, get_ith(simSEs, i), get_ith(expSEs, i), self.av_time)
            self.sub_metrics[self.quantity[i]] = sse
                  
            if sse < mindiff:
                sse = mindiff
            cost += np.log(sse)
            
        if sum_dist:
            for i in range(len(simPdfs)):
                ssd = dist_dist(simPdfs[i], expPdfs[i], dist_metric, mode = 1)
                self.sub_metrics[self.dist[i]] = ssd
                cost += np.log(ssd)
        else:
            for i in range(len(simPdfs)):
                ssd = dist_dist(simPdfs[i], expPdfs[i], dist_metric, mode = 2)
                self.sub_metrics[self.dist[i]] = ssd
                cost += ssd
                
        return cost


    def cost_sum(self, simAvgs = [], expAvgs = [], simPdfs = [], expPdfs = [], dist_metric = 'lsq', 
                qmetric = 'lsq', simSEs = None, expSEs = None):
        '''
            Calculates cost function as sum of errors (which individually are SSE)
            
            Input:
                simAvgs, expAvgs: lists (same length)
                simPdfs, expPdfs: should be lists of what can be 1D or 2D arrays
                Distance metric between distributions: 'lsq'/'kld'/'ksd'/'lLL'
                Distance metric between quantities: 'lsq'/'pls' 
                    (for quantity use least squares or percentage least squares)
        '''
        
        cost = 0
        assert(len(simAvgs) == len(expAvgs))
        assert(dist_metric in dist_metrics)
        
        for i in range(len(simAvgs)):
            sse = quant_dist(simAvgs[i], expAvgs[i], qmetric, get_ith(simSEs, i), get_ith(expSEs, i), self.av_time)
            self.sub_metrics[self.quantity[i]] = sse   
            cost += sse
            
        for i in range(len(simPdfs)):
            ssd = dist_dist(simPdfs[i], expPdfs[i], dist_metric, mode = 1)
            self.sub_metrics[self.dist[i]] = ssd            
            cost += ssd

        return cost



def quant_dist(simAvg, expAvg, qmetric = 'lsq', simSE = None, expSE = None, 
               av_time = False):

    '''
        Distance metric between quantities
            simAvg, expAvg: are given as L length array
            qmetric: 'lsq'/'pls' (for quantity use least squares or percentage least squares)
            / 'els' (for quantity use least squares normalised by the standard error)
    '''
    
    assert(qmetric in qmetrics)
    
    if qmetric == 'lsq':
        qm = np.sum(np.square(simAvg - expAvg))
    elif qmetric == 'pls':
        qm = np.sum(np.square(simAvg - expAvg)/np.square(expAvg))
    elif qmetric == 'els':
        qm = np.sum(np.square(simAvg - expAvg)/np.square(expSE))    

    if av_time:
        return qm/len(simAvg)
    else:
        return qm
    

def dist_dist(simPdf, expPdf, dist_metric = 'lsq', mode = 1, mindiff = 1e-8):
    ''' 
        Distance metric between distributions
            Pdfs: are given as TxL array, where T is time, and L is number of bins, or an L length array (no time) 
            mode: 0 (product between time-points) / 1 (sum between time-points) / 2 (sum(log(prod))) / 3 max
            Distance metric between distributions: 'lsq'/'kld'/'ksd'/'lLL'            
    
    '''
    
    assert(dist_metric in dist_metrics)
    assert(mode >= 0 and mode <= 3)
    
    if   dist_metric == 'lsq':
        dis = lsq_dist(expPdf, simPdf)
    elif dist_metric == 'ksd':
        dis = KSD(getCdf(expPdf), getCdf(simPdf))
    elif dist_metric == 'kld':
        dis = kld(expPdf, simPdf)
        
    #replace equalities with minimum difference to avoid singularitis
    dis[dis < mindiff] = mindiff

    if   mode == 0:
        dis = np.prod(dis)
    elif mode == 1:
        dis = np.sum(dis)
    elif mode == 2:
        dis = np.sum(np.log(dis))
    elif mode == 3:
        dis = np.max(dis)           
        
    return dis
        
    
def lsq_dist(exppdf, simpdf):
    '''
        Input distributions as TxL array, where T is time, and L is number of bins
        Returns lsq for each time-point
    '''     
    return np.sum(np.square((exppdf - simpdf)),-1)
    
    
def getCdf(spdf):
    '''
        Converts pdf to cdf, works provided the values are kept along last dimension,
        Assumes but does not check normalisation
    '''
    return np.cumsum(spdf[...,::-1], axis = -1)[...,::-1]   


def KSD(cdf1, cdf2):
    '''
        Calculates Kolmogorov Smirnov distance from cdfs
        Bins must be the last index so can work across multiple time-points
    '''
    return np.max(np.abs(cdf1 - cdf2),-1)



def kld(exppdf, simpdf):
    '''
        Calculates Kullback-Leibler divergence from pdfs
        Bins must be the last index so can work across multiple time-points
    '''
    
    
    return scipy.stats.entropy(exppdf[...,1:], simpdf[...,1:], axis = -1)

#weighted average mean:
def wam(vector, weights):
    assert(np.shape(vector)[1] == len(weights))
    scale = vector.copy()
    for i in range(len(weights)):
        scale[:,i] = scale[:,i]*weights[i]
    
    return np.sum(scale, 1)/np.sum(weights)
    
    
def getcounts(pdf, nsamples):
    return pdf * nsamples.reshape(-1, *(1,) * (pdf.ndim - 1))
    
    
def is_supra(met):
    '''
        Returns if a metric contains terms for the suprabasal layer
    '''
    return (any(item in ['supraCS', 'totalCS'] for item in met.quantity) or any(item in ['supraCS', 'totalCS'] for item in met.dist))


def get_ith(x, i):
    ''' 
        Returns the ith element of a list else returns the None
    '''
    if x is not None and i < len(x):
        return x[i]
    else:
        return None
    