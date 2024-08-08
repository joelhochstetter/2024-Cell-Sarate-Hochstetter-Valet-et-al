'''
    These provides a way to create the tissue object for pre-defined models 
        (either tissue or tissue0D)
    
    When adding a new model:
        - Add the function
        - Add the name to models0D or models2D
        - Add the number of cell populations to popdict
        
        
    For the governing stem cell division time-scale use tau
    
    For the governing differentiated cell delamination time-scale use mu
    
    For the fraction of asymetric divisions use 1-r

'''

import numpy as np
import sys
sys.path.append("../")
from tissue import *


'''
    0D models:

        S_0D
        SD_0D    
        
        
'''

#List of all models
models0D = ['S_0D', 'SD_0D']
models2D = ['S_mech_switch',  'SD_mech_switch', 'S_mech_switch_CC', 'S_mech_regen', 'SD_mech_regen']

#Dictionary of number of populations per model
popdict = {'S_0D': 1, 'SD_0D': 2, 'S_mech_switch': 1,
           'SD_mech_switch': 2, 'S_mech_switch_CC': 2,
           'S_mech_regen': 2, 'SD_mech_regen': 3}


def models(): #list of all models
    return models0D + models2D

def npops(x): #return number of populations
    if x.__name__ in popdict:
        return popdict[x.__name__]
    else:
        print('Model not defined')
        return 0


def S_0D(ncells, tau, r = 1.0, shed = 0, p = 0.5, useTerm = False, refract = 0.0, sbasrefract = 0, corr = 0.0):
    '''
        Model with Stem cell dividing at rate tau:
            S-> 2S (prob: r*p)
            S->S+u (prob: (1-r))
            S->u+u (prob: r*(1-p))
            
        If useTerm = False then we use a model without terminal divisions:
            S->u (prob: 1-p)
    '''
      
    if r < 1.0:
        useTerm = True
    
    if useTerm == True:
        tiss = tissue0D(poppts = ncells, divtime = (tau - 1)*(1 - refract) + 1, divprob = r*p, assprob = 1 - r, 
                 difprob = 0.0, terprob = r*p, sbasshed = shed, sbasrefract = sbasrefract*shed, popname = ['S'], refract = (tau - 1)*refract,
                 corrSym = corr)
    else: 
        tiss = tissue0D(poppts = ncells, divtime = (tau - 1)*(1 - refract) + 1, divprob = p, assprob = 0.0, 
                difprob = 1 - p, terprob = 0.0, sbasshed = shed, sbasrefract = sbasrefract*shed, popname = ['S'], refract = (tau - 1)*refract)        
    return tiss


def SD_0D(ncells, tau, rho = 0.5, r = 1.0, shed = 0, p = 0.5, mu = None, useTerm = False,
          corrSD = 0.0, corrDD = 0.0, corrSS = 0.0, refractS = 0.0, refractD = 0.0, sbasrefract = 0):
    '''
        Model with stem cell and differentiated cell
        
        Stem cell divides at rate tau:
            S-> 2S (prob: r*p)
            S->S+D (prob: (1-r))
            S->D+D (prob: r*(1-p))
            
        Differentiated cell delaminates at rate mu = tau*(1-rho)/rho
    
        To specify unequal # of initial cells input ncells as a list
            
        If useTerm = False then we use a model without terminal divisions:
            S->u (prob: 1-p)
            
            
        We can encode refractory periods for each cell type as a fraction of decision tims
            refractory period of S = (tau - 1)*refractS
            refractory period of D =  (mu - 1)*refractD
        
    '''
    
    if rho == 1.0:
        print('Using S model')
        return S_0D(ncells, tau, r, shed, p, useTerm, refractS, sbasrefract, corrSS)
    
    if r < 1.0:
        useTerm = True    
    
    if mu is None:
        if useTerm is True:
            mu = tau*(1-rho)/rho
        else:
            mu = tau*(1-rho)/rho*2         
    else:
        if useTerm is True:
            rho = 1/(1 + mu/tau)            
        else:
            rho = 1/(1 + 2*mu/tau)   
        
    if np.ndim(ncells) == 0: 
        nS = int(rho*ncells)
        poppts = [ncells - nS, nS]        
    else:   
        poppts = ncells
            
    if useTerm == True:
        tiss = tissue0D(poppts = poppts, divtime = [(mu - 1)*(1 - refractD) + 1, (tau - 1)*(1 - refractS) + 1], 
                        divprob = [0.0, p*r], assprob = [0.0, 1-r], 
              difprob = [1.0, 0.0], terprob = [0.0, r*(1-p)], sbasshed = shed, sbasrefract = sbasrefract*shed, popname = ['D', 'S'], 
              corrSym = [corrDD, corrSS], corrAss = corrSD,
              refract = [(mu - 1)*refractD, (tau - 1)*refractS])
    else: 
        tiss = tissue0D(poppts = poppts, divtime = [(mu - 1)*(1 - refractD) + 1, (tau - 1)*(1 - refractS) + 1], 
                        divprob = [0.0, p], assprob = 0.0, 
              difprob = [1.0, 1-p], terprob = 0.0, sbasshed = shed, sbasrefract = sbasrefract*shed, popname = ['D', 'S'], 
              corrSym = [corrDD, corrSS], corrAss = corrSD,
              refract = [(mu - 1)*refractD, (tau - 1)*refractS])
              
    return tiss


def S_mech_switch(ncells, L, tau = None, r = 1.0, shed = 0, tadiv = 10, Ahigh = 2.0, hilln = 2.0, 
                  rho_c = 1.0,  switch_profile = 'binary', useTerm = False, rlambda = None, sbasrefract = 0):
    ''' 
        Basal stem cell with division at fixed rate 1/tau1 (for A < Ahigh) and 1/tau2 for (A > Ahigh)
            can use other switch types (e.g. binary, relu)
        
        Area controls the fate balance, p(A*rhoc) = hill(A*rhoc, hilln)
        
        Stem cell dividing at rate tau:
            S-> 2S (prob: r*p(A*rhoc))
            S->S+u (prob: (1-r))
            S->u+u (prob: r*(1-p(A*rhoc)))        
            
        If useTerm = False then we use a model without terminal divisions:
            S->u (prob: 1-(1-p(A*rhoc))   
            
            
        If rlambda is specified, then we use r*lambda is fixed and then r is varied, i.e. the value of tau is not used
    '''
      
    if r < 1.0:
        useTerm = True
        
    if rlambda is not None:
        tau = r/rlambda
        print('Using fixed r*lambda = ', rlambda, ', tau = ', tau, ', varying r = ', r)               
    
    if Ahigh == np.inf:
        ruleS = 'size-fixrate'
    else:
        ruleS = 'size-size'
    
    if useTerm == True:
        tiss = tissue(poppts = ncells, L = L, divtime = tau, divprob = r/2, assprob = 1-r, Alow = 0.0, Ahigh = Ahigh,
                 difprob = 0.0, terprob = r/2, sbasshed = shed, sbasrefract = sbasrefract*shed, rule = ruleS, areadivtime = tadiv, popname = ['S'],
                 rho_c = rho_c, hilln = hilln, profile = switch_profile)
    else: 
        tiss = tissue(poppts = ncells, L = L, divtime = tau, divprob = 0.5, assprob = 0.0, Alow = 0.0, Ahigh = Ahigh,
                 difprob = 0.5, terprob = 0.0, sbasshed = shed, sbasrefract = sbasrefract*shed, rule = ruleS, areadivtime = tadiv, popname = ['S'],
                 rho_c = rho_c, hilln = hilln, profile = switch_profile)
    return tiss



def S_mech_switch_CC(ncells, L, tau, r = 1.0, shed = 0, tadiv = 10, Ahigh = 2.0, hilln = 2.0, fr = 0.5, 
                  rho_c = 1.0,  switch_profile = 'binary', useTerm = False, rlambda = None, sbasrefract = 0):
    ''' 
        S_mech_switch model with internal states
        
        Basal stem cell transitions from S1->S2 at fixed rate 1/tau (for A < Ahigh) and 1/tadiv for (A > Ahigh)
            can use other switch types (e.g. binary, relu)
        
        Then S2 divides after characteristic time fr*tau ( 0 < fr < 1)
        Area at the time of division controls the fate balance, p(A*rhoc) = hill(A*rhoc, hilln)
        
        Stem cell dividing at rate tau:
            S-> 2S (prob: r*p(A*rhoc))
            S->S+u (prob: (1-r))
            S->u+u (prob: r*(1-p(A*rhoc)))        
            
        If useTerm = False then we use a model without terminal divisions:
            S->u (prob: 1-(1-p(A*rhoc))   
            
    '''
      
    if r < 1.0:
        useTerm = True
        
    if rlambda is not None:
        tau = r/rlambda
        print('Using fixed r*lambda = ', rlambda, ', tau = ', tau, ', varying r = ', r)
    
    if np.ndim(ncells) == 0: 
        poppts = [int(fr*ncells), int((1-fr)*ncells)]        
    else:   
        poppts = ncells    
    
    
    if useTerm == True:
        tiss = tissue(poppts = poppts, L = L,  divtime = [tau*fr, tau*(1-fr)], divprob = [0.0, 0.0], assprob = [0.0, 0.0],
            desymprob = [r/2, 0.0], deassprob = [1 - r, 0.0], difprob = [0.0, 1.0], 
            terprob = [r/2, 0.0], rule = ['size-fixrate', 'size'],
            rho_c = rho_c, hilln = hilln, Ahigh = Ahigh, Alow = 0.0, areadivtime = tadiv,
            popname = ['S2', 'S1'], sbasshed = shed, sbasrefract = sbasrefract*shed, profile = switch_profile)
    else: 
        tiss = tissue(poppts = poppts, L = L, divtime = [tau*fr, tau*(1-fr)], divprob = 0.0, assprob = 0.0, Alow = 0.0, Ahigh = Ahigh,
                 difprob = [0.5,1.0], desymprob = [0.5,0.0], rule = ['size-fixrate', 'size'], terprob = 0.0, sbasshed = shed, sbasrefract = sbasrefract*shed, 
                 rho_c = rho_c, hilln = hilln, profile = switch_profile, areadivtime = tadiv, popname = ['S2', 'S1'])
    return tiss


def S_mech_regen(ncells, L, tau, taur, r = 1.0, shed = 0, tadiv = 0, Alow = 0.5, Ahigh = 2.0, hilln = 2.0, 
                  rho_c = 1.0,  switch_profile = 'binary', useTerm = False, rlambda = None, sbasrefract = 0):
    ''' 
        S_mech_switch model with internal states (called Sh and Sr) with a different division rate, 
            but the same fate decision rule
            
        Sh: has stochastic division rate 1/tau
        Sr: has fixed division rate 1/taur
        
        Basal stem cell transitions from Sh->Sr if A > Ahigh at rate 1/tadiv and Sr->Sh if A < Alow at rate 1/tadiv
            can use other switch types (e.g. binary, relu)
        
        Area at the time of division controls the fate balance, p(A*rhoc) = hill(A*rhoc, hilln)
        
        Stem cell dividing at rate tau:
            S-> 2S (prob: r*p(A*rhoc))
            S->S+u (prob: (1-r))
            S->u+u (prob: r*(1-p(A*rhoc)))        
            
        If useTerm = False then we use a model without terminal divisions:
            S->u (prob: 1-(1-p(A*rhoc))   
            
        By default: all cells are in the homeostatic state unless otherwise specified
            
    '''
      
    if r < 1.0:
        useTerm = True
        
    if rlambda is not None:
        tau = r/rlambda
        print('Using fixed r*lambda = ', rlambda, ', tau = ', tau, ', varying r = ', r)
    
    if np.ndim(ncells) == 0: 
        poppts = [ncells, 0]
    else:   
        poppts = ncells    
    
    
    if useTerm == True:
        tiss = tissue(poppts = poppts, L = L,  divtime = [tau, taur], 
                      divprob = r/2, assprob = 1-r, terprob = r/2, 
                      difprob = 0.0, rule = ['size-size', 'size-size'],
                      rho_c = rho_c, hilln = hilln, Ahigh = [Ahigh, np.inf], Alow = [0.0, Alow], areadivtime = tadiv,
                      popname = ['Sh', 'Sr'],   sbasshed = shed, sbasrefract = sbasrefract*shed, profile = switch_profile, daughterTypes = [2, 2], 
                      bigDaughter = [1, None], smallDaughter = [None, 0])
    else: 
        tiss = tissue(poppts = poppts, L = L,  divtime = [tau, taur], 
                      divprob = 0.5, assprob = 0.0, terprob = 0.0, 
                      difprob = 0.5, rule = ['size-size', 'size-size'],
                      rho_c = rho_c, hilln = hilln, Ahigh = [Ahigh, np.inf], Alow = [0.0, Alow], areadivtime = tadiv,
                      popname = ['Sh', 'Sr'],   sbasshed = shed, sbasrefract = sbasrefract*shed, profile = switch_profile, daughterTypes = [2, 2], 
                      bigDaughter = [1, None], smallDaughter = [None, 0])
        
    return tiss




def SD_mech_regen(ncells, L, tau, taur, r = 1.0, rho = 0.5,  shed = 0, tadiv = 0, Alow = 0.5, Ahigh = 2.0, hilln = 2.0, 
                  rho_c = 1.0,  switch_profile = 'binary', useTerm = False, rlambda = None, mu = None, sbasrefract = 0):
    ''' 
        SD_mech_switch model with internal states (called Sh and Sr) with a different division rate, 
            but the same fate decision rule
            
        Sh: has stochastic division rate 1/tau
        Sr: has fixed division rate 1/taur
        
        Basal stem cell transitions from Sh->Sr if A > Ahigh at rate 1/tadiv and Sr->Sh if A < Alow at rate 1/tadiv
            can use other switch types (e.g. binary, relu)
        
        Area at the time of division controls the fate balance, p(A*rhoc) = hill(A*rhoc, hilln)
        
        Stem cell dividing at rate tau:
            S-> 2S (prob: r*p(A*rhoc))
            S->S+D (prob: (1-r))
            S->D+D (prob: r*(1-p(A*rhoc)))        
            
        If useTerm = False then we use a model without terminal divisions:
            S->u (prob: 1-(1-p(A*rhoc))   
            
        By default: all cells are in the homeostatic state unless otherwise specified
            
    '''
      
    if r < 1.0:
        useTerm = True
        
    if rlambda is not None:
        tau = r/rlambda
        print('Using fixed r*lambda = ', rlambda, ', tau = ', tau, ', varying r = ', r)
    
    if np.ndim(ncells) == 0: 
        poppts = [ncells, 0]
    else:   
        poppts = ncells  


    if mu is None:
        if useTerm is True:
            mu = tau*(1-rho)/rho
        else:
            mu = tau*(1-rho)/rho*2
    else:
        if mu == 0:
            mu1 = 100
        else:
            mu1 = mu
            
        if useTerm is True:
            rho = 1/(1+mu1/tau)            
        else:
            rho = 1/(1+2*mu1/tau)

    if np.ndim(ncells) == 0: 
        nS = int(rho*ncells)
        poppts = [ncells - nS, nS, 0]        
    else:   
        poppts = ncells
    
    
    if useTerm == True:
        tiss = tissue(poppts = poppts, L = L,  divtime = [mu, tau, taur], 
                      divprob = [0.0, r/2, r/2], assprob = [0,1-r,1-r], terprob = [0,r/2,r/2], 
                      difprob = [1.0, 0.0, 0.0], rule = ['stochastic', 'size-size', 'size-size'],
                      rho_c = rho_c, hilln = hilln, Ahigh = [np.inf, Ahigh, np.inf], Alow = [0.0, 0.0, Alow], areadivtime = tadiv,
                      popname = ['D', 'Sh', 'Sr'],   sbasshed = shed, sbasrefract = sbasrefract*shed, profile = switch_profile, daughterTypes = [3, 0, 0], 
                      bigDaughter = [None, 2, None], smallDaughter = [None, None, 1])
    else: 
        r = 1.0
        tiss = tissue(poppts = poppts, L = L,  divtime = [mu, tau, taur], 
                      divprob = [0.0, 1/2, 1/2], assprob = 0, terprob = 0, 
                      difprob = [1.0, 1/2, 1/2], rule = ['stochastic', 'size-size', 'size-size'],
                      rho_c = rho_c, hilln = hilln, Ahigh = [np.inf, Ahigh, np.inf], Alow = [0.0, 0.0, Alow], areadivtime = tadiv,
                      popname = ['D', 'Sh', 'Sr'],   sbasshed = shed, sbasrefract = sbasrefract*shed, profile = switch_profile, daughterTypes = [3, 0, 0], 
                      bigDaughter = [None, 2, None], smallDaughter = [None, None, 1])
        
    return tiss    




def SD_mech_switch(ncells, L, tau, r = 1.0, rho = 0.5, shed = 0, tadiv = 10, Ahigh = np.inf, hilln = 2.0, 
                  rho_c = 1.0, mu = None, switch_profile = 'binary', useTerm = False,
                  mechDelam = False, Alow = 0.05, A0 = 1.0,
                  corrSD = 0.0, corrDD = 0.0, corrSS = 0.0, refractS = 0.0, refractD = 0.0, rlambda = None, sbasrefract = 0):
    ''' 
        Basal stem cell with division at fixed rate 1/tau1 (for A < Ahigh) and 1/tau2 for (A > Ahigh)
            can use other switch types (e.g. binary, relu)
        
        Area controls the fate balance, p(A*rhoc) = hill(A*rhoc, hilln)
        
        Stem cell dividing at rate tau:
            S-> 2S (prob: r*p(A*rhoc))
            S->S+D (prob: (1-r))
            S->D+D (prob: r*(1-p(A*rhoc)))        
            
        Differentiated cell delaminates at rate mu = tau*(1-rho)/rho            
            
        If useTerm = False then we use a model without terminal divisions:
            S->D (prob: 1-(1-p(A*rhoc)) 
            
        Use mechDelam = True for a mechanical rule on delamination (D->u)
            need to specify Alow (defaults to 0.05)
            can also specify A0 (defaults to 1.0)
        
    '''

    
    if rho == 1.0:
        print('Using S model')
        return S_mech_switch(ncells, L, tau, r, shed, tadiv, Ahigh, hilln, rho_c, switch_profile, useTerm, rlambda = rlambda, sbasrefract = sbasrefract)
    
      
    #Set the rule on Stem-cells
    if Ahigh == np.inf:
        ruleS = 'size-fixrate'
    else:
        ruleS = 'size-size'
      
    #Set the rule on differentiated cells
    if mechDelam == True:
        Alow = Alow
        ruleD = 'size'
        mu = 0.0
        print('Using mechanical delamination rule, mu ~ 100')
        sim_params = {'A0': [0.0, A0]}
    else:
        Alow = 0.0
        ruleD = 'stochastic'
        sim_params = None
      
    if r < 1.0:
        useTerm = True   
        
    if rlambda is not None:
        tau = r/rlambda
        print('Using fixed r*lambda = ', rlambda, ', tau = ', tau, ', varying r = ', r)         
    
    if mu is None:
        if useTerm is True:
            mu = tau*(1-rho)/rho
        else:
            mu = tau*(1-rho)/rho*2
    else:
        if mu == 0:
            mu1 = 100
        else:
            mu1 = mu
            
        if useTerm is True:
            rho = 1/(1+mu1/tau)            
        else:
            rho = 1/(1+2*mu1/tau)
        
    if np.ndim(ncells) == 0: 
        nS = int(rho*ncells)
        poppts = [ncells - nS, nS]        
    else:   
        poppts = ncells
    
    if useTerm == True:
        tiss = tissue(poppts = poppts, L = L, divtime = [(mu - 1)*(1 - refractD) + 1, (tau - 1)*(1 - refractS) + 1], 
                      divprob = [0,r/2], assprob = [0,1-r], Ahigh = [np.inf, Ahigh],
                 difprob = [1.0, 0.0], terprob = [0.0,r/2], sbasshed = shed, sbasrefract = sbasrefract*shed, rule = [ruleD, ruleS], areadivtime = tadiv, popname = ['D','S'],
                 rho_c = rho_c, hilln = hilln, profile = switch_profile, Alow = [Alow, 0.0],
                 sim_params = sim_params, corrSym = [corrDD, corrSS], corrAss = corrSD,
                 refract = [(mu - 1)*refractD, (tau - 1)*refractS])
    else: 
        tiss = tissue(poppts = poppts, L = L, divtime = [(mu - 1)*(1 - refractD) + 1, (tau - 1)*(1 - refractS) + 1],
                      divprob = [0.0, 0.5], assprob = 0.0, Ahigh = [np.inf, Ahigh],
                 difprob = [1.0, 0.5], terprob = 0.0, sbasshed = shed, sbasrefract = sbasrefract*shed, rule = [ruleD, ruleS], areadivtime = tadiv, popname = ['D','S'],
                 rho_c = rho_c, hilln = hilln, profile = switch_profile, Alow = [Alow, 0.0], 
                 sim_params = sim_params, corrSym = [corrDD, corrSS], corrAss = corrSD,
                 refract = [(mu - 1)*refractD, (tau - 1)*refractS])
        
    return tiss
