import numpy as np
from utils import *
import copy
from sortedcontainers import SortedSet


# cell population class

class cellulation():
    '''
        Cell population class:
            specifies population of cells who have same rules for division vs differentiation
            
            in this version of the code only exponential division with fixed rate and prob is implemented
    '''
    def __init__(self, points, divtime = 1000, divprob = 0.5, totpts = 0, thetas = None, dx = 0.01, 
                 conserve = None, assprob = 0.0, difprob = 0.0, terprob = 0.0, daughterType = -1, 
                 fixchild = True, popidx = None, assymReplace = False, singleDaughter = False, refract = 0, 
                 alwaysPair = False, syncCycles = -1, verbose = False, delayedDeath = 1, 
                 corrtimeSym = 0, corrtimeAss = 0, corrSym = None, corrAss = None,
                 popname = None, dedifprob = 0.0, desymprob = 0.0, deassprob = 0.0,  
                 daughter_tau = 0, **extras):
        #Use number conservation: conserve = None, 'local', 'global': If division promotes death
    
        if popname is None:
            popname = popidx
        self.popname = popname

        self.poptype = 'stochastic'

        self.ncells  = len(points) #current number of cells
        
        if popidx is not None:
            self.popidx = popidx
        else:
            self.popidx = np.zeros(self.ncells, dtype = 'int64')

        self.divtime  = divtime  # division time
        self.divprob  = divprob  #divprob*np.ones(self.ncells), probability to symmetrically divide
        self.refract  = refract #refractory time

        self.assprob = assprob #probability to assymemtrically divide
        self.difprob = difprob #probability to differentiate
        self.terprob = terprob #probability to terminally divide
        
        if corrSym is not None:
            self.corrtimeSym = np.max([(corrSym*(self.divtime - 1)), 0]) #cell decision correlation time for stochastic decisions for sisters born into this population
        else:
            self.corrtimeSym = corrtimeSym #cell decision correlation time for stochastic decisions for sisters born into this population
        
        
        if corrAss is not None and corrAss > 0 and self.assprob > 0:
            assert(daughter_tau > 0)
            self.corrtimeAss = np.max([(corrAss*(np.min([self.divtime, daughter_tau]) - 1)), 0]) #cell decision correlation time for stochastic decisions resulting from asymmetric division from this population
        else:
            self.corrtimeAss = corrtimeAss #cell decision correlation time for stochastic decisions resulting from asymmetric division from this population
                
        
        self.dedifprob = dedifprob #probability to reverse differentiate
        self.desymprob = desymprob #probability to divide into less differentiated cells
        self.deassprob = deassprob #probability to asymetrically divide into one more differentiated and one less differentiated cell-type        
        
        self.assymReplace = assymReplace #assymetrically replace daughter if they are touching
        self.singleDaughter = singleDaughter #attach a single daughter with mother. This is replaced when the mother assymetrically divides, and is lost when the mother is lost

        if singleDaughter is True:
            self.daughter      = {(i + totpts): -1 for i in range(self.ncells)}
            self.alldaughters  = {(i + totpts): -1 for i in range(self.ncells)}            
        
        self.sister     = {}
        self.allsisters = {}
            
        
        self.posunq = {(i + totpts): points[i,:] for i in range(self.ncells)} #position of cell by unique index, allows to keep track of division and differentiation
        
        self.verbose = verbose
        
        self.syncCycles = syncCycles #synchronises cell-cycles on terminal divisions to this value
        self.delayed = delayedDeath
        
        if daughterType < 0:
            if self.popidx == 0:
                self.daughterType = 0
            else:
                self.daughterType = self.popidx - 1 #index of the daughter cell population with A->B+B, A->A+B allowed
        else:
            self.daughterType = daughterType

        self.motherType = self.popidx + 1 #index of the mother cell population with A->B+B, A->A+B allowed
        
        self.fixchild = fixchild  #fix child cell position during divisions
        
        
        if thetas is None:
            thetas = np.zeros(self.ncells)
               
        self.angunq = {(i + totpts): thetas[i] for i in range(self.ncells)}
        self.desc   = {i: [] for i in range(totpts, totpts + self.ncells)} #descendents
        self.ancs   = {i: i  for i in range(totpts, totpts + self.ncells)} #ancestors: at first time ancestor are themself
        self.parent = {i: i  for i in range(totpts, totpts + self.ncells)} #parents: at first time parents are themself
        
        if self.refract > 0:
            self.born   = {i: -np.random.randint(self.refract) for i in range(totpts, totpts + self.ncells)} #birth-time
        else:
            self.born   = {i: -self.refract for i in range(totpts, totpts + self.ncells)}
        
        self.decTime = {} #decision times
        self.dx = dx #distance
        
        self.dieTime = {} #stores times which cells die, and index of cell
        self.dupTime = {} #stores times which cells divide with tuple of parent and daughters
        self.kilTime = {} #stores times which cells were killed
        self.conserve = conserve
        self.totpts  = totpts + self.ncells #total number of points in history of simulation (used for lineage tracing)
        
        self.alwaysPair = alwaysPair
        
        if self.divtime > 0:
            initDecTimes = np.random.exponential(self.divtime - 1, self.ncells) + 1
            #initialisation
            for i in range(self.ncells):
                self.decTime.setdefault(int(initDecTimes[i] + (self.refract - self.born[i + totpts])), []).append(i + totpts)        
        
    def copy(self):
        return copy.deepcopy(self)    
        
    def getIndices(self): #gets arrays of cell indices for alive cells
        return np.array(list(self.posunq.keys()))
    
    def getPoints(self): #gets array of cell-centre positions
        return np.array(list(self.posunq.values()))
        
    def getThetas(self): #gets array of cell-centre angular orientations
        return np.array(list(self.angunq.values()))
    
    def updateTotpts(self, totpts):
        self.totpts = totpts
        return totpts
    
    def getTotpts(self):
        return self.totpts
    
    # update division probabilities according to some rules
    def updateProb(self):
        return self.divprob
        
    # update points, for class
    def updatePoints(self, points):
        self.posunq = dict(zip(self.posunq.keys(), points)) #find a faster way to do this
        return self.posunq

    # update thetas, for class    
    def updateThetas(self, thetas):
        self.angunq = dict(zip(self.angunq.keys(), thetas)) #find a faster way to do this
        return self.angunq
    
    
    def reset(self, oldtime = 0):
        self.basereset(oldtime)
    
    
    def basereset(self, oldtime = 0):
        #Reset division and born times
        
        self.dupTime.clear()
        self.dieTime.clear()
        self.dupTime.clear()

        if oldtime == 0: #complete reset
            self.born.clear()
            if self.refract > 0:
                self.born   = {i: -np.random.randint(self.refract) for i in self.getIndices()}
            else:
                self.born   = {i: 0 for i in self.getIndices()}
            
            self.decTime.clear()
            
            if self.divtime > 0:
                initDecTimes = np.random.exponential(self.divtime - 1, self.ncells) + 1
                      
                for i in range(self.ncells):
                    self.decTime.setdefault(int(initDecTimes[i]), []).append(self.getIndices()[i])

        else: #Partial reset
            #reset birth times
            self.born = dict(zip(self.born.keys(), np.array(list(self.born.values())) - oldtime))

            if  self.divtime > 0 or len(self.decTime) > 0:
                self.decTime = dict(zip(np.array(list(self.decTime.keys())) - oldtime, self.decTime.values()))
                
                deadtimes = np.array(list(self.decTime.keys()))[np.array(list(self.decTime.keys())) <= 0]
                deadcells = []
                for dt in deadtimes:
                    deadcells += self.decTime[dt]
                
                #print(deadcells, self.popidx)
                intersection = [value for value in self.getIndices() if value in deadcells]

                assert(len(intersection) == 0)
                
        #Reset ancestors
        self.ancs   = {i: i  for i in self.getIndices()} #ancestors: at first time ancestor are themself
    
    
    #make cell-decisions:
        #Implement different strategies
    def cellDecision(self, time, tiss = None): #tissue is not needed
        self.stochasticDecision(time, tiss)
        
        
    def isPaired(self, d):
    #For singleDaughter, check if paired, otherwise is paired
        paired = True
        if self.singleDaughter is True and d in self.daughter and self.daughter[d] == -1:
            paired = False
        return paired
        
        
    def stochasticFateChoice(self, d, tiss, time, randfate = -1):
        if d not in list(self.posunq.keys()): #may have made a decision by some other means
            if self.verbose:
                print('Warning: cell', d, 'does not exist')
            return False
            
        if self.assprob > 0 and self.alwaysPair and not self.isPaired(d):
            if self.verbose:                    
                print('1st div assymetric,', d, 'ass divs', randfate, self.divprob + self.difprob +  self.assprob, 'to', self.totpts, self.totpts + 1, 'at', np.round(self.posunq[d],2))            
            success = self.assymDivide(d, time, tiss)
            return success

        if randfate <= self.divprob: #replace with: self.divprob[i] # divide
            success = self.divide(d, time, tiss)  
                
        elif randfate <= self.divprob + self.difprob:             
            success = self.differentiate(d, time, tiss)
            
        elif randfate <= self.divprob + self.difprob + self.assprob:
            if self.verbose:
                print(d, 'ass divs', randfate, self.divprob + self.difprob +  self.assprob, 'to', self.totpts, self.totpts + 1, 'at', np.round(self.posunq[d],2))
            success = self.assymDivide(d, time, tiss)
            
        elif randfate <= self.divprob + self.difprob + self.assprob + self.terprob:
            if self.verbose:
                print(d, 'term divides', randfate, self.divprob + self.difprob + self.assprob + self.terprob, 'at', np.round(self.posunq[d],2))
            
            if self.singleDaughter is True:
                #Case 1: in TA population with a sister, then divides with sister
                if self.motherType == (tiss.npops - 1) and (self.sister[d] != d) and (self.sister[d] >= 0):
                    if (self.sister[d] not in self.getIndices()): #sister is in mother population
                        if self.verbose:
                            print(self.sister[d], 'sister of', d, 'makes a fate decision')  
                        tiss.totpts = self.totpts
                        success = tiss.divideFateChoice(self.motherType, self.sister[d])
                        self.totpts = tiss.totpts
                        #success = tiss.cellpops[self.motherType].divideFateChoice(self.sister[d], tiss, time)
                    else: #sister is in the same population => divides too
                        if self.verbose:
                            print(self.sister[d], 'sister of', d, 'divides too')  
                        success = self.termDivide(self.sister[d], tiss, time)
                        success = self.termDivide(d, tiss, time)
                else: #In stem/Differentiated populations, or in TA with no living sister, 
                    if self.syncCycles > 0 or tiss.cellpops[self.daughterType].divtime > 0:
                        success = self.termDivide(d, tiss, time) 
                    else:
                        success = self.termDivide(d, tiss, time, daughterType = tiss.npops) 
                                        
                    #if self.daughterType is not tiss.npops and success:
                    if self.popidx == (tiss.npops - 1) and success:
                        if self.daughter[d] >= 0:
                            daughterpos = tiss.cellpops[self.daughterType].posunq[self.daughter[d]]
                            
                        success = self.termDivDaughter(d, tiss, time)
                        if success and self.verbose:
                            print(self.daughter[d], 'daughter of', d, 'term divides at', np.round(daughterpos,2))
                        self.daughter.pop(d) 
                        assert(len(list(self.daughter.keys())) == self.ncells) 

            else:
                success = self.termDivide(d, tiss, time)  
    
        elif randfate <= self.divprob + self.difprob + self.assprob + self.terprob + self.dedifprob:
            if self.verbose:
                print(d, 'reverse differentiates', randfate, self.divprob + self.difprob + self.assprob + self.terprob + self.dedifprob, 'at', np.round(self.posunq[d],2))
            assert(self.popidx < tiss.npops)
            success = self.differentiate(d, time, tiss, daughterType = self.popidx + 1)
            
        elif randfate <= self.divprob + self.difprob + self.assprob + self.terprob + self.dedifprob + self.desymprob:
                if self.verbose:
                    print(d, 'reverse divides', randfate, self.divprob + self.difprob + self.assprob + self.terprob + self.dedifprob + self.desymprob, 'at', np.round(self.posunq[d],2))
                assert(self.popidx < tiss.npops)
                success = self.customDivide(d, tiss, time, daughter1type = self.popidx + 1, daughter2type = self.popidx + 1)
            
        elif randfate <= self.divprob + self.difprob + self.assprob + self.terprob + self.dedifprob + self.desymprob + self.deassprob:
                if self.verbose:
                    print(d, 'reverse assdivides', randfate, self.divprob + self.difprob + self.assprob + self.terprob + self.dedifprob + self.desymprob + self.deassprob, 'at', np.round(self.posunq[d],2))
                assert(self.popidx < tiss.npops)
                success = self.customDivide(d, tiss, time, daughter1type = self.popidx - 1, daughter2type = self.popidx + 1)            
            
        else: #die
            if self.verbose:
                print(d, 'delaminates', randfate, self.divprob + self.difprob + self.assprob + self.terprob + self.dedifprob + self.desymprob + self.deassprob, 'at', np.round(self.posunq[d],2))
            if self.assymReplace is True:
                self.replaceByParent(d, self.totpts, d, tiss, time)
            
            success = self.die(d, time) 
            
            if success and self.singleDaughter is True and (self.popidx == tiss.npops - 1):
                if self.terprob == 0.0:
                    success = self.removeDaughter(d, tiss, time)
                else: 
                    success = self.termDivDaughter(d, tiss, time)
                if success and self.verbose:
                    print('delaminates', self.daughter[d], 'from', self.daughterType, 'daughter of', d, 'at', np.round(self.posunq[d],2))
                if self.daughter[d] >= 0:
                    tiss.cellpops[self.daughterType].sister.pop(self.daughter[d])
                self.daughter.pop(d)    

        return success
    
    def divideFateChoice(self, d, tiss, time, randfate = -1):
        return self.stochasticFateChoice(d, tiss, time, randfate)
    

    def stochasticDecision(self, time, tiss = None):
        if time in self.decTime:
            ndec = len(self.decTime[time]) #number of decisions
            #based on position calculate division / differentation prob    
            rnums = np.random.random(ndec) # generate random number
            
            if self.verbose:
                print('Stochastic: Pop', self.popidx, 'cells', self.decTime[time], 'are deciding at time', time)
            
            if self.conserve == None:
                for i in range(ndec):
                    d = self.decTime[time][i]
                    self.divideFateChoice(d, tiss, time, randfate = rnums[i])
                        
            elif self.conserve == 'local':
                for i in range(ndec):
                    d = self.decTime[time][i]                
                                        
                    self.divide(d, time)
                    #need to count neighbours accounting for periodic boundary conditions
                    #if two neighbours are in the list we handle differently 
                    #(do we have an adjacency matrix)
                    
                    
            elif self.conserve == 'global':
                for i in range(ndec):
                    d = self.decTime[time][i]     
                    
                    if rnums[i] <= self.divprob*2: #replace with: self.divprob[i] # divide
                        if self.divide(d, time):
                            self.deleteRandom(time)
                        #print(d, 'divides, we have', self.ncells, 'cells left')

                    elif rnums[i] <= self.divprob*2 + self.difprob:
                        self.differentiate(d, time, tiss)
                        
                    elif rnums[i] <= self.divprob*2 + self.difprob + self.assprob:
                        # We need to delete a cell in the population that we create

                        if self.assymDivide(d, time, tiss):
                            tiss.removeRandomCell(self.daughterType)
                        
                    else: #rnums[i] < self.divprob*2 + self.difprob + self.assprob + self.terprob:  
                        # We need to delete a cell in the population that we create                     
                        if self.termDivide(d, tiss, time):
                            tiss.removeRandomCell(self.daughterType)
        
        #random cell death
        if tiss.killprofile is not None and time > 0:
            tokill = self.getIndices()[(np.random.random(self.ncells) < tiss.killprofile[time - 1]).nonzero()[0]]
            
            if len(tokill) > 0 and self.verbose:
                print('Killing', tokill, 'from', self.popidx)
                
            
            for d in tokill:
                if time - self.born[d] < 1: #self.refract: #just born
                    continue
                
                success = self.die(d, time) 
                if success:
                    self.kilTime.setdefault(time, []).append(d)

                #Pair model has some associated complexities:
                if tiss.npops > 1 and self.singleDaughter is True and success:
                    #If is the parent celltype kill the cell and it's daughter                
                    if self.popidx == (tiss.npops - 1):
                        if self.daughter[d] >= 0:
                            if self.verbose:
                                print(d, 'has daughter', self.daughter[d], 'at', np.round(self.posunq[d],2))
                                
                            if self.terprob == 0.0:
                                success = self.removeDaughter(d, tiss, time)
                                if success:
                                    tiss.cellpops[self.daughterType].sister.pop(self.daughter[d])
                            else:
                                success = self.termDivDaughter(d, tiss, time, delayed = self.delayed)

                            if self.verbose:
                                print('delaminates', self.daughter[d], 'from', self.daughterType, 'daughter of', d, 'at', np.round(self.posunq[d],2))
                                #We make the aesthetic choice to associate death of a mother with a delamination / terminal division.
                                #tiss.cellpops[self.daughterType].kilTime.setdefault(time, []).append(self.daughter[d])
                            
                        self.daughter.pop(d)

                        assert(len(list(self.daughter.keys())) == self.ncells)
                        assert(len(list(tiss.cellpops[self.daughterType].sister.keys())) == tiss.cellpops[self.daughterType].ncells)
                    
                    #If is the daughter celltype, remove as daughter from parent
                    if self.motherType == (tiss.npops - 1):
                        if self.verbose:
                            print(d, 'has sister', self.sister[d])
                        
                        #If the sister is of different type
                        if success and tiss.isInPop(self.sister[d], self.motherType): #tiss.isAlive(self.sister[d]):
                            tiss.cellpops[self.motherType].daughter[self.sister[d]] = -1
                            
                        self.sister.pop(d)
                        assert(len(list(tiss.cellpops[self.motherType].daughter.keys())) == tiss.cellpops[self.motherType].ncells)
                        assert(len(list(self.sister.keys())) == self.ncells)
                        #tiss.cellpops[self.motherType].daughter
                    
                
                  
    
    def divide(self, d, time, tiss = None): # cell d divides at time t
        success = False
        if d in list(self.posunq.keys()): #check the cell index is valid
            if self.verbose:
                print(d, 'divides into', [self.totpts, self.totpts + 1], 'at', np.round(self.posunq[d],2))
                
            #set descendents
            self.desc[d] = [self.totpts, self.totpts + 1] 
            self.desc[self.totpts] = []
            self.desc[self.totpts + 1] = []    

            #set ancestors
            self.ancs[self.totpts] = self.ancs[d]
            self.ancs[self.totpts + 1] = self.ancs[d]
            
            self.parent[self.totpts]     = d
            self.parent[self.totpts + 1] = d

            self.born[self.totpts]     = time
            self.born[self.totpts + 1] = time
                        
            #inherit daughters from parent
            if self.singleDaughter is True:
                self.daughter[self.totpts]         = self.daughter[d]
                self.alldaughters[self.totpts]     = self.daughter[d]                
                self.daughter[self.totpts + 1]     = -1
                self.alldaughters[self.totpts + 1] = -1                 
                
                if self.daughter[d] >= 0:
                    tiss.cellpops[self.daughterType].sister[self.daughter[d]] = self.totpts             
                self.daughter.pop(d)                
                
                assert(len(list(self.daughter.keys())) == self.ncells + 1)
                assert(len(list(tiss.cellpops[self.daughterType].sister.keys())) == tiss.cellpops[self.daughterType].ncells)
   

            #remove particle
            posd = self.posunq[d]
            self.posunq.pop(d)

            angd = self.angunq[d]
            self.angunq.pop(d)

            #add new particles positions using random orientation
            theta = np.random.uniform(2*np.pi)
            dv = self.dx*np.array([np.cos(theta), np.sin(theta)])
            
            if self.fixchild:
                theta = np.random.uniform(2*np.pi)
                self.posunq[self.totpts]     = posd                
                self.angunq[self.totpts]     = angd
                self.angunq[self.totpts + 1] = angd                  
            else:
                self.posunq[self.totpts]     = posd + dv
                self.angunq[self.totpts]     = theta
                self.angunq[self.totpts + 1] = np.pi - theta

            self.posunq[self.totpts + 1] = posd - dv
            

            #set decision times
            if self.divtime > 0:
                dtimes = 1 + np.random.exponential(self.divtime - 1 - self.corrtimeSym, 2) + time + self.refract 
                if self.corrtimeSym > 0:
                    dtimes +=   np.random.exponential(self.corrtimeSym - 1, 1)

                self.decTime.setdefault(int(dtimes[0]), []).append(self.totpts)
                self.decTime.setdefault(int(dtimes[1]), []).append(self.totpts + 1)
            
            #store time of division
            self.dupTime.setdefault(time, []).append([d, self.totpts, self.totpts + 1])
            
            #update number of points
            self.totpts += 2
            self.ncells += 1
            
            success = True
        #else:
        #    print('Time:', time, ', ', d, ' cannot divide, already dead')

        return success


    
    def die(self, d, time = 0): # cell d dies
        success = False
           
        if d in list(self.posunq.keys()):
            if self.verbose:
                print('Removing', d, 'from', self.popidx, 'at', time, 'at', np.round(self.posunq[d],2)) #self.cellidx
        
            #remove particle
            self.posunq.pop(d)
            self.angunq.pop(d)
            self.dieTime.setdefault(time, []).append(d) # self.dupTime
            self.ncells -= 1

            success = True            

        return success

    def deleteRandom(self, time): #pick a cell at random from population to detete
        if self.ncells > 1:
            toDel = np.random.randint(0, self.ncells)
            #print(toDel, self.ncells)
            self.die(list(self.posunq.keys())[toDel], time)
        else:
            print('Cells left:', self.posunq)
            
         
    def deleteOther(self, d, time = 0): # select other cell at random and kill
        if self.ncells > 1:
            toDel = np.random.randint(0, self.ncells - 1)
            if toDel >= d:
                toDel += 1
            self.die(list(self.posunq.keys())[toDel], time)
        else:
            print('Cells left:', self.posunq)
    
        
    #requires attributes from the tissue class (vertices corresponding to each point)
    #overcome this => pass in tissue class, defaults to None
    def divideNoTrack(self, p, longax = True): #vertices is vertices corresponding to point p
        if longax:
            mat1 = np.ones((self.nv, self.nv))
            A = ((self.vertices[:,0]*mat1).transpose() - self.vertices[:,0])**2 + ((self.vertices[:, 1]*mat1).transpose() - self.vertices[:,1])**2
            loc = np.unravel_index(np.argmax(A), A.shape)
            dv = (self.vertices[loc[0],:] - self.vertices[loc[1],:])/np.sqrt(A[loc[0],loc[1]])
        else: #random orientation 
            theta = np.random.uniform(2*np.pi)
            dv = self.dx*np.array([np.cos(theta), np.sin(theta)])
        self.cellpoints = np.vstack([self.cellpoints, self.cellpoints[p,:] - dv])
        self.cellpoints[p,:] += dv
        self.ncells += 1
        
        #update angle
        
        return self.cellpoints        
    
    
    def differentiate(self, d, time, tiss, daughterType = None):
        success = False    
        if d in list(self.posunq.keys()): #check the cell index is valid
            if daughterType is None:
                daughterType = self.daughterType

            #set descendents
            self.desc[d] = [self.totpts]

            if self.verbose:
                print(d, 'differentiates from', self.popidx, 'to', daughterType, 'at', time, 'at', np.round(self.posunq[d],2)) #self.cellidx           

            #remove particle
            posd = self.posunq[d]
            self.posunq.pop(d)

            angd = self.angunq[d]
            self.angunq.pop(d)

            #store time of decision
            self.dupTime.setdefault(time, []).append([d, self.totpts])
            
            self.ncells -= 1
            
            # add cell of other type
            tiss.addCell(daughterType, self.totpts, posd, angd, self.ancs[d], d)    

            #update number of points
            self.totpts += 1
                
            success = True            
       
        return success

    
            
    def replaceByParent(self, parent, daughter, tiss, time = -1, touch = False):
        ''' If a cell dies and it is touching is parent then we can replace the lost cell
        '''
        success = False
        if touch is True:
            if tiss.isNeighbour(parent, daughter):
                self.addCell(self.totpts, pos = self.posunq[daughter], 
                    theta = np.random.uniform(2*np.pi), time = time, 
                    anc = self.ancs[parent], parent = parent)    
                        
            success = True
            self.totpts += 1            
            
        else:
            self.addCell(self.totpts, pos = self.posunq[daughter], 
                theta = np.random.uniform(2*np.pi), time = time, 
                anc = self.ancs[parent], parent = parent)    
                        
            success = True
            self.totpts += 1            
        #self.die(daughter, time) #this is done elsewhere
        
        return success
    
    
    def removeDaughter(self, parent, tiss, time = -1, touch = False):
        #implement daughter
        success = False
        
        if self.daughter[parent] >= 0:
            if touch is True:
                if tiss.isNeighbour(parent, self.daughter[parent]): #doesn't work if off the screen
                    success = tiss.removeCell(self.daughterType, self.daughter[parent])

            else:
                success = tiss.removeCell(self.daughterType, self.daughter[parent])

        return success   
        
    def termDivDaughter(self, parent, tiss, time = -1, touch = False, delayed = 0):
        #implement daughter
        '''
            delayed = 0: Send immediately
            delayed = 1: Send at the original (stochastic) time
            delayed = 2: Send after refractory period
        '''
        
        
        if self.popidx != (tiss.npops - 1): 
            return False
        
        tiss.totpts = self.totpts
        success = False
        
        if self.daughter[parent] >= 0 and touch is False:
            if delayed == 1:
                if self.verbose:
                    print('Scheduling terminal division of', self.daughter[parent], 'daughter of', parent, 'at', np.round(self.posunq[d],2))
                    
                success = False
                tiss.scheduleEvent(self.daughterType, self.daughter[parent], dtime = -1, drate = self.syncCycles)
            
            elif delayed == 2:
                if self.verbose:
                    print('Scheduling terminal division of', self.daughter[parent], 'daughter of', parent, 'at', np.round(self.posunq[d],2))
                    
                success = False
                tiss.scheduleEvent(self.daughterType, self.daughter[parent], dtime = time + self.refract, drate = -1)                
                 
            elif delayed == 0: #delayed is false
                if self.verbose:
                    print('Instant terminal division of', self.daughter[parent], 'daughter of', parent)
                success = tiss.termDivide(self.daughterType, self.daughter[parent])

        self.totpts = tiss.totpts
        
        return success           
                    
    
    def setDaughter(self, daughterList = None, tiss = None):
        if daughterList is None:
            daughterList = -np.ones(self.ncells, dtype = np.int32)
        assert(len(daughterList) == self.ncells)
        self.daughter     = dict(zip(list(self.getIndices()), daughterList))
        self.alldaughters = dict(zip(list(self.getIndices()), daughterList))

        if tiss is not None:
            #set the sisters
            for i in range(self.ncells):
                d = daughterList[i]

                if d >= 0:
                    tiss.cellpops[self.daughterType].sister[d]     = list(self.getIndices())[i]
                    tiss.cellpops[self.daughterType].allsisters[d] = list(self.getIndices())[i]                    

            #set escape times for TA cells which do not have sisters
            hassister = tiss.cellpops[self.daughterType].sister.keys()
            idx2 = list(tiss.cellpops[self.daughterType].getIndices())

            #Add division times
            tiss.cellpops[self.daughterType].decTime.clear()

            if self.syncCycles > 0:
                divtime = self.syncCycles
                
            elif tiss.cellpops[self.daughterType].divtime > 0:
                divtime = tiss.cellpops[self.daughterType].divtime
            else:
                divtime = self.divtime

            for i in idx2:
                if i not in hassister:

                    #set is own sister, we don't have the pair correlation
                    tiss.cellpops[self.daughterType].sister[i]     = i
                    tiss.cellpops[self.daughterType].allsisters[i] = i                    
                    
                    if divtime > 0:
                        dtimes = 1 + np.random.exponential(divtime - 1, 1) + self.refract - tiss.cellpops[self.daughterType].born[i]
                        tiss.cellpops[self.daughterType].decTime.setdefault(int(dtimes[0]), []).append(i)                
                    
                
        #assert(tiss.cellpops[self.daughterType].ncells == 
    
        
    def setParent(self, daughterList, parentList):
        self.parent = dict(zip(daughterList, parentList))


            
    def assymDivide(self, d, time, tiss):
    # assymetric division
    # fixchild = True for daughter cell of same type position fixed
        success = False
        
        if d in list(self.posunq.keys()): #check the cell index is valid
            #set descendents
            self.desc[d] = [self.totpts, self.totpts + 1] 
            self.desc[self.totpts] = []

            #set ancestors
            self.ancs[self.totpts] = self.ancs[d]
            self.parent[self.totpts] = d
            self.sister[self.totpts] = self.totpts + 1
            self.allsisters[self.totpts] = self.totpts + 1
            
            if self.verbose:
                print(d, 'ass divides from', self.popidx, 'to', self.daughterType, 'giving', self.desc[d], 'with ancestor', self.ancs[d], 'at', np.round(self.posunq[d],2))
            
            self.born[self.totpts] = time
            
            posd = self.posunq[d]
            angd = self.angunq[d]
                
            
            #remove particle
            self.posunq.pop(d)
            self.angunq.pop(d)            
            
            #add new particles positions using random orientation
            theta = np.random.uniform(2*np.pi)
            dv = self.dx*np.array([np.cos(theta), np.sin(theta)])
            
            if self.fixchild:
                self.posunq[self.totpts]     = posd                
                self.angunq[self.totpts]     = angd    
                            
            else:
                self.posunq[self.totpts]     = posd + dv
                self.angunq[self.totpts]     = theta


            if self.corrtimeAss > 0:
                delay = np.random.exponential(self.corrtimeAss, 1)
                drate = tiss.cellpops[self.daughterType].divtime - self.corrtimeAss
                if drate <= 0:
                    drate = 1
            else:
                delay = 0
                drate = None

            #set decision times
            if self.divtime > 0:
                dtimes = 1 + np.random.exponential(self.divtime - 1 - self.corrtimeAss, 1) + time + self.refract + delay
                self.decTime.setdefault(int(dtimes[0]), []).append(self.totpts)
                        
            #store time of division
            self.dupTime.setdefault(time, []).append([d, self.totpts, self.totpts + 1])

            
            if self.singleDaughter is True and ((tiss.cellpops[self.daughterType] == 'stochastic') or (tiss.cellpops[self.daughterType] == '0D')): #unidirectional
                dtime = -1
            else:
                dtime = None
                        
            #add extra cell to the tissue
            tiss.addCell(self.daughterType, self.totpts + 1, posd - dv, np.pi - theta, self.ancs[d], d, self.totpts, dtime, delay, drate)    
               
            
            ogtotpts = self.totpts
            
            #update number of points
            #self.ncells += 1
            self.totpts += 2
            
            #update single daughter
            if self.singleDaughter is True and self.popidx == (tiss.npops - 1):
                #if self.daughter[d] >= 0: #check is done elsewhere
                self.daughter[ogtotpts]     = ogtotpts + 1
                self.alldaughters[ogtotpts] = ogtotpts + 1                               
                
                if tiss.cellpops[self.daughterType].terprob == 0.0:
                    success = self.removeDaughter(d, tiss, time)
                    if success:
                        tiss.cellpops[self.daughterType].sister.pop(self.daughter[d])

                else: 
                    #simultaneous delaminate
                    success = self.termDivDaughter(d, tiss, time, delayed = 0)
               
                if success and self.verbose:
                    print(self.daughter[d], 'daughter of', d, 'term divides')

                
                    
                self.daughter.pop(d)    
                assert(len(list(self.daughter.keys())) == self.ncells)         

            success = True            
            
        return success   

    def customDivide(self, d, tiss, time, daughter1type = None, daughter2type = None): 
    # "terminal" division of cell-type A into cell-type B
        
        if daughter1type is None:
            daughter1type = self.daughterType
            
        if daughter2type is None:
            daughter2type = self.daughterType        
        
        success = False
       
        if d in list(self.posunq.keys()): #check the cell index is valid
            #set descendents
            self.desc[d] = [self.totpts, self.totpts + 1] 
            self.desc[self.totpts] = []

            if self.verbose:
                print(d, 'term divides from', self.popidx, 'to', daughter1type, ',',  daughter2type, 'giving', self.desc[d], 'with ancestor', self.ancs[d], 'at', np.round(self.posunq[d],2))
                
            #remove particle
            posd = self.posunq[d]
            self.posunq.pop(d)

            angd = self.angunq[d]
            self.angunq.pop(d)

            #add new particles positions using random orientation
            theta = np.random.uniform(2*np.pi)
            dv = self.dx*np.array([np.cos(theta), np.sin(theta)])
            
            if self.fixchild:
                pos1 = posd                
                ang1 = angd 
                pos2 = posd - dv
                ang2 = np.pi - theta 
                            
            else:
                pos1 = posd                
                ang1 = angd 
                pos2 = posd - dv
                ang2 = np.pi - theta 

            #store time of division
            self.dupTime.setdefault(time, []).append([d, self.totpts, self.totpts + 1])
            
            dtime = None
                                
            if np.max([daughter1type, daughter2type]) != tiss.npops:
                sisA = self.totpts + 1
                sisB = self.totpts   
            else:
                sisA = -1
                sisB = -1    

            if tiss.cellpops[self.daughterType].corrtimeSym > 0:
                delay = np.random.exponential(tiss.cellpops[self.daughterType].corrtimeSym, 1)
                drate = tiss.cellpops[self.daughterType].divtime - tiss.cellpops[self.daughterType].corrtimeSym
                if drate <= 0:
                    drate = 1
            else:
                delay = 0
                drate = None
     
            #popidx, d, pos, theta, anc = -1, parent = -1, sister = -1, dtime = None
            tiss.addCell(daughter1type, self.totpts,     pos1, ang1, self.ancs[d], d, 
                sister = sisA, dtime = dtime, delay = delay, drate = drate)                   
            tiss.addCell(daughter2type, self.totpts + 1, pos2, ang2, self.ancs[d], d, 
                sister = sisB, dtime = dtime, delay = delay, drate = drate) 
            
            #if self.singleDaughter and self.popidx == 0:
            if self.singleDaughter and self.motherType == (tiss.npops - 1):            
                #print('Removing sister of', d, self.sister[d])
                if self.sister[d] in self.getIndices():
                    self.sister[self.sister[d]] = -1
                #elif self.sister[d] in tiss.cellpops[self.motherType].getIndices():
                self.sister.pop(d)
            
            #one less cell of this cell-type
            self.ncells -= 1 
            self.totpts += 2               
                  
            success = True            

        return success  
    
    def termDivide(self, d, tiss, time, daughterType = None): 
    # "terminal" division of cell-type A into cell-type B
        
        if daughterType is None:
            daughterType = self.daughterType
        
        success = False
       
        if d in list(self.posunq.keys()): #check the cell index is valid
            #set descendents
            self.desc[d] = [self.totpts, self.totpts + 1] 
            self.desc[self.totpts] = []

            if self.verbose:
                print(d, 'term divides from', self.popidx, 'to', daughterType, 'giving', self.desc[d], 'with ancestor', self.ancs[d], 'at', np.round(self.posunq[d],2))
                
            #remove particle
            posd = self.posunq[d]
            self.posunq.pop(d)

            angd = self.angunq[d]
            self.angunq.pop(d)

            #add new particles positions using random orientation
            theta = np.random.uniform(2*np.pi)
            dv = self.dx*np.array([np.cos(theta), np.sin(theta)])
            
            if self.fixchild:
                pos1 = posd                
                ang1 = angd 
                pos2 = posd - dv
                ang2 = np.pi - theta 
                            
            else:
                pos1 = posd                
                ang1 = angd 
                pos2 = posd - dv
                ang2 = np.pi - theta 

            #store time of division
            self.dupTime.setdefault(time, []).append([d, self.totpts, self.totpts + 1])
            
            dtime = None
            #if self.syncCycles > 0 and self.daughterType is not tiss.npops:
            if self.syncCycles > 0 and self.daughterType == 0:#(self.popidx == tiss.npops - 1):            
                #add extra cells to the tissue
                #Currently do this with our cell-cycle rates
                #daughDivTime = tiss.cellpops[self.daughterType].divtime
                #if daughDivTime <= 0:
                #    daughDivTime = self.divtime                
                dtime = 1 + np.random.exponential(self.syncCycles - 1, 1) + time + self.refract
                
                if self.verbose:
                    print('Syncing cell-cycles of', [self.totpts, self.totpts + 1], 'from pop', self.daughterType)
                
                sisA = self.totpts + 1
                sisB = self.totpts
                                
            elif self.daughterType != tiss.npops:
                sisA = self.totpts + 1
                sisB = self.totpts   
            else:
                sisA = -1
                sisB = -1    

            if tiss.cellpops[self.daughterType].corrtimeSym > 0:
                delay = np.random.exponential(tiss.cellpops[self.daughterType].corrtimeSym, 1)
                drate = tiss.cellpops[self.daughterType].divtime - tiss.cellpops[self.daughterType].corrtimeSym
                if drate <= 0:
                    drate = 1
            else:
                delay = 0
                drate = None
     
            #popidx, d, pos, theta, anc = -1, parent = -1, sister = -1, dtime = None
            tiss.addCell(daughterType, self.totpts,     pos1, ang1, self.ancs[d], d, 
                sister = sisA, dtime = dtime, delay = delay, drate = drate)                   
            tiss.addCell(daughterType, self.totpts + 1, pos2, ang2, self.ancs[d], d, 
                sister = sisB, dtime = dtime, delay = delay, drate = drate) 
            
            #if self.singleDaughter and self.popidx == 0:
            if self.singleDaughter and self.motherType == (tiss.npops - 1):
                #if self.verbose:            
                #    print('Removing sister of', d, self.sister[d])
                if self.sister[d] in self.getIndices():
                    self.sister[self.sister[d]] = -1
                elif self.sister[d] in tiss.cellpops[self.motherType].getIndices():
                    tiss.cellpops[self.motherType].daughter[self.sister[d]] = -1
                         
                self.sister.pop(d)
            
            
            #one less cell of this cell-type
            self.ncells -= 1 
            self.totpts += 2               
                  
            success = True            

        return success       
    
    
    def addCell(self, d, pos, theta, time, anc, parent, sister = -1, dtime = None, delay = 0, drate = None):
        #dtime < 0: cell does not divide
        #note: this does not update totpts        
        
        self.posunq[d]     = pos
        self.angunq[d]     = theta

        self.desc[d] = []
        self.ancs[d] = anc
        self.parent[d]  = parent
        self.born[d] = time
       
        self.ncells += 1   
        
        if sister >= 0:
            self.sister[d]  = sister
            self.allsisters[d] = sister
        
        if self.verbose:
            print('Adding', d, 'to pop', self.popidx, 'at', time, 'has ancestor', anc, 'parent', parent, 'at', np.round(self.posunq[d],2)) #self.cellidx     
        
        if drate is None:
            drate = self.divtime #this is a time not a rate

        if dtime is None:
            #set decision times
            if self.divtime > 0:
                dtime = 1 + np.random.exponential(drate - 1, 1) + time + self.refract + delay
                self.decTime.setdefault(int(dtime[0]), []).append(d)
                #if self.verbose:
                #    print('cell', d, 'divides at rate', self.divtime)
            
        elif dtime > 0:
            self.decTime.setdefault(int(dtime[0]), []).append(d) 
            if self.verbose:
                print('cell', d, 'will divide at', dtime[0], int(dtime[0]), 'see', self.decTime[int(dtime[0])])
        else:
            if self.verbose:
                print('No decision made by cell', d)         
         
        #print(anc, 'makes', d)

    def posToDens(self, perArea = True, centre = False, tiss = None, gridsz = 0, h = 1, L = 1, Ly = None, gridszy = None): #maps position vector to density, scaled by cell area
        '''
        if   perArea == True  and self.consumArea == 0.0:
            return np.zeros([gridsz, gridsz])
            
        elif perArea == False and self.consumCell == 0.0:
            return np.zeros([gridsz, gridsz])
        '''
        
        if Ly is None:
            Ly = L
        
        if gridszy is None:
            gridszy = gridsz
        
        if tiss == None or perArea == False:
            areas = np.ones(self.ncells)
        else:
            areas = np.abs(tiss.popAreas(self.popidx))
        
        
        if centre:
            rho = np.zeros([gridsz, gridszy])            
            locs = np.floor((self.getPoints() % (np.array([self.L, self.Ly])[None,:]))/h).astype('int')

            for i in range(len(locs)):
                rho[locs[i,0], locs[i,1]] += areas[i]/h**2
            
            
        else: #consumes evenly over cell, by default is proportional to area     
            rho = rasteriseCells(np.arange(tiss.splits[self.popidx], 
                tiss.splits[self.popidx + 1]), tiss.vertices, tiss.regions, 
                tiss.point_region, perArea, h, L, Ly)  

        return rho
        

class sizulation(cellulation):
    ''' 
        cellulation with division rules based on determinisitic dynamics based on 
            size of cell exceeding a critical value, or from random dynmics
            
        To specify different fate decisions rules for size and stochastic decisions use:
            bigDaughter or smallDaughter. This only enables 1 reaction type on each condition.
            If bigDaughter/smallDaughter is an integer, then we differentiate to this population
            Else if a tuple/list of length 2 is provided we use a custom decision of a certain type

            If None is provided, then we use the same rules as divideFateChoice
            
    '''

    def __init__(self, points, divtime = 1000, divprob = 1.0, totpts = 0, thetas = None, 
        dx = 0.01, Alow = 0.0, Ahigh = 1000, conserve = None, assprob = 0.0, difprob = 0.0, 
        terprob = 0.0, daughterType = -1, fixchild = True, popidx = 0, assymReplace = False, 
        singleDaughter = False, areadivtime = -1, refract = 0, alwaysPair = False, 
        syncCycles = -1, profile = 'binary', divhilln = 2, Ascale = 1.0, 
        dedifprob = 0.0, desymprob = 0.0, deassprob = 0.0, useRandom = None,
        forceRandFate = False, bigDaughter = None, smallDaughter = None, **extras):
        
        if assprob < 0.0: #use assprob < 0.0 for assymetric division, and probabilistic death
            self.assdiv = True
            assprob = 0.0
        else:
            self.assdiv = False    
    
        cellulation.__init__(self, points, divtime, divprob, totpts, thetas, dx, conserve, 
            assprob, difprob, terprob, daughterType, fixchild, popidx, assymReplace, 
            singleDaughter, refract, alwaysPair, syncCycles, dedifprob = dedifprob, desymprob = desymprob, 
            deassprob = deassprob, **extras)
            
        self.poptype = 'size'
        self.Alow  = Alow
        self.Ahigh = Ahigh
        
        #Set daughter types from 
        self.bigDaughter = bigDaughter
        assert((self.bigDaughter   is None) or (np.ndim(self.bigDaughter)   == 0) or (len(self.bigDaughter)   == 2))
        self.smallDaughter = smallDaughter
        assert((self.smallDaughter is None) or (np.ndim(self.smallDaughter) == 0) or (len(self.smallDaughter) == 2))
        
        if areadivtime > 0:
            self.areadivprob = 1/areadivtime
        else:
            self.areadivprob = 1.0
        
        if useRandom is None:
            self.useRandom = divtime > 0 #whether to use stochastically timed decisions as well as determinitic
        else:
            self.useRandom = useRandom

        self.forceRandFate = forceRandFate #if true then we use a stochastic fate choice

        '''
        For profile of division we have the following options:
            (equivalent in reverse for A < Alow)
            profile = 'binary':  this is just a binary switch, fixed rate = divtime for A < Ahigh, and 1/rate = 1/areadivtime for A > Ahigh
            
            profile = 'relu':  increases linearly above Ahigh, fixed rate = divtime for A < Ahigh, and 1/rate = (A - Ah)/Ah/areadivtime + 1/divtime for A > Ahigh
            
            profile = 'hill':   hill function increasing rate, fixed rate = divtime for A < Ahigh, and 1/rate = hill((A - Ah)/Asc,n) *(1/areadivtime - 1/divtime) + 1/divtime for A > Ahigh
                hill: defaults to Ahigh = 0, n = divhilln = 2, and Asc = Ascale is scaling parameter on hill function
            
            profile = 'dual':      linearly between two rates, fixed rate = divtime for A < Alow, 1/rate = 1/areadivtime for A > Ahigh, linearly interporated in middle
            
            Note: unless binary is selected then there is no possibility of cells being "too small",
                this is a future possibility to implement, e.g. a profile of delamination

        '''

        self.profile = profile

        if profile == 'relu':
            self.areadivprob /= self.Ahigh

        elif profile == 'hill':
            self.divhilln = divhilln
            self.Ascale      = Ascale #scaling parameter, setting when hill = 0.5, so rate is halfway between each value

            if self.divtime  > 0:
                self.areadivprob = 1/self.areadivtime - 1/self.divtime
            #else just: 1/self.areadivtime

        elif profile == 'dual':
            self.Aupper = self.Ahigh
            self.Ahigh  = self.Alow
            if self.divtime  > 0:
                self.areadivprob = (1/self.areadivtime - 1/self.divtime)/(self.Aupper - self.Ahigh)


    def smallprob(self, smallSizes):
        if self.profile == 'binary':
            probs = self.areadivprob
        else:
            probs = 0.0

        return probs
    
    def bigprob(self, bigSizes):
        if self.profile == 'binary':
            probs = self.areadivprob
        elif self.profile == 'relu':
            probs = relu(bigSizes, m = self.areadivprob, x0 = self.Ahigh)
        elif self.profile == 'hill':
            probs = relu(hill((bigSizes - self.Ahigh)/self.Ascale, self.divhilln)*self.areadivprob)
        elif self.profile == 'dual':
            probs = np.zeros(len(bigSizes))
            probs[bigSizes > self.Aupper] = self.areadivprob
            probs[bigSizes < self.Aupper] = relu(bigSizes, m = self.areadivprob, x0 = self.Ahigh)
        else:
            probs = 0.0
        return probs


    def getSmall(self, area, keys):
        if self.Alow == 0.0 or self.profile == 'dual':
            tooSmall   =  np.array([])
            smallSizes = np.array([])
        elif self.Alow >= 1000:
            tooSmall   = keys
            smallSizes = area
        else:           
            smallidx = np.argwhere(area < self.Alow ).transpose()[0]
            tooSmall   = keys[smallidx]
            smallSizes = area[smallidx]
        return tooSmall, smallSizes
    

    def getBig(self, area, keys):
        if self.Ahigh >= 1000:
            tooBig   =  np.array([])
            bigSizes = np.array([])   
        elif self.Ahigh == 0.0:
            tooBig   = keys
            bigSizes = area          
        else:           
            bigidx = np.argwhere(area > self.Ahigh).transpose()[0]
            tooBig   = keys[bigidx]
            bigSizes = area[bigidx]         
        return tooBig, bigSizes    


    def smallDecision(self, d, time, tiss):
        '''
            Decision that we execute when cells are smaller than a certain area
        '''

        if self.verbose:
            print(d, 'from pop', self.popidx, 'is too small  at ', np.round(self.posunq[d],2), 'time =', time)                    

        
        if self.smallDaughter is None:
            if self.difprob > 0.0:
                self.differentiate(d, time, tiss) 
            elif self.dedifprob > 0.0:
                self.differentiate(d, time, tiss, self.popidx + 1)                        
            else:                                     
                self.die(d, time)       
                
        else:
            if np.ndim(self.smallDaughter) == 0:
                self.differentiate(d, time, tiss, daughterType = self.smallDaughter)
            else:
                self.customDivide(d, tiss, time, self.smallDaughter[0], self.smallDaughter[1])


    def bigDecision(self, d, time, tiss):
        '''
            Decision that we execute when cells are larger than a certain area
        '''
        
        if self.verbose:
            print(d, 'from pop', self.popidx, 'is too big  at ', np.round(self.posunq[d],2), 'time =', time)                    
        
        
        if self.bigDaughter is None:
            self.divideFateChoice(d, tiss, time, randfate = -1)
        else:
            if np.ndim(self.bigDaughter) == 0:
                self.differentiate(d, time, tiss, daughterType = self.bigDaughter)
            else:
                self.customDivide(d, tiss, time, self.bigDaughter[0], self.bigDaughter[1])




    def cellDecision(self, time, tiss):
        #This doesn't work in principle once you change the number of cells
        area = np.abs(tiss.popAreas(self.popidx))#abs(np.array(tiss.areas[:self.ncells]))
        keys = tiss.getPopIndices(self.popidx)
        #keys = np.array(list(self.posunq.keys()))
        assert(len(area) == len(keys))


        tooSmall, smallSizes = self.getSmall(area, keys)
        tooBig,     bigSizes = self.getBig(area, keys)

        if self.areadivprob >= 1.0:
            for d in tooSmall:
                if d not in list(self.posunq.keys()):
                    if self.verbose:
                        print(d, 'already divided')
                    continue                
                
                if time - self.born[d] > self.refract:   
                    self.smallDecision(d, time, tiss)
                    
            for d in tooBig:
                if d not in list(self.posunq.keys()):
                    if self.verbose:
                        print(d, 'already divided')
                    continue                
                
                if time - self.born[d] > self.refract:                            
                    self.bigDecision(d, time, tiss)
            
        else:
            if len(tooSmall) > 0:
                rands = np.random.rand(len(tooSmall))
                for d in tooSmall[rands < self.smallprob(smallSizes)]:
                    if d not in list(self.posunq.keys()):
                        if self.verbose:
                            print(d, 'already divided')
                        continue
                                        
                    if time - self.born[d] > self.refract:
                        self.smallDecision(d, time, tiss)
            
            if len(tooBig) > 0:
                randb = np.random.rand(len(tooBig))

                for d in tooBig[randb < self.bigprob(bigSizes)]:
                    if d not in list(self.posunq.keys()):
                        if self.verbose:
                            print(d, 'already divided')
                        continue

                    if time - self.born[d] > self.refract:                
                        self.bigDecision(d, time, tiss)
                        
        if self.useRandom:
            self.stochasticDecision(time, tiss)


    def divideFateChoice(self, d, tiss, time, randfate = -1):
        #    randfate < 0: specifies to apply the rule for large cell-size
        #0 <= randfate <= 1: speciies making the fate decision as for a stochastic fate choice
        # randfate >= 1: forces cell death
        
        if self.forceRandFate is True:
            randfate = np.random.rand(1)
        
        if randfate < 0:
            if self.assdiv: #set for assymetric division
                success = self.assymDivide(d, time, tiss)        
            else:
                if not self.singleDaughter:
                    if self.terprob > 0.0:
                        if self.verbose:
                            area = np.abs(tiss.popAreas(self.popidx))#abs(np.array(tiss.areas[:self.ncells]))
                            keys = tiss.getPopIndices(self.popidx)                        
                            areas = dict(zip(keys, area))
                            print(d, 'from pop', self.popidx, 'is big, term divs to', self.totpts, self.totpts + 1, 'at', np.round(self.posunq[d],2), 'time =', time, 'size =', np.round(areas[d],2), 'daughterType', self.daughterType)    
                        success = self.termDivide(d, tiss, time)
                        
                    elif self.difprob > 0.0:
                        success = self.differentiate(d, time, tiss)
                        
                    elif self.desymprob > 0.0:
                        if self.verbose:
                            area = np.abs(tiss.popAreas(self.popidx))#abs(np.array(tiss.areas[:self.ncells]))
                            keys = tiss.getPopIndices(self.popidx)                        
                            areas = dict(zip(keys, area))
                            print(d, 'from pop', self.popidx, 'is big, symmetric de-divs to', self.totpts, self.totpts + 1, 'at', np.round(self.posunq[d],2), 'time =', time, 'size =', np.round(areas[d],2), 'daughterType', self.daughterType)    
                        success = self.customDivide(d, tiss, time, self.popidx + 1, self.popidx + 1)

                    elif self.deassprob > 0.0:                    
                        if self.verbose:
                            area = np.abs(tiss.popAreas(self.popidx))#abs(np.array(tiss.areas[:self.ncells]))
                            keys = tiss.getPopIndices(self.popidx)                        
                            areas = dict(zip(keys, area))
                            print(d, 'from pop', self.popidx, 'is big, asymmetrically de-divs to', self.totpts, self.totpts + 1, 'at', np.round(self.posunq[d],2), 'time =', time, 'size =', np.round(areas[d],2), 'daughterType', self.daughterType)    
                        success = self.customDivide(d, tiss, time, self.popidx + 1, self.popidx - 1)                    
                    
                    else:
                        if self.verbose:
                            areas = dict(zip(keys, area))
                            print(d, 'from pop', self.popidx, 'is big, symm divs to', self.totpts, self.totpts + 1, 'at', np.round(self.posunq[d],2), 'time =', time, 'size =', np.round(areas[d],2))    
                        success = self.divide(d, time, tiss)
                                                                            
                elif self.alwaysPair and not self.isPaired(d):
                    if self.verbose:
                        areas = dict(zip(keys, area))                                
                        print('First division assymetric', d, 'from pop', self.popidx, 'is big, ass divs to', self.totpts, self.totpts + 1, 'at', np.round(self.posunq[d],2), 'time =', time, 'size =', np.round(areas[d],2))        
                    success = self.assymDivide(d, time, tiss)
                    
                else:
                    if self.terprob > 0.0:
                        if self.motherType == (tiss.npops - 1) and (self.sister[d] != d) and (self.sister[d] >= 0):
                            if (self.sister[d] not in self.getIndices()): #sister is in mother population
                                if self.verbose:
                                    print(self.sister[d], 'sister of', d, 'makes a fate decision')  
                                tiss.totpts = self.totpts
                                success = tiss.divideFateChoice(self.motherType, self.sister[d])
                                self.totpts = tiss.totpts
                                #success = tiss.cellpops[self.motherType].divideFateChoice(self.sister[d], tiss, time)
                            else: #sister is in the same population => divides too
                                if self.verbose:
                                    print(self.sister[d], 'sister of', d, 'divides too')  
                                success = self.termDivide(self.sister[d], tiss, time)
                                success = self.termDivide(d, tiss, time)

                        else: #if either no sister exists, or sister is in population
                            if self.verbose:
                                areas = dict(zip(keys, area))
                                print(d, 'from pop', self.popidx, 'is big, term divs to', self.totpts, self.totpts + 1, 'at', np.round(self.posunq[d],2), 'time =', time, 'size =', np.round(areas[d],2), 'daughterType', self.daughterType)    
                            success = self.termDivide(d, tiss, time)

                    else:
                        if self.verbose:
                            areas = dict(zip(keys, area))                                
                            print(d, 'from pop', self.popidx, 'is big, symm divs to', self.totpts, self.totpts + 1, 'at', np.round(self.posunq[d],2), 'time =', time, 'size =', np.round(areas[d],2))        
                        success = self.divide(d, time, tiss)
        
        else:
            success = self.stochasticFateChoice(d, tiss, time, randfate)
        
        return success
    
       

class sizulationFR(cellulation):
    #area-based dynamics with fixed stochastic division rate

    def __init__(self, points, rho_c = 1.0, hilln = 2, divtime = 0, totpts = 0, thetas = None, dx = 0.01, 
                 popidx = 0, assprob = 0.0, difprob = 0.0, terprob = 0.0, corrtimeSym = 0, corrtimeAss = 0, **extras):
        
        cellulation.__init__(self, points, divtime, totpts = totpts, thetas = thetas, dx = dx, popidx = popidx, assprob = assprob, difprob = difprob, terprob = terprob,
                             corrtimeSym = corrtimeSym, corrtimeAss = corrtimeAss, **extras)
        
        self.rho_c  = rho_c
        self.hilln  = hilln

        self.poptype = 'size-fixrate'


    def cellDecision(self, time, tiss): 
        #note the probability should be the integral over the trajectory        

        if time in self.decTime:

            ndec = len(self.decTime[time]) #number of decisions
            #based on position calculate division / differentation prob    
            rnums = np.random.random(ndec) # generate random number
            
            for i in range(ndec):
                d = self.decTime[time][i]
                area = abs(tiss.areaById(d))
                
                self.divideFateChoice(d, tiss, time, rnums[i], area)
                
    def symmProb(self, d = None, tiss = None, area = None):
        '''
            Probability that cell d executes a duplicative division
                given it choses a symmetric fate decision
        '''
        if area is None:
            area = abs(tiss.areaById(d))
            
        return hill(self.rho_c*area, self.hilln)

    def divideFateChoice(self, d, tiss, time, randfate = -1, area = None):
        #Makes decision based on hill function of cell-area
        if randfate < -1:
            randfate = np.random.rand(1)
            
        if d not in list(self.posunq.keys()): #may have died from some other means
            if self.verbose:
                print('Warning: cell', d, 'does not exist')
            return False
        
        if area is None:
            area = abs(tiss.areaById(d))        
        
        if randfate < self.assprob:
            success = self.assymDivide(d, time, tiss)
        
        elif randfate < self.assprob + self.deassprob:
            assert(self.popidx < tiss.npops)
            success = self.customDivide(d, tiss, time, daughter1type = self.popidx - 1, daughter2type = self.popidx + 1)            
            
        
        elif (randfate - (self.assprob + self.deassprob))/(1 - (self.assprob + self.deassprob)) < self.symmProb(d, tiss, area = area):
            if self.desymprob > 0.0:
                assert(self.popidx < tiss.npops)
                success = self.customDivide(d, tiss, time, daughter1type = self.popidx + 1, daughter2type = self.popidx + 1)
            else:
               success = self.divide(d, time, tiss)
                            
        else: #differentiate
            if self.difprob > 0.0:
                success = self.differentiate(d, time, tiss)
            elif self.dedifprob > 0.0:
                assert(self.popidx < tiss.npops)
                success = self.differentiate(d, time, tiss, daughterType = self.popidx + 1)
            elif self.terprob > 0.0:
                success = self.termDivide(d, tiss, time)
            else:
                success = self.die(d, time)   
                
        return success
    


class sizulation_sz(sizulation):
    '''
        Decision timing driven by an area rule and balance between cell-fate decisions
            occurs as a hill function of cell-area
    '''
    
    def __init__(self, points, rho_c = 1.0, hilln = 2, divtime = 1000, divprob = 1.0, totpts = 0, thetas = None, 
        dx = 0.01, Alow = 0.0, Ahigh = 1000, conserve = None, assprob = 0.0, difprob = 0.0, 
        terprob = 0.0, daughterType = -1, fixchild = True, popidx = 0, assymReplace = False, 
        singleDaughter = False, areadivtime = -1, refract = 0, alwaysPair = False, 
        syncCycles = -1, profile = 'binary', divhilln = 2, Ascale = 1.0, 
        dedifprob = 0.0, desymprob = 0.0, deassprob = 0.0, useRandom = None, **extras):

        sizulation.__init__(self, points, divtime, divprob, totpts, thetas, dx, Alow, Ahigh, conserve, assprob, difprob, terprob, daughterType, fixchild, popidx, assymReplace, singleDaughter, areadivtime, refract, alwaysPair, syncCycles, profile, divhilln, Ascale, dedifprob, desymprob, deassprob, useRandom, **extras)
    
        self.rho_c  = rho_c
        self.hilln  = hilln

        self.poptype = 'size-size'   
         
    
    def symmProb(self, d = None, tiss = None, area = None):
        '''
            Probability that cell d executes a duplicative division
                given it choses a symmetric fate decision
        '''
        if area is None:
            area = abs(tiss.areaById(d))
            
        return hill(self.rho_c*area, self.hilln)
        
    

    def divideFateChoice(self, d, tiss, time, randfate = -1, area = None):
        if d not in list(self.posunq.keys()): #may have died from some other means
            if self.verbose:
                print('Warning: cell', d, 'does not exist')
            return False        
        
        
        if randfate < 0:
            randfate = np.random.rand(1)
        
            #self.daughter.pop(d)
        if randfate < self.assprob:
            if self.verbose:
                print('rnd:', randfate, 'ass prob:', self.assprob, '=> ass divides')
            success = self.assymDivide(d, time, tiss)
        
        elif randfate < self.assprob + self.deassprob:
            if self.verbose:
                print('rnd:', randfate, 'de-ass prob:', self.deassprob, '=> de-ass divides')
            
            assert(self.popidx < tiss.npops)
            success = self.customDivide(d, tiss, time, daughter1type = self.popidx - 1, daughter2type = self.popidx + 1)            
        
        
        elif (randfate - (self.assprob + self.deassprob))/(1 - (self.assprob + self.deassprob)) < self.symmProb(d, tiss = tiss):
            if self.verbose:
                print('rnd:', (randfate - self.assprob)/(1 - self.assprob), 'symm prob:', self.symmProb(d, tiss = tiss), '=> symm divides')            
            if self.singleDaughter is True and self.popidx == (tiss.npops - 1):          
                success = self.termDivDaughter(d, tiss, time)
                if success:
                    self.daughter[d] = -1

            if self.desymprob > 0.0:
                assert(self.popidx < tiss.npops)
                success = self.customDivide(d, tiss, time, daughter1type = self.popidx + 1, daughter2type = self.popidx + 1)
            else:
               success = self.divide(d, time, tiss)

            if self.verbose and self.singleDaughter is True:
                print(self.desc[d], 'have daughters', self.daughter[self.desc[d][0]], self.daughter[self.desc[d][1]])
            '''
            if success is True and self.singleDaughter is True:
                self.daughter[d] = self.alldaughters[d] #repair for case you want to S->SS to drive T divisions
            '''
        else: #differentiate
            if self.verbose:
                print('rnd:', (randfate - self.assprob)/(1 - self.assprob), 'symm prob:', self.symmProb(d, tiss = tiss), '=> diff/term divides')            

            if self.difprob > 0.0:
                success = self.differentiate(d, time, tiss)  
            elif self.dedifprob > 0.0:
                assert(self.popidx < tiss.npops)
                success = self.differentiate(d, time, tiss, daughterType = self.popidx + 1)
                
            elif self.terprob > 0.0:
                if self.singleDaughter is True and self.popidx == (tiss.npops - 1):          
                    success = self.termDivDaughter(d, tiss, time)

                success = self.termDivide(d, tiss, time)
                if success and self.singleDaughter is True:
                    self.daughter.pop(d)     
                    assert(len(list(self.daughter.keys())) == self.ncells)

            else:
                success = self.die(d, time)          
        
        return success         
    
    
#non-spatial cellulation
class cellulation0D(cellulation):
    '''
        Cell population class:
            specifies population of cells who have same rules for division vs differentiation
            
            in this version of the code only exponential division with fixed rate and prob is implemented
    '''
    def __init__(self, cellidx = set(), divtime = 1000, divprob = 0.5, totpts = 0, conserve = None, 
                 assprob = 0.0, difprob = 0.0, terprob = 0.0, daughterType = -1, popidx = 0, 
                 assymReplace = False, singleDaughter = False, refract = 0, motherType = -1, verbose = True,
                 corrtimeSym = 0, corrtimeAss = 0, corrSym = None, corrAss = None,
                 dedifprob = 0.0, desymprob = 0.0, deassprob = 0.0, daughter_tau = 0, **extras):
        #Use number conservation: conserve = None, 'local', 'global': If division promotes death

        self.poptype = '0D'

        singleDaughter = False
        
        self.verbose = verbose
        
        self.popidx = popidx
        
        self.cellidx = set(list(cellidx))
        self.ncells  = len(cellidx) #current number of cells

        self.divtime = divtime  # division time
        self.divprob = divprob  #divprob*np.ones(self.ncells), probability to symmetrically divide
        self.refract = refract #refractory time
        
        self.assprob = assprob #probability to assymemtrically divide
        self.difprob = difprob #probability to differentiate
        self.terprob = terprob #probability to terminally divide
        
        if corrSym is not None:
            self.corrtimeSym = np.max([(corrSym*(self.divtime - 1)), 0]) #cell decision correlation time for stochastic decisions for sisters born into this population
        else:
            self.corrtimeSym = corrtimeSym #cell decision correlation time for stochastic decisions for sisters born into this population
        
        
        if corrAss is not None and self.assprob > 0:
            assert(daughter_tau > 0)
            self.corrtimeAss = np.max([(corrAss*(np.min([self.divtime, daughter_tau]) - 1)), 0]) #cell decision correlation time for stochastic decisions resulting from asymmetric division from this population
        else:
            self.corrtimeAss = corrtimeAss #cell decision correlation time for stochastic decisions resulting from asymmetric division from this population
                        
        
        self.dedifprob = dedifprob #probability to reverse differentiate
        self.desymprob = desymprob #probability to divide into less differentiated cells
        self.deassprob = deassprob #probability to asymetrically divide into one more differentiated and one less differentiated cell-type           
        
        self.assymReplace = assymReplace #assymetrically replace daughter if they are touching
        self.singleDaughter = singleDaughter #attach a single daughter with mother. This is replaced when the mother assymetrically divides, and is lost when the mother is lost

        if singleDaughter is True:
            self.daughter = {(i + totpts): -1 for i in range(self.ncells)}
                
        if daughterType < 0:
            if self.popidx == 0:
                self.daughterType = 0
            else:
                self.daughterType = self.popidx - 1 #index of the daughter cell population with A->B+B, A->A+B allowed
        else:
            self.daughterType = daughterType
        
        if motherType < 0:
            self.motherType = self.popidx + 1
        else: 
            self.motherType = motherType #self.popidx + 1 #index of the mother cell population with A->B+B, A->A+B allowed
                     
        self.desc   = {i: [] for i in range(totpts, totpts + self.ncells)} #descendents
        self.ancs   = {i: i  for i in range(totpts, totpts + self.ncells)} #ancestors: at first time ancestor are themself
        self.parent = {i: i  for i in range(totpts, totpts + self.ncells)} #parents: at first time parents are themself
        self.born   = {i: -refract  for i in range(totpts, totpts + self.ncells)} #birth-time
        
        self.decTime = {} #decision times
        
        
        
        self.dieTime = {} #stores times which cells die, and index of cell
        self.dupTime = {} #stores times which cells divide with tuple of parent and daughters
        self.kilTime = {} #stores times which cells were killed
        self.conserve = conserve
        self.totpts  = totpts + self.ncells #total number of points in history of simulation (used for lineage tracing)
        
        if self.divtime > 0:
            initDecTimes = np.random.exponential(self.divtime - 1 + self.refract, self.ncells) + 1
            for i in range(self.ncells):
                self.decTime.setdefault(int(initDecTimes[i]), []).append(i + totpts)  

        self.futureTimes = SortedSet(list(self.decTime.keys())) #stores division time for future events, useful for simulating longer times

        
    def getIndices(self): #gets arrays of cell indices for alive cells
        return np.array(list(self.cellidx), dtype = np.int32)
        
    def updateTotpts(self, totpts):
        self.totpts = totpts
        return totpts
    
    def getTotpts(self):
        return self.totpts
        
    #make cell-decisions:
        #Implement different strategies
    def cellDecision(self, time, tiss = None): #tissue is not needed
        self.stochasticDecision(time, tiss)
        
        
    def nextEventTime(self):
        if self.ncells > 0 and self.divtime > 0:
            return self.futureTimes[0]
        else:
            return np.inf


    def stochasticDecision(self, time, tiss = None):
        if time in self.decTime:
            ndec = len(self.decTime[time]) #number of decisions
            #based on position calculate division / differentation prob    
            rnums = np.random.random(ndec) # generate random number
            
            if self.conserve == None:
                for i in range(ndec):
                    d = self.decTime[time][i]
                    
                    if d not in self.cellidx: #may have died from some other means
                        continue

                    if rnums[i] <= self.divprob: #replace with: self.divprob[i] # divide
                        self.divide(d, time)  
                         
                    elif rnums[i] <= self.divprob + self.difprob:
                        self.differentiate(d, time, tiss)
                        
                    elif rnums[i] <= self.divprob + self.difprob + self.assprob:
                        self.assymDivide(d, time, tiss)
                        
                    elif rnums[i] <= self.divprob + self.difprob + self.assprob + self.terprob:   
                        self.termDivide(d, tiss, time) 
                        
                    elif rnums[i] <= self.divprob + self.difprob + self.assprob + self.terprob + self.dedifprob:
                        assert(self.popidx < tiss.npops)
                        success = self.differentiate(d, time, tiss, daughterType = self.popidx + 1)                        
                        
                    elif rnums[i] <= self.divprob + self.difprob + self.assprob + self.terprob + self.dedifprob + self.desymprob:
                            assert(self.popidx < tiss.npops)
                            success = self.customDivide(d, tiss, time, daughter1type = self.popidx + 1, daughter2type = self.popidx + 1)
                        
                    elif rnums[i] <= self.divprob + self.difprob + self.assprob + self.terprob + self.dedifprob + self.desymprob + self.deassprob:
                            assert(self.popidx < tiss.npops)
                            success = self.customDivide(d, tiss, time, daughter1type = self.popidx - 1, daughter2type = self.popidx + 1)
                        
                    else: #delaminate
                        if self.assymReplace is True:
                            self.replaceByParent(d, self.totpts, d, tiss, time)
                        
                        success = self.die(d, time) 
                        
                        if success and self.singleDaughter is True:
                            self.removeDaughter(d, tiss, time)
                            self.daughter.pop(d)
                                                               
            elif self.conserve == 'global':
                for i in range(ndec):
                    d = self.decTime[time][i]     
                    
                    if rnums[i] <= self.divprob*2: #replace with: self.divprob[i] # divide
                        if self.divide(d, time):
                            self.deleteRandom(time)
                        #print(d, 'divides, we have', self.ncells, 'cells left')

                    elif rnums[i] <= self.divprob*2 + self.difprob:
                        self.differentiate(d, time, tiss)
                        
                    elif rnums[i] <= self.divprob*2 + self.difprob + self.assprob:
                        # We need to delete a cell in the population that we create

                        if self.assymDivide(d, time, tiss):
                            tiss.removeRandomCell(self.daughterType)
                        
                    else: #rnums[i] < self.divprob*2 + self.difprob + self.assprob + self.terprob:  
                        # We need to delete a cell in the population that we create                     
                        if self.termDivide(d, tiss, time):
                            tiss.removeRandomCell(self.daughterType)                    

            self.futureTimes.pop(0)
                
                
    def divide(self, d, time, tiss = None): # cell d divides at time t
        success = False

        if d in self.cellidx: #check the cell index is valid
            #set descendents
            self.desc[d] = [self.totpts, self.totpts + 1] 
            self.desc[self.totpts] = []
            self.desc[self.totpts + 1] = []    

            #set ancestors
            self.ancs[self.totpts] = self.ancs[d]
            self.ancs[self.totpts + 1] = self.ancs[d]
            
            self.parent[self.totpts]     = d
            self.parent[self.totpts + 1] = d

            self.born[self.totpts]     = time
            self.born[self.totpts + 1] = time
                        
            #inherit daughters from parent
            if self.singleDaughter is True:
                self.daughter[self.totpts]         = self.daughter[d]
                self.daughter[self.totpts + 1]     = -1
                self.alldaughters[self.totpts]     = self.daughter[d]
                self.alldaughters[self.totpts + 1] = -1                
                self.daughter.pop(d)

            #remove particle
            self.cellidx.remove(d)

            #add new particle
            self.cellidx.add(self.totpts)
            self.cellidx.add(self.totpts + 1)  
            
            #set decision times
            if self.divtime > 0:
                dtimes = 1 + np.random.exponential(self.divtime - 1 - self.corrtimeSym, 2) + time + self.refract
                if self.corrtimeSym > 0:
                    dtimes +=   np.random.exponential(self.corrtimeSym, 1)

                self.decTime.setdefault(int(dtimes[0]), []).append(self.totpts)
                self.decTime.setdefault(int(dtimes[1]), []).append(self.totpts + 1)
                self.futureTimes.add(int(dtimes[0]))
                self.futureTimes.add(int(dtimes[1]))

            
            #store time of division
            self.dupTime.setdefault(time, []).append([d, self.totpts, self.totpts + 1])
            
            #update number of points
            self.totpts += 2
            self.ncells += 1
            
            success = True

        return success


    
    def die(self, d, time = 0): # cell d dies
        success = False
           
        if d in self.cellidx:
            #remove particle
            self.cellidx.remove(d)            
            self.dieTime.setdefault(time, []).append(d) # self.dupTime
            self.ncells -= 1
            
            if self.verbose:
                print('Removing', d, 'from', self.popidx, 'at', time) #self.cellidx
            
            success = True            

        return success
        

        
    def differentiate(self, d, time, tiss, daughterType = None):
        success = False    
        
        if d in self.cellidx:
            if daughterType is None:
                daughterType = self.daughterType
            
            #set descendents
            self.desc[d] = [self.totpts]

            #remove particle
            self.cellidx.remove(d)

            #store time of decision
            self.dupTime.setdefault(time, []).append([d, self.totpts])
            
            self.ncells -= 1
            
            # add cell of other type
            tiss.addCell(daughterType, self.totpts, None, None, self.ancs[d], d)            

            #update number of points
            self.totpts += 1
                
            success = True            
       
        return success  
        
        
    def assymDivide(self, d, time, tiss):
    # assymetric division
    # fixchild = True for daughter cell of same type position fixed
        success = False
        
        if d in self.cellidx:
            #set descendents
            self.desc[d] = [self.totpts, self.totpts + 1] 
            self.desc[self.totpts] = []

            #set ancestors
            self.ancs[self.totpts] = self.ancs[d]
            self.parent[self.totpts] = d
            
            self.born[self.totpts] = time
            
            
            #update single daughter
            if self.singleDaughter is True:
                #if self.daughter[d] >= 0: #check is done elsewhere
                self.daughter[self.totpts]     = self.totpts + 1
                self.alldaughters[self.totpts] = self.totpts + 1                
                self.removeDaughter(d, tiss, time)
                self.daughter.pop(d) 
                
           
        
            #remove particle
            self.cellidx.remove(d)
            
            #add new particle of same type
            self.cellidx.add(self.totpts)


            if self.corrtimeAss > 0:
                delay = np.random.exponential(self.corrtimeAss, 1)
                drate = tiss.cellpops[self.daughterType].divtime - self.corrtimeAss
                if drate <= 0:
                    drate = 1
            else:
                delay = 0
                drate = None
                                 
            #set decision times
            if self.divtime > 0:
                dtimes = 1 + np.random.exponential(self.divtime - 1 - self.corrtimeAss, 1) + time + self.refract + delay
                self.decTime.setdefault(int(dtimes[0]), []).append(self.totpts)
                self.futureTimes.add(int(dtimes[0]))              

            #store time of division
            self.dupTime.setdefault(time, []).append([d, self.totpts, self.totpts + 1])
                        
            #add extra cell to the tissue
            #popidx, d, pos, theta, anc = -1, parent = -1, sister = -1, dtime = None, delay = 0, drate = None)
            tiss.addCell(self.daughterType, self.totpts + 1, None, None, self.ancs[d], d, self.totpts, None, delay, drate)

            #update number of points
            #self.ncells += 1
            self.totpts += 2

            success = True            
            
        return success   


    def customDivide(self, d, tiss, time, daughter1type = None, daughter2type = None): 
        # custom division
        if daughter1type is None:
            daughter1type = self.daughterType
            
        if daughter2type is None:
            daughter2type = self.daughterType   

        success = False
       
        if d in self.cellidx:
            #set descendents
            self.desc[d] = [self.totpts, self.totpts + 1] 
            self.desc[self.totpts] = []

            #remove particle
            self.cellidx.remove(d)

            #store time of division
            self.dupTime.setdefault(time, []).append([d, self.totpts, self.totpts + 1])

            #add extra cells to the tissue
            tiss.addCell(daughter1type, self.totpts,     None, None, self.ancs[d], d, self.totpts, None, 0, None)                   
            tiss.addCell(daughter2type, self.totpts + 1, None, None, self.ancs[d], d, self.totpts, None, 0, None)   
            
            #one less cell of this cell-type
            self.ncells -= 1 
            self.totpts += 2   
            
            if self.verbose:   
                print(d, 'custom divides from', self.popidx, 'to', daughter1type, daughter2type, 'giving', self.desc[d], 'with ancestor', self.ancs[d])  
                  
            success = True            

        return success      
    
    def termDivide(self, d, tiss, time): 
    # "terminal" division of cell-type A into cell-type B

        success = False
       
        if d in self.cellidx:
            #set descendents
            self.desc[d] = [self.totpts, self.totpts + 1] 
            self.desc[self.totpts] = []

            #remove particle
            self.cellidx.remove(d)

            #store time of division
            self.dupTime.setdefault(time, []).append([d, self.totpts, self.totpts + 1])

            if tiss.cellpops[self.daughterType].corrtimeSym > 0:
                delay = np.random.exponential(tiss.cellpops[self.daughterType].corrtimeSym, 1)
                drate = tiss.cellpops[self.daughterType].divtime - tiss.cellpops[self.daughterType].corrtimeSym
                if drate <= 0:
                    drate = 1
            else:
                delay = 0
                drate = None

            #add extra cells to the tissue
            tiss.addCell(self.daughterType, self.totpts,     None, None, self.ancs[d], d, self.totpts, None, delay, drate)                   
            tiss.addCell(self.daughterType, self.totpts + 1, None, None, self.ancs[d], d, self.totpts, None, delay, drate)   
            
            #one less cell of this cell-type
            self.ncells -= 1 
            self.totpts += 2               
                  
            success = True            

        return success       
        
         
        
    def addCell(self, d, pos = None, theta = None, time = 0, anc = -1, parent = -1, sister = -1, dtime = None, delay = 0, drate = None):
        #note: this does not update totpts       

        self.desc[d] = []
        self.ancs[d] = anc
        self.parent[d]  = parent
        self.born[d] = time
        
        if self.verbose:
            print('Adding', d, 'to pop', self.popidx, 'at', time, 'has ancestor', anc, 'parent', parent) #self.cellidx
        
        self.cellidx.add(d)
        
        self.ncells += 1   

        if drate is None:
            drate = self.divtime #this is a time not a rate
        
        #set decision times
        if self.divtime > 0:
            dtime = 1 + np.random.exponential(drate - 1, 1) + time + self.refract + delay
            self.decTime.setdefault(int(dtime[0]), []).append(d)
            self.futureTimes.add(int(dtime[0]))              


            
    def replaceByParent(self, parent, daughter, tiss, time = -1):
        ''' If a cell dies and it is touching is parent then we can replace the lost cell
    '''

        self.addCell(self.totpts, pos = None, theta = None, time = time, 
            anc = self.ancs[parent], parent = parent)    
                    
        success = True
        self.totpts += 1

        return success        
        
        
    def clear(self):
        self.cellidx = set()
        self.desc   = dict() #descendents
        self.ancs   = dict() #ancestors: at first time ancestor are themself
        self.parent = dict() #parents: at first time parents are themself
        self.born   = dict() #birth-time
        
        self.decTime = {} #decision times
        
        self.dieTime = {} #stores times which cells die, and index of cell
        self.dupTime = {} #stores times which cells divide with t        
        self.futureTimes.clear()

        self.ncells = 0
        
    def copy(self):
        return copy.deepcopy(self)
                
            
                
                
    def evolvecopy(self, tsteps, divtime, skip = 10, 
                   deathprofile = None, tstart = 0,
                   useinstantdeath2 = False, born = None,
                   refract = 0):
        ''' 
            Evolve and extract results with a different decay (division) time
            Use instant death kills every 2nd cell instantly then the other comes later: simulating skipping a layer
            Can pass in born times
        '''
        
        #Exclude old cells:
        if born is not None:
            self.born = born
            
        self.born = {k:v for (k,v) in self.born.items() if v >= tstart}

        self.cellidx = set() #set(list(self.born.keys()))
        self.ncells = 0 

        self.divtime = divtime
        ncells = len(list(self.born.keys()))

        self.decTime = {}
        if divtime > 0:
            dectime = (np.array(list(self.born.values())) + 1 + np.random.exponential(divtime - 1, ncells)).astype('int') + refract

            for i in range(ncells):
                self.decTime.setdefault(int(dectime[i]), []).append(list(self.born.keys())[i])

        self.dieTime = {} #stores times which cells die, and index of cell
        self.dupTime = {} #stores times which cells divide with t        

        cellids = []
        ncells = np.zeros(tsteps + 1 - tstart, dtype = np.int64)
        #ncells[0] = self.ncells

        birthtime = {}
        for b in self.born.keys():
            birthtime.setdefault(self.born[b], []).append(b)

        #print(birthtime)

        if not useinstantdeath2:
            for time in range(tstart, tsteps + 1):#eventimes:    
                if time in birthtime.keys():
                    #add cells
                    for d in birthtime[time]:
                        #print('Add', d, 'at time', time)                
                        self.cellidx.add(d)
                        self.ncells += 1 
                
                if time in self.decTime:
                    for d in self.decTime[time]:
                        #print('Kill', d, 'at time', time)    
                        success = self.die(d, time)
                        
                if deathprofile is not None and time > 0:
                    tokill = self.getIndices()[(np.random.random(self.ncells) < deathprofile[time - 1]).nonzero()[0]]

                    for d in tokill:
                        success = self.die(d, time)

                ncells[time] = int(self.ncells)
                assert(len(self.cellidx) == self.ncells)
                #print(time, self.cellidx, len(self.cellidx), self.ncells)
                if time % skip == 0: #saves data every 10 time-points
                    cellids.append(list(self.cellidx).copy()) 
        
        elif useinstantdeath2:
            for time in range(tstart, tsteps + 1):#eventimes:    
                if time in birthtime.keys():
                    #add cells
                    #print(birthtime[time])
                    assert(len(birthtime[time]) % 2 == 0)
                    for i in range(len(birthtime[time])):
                        #print('Add', d, 'at time', time)   
                        d = birthtime[time][i] 
                        self.cellidx.add(d)
                        self.ncells += 1                         
                        if i % 2 == 0:
                            success = self.die(d, time)

                if time in self.decTime:
                    for d in self.decTime[time]:
                        #print('Kill', d, 'at time', time)    
                        success = self.die(d, time)                        

                ncells[time] = int(self.ncells)
                assert(len(self.cellidx) == self.ncells)
                #print(time, self.cellidx, len(self.cellidx), self.ncells)
                if time % skip == 0: #saves data every 10 time-points
                    cellids.append(list(self.cellidx).copy())              
        
        if len(cellids) != int((tsteps - tstart)/skip) + 1:
            print('Length is off', len(cellids), int((tsteps - tstart)/skip) + 1)
        
        assert(len(cellids) == int((tsteps - tstart)/skip) + 1)
        
        return {'ncells': ncells, 'cellid': cellids}  
    
            
        
    def reset(self, oldtime = 0):
        self.basereset(oldtime)
        
        decTimeKeys = np.array(list(self.decTime.keys()))        
        self.futureTimes = SortedSet(decTimeKeys[decTimeKeys > 0])
        
