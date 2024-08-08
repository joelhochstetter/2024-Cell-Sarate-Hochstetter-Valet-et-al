import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import time
from utils import *
from cellulation import *
import copy
import numbers

#tissue class
class tissue(Voronoi):
    ''' tissue class defines tissue by Voronoi tessalation of points and evolves 
        according to rules of a self-propelled Voronoi model
        
        Calls cellulation class to implement division and differentiation
    '''

    def __init__(self, points = None, thetas = None, L = 1, PBC = True, pop_idx = 0, divtime = 0, 
        divprob = 0.0, assprob = 0.0, difprob = 0.0, terprob = 0.0, sbasshed = 0, sbasrefract = 0, rule = 'stochastic', 
        poppts = None, seed = None, Ly = None, sim_params = None, **params):
        
        '''
            Other params:
            Alow = None, Ahigh = None, 
            conserve = None, assymReplace = False, 
            singleDaughter = False, areadivtime = 0.0, refract = 0, sbasshed = -1, alwaysPair = False, syncCycles = -1, delayedDeath = 1,
            shapeAlign = 0.0, velAlign = 0.0, neighAlign = 0.0, rho_c = None, hilln = 2, daughterTypes = None, 
            corrtimeSym = 0, corrtimeAss = 0, Ly = None):

            Arguments for all tissues:
                poppts (array or number): will generate tissue with points at random coordinates
                points (Nx2 array): (x,y) coordinates for N points 
                thetas (Nx1 array): angular orientation of initial velocity, 
                    either None (generated from uniform distribution) or a Nx1 array
                L: linear dimension of box
                PBC: whether to use periodic boundary conditions. Non-PBCs may not work in this version of code
                
            For tissues with division and death:
                For a certain number of cell populations, Each cell is assigned a population index "pop_idx"
                If you want only one cell population, leave this as 0. 
                If you want m populations (for m > 1), set pop_idx as Nx1 array where elements are 0,..., m-1
                    corresponding to index of each population
                For one population:
                    divtime (positive double): is the average division time
                    divprob (double): is the probability to divide (instead of delaminate) 0.0 <= divprob <= 1.0
                For one or more populations:
                    Enter divtime, divprob as lists where values correspond to each population
                    E.g. divtime = [5, 10], divprob = [1.0, 0.0] specifies one renewing and one dying population
                    
                    
            sim_params (dict): dictionary of parameters for simulation, e.g. mu, kA, kP, A0, P0, Dr, v0, dt
                This will fixe the parameters for the simulation, overriding any values passed in
        '''
        
        self.dim = 2 #we have a 2D tissue

        if seed is not None:
            seed = np.random.seed(seed)

        #Set-up initial cells and box
        self.L = L
        if Ly is None:
            Ly = L
        self.Ly = Ly

        if poppts is not None:
            pop_idx = self.randomConfig(poppts)
        else:
            self.cellpoints = points #correspond to point locations
            
        self.oldpoints  = self.cellpoints.copy() #store "old points" for calculating flocking, at previous time-point

        
        self.npts   = len(self.cellpoints)
        self.totpts = 0 #total number of cells in history of simulation
        self.splits = [0, self.npts] # default splits between points

        
        self.E = None
        self.time = 0
                
        if thetas is None:
            self.thetas = np.random.uniform(0, 2*np.pi, self.npts)
        else:
            self.thetas = thetas
        
        #Suprabasal shedding time-scale < 0: no suprabasal layer
        if sbasshed < 0:
            self.suprabasal = False
            self.nsbcells = 0
        else:
            self.suprabasal = True
            self.sbasshed   = sbasshed
            self.sbasrefract = sbasrefract
                
        
        # Set-up populations
        self.makeParams(pop_idx, divtime, divprob, assprob, difprob, terprob, rule, **params)
        self.setupPopulations(pop_idx, divtime, rule, **params)

        #Add present only PBCs are coded
        if PBC:
            self.addPBC()
        else:
            self.fullpts = self.cellpoints
        
        #apply PBCs
        self.oldpoints = np.zeros((self.npts,2))
        self.oldnpts = self.npts
        self.updateVoronoi() #take Voronoi tessalation and calculate relevant attributes
    
        #all profile of global rate of cell-death:
        self.killprofile = None
        
        #For flocking dynamics
        self.setupFlocking(**params)
        
        #Set fixed mechanical parameters, overrides values passed in
        self.sim_params = {}
        if sim_params is None:
            self.default_sim_params = {}
        else:
            self.default_sim_params = sim_params


    def randomConfig(self, poppts):
        '''
            If input number of cells by population, initialises with randomly places cells:
        '''
             
        if np.ndim(poppts) == 0: #if input as a number
            poppts = [poppts]
        
        npts = np.sum(poppts)
        
        self.cellpoints = randomConfig(npts, self.L, Ly = self.Ly) #initialisePositions
        self.poppts = poppts
        splits = self.updateSplits()

        popidx = np.zeros(npts, dtype = np.int32)
        for i in range(len(poppts)):
            popidx[splits[i]:splits[i+1]] = i
            
        return popidx

  
    
    def updateVoronoi(self, tolerance = 0.0): #update attributes for changes to points

        self.getVoronoi()
        self.ridge_vertices = np.array(self.ridge_vertices)        
        self.npbcpts   = len(self.fullpts) #number of pbc points 
        self.nv = len(self.vertices)
        self.tri = self.triangles() #updates triangles and interior sites 
        #always update areas, perimeters
        self.edgeLengths = self.ridgeLengths() #check how to correctly use functions to update attributes themself
        
        self.areas = self.cellAreas()
        self.perim = self.cellPerimeters()


    def getVoronoi(self):
        Voronoi.__init__(self, self.fullpts)
        
        
    def copy(self):
        return copy.deepcopy(self)
        
    
    def reset(self, full = False):
        '''reset's populations retaining final point locations, thetas, 
            cells of each population type and

            Note: the dead cell will be left in decision time, unless a full reset is used.
        '''
        if full == True:
            oldtime = 0
        else:
            oldtime = self.time

        print('Resetting time and division times')        
        
        for i in range(self.npops):
            self.cellpops[i].reset(oldtime)
            
        self.cellid = self.cellid[:self.npts]
        
        if self.suprabasal:
            self.sbas.reset(oldtime)
            #self.initSuprabasal()
            self.sbas.clear()
            self.nsbcells = 0
                   
        self.time = 0      

        self.updateVoronoi() #take Voronoi tessalation and calculate relevant attributes
  
        
    def addPBC(self, rinc = None): #adds periodic boundary conditions
        #rinc is inclusion radius for PBCs
        if rinc == None:
            rinc = self.L
            
        vx = self.L *np.array((1,0), dtype = np.float64)
        vy = self.Ly*np.array((0,1), dtype = np.float64)
        boxpoints = self.cellpoints % (np.array([self.L, self.Ly])[None,:])
        left  = boxpoints[:,0] < rinc
        right = boxpoints[:,0] > self.L  - rinc
        up    = boxpoints[:,1] > self.Ly - rinc
        down  = boxpoints[:,1] < rinc
        self.fullpts = np.vstack([boxpoints, 
                            boxpoints[left ].copy() + vx, 
                            boxpoints[right].copy() - vx, 
                            boxpoints[up   ].copy() - vy, 
                            boxpoints[down ].copy() + vy, 
                            boxpoints[left  & down].copy() + vx + vy, 
                            boxpoints[right & up  ].copy() - vx - vy, 
                            boxpoints[left  & up  ].copy() + vx - vy, 
                            boxpoints[right & down].copy() - vx + vy])
                            
        self.PBC  = True
        self.npbcpts = len(self.fullpts)
        
        #store indices
        self.pbcidx = np.concatenate([np.array(np.arange(self.npts)), 
                                    np.nonzero(left)[0], np.nonzero(right)[0],
                                    np.nonzero(up)[0], np.nonzero(down)[0], 
                                    np.nonzero(left  & down)[0], np.nonzero(right & up)[0],
                                    np.nonzero(left  & up)[0], np.nonzero(right & down)[0]])
        
        
        return self.fullpts
    

     
     
    def ridgeLengths(self): #calculates lengths of ridges around points
        '''
            Stores the lengths of edges by vertex indices
        '''
        #use ridge_vertices as np.array
        dist = np.linalg.norm(self.vertices[self.ridge_vertices[:,0],:] - self.vertices[self.ridge_vertices[:,1],:], axis = 1)
        rl = np.zeros((self.nv + 1, self.nv + 1), dtype = np.float64)
        rl[self.ridge_vertices[:,0], self.ridge_vertices[:,1]] = dist
        rl[self.ridge_vertices[:,1], self.ridge_vertices[:,0]] = dist
        return rl       
    
    def junctionLengths(self, incpbc = True, useEdgeLengths = False): #calculates lengths of ridges around points
        '''
            Stores the lengths of edges by cell indices
            
            jl[:npts, :npts] 
            
            if useEdges is True, we assume self.edgeLengths = self.ridgeLenghts() is up to date
        '''
        
        jl = np.zeros((self.npbcpts, self.npbcpts), dtype = np.float64)        
        if useEdgeLengths:
            jl = self.edgeLengths
            #jl[self.ridge_points[:,0], self.ridge_points[:,1]] = self.edgeLengths[self.ridge_vertices[:,0],self.ridge_vertices[:,1]]
        else:
            dist = np.linalg.norm(self.vertices[self.ridge_vertices[:,0],:] - self.vertices[self.ridge_vertices[:,1],:], axis = 1)
            jl[self.ridge_points[:,0], self.ridge_points[:,1]] = dist
            #jl[self.ridge_points[:,1], self.ridge_points[:,0]] = dist
            jl += jl.transpose()
        if incpbc is False:
            jl = jl[:self.npts, :self.npts]
            
        return jl
    
        
    def cellAreas(self): #signed area for all cells
        return [PolyArea(self.vertices[self.regions[v]], np.array(self.regions[v])) for v in self.point_region[:self.npts]]
    
    '''
    def cellPerimeters(self): #perimeters for all cells
        return PolyPerim(self.edgeLengths[:self.npts, :])#, self.ridge_points)
    ''' 
    
    def cellPerimeters(self): #perimeters for all cells
        return [PolyPerim(self.edgeLengths, np.array(self.regions[v]))  for v in self.point_region[:self.npts]]
        
    def areaById(self, i):
        return self.areas[self.inverseid[i]]
        
    def popAreas(self, i): #return the area of all cells in a specified population
        self.updateSplits()
        return np.array(self.areas[self.splits[i]:self.splits[i+1]])
    
    def popPerims(self, i): #return the area of all cells in a specified population
        self.updateSplits()
        return self.perim[self.splits[i]:self.splits[i+1]]    
    
    def updateSplits(self):
        self.splits = list(np.cumsum(self.poppts))
        self.splits.insert(0, 0)  
        return self.splits
    
    def triangles(self): #triangles of points
        #stores point indices for each vertex
        #last element corresponds to points which lie in boundary regions  
        #updates interior sites
        tri = [set() for i in range(self.nv + 1)] 
        for i in range(len(self.ridge_vertices)):
            tri[self.ridge_vertices[i,0]].add(self.ridge_points[i,0])
            tri[self.ridge_vertices[i,0]].add(self.ridge_points[i,1])
            tri[self.ridge_vertices[i,1]].add(self.ridge_points[i,0])
            tri[self.ridge_vertices[i,1]].add(self.ridge_points[i,1])
        
        if np.max([len(tt) for tt in tri[:-1]]) == 4:
            print('WARNING: 4 fold vertices detected, repairing cell positions with noise at time ' + str(self.time))
            self.cellpoints += np.random.random((self.npts,2))*1e-7
            self.addPBC()
            self.updateVoronoi()
            return self.tri
            
        self.interior = self.interiorSites(tri)

        return np.array([list(t) for t in tri[:-1]])
    
    def interiorSites(self, triList): #returns interior sites as true and exterior as false
        interior = np.ones(self.npbcpts, dtype = bool)
        interior[np.array(list(triList[-1]))] = False
        return interior
    
    def get_dhdr(self): #all_dhdr(points, tri, npts, nv, vertices)
        #return all_dhdr(self.fullpts, self.tri, self.npbcpts, self.nv, self.vertices)
        return all_dhdr(self.fullpts, self.tri, self.npts, self.nv, self.vertices)

    
    #For optimised code:
    def get_dEdh(self, KA, KP, A0, P0): 
        return all_dEdh(self.vertices, self.regions, self.point_region, self.points, self.areas, self.perim, KA, KP, A0, P0, self.npts, self.interior, self.pbcidx, self.npbcpts, self.ridge_points, np.array(self.splits))
        #return all_dEdh(self.vertices, self.regions, self.point_region, self.areas, self.perim, KA, KP, A0, P0, self.npts, self.interior)
    
    '''
    #For old code
    def get_dEdh(self, KA, KP, A0, P0): #all_dEdh(vertices, regions, point_region, Ac, Pc, KA, KP, A0, P0, npts, interior)
        return all_dEdh(self.vertices, self.regions, self.point_region, self.areas, self.perim, KA, KP, 
                        A0, P0, self.npbcpts, self.interior, self.pbcidx)
        #return all_dEdh(self.vertices, self.regions, self.point_region, self.areas, self.perim, KA, KP, A0, P0, self.npts, self.interior)        
    '''
    
    def getForces(self, dEdh, dhdr): #computeForces(ridge_points, ridge_vertices, point_region, dEdh, dhdr, npts, interior)
        #return computeForces(self.ridge_points, self.ridge_vertices, self.point_region, dEdh, dhdr, self.npts, self.interior)
        return computeForces(self.ridge_points, self.ridge_vertices, self.point_region, dEdh, dhdr, self.npts, self.interior, self.pbcidx)    
            
    def backInBox(self): #move points back in box
        return self.cellpoints % self.L
    
    def setupFlocking(self, shapeAlign = 0.0, velAlign = 0.0, neighAlign = 0.0, **extras):
        self.flockType = 0 #0: No flocking, 1: aligns with long axis of cell, 2: polarity aligns with velocity vector, assigns with polarity of neighbours
        
        if self.npops == 1:
            self.shapeAlign = shapeAlign
            self.velAlign   = velAlign
            self.neighAlign = neighAlign
            if shapeAlign > 0.0:
                self.flockType = 1
            elif velAlign > 0.0:
                self.flockType = 2
            elif neighAlign > 0.0:
                self.flockType = 3        
        else:
            self.shapeAlign = np.zeros(self.npops)
            self.velAlign   = np.zeros(self.npops)
            self.neighAlign = np.zeros(self.npops)
            
            if isinstance(shapeAlign, numbers.Number):
                if shapeAlign > 0:
                    self.shapeAlign *= shapeAlign
                    self.flockType = 1
            elif len(shapeAlign) == self.npops:
                self.shapeAlign = shapeAlign
                self.flockType = 1            
            else:
                print('Incorrect size of flocking strengths specified')
                assert(0)
            
            if isinstance(velAlign, numbers.Number):
                if velAlign > 0:
                    self.velAlign *= velAlign
                    self.flockType = 2        
            elif len(velAlign) == self.npops:
                self.velAlign = velAlign
                self.flockType = 2       
            else:
                print('Incorrect size of flocking strengths specified')   
                
            if isinstance(neighAlign, numbers.Number):
                if neighAlign > 0:
                    self.neighAlign *= neighAlign
                    self.flockType = 3
            elif len(neighAlign) == self.npops:
                self.neighAlign = neighAlign
                self.flockType = 3           
            else:
                print('Incorrect size of flocking strengths specified')
                
        if self.flockType == 1:
            print('Using polarity aligning with long-axis')
        elif self.flockType == 2:
            print('Using polarity aligning with velocity')
        elif self.flockType == 3:
            print('Using polarity aligning with neigbours')
    
    
    def addFlocking(self): #add aligning interactions
        alignJ = 0.0 #aligning interaction
        delta = np.zeros(self.npts) #aligning angle: (theta-phi) for both polar, 2*(theta - phi) for one nematic
        
        if self.flockType == 0:
            return
        
        elif self.flockType == 1: #shape aligning
            velor  = np.zeros(self.npts)  
            alignJ = self.shapeAlign
            for i in range(self.npts):
                verts = self.vertices[self.regions[self.point_region[i]]]
                mat1 = np.ones((len(verts), len(verts)))
                A = ((verts[:,0]*mat1).transpose() - verts[:,0])**2 + ((verts[:, 1]*mat1).transpose() - verts[:,1])**2
                loc = np.unravel_index(np.argmax(A), A.shape)
                velor[i] = np.arctan2(verts[loc[0],0] - verts[loc[1], 0], verts[loc[0],1] - verts[loc[1],1])
                        
            alignJ = self.shapeAlign
            delta = 2*(self.thetas - velor) #nematic-type alignment
            
        elif self.flockType == 2: #velocity aligning
            cellvel = self.cellpoints - self.oldpoints 
            velor   = np.arctan2(cellvel[:,0], cellvel[:,1])   #orientation of cell velocity vector
            alignJ = self.velAlign
            delta = self.thetas - velor
                                
        elif self.flockType == 3: #neigbour aligning
            #Vicsek-type aligning
            cellvel = self.cellpoints - self.oldpoints 
            velor   = np.arctan2(cellvel[:,0], cellvel[:,1])   #orientation of cell velocity vector
            
            avgneighangle = np.zeros(self.npts)
            
            #Compute average of neighbours
            for i in range(self.npts):
                #average angle of self and neighbours
                avgneighangle[i] = np.angle(np.exp(1j*self.thetas[i]) + np.sum(np.exp(1j*self.thetas[self.getNeighbours(i)])))
                
            #classic vicsek
            if self.neighAlign == np.inf:
                self.thetas = avgneighangle
                return
            
            alignJ = self.neighAlign
            delta = self.thetas - avgneighangle            
            
        #Aligning model:
        if self.npops == 1:
            self.thetas += -alignJ*np.sin(delta)
        else:
            for p in range(self.npops):
                self.thetas[self.splits[p]:self.splits[p+1]] += -alignJ[p]*np.sin(delta[self.splits[p]:self.splits[p+1]])
             
                    
    def rotDiff(self, Dr, dt, v0): #add rotational diffusion (perstent RW of centres)
        #Dr is rotational diffusion coefficient, dt is time-step, v0 is cell velocity
        #further major optimisation from pre-generating random numbers
        
        if Dr > 0.0:
            Drdt = np.sqrt(2*Dr*dt)
            self.thetas += Drdt*np.random.normal(0, 1, self.npts) #Add rotationaldiffusion

        if v0 > 0.0:
            self.cellpoints[:,0] += v0*dt*np.cos(self.thetas)
            self.cellpoints[:,1] += v0*dt*np.sin(self.thetas)
        
        return self.cellpoints
    
    def rotDiffMulti(self, Dr, dt, v0): #add rotational diffusion (perstent RW of centres)
        #Dr is rotational diffusion coefficient, dt is time-step, v0 is cell velocity
        #further major optimisation from pre-generating random numbers

        for i in range(self.npops):
            if self.poppts[i] == 0:
                continue
            
            if Dr[i] > 0.0:
                Drdt = np.sqrt(2*Dr[i]*dt)
                self.thetas[self.splits[i]:self.splits[i+1]] += Drdt*np.random.normal(0, 1, self.poppts[i]) #Add rotationaldiffusion
                     
            if v0[i] > 0:
                self.cellpoints[self.splits[i]:self.splits[i+1],0] += v0[i]*dt*np.cos(self.thetas[self.splits[i]:self.splits[i+1]])
                self.cellpoints[self.splits[i]:self.splits[i+1],1] += v0[i]*dt*np.sin(self.thetas[self.splits[i]:self.splits[i+1]])
        
        
        return self.cellpoints    
        
    def addForces(self, mu, dt, KA, KP, A0, P0):
        #add forces onto each cell
        dhdr   = self.get_dhdr()
        dEdh   = self.get_dEdh(KA, KP, A0, P0)
        forces = self.getForces(dEdh, dhdr)  #computeForce vector
        self.oldpoints = self.cellpoints.copy()
        self.cellpoints += mu*forces[:self.npts, :]*dt
        return self.cellpoints
        
    def addForcesMulti(self, mu, dt, KA, KP, A0, P0):
        # Add forces for multiple cell populations
        
        dhdr   = self.get_dhdr()
        dEdh   = self.get_dEdh(KA, KP, A0, P0)
        forces = self.getForces(dEdh, dhdr)  #computeForce vector
        
        self.oldpoints = self.cellpoints.copy()  #save old points      

        for i in range(self.npops):
            if len(self.cellpoints[self.splits[i]:self.splits[i+1],0]) != len(forces[self.splits[i]:self.splits[i+1],0]):
                print(i, self.splits, np.shape(self.cellpoints), np.shape(forces))
            assert(len(self.cellpoints[self.splits[i]:self.splits[i+1],0]) == len(forces[self.splits[i]:self.splits[i+1],0]))
            self.cellpoints[self.splits[i]:self.splits[i+1],:] += mu[i]*forces[self.splits[i]:self.splits[i+1], :]*dt
        
        return self.cellpoints        
        
        
        
    def energy(self, kA, A0, kP, P0): #energy per cell
        if self.npops == 1:
            return np.sum((kA*(abs(np.array(self.areas[:self.npts])) - A0)**2 + kP*(np.array(self.perim[:self.npts]) - P0)**2))/self.npts
        else: #multiple populations
            E = 0
            for p in range(self.npops):
                E += np.sum((kA[p]*(abs(np.array(self.areas[self.splits[p]:self.splits[p+1]])) - A0[p])**2 + kP[p]*(np.array(self.perim[self.splits[p]:self.splits[p+1]]) - P0[p])**2))/self.npts
            return E
    
    def PBCevolve(self, Dr, v0, mu, kA, kP, A0, P0, dt = 1e-2, plot = False): 
        #evolve SPV model one time-step with periodic boundary conditions
        if self.npts == 0:
            return
        
        #0.5 is guaranteed not to break
        rinc = 0.55*np.max(self.perim[:self.npts]) #formerly 1.5 #

        #self.addPBC()        
        self.addPBC(rinc)
        self.updateVoronoi()
                            
        if self.npops > 1:
            self.addForcesMulti(mu, dt, kA, kP, A0, P0)
            self.rotDiffMulti(Dr, dt, v0)
            self.addFlocking()
        else:
            self.addForces(mu, dt, kA, kP, A0, P0)
            self.rotDiff(Dr, dt, v0)
            self.addFlocking()            
                    
        self.E = self.energy(kA, A0, kP, P0)
        
        if plot: #plot for time-steps
            fig = voronoi_plot_2d(self, show_vertices=False)
            plt.xlim([0, self.L])
            plt.ylim([0, self.Ly])    
            plt.show()

    
    def makeParams(self, pop_idx, divtime, divprob, assprob, difprob, terprob, rule, **params):
        keys = ['pop_idx', 'divtime', 'divprob', 'assprob', 'difprob', 'terprob', 'rule'] 
        values = [pop_idx, divtime, divprob, assprob, difprob, terprob, rule]             
        self.params = {**params, **dict(zip(keys, values))}
        return self.params       
    
    #From parameters dictionary, get the parameters for the i-th population
    #This corresponds to taking the i-th element of parameters that are lists
    #Future add for suprabasal layer
    def popParams(self, i):
        popParams = {}
        if i < self.npops: 
            for key in self.params:
                if type(self.params[key]) is list or type(self.params[key]) is np.ndarray:
                    if len(self.params[key]) > self.npops: #don't include variables which are specified by point, e.g. pop_idx
                        continue
                    assert(len(self.params[key]) == self.npops)
                    popParams[key] = self.params[key][i]
                else:
                    popParams[key] = self.params[key]

        #Pass the daughter division time in, useful for enforcing sister correlations
        if i > 0:
            popParams['daughter_tau'] = self.params['divtime'][i-1]
        else:
            popParams['daughter_tau'] = self.sbasshed
        
        

        return popParams


    
    def setupPopulations(self, pop_idx, divtime, rule, **params):   
        if self.time > 0:
            self.time = 0
            print('Time reset to 0')
        
        # if we have division and differentiation
        self.npops = 1
        if type(divtime) is list:
            self.npops = len(divtime)
            if hasattr(self, 'poppts'):
                assert(self.npops == len(self.poppts))
            
        #total number of points in each cell population:
        self.poppts = np.zeros(self.npops, dtype = np.int32)  
        
        self.cellpops = [] #list of cell populations, using cellulation objects

        self.makePopulations(pop_idx, rule, **params)

        #Initialise suprabasal population             
        if self.suprabasal:
            self.initSuprabasal()   
        


    def makePopulations(self, pop_idx, rule = 'stochastic', daughterTypes = None, **params):
    
        '''
            Re-orders points, thetas, so one population lies ahead of other populations
            and creates "cellulation" object for each population
            
                            
        '''
        if type(rule) is not list:
            rule = [rule for i in range(self.npops)]
            
        #Parameters that can be same across populations (e.g. corrtimeSym) is deprecated
        # Now must specify for each population, or defaults to all zero

        #Set-up cell points and thetas
        if self.npops > 1:
            parts = partitionTypes(pop_idx, nclasses = self.npops)
        else:
            parts = [np.arange(self.npts)]
        
        if self.dim > 0:
            newpoints = np.reshape(np.array([]), (0,2))
            newthetas = np.array([])
        
        for i in range(self.npops):
            if self.npops == 1:
                if self.dim > 0:
                    poppoints = self.cellpoints
                    popthetas = self.thetas
                    newpoints = poppoints
                    newthetas = popthetas
                self.poppts[0] = self.npts

            else:
                if self.dim > 0:
                    if len(parts[i]) > 0:
                        poppoints = self.cellpoints[np.array(parts[i]), :]
                        popthetas = self.thetas[np.array(parts[i])]
                        newpoints = np.vstack([newpoints, poppoints])
                        newthetas = np.hstack([newthetas, popthetas])
                    else:
                        poppoints = np.zeros((0,2))
                        popthetas = np.zeros((0,2))    
                                        
                self.poppts[i] = len(parts[i])
            

            #Set daughter populations
            if type(daughterTypes) is list:
                assert(len(daughterTypes) == self.npops)
                daughterType = daughterTypes[i]
            else:
                if i == 0 and self.suprabasal is True:
                    daughterType = self.npops
                else:
                    daughterType = -1
            
            if   rule[i] == 'stochastic':
                self.cellpops.append(cellulation(poppoints, popidx = i, totpts = self.totpts, popthetas = popthetas, daughterType = daughterType, **self.popParams(i)))

            elif rule[i] == 'size':
                self.cellpops.append(sizulation(poppoints, popidx = i, totpts = self.totpts, popthetas = popthetas, daughterType = daughterType, **self.popParams(i)))            

            elif rule[i] == 'size-fixrate': #constant rate, fate determination based on size
                self.cellpops.append(sizulationFR(poppoints, popidx = i, totpts = self.totpts, popthetas = popthetas, daughterType = daughterType, **self.popParams(i)))
            
            elif rule[i] == 'size-size': #area rule for cell decisions, fate determination based on size
                self.cellpops.append(sizulation_sz(poppoints, popidx = i, totpts = self.totpts, popthetas = popthetas, daughterType = daughterType, **self.popParams(i)))
            
            elif rule[i] == '0D':
                self.cellpops.append(cellulation0D(cellidx = np.arange(self.totpts, self.totpts + self.poppts[i]), popidx = i, totpts = self.totpts, 
                                                   daughterType = daughterType, **self.popParams(i)))

            else:
                print('Invalid rule')
                assert(0)

            self.totpts += self.poppts[i]

        if self.dim > 0:
            self.cellpoints = newpoints #store number of cells (which is boundaries between populations)
            self.thetas     = newthetas

        self.splits = list(np.cumsum(self.poppts))
        self.splits.insert(0, 0)            
        self.totpts    = self.npts
        assert(self.totpts == self.npts)
        self.rule = rule
        self.cellid = np.arange(self.totpts, dtype = np.int32)
        
        return self.cellpops

    
    def updatePopulations(self, time):
        '''
            Update cell populations by allowing cells to divide and differentiate
        '''
        
        if self.npops > 1:
            newpoints = np.reshape(np.array([]), (0,2))
            newthetas = np.array([])
            newcellid = np.array([], dtype = np.int32)   
            
            for i in range(self.npops):
                self.cellpops[i].updatePoints(self.cellpoints[self.splits[i]:self.splits[i+1]])
                self.cellpops[i].updateThetas(self.thetas[self.splits[i]:self.splits[i+1]])                


            for i in range(self.npops):
                #update points, update thetas for cellulation class
                                    
                self.cellpops[i].updateTotpts(self.totpts)             
                 
                if time > 0:
                    self.cellpops[i].cellDecision(time, self) #cell-fate decisions
                
                self.totpts = self.cellpops[i].getTotpts() # get total points

                
            for i in range(self.npops):                
                #get points, get thetas 
                self.poppts[i]  = self.cellpops[i].ncells
                if self.poppts[i] > 0:              
                    newpoints = np.vstack([newpoints, self.cellpops[i].getPoints()])
                    newthetas = np.hstack([newthetas, self.cellpops[i].getThetas()])
                    newcellid = np.hstack([newcellid, self.cellpops[i].getIndices()])                  
                              
            self.npts = np.sum(self.poppts)
            self.cellpoints = newpoints
            self.thetas = newthetas
            self.cellid = newcellid
            self.splits = list(np.cumsum(self.poppts))
            self.splits.insert(0, 0)            
            
        else: # n == 1
            #update points, update thetas    
            self.cellpops[0].updatePoints(self.cellpoints)
            self.cellpops[0].updateThetas(self.thetas)            
            
            self.cellpops[0].cellDecision(time, self) #cell fate decisions
            
            #get points, get thetas 
            self.cellpoints = self.cellpops[0].getPoints()
            self.thetas = self.cellpops[0].getThetas() 
            self.cellid = self.cellpops[0].getIndices()   
            self.poppts[0]  = self.cellpops[0].ncells            
            self.npts = int(self.cellpops[0].ncells)
            
        self.inverseid  = dict(zip(self.cellid, np.arange(self.npts)))
        
        
        if self.suprabasal:
            self.updateSuprabasal(time)
        
        #print('Time: ', time, 'ncells:', self.poppts, 'nsb:', self.nsbcells, 'Totpts:', self.totpts)
        
        return self.cellpops


    def initSuprabasal(self):

        self.nsbcells = 0 #start with zero suprabasal cells
        self.sbas = cellulation0D(cellidx = set(), divtime = self.sbasshed, refract = self.sbasrefract,
            divprob = 0.0, totpts = self.totpts, conserve = None, assprob = 0.0, difprob = 0.0, 
            terprob = 0.0, daughterType = 10*self.npops, popidx = self.npops, motherType = 0)
        
        
        #Create the cellulation and add to end of cellpops
        self.cellpops = self.cellpops[:self.npops]
        self.cellpops.append(self.sbas)
        
        
    def updateSuprabasal(self, time):
        #Update the cellulation
         
        self.sbas.updateTotpts(self.totpts)
        
        if time > 0:
            self.sbas.cellDecision(time, self) #cell-fate decisions
        
        self.totpts = self.sbas.getTotpts() # get total points      
        self.nsbcells = self.sbas.ncells 
        
        assert(len(self.sbas.cellidx) == self.sbas.ncells)
        self.cellid = np.hstack([self.cellid, self.sbas.getIndices()]) 


    def addCell(self, popidx, d, pos, theta, anc = -1, parent = -1, sister = -1, dtime = None, delay = 0, drate = None):
        #dtime = None: use default division rate, 
        #dtime <= 0: daughter cannot divide (e.g. in singleDaughter model)
        #dtime > 0: sets a time for the cell to divide

        #drate = None: use default division rate
        #drate  > 0: use adjusted division rate

        self.cellpops[popidx].addCell(d, pos, theta, self.time, anc, parent, sister, dtime, delay, drate)
        
        
    def removeCell(self, popidx, d):
        return self.cellpops[popidx].die(d, self.time)
        
    def termDivide(self, popidx, d):
        self.cellpops[popidx].updateTotpts(self.totpts)
        success = self.cellpops[popidx].termDivide(d, self, self.time)        
        self.totpts = self.cellpops[popidx].getTotpts()
        return success
        
    def divideFateChoice(self, popidx, d):
        self.cellpops[popidx].updateTotpts(self.totpts)
        success = self.cellpops[popidx].divideFateChoice(d, self, self.time)
        self.totpts = self.cellpops[popidx].getTotpts()
        return success



    def scheduleEvent(self, popidx, d, dtime = -1, drate = -1):
        if dtime < 0: #use random time
            if drate <= 0:
                drate = self.cellpops[popidx].divtime
                if drate <= 0:
                    drate = self.cellpops[self.npops - 1].divtime
                    
            if drate <= 0: #If still no time, we do not schedule
                return 
            
            if self.time - self.cellpops[popidx].born[d] > self.cellpops[popidx].refract:
                dtime = 1 + np.random.exponential(drate - 1, 1) + self.time
            else:
                dtime = 1 + np.random.exponential(drate - 1, 1) + self.time + (self.cellpops[popidx].refract + self.cellpops[popidx].born[d] - self.time)
            #self.cellpops[popidx].born[d]
        
        self.cellpops[popidx].decTime.setdefault(int(dtime), []).append(d)
    
    
    
    def removeRandomCell(self, popidx):
        self.cellpops[popidx].deleteRandom(self.time)
    
    
    def lineagePopulations(self): # get lineages
        if self.npops > 1 or self.suprabasal:
            descendents = [cp.desc for cp in self.cellpops]
        else:
            descendents = self.cellpops[0].desc
            
        if self.suprabasal:
            descendents.append(self.sbas.desc)
        
        return descendents
            
    def getAncestors(self):
        if self.npops > 1 or self.suprabasal:
            ancestors = [cp.ancs for cp in self.cellpops]
        else:
            ancestors = self.cellpops[0].ancs
            
        if self.suprabasal:
            ancestors.append(self.sbas.ancs)            
            
        return ancestors
    
    def getPopIndices(self, i):
        return self.cellid[self.splits[i]:self.splits[i+1]]
    
    def getDivisions(self):
        if self.npops > 1 or self.suprabasal:
            divs = [cp.dupTime for cp in self.cellpops]
        else:
            divs = self.cellpops[0].dupTime
            
        if self.suprabasal:
            divs.append(self.sbas.dupTime)                
            
        return divs
        
        
    def getDeaths(self):
        if self.npops > 1 or self.suprabasal:
            deaths = [cp.dieTime for cp in self.cellpops]
        else:
            deaths = self.cellpops[0].dieTime
            
        if self.suprabasal:
            deaths.append(self.sbas.dieTime)                  
            
        return deaths
     
    def getNeighbours(self, i): #returns as points idx, not cell idxs
        #Gets all the neighbours of
        #if i not in self.point_region:
            
        return self.pbcidx[self.regions[self.point_region[i]]]
        
    def isNeighbour(self, i, j):
        #Checks if i and j are neighbours
        # i and j are the unique cell indices
        return self.inverseid[i] in self.getNeighbours(self.inverseid[j])
    
    def isAlive(self, p):
        return p in self.cellid
        
    def isInPop(self, p, popidx):
        return p in self.cellpops[popidx].getIndices()        
    
    def initCellDeath(self, killprofile, tsteps):
        if killprofile is not None:
            if type(killprofile) == int or type(killprofile) == float:
                self.killprofile = np.ones(tsteps)*killprofile
                print('Using constant cell death')
                
            elif len(killprofile) == tsteps:
                self.killprofile = killprofile
                print('Using specified cell death profile')  
                
            else:
                print('Kill profile provided invalid')
    
    def setVerbose(self, verbose):
        for j in range(len(self.cellpops)):
            self.cellpops[j].verbose = verbose

    
    def multify(self, x):
        if np.ndim(x) == 0 and self.npops > 1:
            x = np.ones(self.npops)*x
        return x
    
    
    def setup_sim_params(self, Dr = 0.0, v0 = 0.0, mu = 1.0, kA = 1.0, kP = 1.0, A0 = 1.0, p0 = 3.5):
        ''' 
            Sets up the simulation parameters, and multifies
        '''
        
        self.sim_params = self.default_sim_params.copy()
        
        if 'Dr' not in self.sim_params:
            self.sim_params['Dr'] = Dr
        
        if 'v0' not in self.sim_params:
            self.sim_params['v0'] = v0
            
        if 'mu' not in self.sim_params:
            self.sim_params['mu'] = mu
            
        if 'kA' not in self.sim_params:
            self.sim_params['kA'] = kA
            
        if 'kP' not in self.sim_params:
            self.sim_params['kP'] = kP
            
        if 'A0' not in self.sim_params:
            self.sim_params['A0'] = A0
            
        if 'p0' not in self.sim_params:
            self.sim_params['p0'] = p0
        
        #Ensure parameters are right dimension
        self.sim_params['Dr'] = self.multify(self.sim_params['Dr'])
        self.sim_params['v0'] = self.multify(self.sim_params['v0'])
        self.sim_params['mu'] = self.multify(self.sim_params['mu'])
        self.sim_params['kA'] = self.multify(self.sim_params['kA'])
        self.sim_params['kP'] = self.multify(self.sim_params['kP'])
        self.sim_params['A0'] = self.multify(self.sim_params['A0'])
        self.sim_params['p0'] = self.multify(self.sim_params['p0'])
        self.sim_params['P0'] = self.sim_params['p0']*np.sqrt(self.sim_params['A0']) #target perimeter
        
        return self.sim_params
    
    
    def shape_data(self):
        ''' 
            Calculates shape data from the current state of the tissue
        '''
        
        shape_tensor = np.zeros((self.npts,2,2))
        circularity  = np.zeros(self.npts)
        Saniso       = np.zeros(self.npts)
        Sorientation = np.zeros(self.npts)

        for p in range(self.npts):
            vertices = self.vertices[np.array(self.regions[self.point_region[p]]),:]
            centroid = np.mean(vertices,0)
            rel_vert = vertices - centroid
            shape_tensor[p,:,:] = np.mean(np.array([np.tensordot(rel_vert[v,:], rel_vert[v,:], axes = 0) for v in range(len(vertices))]),0)
            
            evals, evecs = np.linalg.eig(shape_tensor[p,:,:])
            idx = evals.argsort()#[::-1]   
            evals = evals[idx]
            evecs = evecs[:,idx]

            circularity[p] = evals[0]/evals[1]
            Saniso[p] = (evals[1] - evals[0])/np.sqrt(np.sum(np.square(evals)))
            Sorientation[p] = np.arctan2(*evecs[:,1]) % np.pi
            
        #Nematic order parameter, local and global
        edges = self.ridge_points[np.max(self.ridge_points,1) < self.npts]
        Qglobal = np.abs(np.mean(np.exp(2j*Sorientation)))
        rel_angle = Sorientation[edges[:,0]] - Sorientation[edges[:,1]] # + np.pi/2)%np.pi - np.pi/2
        Qlocal  = np.mean(np.cos(2*rel_angle))
    
        return {'Qlocal': Qlocal, 'Qglobal': Qglobal, 'circularity': circularity,
                'Saniso': Saniso, 'Sorientation': Sorientation, 
                'shape_tensor': shape_tensor}
    
    
    
    def save_shape_data(self, shapes = {}, save_shape = True):
        ''' 
            Obtains a dictionary with cell-shape data of the current state, 
                when requested to save
        '''
        
        shapes.setdefault('areas', []).append(self.areas[:self.npts].copy())
        shapes.setdefault('jlengths', []).append(self.junctionLengths(incpbc = False).copy())
        shapes.setdefault('perims', []).append(self.perim[:self.npts].copy())
        
        if save_shape:
            sd = self.shape_data()
            for k in sd.keys():
                shapes.setdefault(k,[]).append(sd[k].copy())
        
        return shapes    
        

    def sim(self, tsteps = 100, dt = 1e-2, Dr = 0.0, v0 = 0.0, mu = 1.0, kA = 1.0, kP = 1.0, A0 = 1.0, p0 = 3.5, plot = False, skip = 10, verbose = False,
            save_shape = False, **kwargs):
        '''simulator to use without division and death
            Dr: rotational diffusion coefficient
            v0: magnitude of self-propulsion velocity of cells
            mu: mobility, inverse of frictional drag
            Ac, Pc: actual cell area/perimeter
            kA, kP: corresponding spring constants
            A0, P0: preferred cell area, perimeter
            plot: whether or not to plot at each time-step
        '''
        
        #Set-up storage
        cellpos = [self.cellpoints.copy()]
        E = np.zeros(tsteps + 1)
        
        #areas = [self.areas[:self.npts]]
        #jlengths = [self.junctionLengths(incpbc = False)] #junction lengths
        shapes = self.save_shape_data(shapes = {}, save_shape = save_shape)
        
        self.time = 0
        self.oldnpts = 0
        self.dt = dt
        
        
        #Set the simulation parameters, and multify
        self.setup_sim_params(Dr, v0, mu, kA, kP, A0, p0)
        
        #Initial energy
        E[0] = self.E
                
        
        print('Starting sim with', self.npts, 'cells at', time.asctime())

        for i in range(1, tsteps + 1):
            self.PBCevolve(self.sim_params['Dr'], self.sim_params['v0'], self.sim_params['mu'], 
                           self.sim_params['kA'], self.sim_params['kP'], self.sim_params['A0'], self.sim_params['P0'], dt, plot)
            E[i] = self.E  
            self.time += 1
            
            if i % skip == 0: #saves data every 10 time poitnts
                cellpos.append(self.cellpoints.copy())
                shapes = self.save_shape_data(shapes, save_shape = save_shape)
                
                #areas.append(self.areas[:self.npts])
                #jlengths.append(self.junctionLengths(incpbc = False))
        
        assert(len(shapes['jlengths']) == len(cellpos))
        # Update population positions
        self.updatePopulations(-1) #negative time ensures no cell divisions
        
        print('Ending sim with', self.npts, 'cells at', time.asctime())            
        return {'cellpos': cellpos, 'E': E, 'shapes': shapes, #'areas': areas, 'jlengths': jlengths,
                'skip': skip}



    # simulations with division and death using cellulation class
    def simFull(self, tsteps = 100, dt = 1e-2, Dr = 0.0, v0 = 0.0, mu = 1.0, kA = 1.0, kP = 1.0, A0 = 1.0,
                p0 = 3.5, plot = False, skip = 10, maxCells = np.inf, minCells = 5, killprofile = None, 
                verbose = False, save_shape = False,  **kwargs):
        
        ''' simulator to use with division and death 
            Dr: rotational diffusion coefficient
            v0: magnitude of self-propulsion velocity of cells
            mu: mobility, inverse of frictional drag
            Ac, Pc: actual cell area/perimeter
            kA, kP: corresponding spring constants
            A0, P0: preferred cell area, perimeter
            plot: whether or not to plot at each time-step
        '''        
             
        #Set-up storage
        cellpos = [self.cellpoints.copy()]
        cellids = [self.cellid.copy()]
        E = np.zeros(tsteps + 1)
        #areas = [self.areas[:self.npts]]
        #jlengths = [self.junctionLengths(incpbc = False)]
        shapes = self.save_shape_data(shapes = {}, save_shape = save_shape)

        
        ncells = np.zeros((tsteps + 1, self.npops))
        ncells[0,:] = self.poppts
        nsbcells = np.zeros(tsteps + 1)
        nsbcells[0] = self.nsbcells
        E[0] = self.E
        self.time = 0
        self.dt = dt
        self.oldnpts = 0             
        
        #Set the simulation parameters, and multify
        self.setup_sim_params(Dr, v0, mu, kA, kP, A0, p0)
        
        #setting verbosity
        self.setVerbose(verbose)        

        self.initCellDeath(killprofile, tsteps)
        
        print('Starting sim with', self.npts, 'cells at', time.asctime())
    
        
        for i in range(1, tsteps + 1):
            self.time += 1
            
            self.updatePopulations(i) #updates populations
            
            ncells[i, :] = self.poppts   
            nsbcells[i]  = self.nsbcells                     
            
            if self.npts >= maxCells or self.npts <= minCells:
                break        
        
            self.PBCevolve(self.sim_params['Dr'], self.sim_params['v0'], self.sim_params['mu'], 
                           self.sim_params['kA'], self.sim_params['kP'], self.sim_params['A0'], 
                           self.sim_params['P0'], dt, plot)
                    
            E[i] = self.E
            
            if i % skip == 0: #saves data every 10 time-points
                cellpos.append(self.cellpoints.copy())
                cellids.append(self.cellid.copy())
                shapes = self.save_shape_data(shapes, save_shape = save_shape)
                

                
        assert(len(shapes['jlengths']) == len(cellids))
        
        print('Ending sim with', self.npts, 'cells at', time.asctime()) 
          
        return {'cellpos': cellpos, 'E': E, 'shapes': shapes, #'areas': areas, 'jlengths': jlengths,
         'ncells': ncells.astype('int'), 'desc': self.lineagePopulations(), 
         'anc': self.getAncestors(), 'cellid': cellids, 'dupTimes': self.getDivisions(),
          'dieTimes': self.getDeaths(), 'nsbcells': nsbcells.astype('int'), 'skip': skip}    
    
    
    def runsuprabasal2(self, results, tsteps, intshed, sbasshed, skip = 10, npops = None, instantdelam2 = False):
        #Run with 2 suprabasal layers:
        #Run delamination to the suprabasal layer occuring via an intermediate state
        #Note that we don't relabel cells depending on which "suprabasal level" they're in
        
        
        #If no intermediate shedding provided, default to a single suprabasal layer
        if intshed == 0 and sbasshed > 0:        
            return self.rerunsuprabasal(results, tsteps, sbasshed, skip)
        
        if npops == None:
            npops = self.npops


        self.intshed  =  intshed #delamination rate for the intermediate cell-type
        self.sbasshed = sbasshed #shedding rate from the suprabasal layer
        
        newresults = results.copy()
        ncells = results['ncells'][:, :npops]
        cellids = results['cellid'].copy()
        
        
        #If already has intermediate layer:
        if len(self.cellpops) == npops + 2:
            self.sbas = self.cellpops[-2]
            
        
        #run the intermediate layer
        intlayer = self.sbas.copy() #intermediate layer
        intresults = intlayer.evolvecopy(tsteps, intshed, skip, instantdelam2 = instantdelam2, born = self.sbas.born)
        nintcells = intresults['ncells']
        self.totpts = intlayer.getTotpts()
        
        #Make the suprabasal layer
        born = {}
        for t in intlayer.dieTime.keys():
            for d in intlayer.dieTime[t]:
                born[d] = t
        
        #Run the suprabasal layer
        sbas = self.sbas
        sbasresults = sbas.evolvecopy(tsteps, sbasshed, skip, instantdelam2 =  False, born = born)     
        nsbcells = sbasresults['ncells']   
       
        
        #Check populations work
        assert(len(intresults['cellid'])  == len(cellids))
        assert(len(sbasresults['cellid']) == len(cellids))
        
        #Append all cellid's together
        for tt in range(0, int(tsteps/skip) + 1): 
            cellids[tt] = np.hstack([cellids[tt][:np.sum(ncells[tt*skip,:])], 
                             intresults['cellid'][tt],
                             sbasresults['cellid'][tt]]).astype('int64')
            #print(tt, sbasresults['cellid'][tt], nsbcells[tt*skip])
            #assert(len(sbasresults['cellid'][tt]) == nsbcells[tt*skip])
            assert(len(cellids[tt]) == np.sum(results['ncells'][tt*skip,:]) + nintcells[tt*skip] + nsbcells[tt*skip])

        #Append the populations     
        self.cellpops = self.cellpops[:npops]
        self.cellpops.append(intlayer)        
        self.cellpops.append(sbas)
        ncells = np.hstack([ncells, np.reshape(nintcells.copy(),(tsteps + 1,1))])        
        
        #Save the data
        newresults['nsbcells'] = nsbcells.copy()
        newresults['cellid']   = cellids
        newresults['ncells']   = ncells
        
        self.nsbcells = int(self.sbas.ncells)
        self.totpts = self.sbas.getTotpts()
        self.cellid = self.cellid[:self.npts]
        self.cellid = np.hstack([self.cellid, intlayer.getIndices(), self.sbas.getIndices()]) 
        #print(len(self.cellid), self.sbas.ncells + self.npts)
        #print(len(cellids[-1]), len(self.cellid), tt)        
        assert(len(self.cellid) == self.sbas.ncells + self.npts + intlayer.ncells)

        self.npops = npops + 1
        
        return newresults    
    
    

    def rerunsuprabasal(self, results, tsteps, divtime, skip = 10, 
                        shed_profile = None, tstart = 0, refract = 0):
        #evolve and extract results with a different decay (division) time
        
        self.sbasshed = divtime
        
        newresults = results.copy()
        ncells = results['ncells']

        if refract > 0:
            print('Running shedding with refract:', refract, ', shed', divtime)

        sbasresults = self.sbas.evolvecopy(tsteps, divtime, skip, shed_profile, tstart, refract = refract)     
        nsbcells = sbasresults['ncells']
        
        cellids = results['cellid'].copy()

        assert(len(sbasresults['cellid']) == len(cellids))
        assert(len(nsbcells) == len(results['ncells']))
        
        for tt in range(0, int(tsteps/skip) + 1): 
            cellids[tt] = np.hstack([cellids[tt][:np.sum(ncells[tt*skip,:])], 
                             sbasresults['cellid'][tt]]).astype('int64')
            #print(tt, sbasresults['cellid'][tt], nsbcells[tt*skip])
            #assert(len(sbasresults['cellid'][tt]) == nsbcells[tt*skip])
            assert(len(cellids[tt]) == np.sum(results['ncells'][tt*skip,:]) + nsbcells[tt*skip])

        
        newresults['nsbcells'] = nsbcells.copy()
        newresults['cellid']   = cellids
        
        self.nsbcells = int(self.sbas.ncells)
        self.totpts = self.sbas.getTotpts()
        self.cellid = self.cellid[:self.npts]
        self.cellid = np.hstack([self.cellid, self.sbas.getIndices()]) 
        #print(len(self.cellid), self.sbas.ncells + self.npts)
        #print(len(cellids[-1]), len(self.cellid), tt)        
        assert(len(self.cellid) == self.sbas.ncells + self.npts)
        
        return newresults
        
        
    def fixedMutate(self, frompop, topop, moveids):       
        for d in moveids:
            success = self.cellpops[frompop].differentiate(d, self.time, self, daughterType = topop)
            assert(success == True)

        #Repair cell populations

        self.totpts = self.cellpops[frompop].getTotpts() # get total points
        for i in range(self.npops):
            #update points, update thetas for cellulation class               
            self.cellpops[i].updateTotpts(self.totpts)        

        newpoints = np.reshape(np.array([]), (0,2))
        newthetas = np.array([])
        newcellid = np.array([], dtype = np.int32)

        for i in range(self.npops):                
            #get points, get thetas 
            self.poppts[i]  = self.cellpops[i].ncells
            if self.poppts[i] > 0:              
                newpoints = np.vstack([newpoints, self.cellpops[i].getPoints()])
                newthetas = np.hstack([newthetas, self.cellpops[i].getThetas()])
                newcellid = np.hstack([newcellid, self.cellpops[i].getIndices()])                  
                            
        self.npts = np.sum(self.poppts)
        self.cellpoints = newpoints
        self.thetas = newthetas
        self.cellid = newcellid
        self.splits = list(np.cumsum(self.poppts))
        self.splits.insert(0, 0)  
        

    def randomMutate(self, frompop, topop, numcells):
        '''
            moved (numcells) cells at random from one cell population (frompop) to another population (topop)
        '''

        #randomly generate indices
        assert(self.poppts[frompop] >= numcells)
        assert(len(self.cellpops[frompop].getIndices()) == self.poppts[frompop])
        chosen = self.cellpops[frompop].getIndices()[np.array(np.random.sample(range(self.poppts[frompop]), numcells))]
        
        self.fixedMutate(frompop, topop, chosen)


    def plot(self): #Plot snapshot at end of tissue
        fig = voronoi_plot_2d(self, show_vertices = False, show_points = False)
        markers = ['o',  '^', 'o', '*', 'h', 'x']
        colors  = ['w', 'b', 'g', 'r', 'm', 'k']
        
        plt.plot(self.cellpops[0].getPoints()[:,0] % self.L, self.cellpops[0].getPoints()[:,1] % self.L, markers[0],  
                    label = self.cellpops[0].popname, color = colors[p], markersize = 0.000001)  
              
        for p in range(1,self.npops):
            plt.plot(self.cellpops[p].getPoints()[:,0] % self.L, self.cellpops[p].getPoints()[:,1] % self.L, markers[p],  
                     label = self.cellpops[p].popname, color = colors[p])

        if self.npops > 1:
            plt.legend()
            
        plt.axis('square')
        plt.xlim([0, self.L])
        plt.ylim([0, self.Ly])
        #plt.show()

        return fig

    def popnames(self):
        pn = []
        for p in range(self.npops):
            pn.append(self.cellpops[p].popname)

        return pn


    def init_clones(self):
        #Initialise clones
        self.clone


class tissue0D(tissue):

    def __init__(self, pop_idx = np.zeros([0]), divtime = 0, divprob = 0.0, assprob = 0.0, difprob = 0.0, terprob = 0.0, rule = '0D', 
                 sbasshed = 0, seed = None, poppts = None, sbasrefract = 0, **params):
        
        '''
            Removed default parameters (not in tissue class): points, thetas, L, PBC
        '''
        
        
        self.dim  = 0
        

        if seed is not None:
            seed = np.random.seed(seed)
            
        if poppts is not None:
            pop_idx = self.randomConfig(poppts)

        self.npts   = len(pop_idx)
        self.totpts = 0 #total number of cells in history of simulation
        
        self.time = 0
        
        #Suprabasal shedding time-scale < 0: no suprabasal layer
        if sbasshed < 0:
            self.suprabasal = False
            self.nsbcells = 0
        else:
            self.suprabasal = True
            self.sbasshed   = sbasshed
            self.sbasrefract = sbasrefract
                
        self.splits = [0, self.npts] # default splits between points
        
        # Set-up populations
        self.makeParams(pop_idx, divtime, divprob, assprob, difprob, terprob, rule, **params)
        self.setupPopulations(pop_idx, divtime, rule, **params)


    def randomConfig(self, poppts):
        '''
            If input number of cells by population, initialises with randomly places cells:
        '''
        if np.ndim(poppts) == 0: #if input as a number
            poppts = [poppts]
        
        npts = np.sum(poppts)
        self.poppts = poppts
        splits = self.updateSplits()

        popidx = np.zeros(npts, dtype = np.int32)
        for i in range(len(poppts)):
            popidx[splits[i]:splits[i+1]] = i
        return popidx



    def sim(self, tsteps = 100, dt = 1e-2, skip = 10, maxCells = np.inf, minCells = 5, killprofile = None, verbose = False, **kwargs):
        print('No spatial evolution. Use simFull')
        return



    def simFull(self, tsteps = 100, dt = 1e-2, skip = 10, maxCells = np.inf, minCells = 5, killprofile = None, verbose = False, **kwargs):
            ''' simulator to use with division and death 
            '''        
                
            
            cellids = [] #[self.cellid.copy()]
            ncells = np.zeros((tsteps + 1, self.npops))
            ncells[0,:] = self.poppts
            
            nsbcells = np.zeros(tsteps + 1)
            nsbcells[0] = self.nsbcells
                        
            self.dt = dt
            self.oldnpts = 0   
            self.time = 0     
            
            #setting verbosity
            self.setVerbose(verbose)
                                
            self.initCellDeath(killprofile, tsteps)
            
            print('Starting sim with', self.npts, 'cells at', time.asctime())
            
            #Initialise time
            lasttime  = 0            
            
            while lasttime <= tsteps:
                ncells[lasttime:self.time, :] = self.poppts[None,:]  #update for all intermediate times
                nsbcells[lasttime:self.time]  = self.nsbcells  
                
                #Add cellids in between times
                for i in np.arange(int(np.ceil((lasttime)/skip))*skip, self.time, skip):
                    cellids.append(self.cellid.copy())                   
                
                self.updatePopulations(self.time) #updates population
                lasttime = self.time
                self.time = np.min([self.jumpToNextEvent(), tsteps + 1])  
                    
                if self.npts >= maxCells or self.npts <= minCells:
                    break
                                
            self.time = tsteps
                                
            print('Ending sim with', self.npts, 'cells at', time.asctime()) 
            
            return {'ncells': ncells.astype('int'), 'desc': self.lineagePopulations(), 
            'anc': self.getAncestors(), 'cellid': cellids, 'dupTimes': self.getDivisions(),
            'dieTimes': self.getDeaths(), 'nsbcells': nsbcells.astype('int'), 'skip': skip}   



    def updatePopulations(self, time):
        '''
            Update cell populations by allowing cells to divide and differentiate
        '''
        
        if self.npops > 1:
            newcellid = np.array([], dtype = np.int32)   

            for i in range(self.npops):
                #update points, update thetas for cellulation class
                                    
                self.cellpops[i].updateTotpts(self.totpts)
                 
                if time > 0:
                    self.cellpops[i].cellDecision(time, self) #cell-fate decisions
                
                self.totpts = self.cellpops[i].getTotpts() # get total points

                
            for i in range(self.npops):                
                #get points, get thetas 
                self.poppts[i]  = self.cellpops[i].ncells
                if self.poppts[i] > 0:              
                    newcellid = np.hstack([newcellid, self.cellpops[i].getIndices()])                  
                              
            self.npts = np.sum(self.poppts)
            self.cellid = newcellid
            self.splits = list(np.cumsum(self.poppts))
            self.splits.insert(0, 0)            
            
        else: # n == 1
            self.cellpops[0].cellDecision(time, self) #cell fate decisions
            
            #get points, get thetas 
            self.cellid = self.cellpops[0].getIndices()   
            self.poppts[0]  = self.cellpops[0].ncells            
            self.npts = int(self.cellpops[0].ncells)
            
        self.inverseid  = dict(zip(self.cellid, np.arange(self.npts)))
        
        
        if self.suprabasal:
            self.updateSuprabasal(time)
        
        #print('Time: ', time, 'ncells:', self.poppts, 'nsb:', self.nsbcells, 'Totpts:', self.totpts)
        
        return self.cellpops
    
    def jumpToNextEvent(self):
        if self.suprabasal is True:
            npops = self.npops + 1
        else:
            npops = self.npops
       
        newtime = int(np.min([self.cellpops[p].nextEventTime() for p in range(npops)]))      

        if newtime <= self.time:
            print(newtime, self.time, 'ERROR: trying to jump the past')
            assert(0)

        return newtime
    
    
    def reset(self, full = False):
        '''reset's populations retaining final point locations, thetas, 
            cells of each population type and

            Note: the dead cell will be left in decision time, unless a full reset is used.
        '''
        if full == True:
            oldtime = 0
        else:
            oldtime = self.time

        print('Resetting time and division times')        
        
        for i in range(self.npops):
            self.cellpops[i].reset(oldtime)
            
        self.cellid = self.cellid[:self.npts]
        
        if self.suprabasal:
            self.sbas.reset(oldtime)
            #self.initSuprabasal()
            self.sbas.clear()
            self.nsbcells = 0
                   
        self.time = 0      