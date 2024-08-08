'''
    Plotting suite for tissue

'''

#Use: import tissplot as tplt

from tissue import *
import numpy as np
import matplotlib.pyplot as plt
import analysis
import random

from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage






def plotBCD(results, dens0 = 1, plotpops = True, plottot = True, tscale = 1, tiss = None, tshift = 0):
    #Reference density: dens0, if dens0 = 0 then we use time-average of total # of cells
    #scale = 1/dt if you want time, or 1/rate

    colors  = ['0.75', 'b', 'g', 'r', 'm', 'k']

    npops = np.shape(results['ncells'])[1]

    if dens0 == 0:
        dens0 = np.mean(np.sum(results['ncells'],1))

    ntimes = np.shape(results['ncells'])[0]
    timevec = np.arange(ntimes)/tscale + tshift

    labels = []
    if tiss is None:
        plabels = list(np.arange(npops))
    else:
        plabels = tiss.popnames()

    plt.figure()
    if plotpops is True:
        for p in range(npops):
            plt.plot(timevec, results['ncells'][:,p]/dens0, color = colors[p])
        labels += plabels
    if plottot is True:
        plt.plot(timevec, np.sum(results['ncells']/dens0, 1), 'k-')
        labels += ['total']
    
    plt.legend(labels)
    plt.ylabel('Basal cell density')
    plt.xlabel('Time')

    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)


def plot_av_cs():
    return

def plot_cs_dist(cdf = False):
    return


def clonal_snapshot(tiss, results, times = [-1], track = 4, sample_mode = -10, cloneResults = None, 
                    skip = None, titles = None, markpops = True, use_legend = True):
    #Plots a snapshot of tissue across different times with the same clones tracked
    '''
        Pass in:
               tiss: tissue object
            results: results struct
            clone_results: clone_results object if you have it, otherwise will re-calculate
            
            ntimes: list of times, if negative numbers determines from reverse snapshots
        
            track:
                if type is  int: is number of clones to track and we randomly sample based on sample mode
                if type is list: track specific clones

            sample_mode
                 n: For n >= 0, samples from all cells from a specific cell population
              -n+1: For n < npops, samples from all surviving cells from a that population 
                    at latest time-point plotted
                -10: sample from all clones
                -20: sample from all surviving clones at latest time-point plotted
                -30: sample from extinct clones
    '''
    
    ntimes = len(results['ncells'])
    L = tiss.L
    
    if np.min(times) < 0: #repair times, so can specifiy points at end
        tarr = np.array(times, dtype = 'int')
        tarr[tarr < 0] = np.arange(0, ntimes, skip)[tarr[tarr < 0]]
        times = list(tarr)

    if skip is None: #calculate skip if specified
        skip = analysis.calculateSkip(results)
        if np.isnan(skip):
            print('Skip not defined')
            assert(0)
            
    if cloneResults is None:
        cloneResults = analysis.clonestats(results, tiss, tvals = None, skip = skip)

    if titles is None:
        titles = [str(t) for t in times]
        
    positions = results['cellpos']
    ancs = results['anc']
    cellid = results['cellid']
    ncells = np.sum(results['ncells'],1)
    
    survivors = analysis.survivors(cloneResults, time = times[-1])
    diers     = analysis.diers(cloneResults,     time = times[-1])
    labeled   = cellid[0]
    
    #get population ids for initial cells
    ncells0 = results['ncells'][0,:]
    splits = list(np.cumsum(ncells0))
    splits = np.array([0] + splits)
    
    cellpopids = [cellid[0][splits[i]:splits[i+1]] for i in range(tiss.npops)] 
    
    ncl   =  len(survivors)    
    
    
    if type(track) is not list:
        if track > 0:
            if sample_mode == -10:
                samplefrom = labeled            
            elif sample_mode == -20:
                samplefrom = survivors
            elif sample_mode == -30:
                samplefrom = diers
            elif np.abs(sample_mode) < tiss.npops + 1:
                if sample_mode < 0:
                    sample_mode = np.abs(sample_mode + 1)
                    samplefrom = np.array(list(set(survivors).intersection(set(cellpopids[sample_mode]))))
                else:
                    samplefrom = cellpopids[sample_mode]
            else:
                track = 0
                samplefrom = np.array([])
            if track > 0:
                track = list(samplefrom[np.array(random.sample(range(len(samplefrom)), track))])
            else:
                track = []
        else:
            track = []
            
    
    #Plotting part    
    plt.figure(figsize = (10.5,10.5/len(times)))
    for tt in range(len(times)):
        t = times[tt]
        ts = int(t/skip)
        plt.subplot(1,len(times),tt + 1)
        t1 = tissue(positions[ts] % L, L = L)
        
        splits = list(np.cumsum(results['ncells'][t,:]))
        splits = np.array([0] + splits)
        
        ax = plt.gca()
        voronoi_plot_2d(t1, ax, show_vertices = False, show_points = False)
        
        col = ['r', 'g', 'y', 'b', 'm', 'c', 'k']
        #Mark clones
        if len(track) > 0:
            clones = cloneResults['allclones'][t]
            for j in range(len(track)):
                print('Clone', track[j], col[j])
                for c in clones[track[j]]:
                    region = t1.regions[t1.point_region[c]]
                    polygon = [t1.vertices[i] for i in region]
                    plt.fill(*zip(*polygon), col[j%len(col)])
        
        
        #Mark cell populations
        if markpops:
            markers = ['o',  '^', 'o', '*', 'h', 'x']
            if tiss.npops <= 2:
                colors  = ['w', 'k']
            else:
                colors  = ['w', 'b', 'g', 'r', 'm', 'k']            
            plt.plot(positions[ts][splits[0]:splits[1],0] % L, positions[ts][splits[0]:splits[1],1] % L, markers[0],
                label = tiss.cellpops[0].popname, color = colors[0], markersize = 0.000001)

            for p in range(1,tiss.npops):
                plt.plot(positions[ts][splits[p]:splits[p+1],0] % L, positions[ts][splits[p]:splits[p+1],1] % L, markers[0],
                    label = tiss.cellpops[p].popname, color = colors[p])

            if tiss.npops > 1 and use_legend:
                plt.legend()
                
        #plt.title('t = ' + str(titles[tt]))
        plt.title(titles[tt], fontweight = 'bold', fontsize = 10)
            
        plt.axis('square')    
        plt.xlim([0, L])
        plt.ylim([0, L])
        plt.xticks([])
        plt.yticks([])
    
    


    
    
    
def clone_movies(positions, ancs, cellid, ncells = None, nsteps = 100, savename = 'sim.mp4', 
                 gifname = None, track = [0,1,2], sample_mode = -10, fps = 10, L = 15, highlight = None, 
                 skip = 1, diebytime = None, delambytime = None, tscale = 1, tshift = 0, 
                 bridges = None, saveskip = 1):    

    
    # to plot animation use: animation.ipython_display(fps = fps, loop = True, autoplay = True)
    print(len(track))
    col = [plt.cm.hsv(i) for i in range(0,200,np.round(200/len(track)).astype('int'))]
    random.shuffle(col)
    
    # matplot subplot
    fig, ax = plt.subplots()

    # method to get frames
    def make_frame(t):

        # clear
        ax.clear()
        t = int(t*skip*fps)
        
        cloneidx = [ancs[i] for i in cellid[t][:np.sum(ncells[int(t*saveskip)])]]
        clones = {i: [] for i in cellid[0][:np.sum(ncells[0])]}
        
        for i in range(len(cloneidx)):
            clones[cloneidx[i]].append(i)
        
        t1 = Voronoi(addPBC(positions[t] % L, L))
        voronoi_plot_2d(t1, ax, show_vertices = False, show_points = False);
        #col = ['r', 'g', 'y', 'b','c', 'm', 'k', 'tab:orange']  
        
        for j in range(len(track)):
            for c in clones[track[j]]:
                region = t1.regions[t1.point_region[c]]
                polygon = [t1.vertices[i] for i in region]
                ax.fill(*zip(*polygon), c = col[j])
        
        if highlight is not None:
            plt.plot(highlight[:,0], highlight[:,1], 'k.', markersize = 15)

        if ncells is not None:
            #highlight population 2 (SC's)
            plt.plot(positions[t][ncells[int(t*saveskip),0]:,0] % L, positions[t][ncells[int(t*saveskip),0]:,1] % L, 'k.', markersize = 15)            
            
        #Mark dying cells
        if diebytime is not None:
            for c in diebytime[t*saveskip]:
                #print(c, t, diebytime[t])
                where = np.argwhere(cellid[t] == c)
                if len(where) > 0: 
                    cid = np.argwhere(cellid[t] == c)[0][0]
                    #print(cid, positions[t][cid])
                    plt.plot(positions[t][cid,0] % L, positions[t][cid,1] % L, 'rx', markersize = 10)

        #Mark delaminating cells
        if delambytime is not None:
            for c in delambytime[t*saveskip]:
                where = np.argwhere(cellid[t] == c)
                if len(where) > 0: 
                    cid = where[0][0]
                    plt.plot(positions[t][cid,0] % L, positions[t][cid,1] % L, 'g^', markersize = 10)
                
        if bridges is not None:
            for c in cellid[t][:ncells[int(t*saveskip),0]]:
                if c in bridges:
                    csis = bridges[c] 
                    cid = np.argwhere(cellid[t] == c)[0][0]
                    if csis in cellid[t]:
                        csisid = np.argwhere(cellid[t] == csis)[0][0]
                        pair = np.array([cid,csisid])
                        if np.max(np.abs(positions[t][cid,:] % L - positions[t][csisid,:] % L)) < L/2:
                            plt.plot(positions[t][pair,0] % L, positions[t][pair,1] % L, 'g--')
            
        
        plt.title('t = '+ "{:.1f}".format(t*saveskip/tscale + tshift))
        ax.axis('square')
        ax.set_xlim([0,L])
        ax.set_ylim([0,L])
        plt.xticks([])
        plt.yticks([])
        fig.set_dpi(200)
        
        #if t == 0:
        #    plt.savefig('Ablation_t=0.png', dpi = 300)

        # returning numpy image
        return mplfig_to_npimage(fig)

    

    # creating animation
    animation = VideoClip(make_frame, duration = nsteps/fps) # ntsteps/20/10
    
    #animation.write_gif('matplotlib.gif', fps=fps)
    if savename is not None:
        animation.write_videofile(savename, fps=fps)
        
    if gifname is not None:
        animation.write_gif(gifname, fps=fps)        

    return animation


def addPBC(points, L, rinc = None): #rinc is inclusion radius for PBCs
    if rinc == None:
        rinc = L
    vx = L*np.array((1,0), dtype = np.float64)
    vy = L*np.array((0,1), dtype = np.float64)
    left  = points[:,0] < rinc
    right = points[:,0] > L - rinc
    up    = points[:,1] > L - rinc
    down  = points[:,1] < rinc
    points = np.vstack([points, 
                        points[left ].copy() + vx, 
                        points[right].copy() - vx, 
                        points[up   ].copy() - vy, 
                        points[down ].copy() + vy, 
                        points[left  & down].copy() + vx + vy, 
                        points[right & up  ].copy() - vx - vy, 
                        points[left  & up  ].copy() + vx - vy, 
                        points[right & down].copy() - vx + vy])
    return points
