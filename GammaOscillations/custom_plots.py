"""
custom_plots.py

Jose Guzman, sjm.guzman@gmail.com
Claudia Espinoza, claum.espinoza@gmail.com

Last change: Thu Oct  6 20:58:42 CEST 2016

Plots to represent network of cells
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.collections import RegularPolyCollection

import itertools
import numpy as np

def GC_plot(ncells, active):
    """
    Creates a collection of hexagons
    ncells  -- (int) number of cells
    active  -- (list) a list with the ID of the active neurons
    """
    fig = figure(figsize=(9,7), dpi = 92, facecolor = "white")
    ax = fig.add_subplot(111)
    
    # sizes of hexagon
    mysizes = np.ones(ncells)*100 

    # offsetx location
    myoffsets = list()
    n_sqrt = int(np.sqrt(ncells))
    x = np.arange(n_sqrt)
    y1 = np.arange(1, n_sqrt,2)
    y2 = np.arange(0, n_sqrt,2)
    for i in itertools.product(x,y1):
        myoffsets.append(i)
    x = x -0.5
    for i in itertools.product(x,y2):
        myoffsets.append(i)
    
    myhex = RegularPolyCollection(numsides = 6, sizes = mysizes,
        offsets = myoffsets, transOffset = ax.transData)

    myfacecolors = ['#FFFF66']*ncells # yellow
    for i in active:
        myfacecolors[i] = 'black'

    myhex.set_facecolors(myfacecolors)
    myhex.set_lw(1)
    myhex.set_edgecolors = ('black',)
    
    ax.add_collection( myhex )
    ax.autoscale_view()
    ax.axis("off")

    return(fig)

def raster(spk_list, color='b'):
    """
    Creates a raster plot from a list of spike times that can be 
    generated from a network of Cellbuilder objects.

    Arguments:
    spk_list    -- (list) of spike times (e.g. list of cell.spk_times)
    color       -- (str) with the color of the lines

    Returns:
    ax : an axis containing the raster plot
    """

    ax = plt.gca() # get current axis

    for n, spk in enumerate( spk_list ): #n is cells in the network
        start = n + 0.5
        end = n + 1.5
        plt.vlines(spk, start, end, color = color, linewidth=10)

    ymin = 0.5, 
    ymax = 0.5 + len(spk_list) 
    plt.ylim(ymin, ymax)
    plt.xlim(0, 150)
    return ax

def raster2(event_times_list, color='k'):
    """
    Creates a raster plot
    Parameters
    ----------
    event_times_list : iterable
                       a list of event time iterables
    color : string
            color of vlines
    Returns
    -------
    ax : an axis containing the raster plot
    """
    ax = plt.gca()
    for ith, trial in enumerate(event_times_list):
        plt.vlines(trial, ith + .5, ith + 19.5, color=color, linewidth=1)
    plt.ylim(.5, len(event_times_list) + .5)
    plt.xlim(0, 150)
    return ax

if __name__ == "__main__":
    #active_cells = range(10,20) 
    #myfig = GC_plot(ncells = 1024, active = active_cells)
    #myfig.show()
    spikes = np.load('PVrasterEI_0.25.npy')
    fig = figure()
    ax = raster(spikes, color='blue')
    plt.title('Example raster plot')
    plt.xlim(0, 150)
    plt.xlabel('time')
    plt.ylabel('Cell index')
    fig.show()
    


