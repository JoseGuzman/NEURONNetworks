"""
custom_plots.py

Jose Guzman, sjm.guzman@gmail.com

Last change: Sun Oct  2 16:05:16 CEST 2016

Plots a network of cells
"""

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

    myfacecolors = ['#FFFF66']*ncells
    for i in active:
        myfacecolors[i] = 'black'

    myhex.set_facecolors(myfacecolors)
    myhex.set_lw(1)
    myhex.set_edgecolors = ('black',)
    
    ax.add_collection( myhex )
    ax.autoscale_view()
    ax.axis("off")

    return(fig)

if __name__ == "__main__":
    active_cells = range(10,20) 
    myfig = GC_plot(ncells = 1024, active = active_cells)
    myfig.show()
    

