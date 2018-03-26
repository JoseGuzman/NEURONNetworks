"""
init.py

Jose Guzman, sjm.guzman@gmail.com

Creates a network of interneurons (PV) that receives excitatory inputs 

To execute in Ipython
%run init.py

directly in a Python shell
$ nrngui init.py -python

The simulation returns 

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm
from matplotlib.pyplot import figure, plot

from neuron import h, gui
from Cell_builder import BCbuilder, GCbuilder
from custom_plots import GC_plot

np.random.seed(10)

h.load_file('stdrun.hoc') # need for h.tstop
h.tstop = 500 
h.v_init = -65

#=========================================================================
# 1. create a network of 200 inhibitory neuron
#=========================================================================
icells = 200 

# check Cell_builder
PV = [BCbuilder( idx = i) for i in range(icells)] 
print('Creating %10d PV'%icells)

#=========================================================================
# 2A. Apply different excitatory drives to inhibitory cells 
#=========================================================================

def inject_excitation(celllist, rv):
    """
    Injects a current of random amplitude and duration to

    Arguments:
    cellobj   -- a cell list with BCbuilder objects with Iclamp mechanism
    rv      -- a scipy.stats random object (e.g., norm, lognorm)
    I_mu    -- mean excitatory drive , 0.001 corresponding to 1 uA/cm^2

    """
    
    for cell in celllist:
        stim = cell.IClamp # access IClamp at soma 

        stim.amp = rv.rvs()
        stim.delay = np.abs( np.random.normal( loc = 150, scale = 50 ) )
        stim.dur   = h.tstop - stim.delay

#=========================================================================
# 3. Custom connections between all cells  
#=========================================================================
def recurrent_inhibitory_connections(cell_list, weight = None):
    """
    Connects all the cells in the list via inhibitory recurrent synapses 

    Arguments:
    cell_list   -- a list with cell objects
    """
    if weight is None:
        myweight = 0.0001/100
    else:
        myweight = weight

    for i in range( len(cell_list) ) :
        for j in np.delete( range(len(cell_list)), i): # avoid auptapse 
           cell_list[i].connect2target( cell_list[j].isyn, myweight )
    print("Adding recurrent inhibition")

#=========================================================================
# 4. Visualize
#=========================================================================

h.load_file('gui/gSingleGraph.hoc')
h.load_file('gui/gRasterPlot.hoc')

#=========================================================================
# 5. Prepare topologies 
#=========================================================================
recurrent_inhibitory_connections(PV)

#=========================================================================
# 6. My custom run 
#=========================================================================
def myrun(rv = None):
    """
    Arguments:
    rv  -- a scipy.stats random object
    Custom run:

    Returns:
    Simulates and plots distribution of excitations
    

    """
    if rv is None:
        I_mu = 0.001 # 1 uA/cm^2
        I_sigma = 0.00003 # 3% heterogeneity
        rv   = norm( loc = I_mu, scale = I_sigma)
        inject_excitation(PV, rv)

    if rv is 'lognorm':
        I_mu = 0.001 # 1 uA/cm^2
        I_sigma = 0.00003 # 3% heterogeneity
        #rv   = lognorm(np.log(2), scale = I_mu, loc = 0)
        rv   = lognorm(np.log(2), scale = I_mu, loc = 0)
        inject_excitation(PV, rv)
            
    h.update_rasterplot() # will call run in gui/gRasterPlot  
 
    fig = figure()
    # plot histogram and probability density function
    ax = fig.add_subplot(111)
    
    ext = [cell.IClamp.amp for cell in PV ]
    ax.hist(ext, normed=True, facecolor='white')
    x = np.linspace(rv.ppf(0.00001), rv.ppf(0.9999), 200)
    ax.plot(x, rv.pdf(x), color = '#A52A2A', lw=2)

    plt.show()
    

    
