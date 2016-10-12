"""
init.py

Creates a network of 10000 principal neurons (PC) 

To execute in Ipython
%run init.py

"""

import numpy as np
from scipy.stats import norm

from neuron import h, gui
from Cell_builder import BCbuilder, PCbuilder

np.random.seed(10)

h.load_file('stdrun.hoc') # need for h.tstop
h.tstop = 150    # in ms
h.v_init = -70.0 # in mV

#=========================================================================
# 1. create a network of 10,000 principal cells and 100 BCs
#=========================================================================
icells = 50
scaling = 100
ecells = icells * scaling 

PC = [PCbuilder( idx = i) for i in range(ecells)] 
PV = [BCbuilder( idx = i) for i in range(icells)]


#=========================================================================
# 2. Apply tonic excitatory drive 
#=========================================================================

def inject_random_current(cell, I_mu = 0.004):
    """
    Injects a current of random amplitude and duration to
    cell    -- a BCbuilder object with Iclamp mechanism
    I_mu    -- mean excitatory drive , 0.001 corresponding to 1 uA/cm^2

    """
    I_sigma = 0.00003 # corresponds to 3% heterogeneity
    
    stim = cell.IClamp
    stim.amp   = np.random.normal( loc = I_mu, scale = I_sigma*I_sigma)
    stim.delay = np.abs( np.random.normal( loc = 5, scale = 4 ) )
    stim.dur   = h.tstop - stim.delay
    stim.dur   = 50 - stim.delay

for cell in PC:
    inject_random_current(cell, I_mu = 0.00045)
for cell in PV:
    inject_random_current(cell)

#=========================================================================
# 3. Connect excitatory cells 
#=========================================================================
pEE = 0.08 # probability of excitatory-excitatory connections
ncells = int(0.08*ecells) # number of connected cells
pre = np.random.choice(range(ecells), ncells)
putative_post = np.delete(range(ecells), pre) # avoid auptases!
post = np.random.choice(putative_post, ncells) # pick from the putative post
for i in pre:
    for j in post:
        PC[i].connect2target( target = PC[j].esyn )
#=========================================================================
# 4. Connect inhibitory to excitatory cells 
#=========================================================================
def inhibition():
    """
    Connects inhibitory cells to excitatory neurons
    accounting for the scaling factor to have the
    same probability of connection.
    """
    pIE = 0.90
    nPVcells = int( pIE*icells ) # number of connected cells
    # connect PV cells with PCs
    netconlist = list()
    for pre in range(nPVcells):
        start = pre * scaling
        end   = start + scaling 
        for post in range(start, end):
            netconlist.append(PV[pre].connect2target( target = PC[post].isyn ))
    return(netconlist)


#=========================================================================
# Visualize
#=========================================================================

h.load_file('gui/gSingleGraph.hoc')
h.load_file('gui/gRasterPlot.hoc')
h.update_rasterplot()
#=========================================================================
# 5. My custom run 
#=========================================================================
def myrun():
    h.update_rasterplot() # in gRasterPlot, will execute run
