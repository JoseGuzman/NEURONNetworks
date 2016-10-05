"""
init.py

Jose Guzman, sjm.guzman@gmail.com
Claudia Espinoza, claumespinoza@gmail.com

Creates a network of basquet cells (PV) and granule cells (GCs)

TODO: complete simulation description

To execute in Ipython
%run init.py

directly in a Python shell
$ nrngui init.py -python

To implement different connections types and probabilities: 
>>> inhibition(pEI = 0.09765, pRI = 0.24, pLI = 0.3283, debug=1)

The simulation returns 
1) The total number of spikes in the GC network            
2) The number GC cells firing in the last 25 ms            

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from neuron import h, gui
from Cell_builder import BCbuilder, GCbuilder
from custom_plots import GC_plot

np.random.seed(10)

h.load_file('stdrun.hoc') # need for h.tstop
h.tstop = 150 
h.v_init = -80

#=========================================================================
# 1. create a network of 100 inhibitory neurons and 10,000 granule cells 
#=========================================================================
icells = 50 
scaling = 100
ecells = icells * scaling # check ModelDB: 124513 

PV = [BCbuilder( idx = i) for i in range(icells)] 
GC = [GCbuilder( idx = i) for i in range(ecells)] 
print('Creating %10d PV'%icells)
print('Creating %10d GC'%ecells)

#=========================================================================
# 2A. Apply tonic excitatory drive to inhibitory cells 
#=========================================================================

def inject_tonic_excitation(cellobj, I_mu = 0.001):
    """
    Injects a current of random amplitude and duration to

    Arguments:
    cellobj   -- a BCbuilder object with Iclamp mechanism
    I_mu    -- mean excitatory drive , 0.001 corresponding to 1 uA/cm^2
    """
    I_sigma = 0.00003 # corresponds to 3% heterogeneity
    
    stim = cellobj.IClamp
    stim.amp   = np.random.normal( loc = I_mu, scale = I_sigma*I_sigma)
    stim.delay = np.abs( np.random.normal( loc = 5, scale = 4 ) )
    stim.dur   = h.tstop - stim.delay


#=========================================================================
# 2B. Apply random spikes to  excitatory cells 
#=========================================================================
def generate_spk(cellobj, freq):
    """
    Generates APs in cells at a given frequency

    Arguments:
    cellobj   -- a Cellbuilder object with ExpSyn mechanism
    freq      -- frequency of APs (in Hz)
    """
    nAPs = int( h.tstop*(freq/1000.) ) # freq in 1000 ms
    AP_times = np.random.choice( range(int(h.tstop)), size = nAPs)

    mynetstim = list()
    for time in AP_times:
        netstim = h.NetStim()
        netstim.number = 1
        netstim.start = time

        mynetstim.append( netstim )

    mynetcon = list()
    for st in mynetstim:
        nc = h.NetCon(st, cellobj.esyn) 
        nc.weight[0] = 1.755e-5 # one AP only

        mynetcon.append( nc )

    return( mynetstim, mynetcon ) # remember to return mechanisms!


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

def inhibition(pEI, pRI, pLI, debug = None):
    """
    connect inhibitory and excitatory network neurons
    Arguments:
    pEI     -- prob of IE connections
    pRI     -- prob of Recurrent inhibition, P(IE | EI)
    pLI     -- prob of Lateral inhibition , P(IE| not EI )

    #===============================
    # Indices of cell subpopulations
    #===============================
    # ecells: total number of GC cells
    # icells: total number of PV cells
    # exc_GC_idx: subpopulation of GCs sending exc. to PV
    # exc_PV_idx: subpopulation of PV receiving exc. from GC
    # RI_PV_idx:  subpopulation of exc_PV_idx sending inh. to exc_GC_idx
    # RI_GC_idx: subpopulation of exc GC_idx receiving inh. from RI_PV_idx
    # nonexc_GC_idx: subpopulation of GC not sending exc. to PV
    # nonexc_PV_idx: subpopulation of PV not receiving exc. from GC
    # LI_PV_idx: subpop. of nonexc_PV_idx sending inh. to nonexc_GC_idx
    """
    
    #===================================================================
    # 1.- Subpopulation of cells involved in excitation 
    #===================================================================
    # 1A.- Interneuron subpopulation
    nexc_PV = int( round( icells * pEI ) )
    exc_PV_idx = np.random.choice( range(icells),  size = nexc_PV)

    # 1AA.-Subpopulation of RI interneurons
    nRI_PV = int( round(nexc_PV * pRI ) )
    RI_PV_idx = np.random.choice(exc_PV_idx, size = nRI_PV)

    # 1B.- Granule cell sending excitation subpopulation
    nexc_GC = nexc_PV * scaling # scaling to have divisible integer number
    exc_GC_idx = np.random.choice( range(ecells), size = nexc_GC)

    # 1C.- Subpopulation of Granule cell receiving inhibtion 
    nRI_GC = nRI_PV * scaling # to avoid floating points
    RI_GC_idx = np.random.choice(exc_GC_idx, size = nRI_GC)

    #===================================================================
    # 2.- Subpopulation of cells not involved in excitation
    #===================================================================
    # 2A.- Interneuron subpopulation
    nonexc_PV_idx = np.delete( range(icells), exc_PV_idx)

    # 2AA.-Subpopulation of LI interneurons
    nLI_PV = int( round(len(nonexc_PV_idx) * pLI ) )
    LI_PV_idx = np.random.choice(nonexc_PV_idx, size = nLI_PV)

    # 3AA. - Granule cell non-sending excitation subpopulation
    nonexc_GC_idx = np.delete( range(ecells), exc_GC_idx )
    
    # 3AB. - Granule cells receiving lateral inhibition
    nLI_GC = nLI_PV * scaling
    LI_GC_idx = np.random.choice(nonexc_GC_idx, size = nLI_GC)

    netcon = list()
    # GC cells converging on to PV cells
    # divide GC into clusters to be projecting to one PV
    nClusters = nexc_GC/scaling
    for pre in range(nClusters):
        start = pre * scaling 
        end = start + scaling
        for post in exc_GC_idx[ start : end ]:
            netcon.append (GC[post].connect2target( target = PV[pre].esyn))

    # connect RI PV cells back to GCs
    for pre in range(nRI_PV):
        start = pre * scaling
        end   = start + scaling 
        for post in RI_GC_idx[ start: end ]:
            netcon.append( PV[pre].connect2target( target = GC[post].isyn ))

    # connect LI PV cells to nonexc_GCs
    for pre in range(nLI_PV):
        start = pre * scaling
        end = start + scaling
        for post in LI_GC_idx[ start : end ]:
            netcon.append( PV[pre].connect2target( target = GC[post].isyn))
        
    if debug:
        print( "exc_GC    = %2d"%nexc_GC )
        print( "nonexc_GC = %2d"%len(nonexc_GC_idx ))
        print( "RI_GC   = %2d"%nRI_GC )
        print( "RI_PV = %2d"%nRI_PV )
        print( "LI_GC   = %2d"%nLI_GC )
        print( "LI_PV = %2d"%nLI_PV )
        print( "netcons       = %2d"%len(netcon) )
    
#=========================================================================
# 4. Visualize
#=========================================================================

h.load_file('gui/gSingleGraph.hoc')
h.load_file('gui/gRasterPlot.hoc')

#=========================================================================
# 5. Prepare topologies 
#=========================================================================
recurrent_inhibitory_connections(PV)

for cell in PV:
    inject_tonic_excitation(cell)

myfreq = 20 # average spiking frequency
l = list() # netcons and netstim must be contained in an object
for cell in GC:
    mu = np.random.normal(loc = myfreq, scale = 0.00003*0.00003)
    l.append(generate_spk(cellobj = cell, freq = int(mu)))

print('%2d Hz average spiking on GC'%myfreq)

#inhibition(pEI = 0.09765, pRI = 0.24, pLI = 0.3283, debug=1)

#=========================================================================
# 6. My custom run 
#=========================================================================
def myrun():
    """
    Custom run:

    freq    -- average spike frequency of GCs

    Returns:

    the average spike frequency of every GC in the network
    """
    h.update_rasterplot() # will call run in gui/gRasterPlot  

    nactive_GC = 0 # number of cells active in the last 25 ms
    active_GC_idx = list() # cells active in the last 25 ms
    for i, cell in enumerate(GC):
        spkt = np.array( cell.spk_times)
        if len(spkt[spkt > (h.tstop - 25)]): # if spikes in the last 25 ms
            active_GC_idx.append(i)
            nactive_GC +=1

    spk_freq = list()
    for cell in GC:
        spk_freq.append( len(cell.spk_times)*(1000/h.tstop) )
        
    print('GC[0] has %d spikes'%len(GC[0].spk_times))
    print('Number of granule cells firing in last 25 ms = %d'%nactive_GC)
    print('Average spike frequency = %f'%np.mean(spk_freq))

    return( spk_freq) 
