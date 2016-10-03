"""
init.py

Creates a network of basquet cells (PV) and granule cells (GCs)

TODO: complete simulation description

To execute in Ipython
%run init.py

directly in a Python shell
$ nrngui init.py -python

To create connections type: 
>>> inhibition(pEI = 0.09765/100, pRI=0.24, pLI=0.3283, debug=1)

The simulation returns 
1) The total number of spikes in the GC network            
2) The number GC cells firing in the last 25 ms            

"""

import numpy as np
from scipy.stats import norm

from neuron import h, gui
from Cell_builder import BCbuilder, GCbuilder
from custom_plots import GC_plot

np.random.seed(10)

h.load_file('stdrun.hoc') # need for h.tstop
h.tstop = 50 

#=========================================================================
# 1. create a network of 100 inhibitory neurons and 10,000 granule cells 
#=========================================================================
icells = 100
scaling = 100
ecells = icells * scaling # check ModelDB: 124513 
#ecells = 100

PV = [BCbuilder( idx = i) for i in range(icells)] 
GC = [GCbuilder( idx = i) for i in range(ecells)] 

#=========================================================================
# 2. Apply tonic excitatory drive 
#=========================================================================

def inject_random_current(nrnSection, I_mu = 0.001):
    """
    Injects a current of random amplitude and duration to
    the section given in nrnSection
    I_mu    -- mean excitatory drive , 0.001 corresponding to 1 uA/cm^2
    """
    I_sigma = 0.00003 # corresponds to 3% heterogeneity

    stim = h.IClamp(0.5, sec = nrnSection )
    stim.amp   = np.random.normal( loc = I_mu, scale = I_sigma*I_sigma)
    stim.delay = np.abs( np.random.normal( loc = 5, scale = 4 ) )
    stim.dur   = h.tstop - stim.delay

    return (stim)

def inject_excitatory_current(cell_list, mean):
    """
    Injects current to the given section
    """
    # clean all previous stimulus
    # at cell with idx = mean will receive 0.0004 current
    # TODO: implement gaussian function, not normal dist!
    rv = norm(loc = mean, scale = 20)
    
    start = rv.ppf(0.0001)
    end   = rv.ppf(0.9999)
    cell_idx = np.arange( int(start), int(end) )
    
    stim_list = list()
    for idx in cell_idx:
        st = h.IClamp(0.5, sec = cell_list[idx].soma)
        st.amp = rv.pdf(idx)/20 # mean will ????receive 4
        st.dur   = h.tstop
        st.delay = 0.0
        stim_list.append( st )
    
    return (stim_list)

stim_icells = [inject_random_current(icell.soma) for icell in PV] 
#stim_ecells = [inject_random_current(cell.soma, 0.00034) for cell in GC] 
#stim_ecells = [inject_random_current(cell.soma, 0.00044) for cell in GC] 
stim_ecells = inject_excitatory_current(cell_list = GC, mean = 000) 

#=========================================================================
# 3. Custom connections between all cells  
#=========================================================================
def recurrent_inhibitory_connections(cell_list):
    """
    Connects all the cells in the list via inhibitory recurrent synapses 

    Arguments:
    cell_list   -- a list with cell objects
    """
    mynetcon = list()
    for i in range( len(cell_list) ) :
        for j in np.delete( range(len(cell_list)), i): # avoid auptapse 
            nc = cell_list[i].connect2target( target = cell_list[j].isyn )
            mynetcon.append( nc )
    print("Adding recurrent inhibition")
    #return (mynetcon)

def inhibition(pEI, pRI, pLI, debug=None):
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
    """
        print( "LI_GC_idx     = %2d"%len(LI_GC_idx))
        print( "LI_PV_idx     = %2d"%len(LI_PV_idx))
    """
    
def inhibition2excitation(prob):
    """
    Connects inhibitory to excitatory cells

    prob    -- probability of inhibitory to excitatory connection
    
    """
    # get indices of the GCs
    GCsize = int(ecells*prob)

    mynetcon = list()
    # connect all inhibitory neurons to selected GC
    for PV_cell in PV:
        idx = np.random.randint(low = 0, high = ecells, size= GCsize)
        for i in idx:
            nc = PV_cell.connect2target( target = GC[i].isyn)
            mynetcon.append( nc ) 

def excitation2inhibition(prob):
    """
    Connects excitatory to excitatory cells

    prob    -- probability of inhibitory to excitatory connection
    
    """
    # get indices of the GCs
    PVsize = int(ecells*prob)

    mynetcon = list()
    # connect all excitatory neurons to selected PV 
    for GC_cell in GC:
        idx = np.random.randint(low = 0, high = icells, size= PVsize)
        for i in idx:
            nc = GC_cell.connect2target( target = PV[i].esyn)
            mynetcon.append( nc ) 
#=========================================================================
# 4. Visualize
#=========================================================================

h.load_file('gui/gSingleGraph.hoc')
h.load_file('gui/gRasterPlot.hoc')

recurrent_inhibitory_connections(PV)
#=========================================================================
# 5. My custom run 
#=========================================================================
def myrun(show_plot=False):
    h.run()
    h.update_rasterplot()

    spk_count = list()
    cell_count = 0 
       
    idx_GC = list() 
    for i, cell in enumerate(GC):
        spk_times = cell.spk_times
        spk_count.append( len(spk_times) )

        x = np.array(spk_times)
        if len(x[x>(h.tstop - 25)]): # n_spikes in last 25 ms
            idx_GC.append(i)
            cell_count +=1
        
        
    net_spikes = np.sum(spk_count)
    
    print('GC[0] has %d spikes'%len(GC[0].spk_times))
    print('Total spikes in GC network = %d'%net_spikes)
    print('Number of cells firing in last 25 ms = %d'%cell_count)
    if show_plot:
        myplot = GC_plot(ncells =1024,  active = idx_GC)
        myplot.show()
        
    #print('Active cells ->%s'%idx_GC)
