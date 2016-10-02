"""
init.py

Creates a network of basquet cells (PV) and granule cells (GCs)

TODO: complete simulation description

To execute in Ipython
%run init.py

directly in a Python shell
$ nrngui init.py -python

To create connections type: 
>> recurrent_inhibitory_connections(PV)
>> inhibition2excitation(0.5)
>> excitation2inhibition(0.3)
>> myrun() " my cust
>> inhibition(pEI = 0.09765, pRI = 0.24, pLI= 0.3283)

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
h.tstop = 150 

#=========================================================================
# 1. create a network of 100 inhibitory neurons and 1000 principal neurons
#=========================================================================
icells = 100
ecells = icells*100 # check ModelDB: 124513 
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
    rv = norm(loc = mean, scale = 10)
    
    start = rv.ppf(0.0001)
    end   = rv.ppf(0.9999)
    print(int(start), int(end))
    cell_idx = np.arange( int(start), int(end) )
    
    stim = list()
    for idx in cell_idx:
        st = h.IClamp(0.5, sec = cell_list[idx].soma)
        st.amp = rv.pdf(idx)/100 # mean will receive 0.0004
        st.dur   = h.tstop
        st.delay = 0.0
        stim.append( st )
    
    return (stim)

stim_icells = [inject_random_current(cell.soma) for cell in PV] 
#stim_ecells = [inject_random_current(cell.soma, 0.00034) for cell in GC] 
#stim_ecells = [inject_random_current(cell.soma, 0.00044) for cell in GC] 
stim_ecells = inject_excitatory_current(cell_list = GC, mean = 0) 

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
    # 1. exc_idx: sub-set of GC sending excitation to PV 
    # 2. nonexc_idx: sub-set of cells lacking excitation to PV 
    # 3. RI_PV: sub-set of PV receiving excitation 
    # 4. LI_PV: sub-set of PV NOT receiving excitation 
    """
    
    # select the first indices for cells that send excitation to PV
    nexc = int( round(ecells * pEI ) )
    exc_GC_idx = np.arange(ecells)[:nexc]

    # the rest are cells that does NOT send excitation to PV
    nonexc_GC_idx = np.arange(ecells)[nexc:]

    # from PV cells receiving excitation, send recurrent inhibition
    nRI_PV = int( round(nexc * pRI ) )
    RI_PV_idx = np.random.choice(exc_GC_idx, size = nRI_PV)

    # GC cells that doesn't have recurrent inhibition
    LI_GC_idx = np.delete( range(ecells), RI_PV_idx ) 

    # from all PV without recurrent inhibition 
    nonRI_PV_idx = np.delete( range(icells), RI_PV_idx ) 
    nLI_PV = int( len(nonRI_PV_idx) * pLI )
    # connect to GC cells without recurrent inhibition
    LI_PV_idx = np.random.choice( nonRI_PV_idx, size = nLI_PV)

    netcon = list()
    # Recurrent inhibition requires idx of PV and GC cells to be the same
    for idx in exc_GC_idx:
        netcon.append( GC[idx].connect2target( target = PV[idx].esyn ) )

    for idx in RI_PV_idx:
        netcon.append( PV[idx].connect2target( target = GC[idx].isyn ) )

    # Lateral inhibition 
    for idx in LI_PV_idx:
        for edx in LI_GC_idx:
            netcon.append( PV[idx].connect2target( GC[edx].isyn ) )

    if debug:
        print( "exc_GC_idx    = %2d"%len(exc_GC_idx) )
        print( "nonexc_GC_idx = %2d"%len(nonexc_GC_idx ))
        print( "RI_PV_idx     = %2d"%len(RI_PV_idx) )
        print( "LI_PV_idx     = %2d"%len(LI_PV_idx))
        print( "LI_GC_idx     = %2d"%len(LI_PV_idx))
        print( "netcons       = %2d"%len(netcon) )
    
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
