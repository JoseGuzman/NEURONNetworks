"""
init.py

Creates a network of basquet cells (BC) and granule cells (GCs)

TODO: complete simulation description

To execute in Ipython
%run init.py

directly in a Python shell
$ nrngui init.py -python

To create connections type: 
>> recurrent_inh(BC)
>> inhibition2excitation(0.5)
>> excitation2inhibition(0.3)
>> myrun() " my cust

The simulation returns 
1) The total number of spikes in the GC network            
2) The number GC cells firing in the last 50 ms            

"""

import numpy as np
from neuron import h, gui
from Cell_builder import BCbuilder, GCbuilder

np.random.seed(10)

h.load_file('stdrun.hoc') # need for h.tstop
h.tstop = 300

#=========================================================================
# 1. create a network of 100 inhibitory neurons and 1000 principal neurons
#=========================================================================
icells = 100
BC = [BCbuilder( idx = i) for i in range(icells)] 

ecells = 100
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

stim_icells = [inject_random_current(cell.soma) for cell in BC] 
stim_ecells = [inject_random_current(cell.soma, 0.0005) for cell in GC] 

#=========================================================================
# 3. Custom connections between all cells  
#=========================================================================
def recurrent_inh(cell_list):
    """
    Connects cells in the list via inhibitory recurrent synapses 
    """
    mynetcon = list()
    for i in range( len(cell_list) ) :
        for j in np.delete( range(len(cell_list)), i): # avoid auptapse
            nc = cell_list[i].connect2target( target = cell_list[j].isyn )
            mynetcon.append( nc )

    #return (mynetcon)

def inhibition(pEI, pRI, pLI):
    """
    connect inhibitory and excitatory network neurons
    Arguments:
    pEI     -- prob of IE connections
    pRI     -- prob of Recurrent inhibition, P(IE | EI)
    pLI     -- prob of Lateral inhibition , P(IE| not EI )
    """

    # Prepare indices of cell subpopulations

    nexc_GCs = int(ecells*pEI)
    # 1. sub-set of GC sending excitation to BC
    exc_GC = np.random.randint(low=0, high=ecells, size=nexc_GCs)

    # 2. sub-set of GC lacking excitation to BC
    nonexc_GC = np.delete( np.arange(ecells), exc_GC )

    # from the sub-set of PV receiving excitation 
    nri_BCs = nexc_GCs*pRI
    
    
    
    for cell in exc_GC:
        GC[cell].connect2target( target = BC[cell].esyn )

    

    
def inhibition2excitation(prob):
    """
    Connects inhibitory to excitatory cells

    prob    -- probability of inhibitory to excitatory connection
    
    """
    # get indices of the GCs
    GCsize = int(ecells*prob)

    mynetcon = list()
    # connect all inhibitory neurons to selected GC
    for BC_cell in BC:
        idx = np.random.randint(low = 0, high = ecells, size= GCsize)
        for i in idx:
            nc = BC_cell.connect2target( target = GC[i].isyn)
            mynetcon.append( nc ) 

def excitation2inhibition(prob):
    """
    Connects excitatory to excitatory cells

    prob    -- probability of inhibitory to excitatory connection
    
    """
    # get indices of the GCs
    BCsize = int(ecells*prob)

    mynetcon = list()
    # connect all excitatory neurons to selected BC 
    for GC_cell in GC:
        idx = np.random.randint(low = 0, high = icells, size= BCsize)
        for i in idx:
            nc = GC_cell.connect2target( target = BC[i].esyn)
            mynetcon.append( nc ) 
#=========================================================================
# 4. Visualize
#=========================================================================

h.run() # run a first simulation without connections
h.load_file('gui/gSingleGraph.hoc')
h.load_file('gui/gRasterPlot.hoc')

#=========================================================================
# 5. My custom run 
#=========================================================================
def myrun():
    h.run()
    h.update_rasterplot()

    spk_count = list()
    cell_count = 0 
       
    idx_GC = list() 
    for i, cell in enumerate(GC):
        spk_times = cell.spk_times
        spk_count.append( len(spk_times) )

        x = np.array(spk_times)
        if len(x[x>(h.tstop - 50)]): # n_spikes in last 50 ms
            idx_GC.append(i)
            cell_count +=1
        
        
    net_spikes = np.sum(spk_count)
    
    print('GC[0] has %d spikes'%len(GC[0].spk_times))
    print('Total spikes in GC network = %d'%net_spikes)
    print('Number of cells firing in last 50 ms = %d'%cell_count)
    print('Active cells ->%s'%idx_GC)

if '__name__' == '__main__':
    pass
