"""
patternseparation_patterns.py

Claudia Espinoza, claumespinoza@gmail.com
Jose Guzman, sjm.guzman@gmail.com

Last change: Tue Nov  8 12:22:19 CET 2016

Simulate random patterns and get the patterns from the network

"""
import numpy as np

from neuron import gui, h
from Cell_builder import BCbuilder, GCbuilder

from PatternGenerator import generator

from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(10)

h.load_file('stdrun.hoc') # need for h.tstop
h.tstop = 125 

#=========================================================================
# 1. Create a network of 50 inhibitory neurons and 5,000 granule cells 
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
    #stim.delay = np.abs( np.random.normal( loc = 5, scale = 4 ) )
    stim.delay = 0
    #stim.dur   = h.tstop - stim.delay
    stim.dur   = h.tstop

#=========================================================================
# 3. Stimulate randomly 500 cells of 5,000 granule cells (10% of ecells)
#=========================================================================
GC_netstim = h.NetStim()
GC_netstim.number = 1
GC_netstim.start = 10

GC_netstim2 = h.NetStim()
GC_netstim2.number = 1
GC_netstim2.start = 92 

iPattern_init = np.concatenate( (np.ones(500), np.zeros(4500)) )

def stimulate_GC(iPattern, myNetConList = None):
    """
    create APs at fixed time in the GC network
    Usage:
    >>> nc = stimulate_GC(pattern)
    >>> nc = stimulate_GC(new_pattern, nc)
    """

    if myNetConList is not None:
        for nc in myNetConList:
            nc.weight[0] = 0.0

    myNetConlist = list()
    for idx in np.nonzero(iPattern)[0]:
        GC_ncstim = h.NetCon(GC_netstim, GC[idx].esyn, sec=GC[idx].soma)
        GC_ncstim.delay = 0
        GC_ncstim.weight[0] = 2.1e-5 # one AP
        myNetConlist.append(GC_ncstim)

        GC_ncstim2 = h.NetCon(GC_netstim2, GC[idx].esyn, sec=GC[idx].soma)
        GC_ncstim2.delay = 0
        GC_ncstim2.weight[0] = 2.1e-5 # one AP
        myNetConlist.append(GC_ncstim2)
    
    return( myNetConlist)
    
#=========================================================================
# 4. Network arquitecture 
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
    
    mydict = dict() # dictionary with the cell indices
    #===================================================================
    # 1.- Subpopulation of cells involved in excitation 
    #===================================================================
    # 1A.- Interneuron subpopulation
    nexc_PV = int( round( icells * pEI ) )
    exc_PV_idx = np.random.choice( range(icells),  size = nexc_PV)
    mydict['exc_PV_idx'] = exc_PV_idx

    # 1AA.-Subpopulation of RI interneurons
    nRI_PV = int( round(nexc_PV * pRI ) )
    RI_PV_idx = np.random.choice(exc_PV_idx, size = nRI_PV)
    mydict['RI_PV_idx'] = RI_PV_idx

    # 1B.- Granule cell sending excitation subpopulation
    nexc_GC = nexc_PV * scaling # scaling to have divisible integer number
    exc_GC_idx = np.random.choice( range(ecells), size = nexc_GC)
    mydict['exc_GC_idx'] = exc_GC_idx

    # 1C.- Subpopulation of Granule cell receiving inhibtion 
    nRI_GC = nRI_PV * scaling # to avoid floating points
    RI_GC_idx = np.random.choice(exc_GC_idx, size = nRI_GC)
    mydict['RI_GC_idx'] = RI_GC_idx

    #===================================================================
    # 2.- Subpopulation of cells not involved in excitation
    #===================================================================
    # 2A.- Interneuron subpopulation
    nonexc_PV_idx = np.delete( range(icells), exc_PV_idx)
    mydict['nonexc_PV_idx'] = nonexc_PV_idx

    # 2AA.-Subpopulation of LI interneurons
    nLI_PV = int( round(len(nonexc_PV_idx) * pLI ) )
    LI_PV_idx = np.random.choice(nonexc_PV_idx, size = nLI_PV)
    mydict['LI_PV_idx'] = LI_PV_idx

    # 3AA. - Granule cell non-sending excitation subpopulation
    nonexc_GC_idx = np.delete( range(ecells), exc_GC_idx )
    mydict['nonexc_GC_idx'] = nonexc_GC_idx
    
    # 3AB. - Granule cells receiving lateral inhibition
    nLI_GC = nLI_PV * scaling
    LI_GC_idx = np.random.choice(nonexc_GC_idx, size = nLI_GC)
    mydict['LI_GC_idx'] = LI_GC_idx

    netcon = list()
    # GC cells converging on to PV cells
    # divide GC into clusters to be projecting to one PV
    nClusters = nexc_GC/scaling
    for pre in range(nClusters):
        start = pre * scaling 
        end = start + scaling
        for post in exc_GC_idx[ start : end ]:
            netcon.append (GC[post].connect2target( target = PV[pre].esyn ))

    # connect RI PV cells back to GCs
    for pre in range(nRI_PV):
        start = pre * scaling
        end   = start + scaling 
        for post in RI_GC_idx[ start: end ]:
            netcon.append( PV[pre].connect2target( target = GC[post].isyn, weight=1.235e-4 ))

    # connect LI PV cells to nonexc_GCs
    for pre in range(nLI_PV):
        start = pre * scaling
        end = start + scaling
        for post in LI_GC_idx[ start : end ]:
            netcon.append( PV[pre].connect2target( target = GC[post].isyn, weight=1.235e-4))
        
    if debug:
        print( "exc_GC    = %2d"%nexc_GC )
        print( "nonexc_GC = %2d"%len(nonexc_GC_idx ))
        print( "RI_GC   = %2d"%nRI_GC )
        print( "RI_PV = %2d"%nRI_PV )
        print( "LI_GC   = %2d"%nLI_GC )
        print( "LI_PV = %2d"%nLI_PV )
        print( "netcons       = %2d"%len(netcon) )

    #return (mydict)
    return (netcon)

#mydict = inhibition(pEI = 0.09765, pRI = 0.24, pLI = 0.3283, debug=1)

#=========================================================================
# 4. Visualize
#=========================================================================

h.load_file('gui/gSingleGraph.hoc')
h.load_file('gui/gRasterPlot.hoc')

#=========================================================================
# 6. My custom run 
#=========================================================================
recurrent_inhibitory_connections(PV)
# tonic excitation
for cell in PV:
    inject_tonic_excitation(cell)
for cell in GC:
    inject_tonic_excitation(cell, I_mu = 0.0001) # ~1uA/cm^2

mynetcon = inhibition(pEI = 0.09765, pRI = 0.24, pLI = 0.3283, debug=1)
#mynetcon = inhibition(pEI = 1.0, pRI = 0.24, pLI = 0.3283, debug=1)

def myrun():
    """
    Executes a simulation by introducing different patterns 

    Example:
    """
    
    h.v_init = -80

    mync = stimulate_GC( iPattern = iPattern_init )
    h.update_rasterplot() # will call run in gui/gRasterPlot  
    oPattern_init = [1 if cell.spk_times.size>1 else 0 for cell in GC]

    # generate degraded patterns
    rInput, rOutput = list(), list()
    for i in range(1,100,10):
        iPattern = generator( i )
        mync = stimulate_GC( iPattern = iPattern, myNetConList = mync)
        h.update_rasterplot() # will call run in gui/gRasterPlot  
        oPattern = [1 if cell.spk_times.size>1 else 0 for cell in GC]
        rinput  =  cosine_similarity(iPattern_init, iPattern)[0][0]
        routput =  cosine_similarity(oPattern_init, oPattern)[0][0]

        print( rinput, routput ) 
        rInput.append( rinput )
        rOutput.append( routput )
    
    np.savetxt('./data/rInputnew.txt', rInput, fmt = '%f')
    np.savetxt('./data/rOutputnew.txt', rOutput, fmt = '%f')


