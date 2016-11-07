"""
patterncompletion.py

Claudia Espinoza, claumespinoza@gmail.com
Jose Guzman, sjm.guzman@gmail.com

Last change:Mon Nov  7 11:43:52 CET 2016

"""
import numpy as np

from neuron import gui, h
from Cell_builder import BCbuilder, GCbuilder

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
# 2. Stimulate randomly  5,000 granule cells 
#=========================================================================



#=========================================================================
# 2. Network arquitecture 
# by default inhibition(pEI = 0.09765, pRI = 0.24, pLI = 0.3283, debug=1)
#=========================================================================
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

inhibition(pEI = 0.09765, pRI = 0.24, pLI = 0.3283, debug=1)
