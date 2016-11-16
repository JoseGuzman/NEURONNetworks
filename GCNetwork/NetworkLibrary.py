"""
NetworkLibrary.py

Claudia Espinoza, claumespinoza@gmail.com
Jose Guzman, sjm.guzman@gmail.com

Last change: Thu Nov 10 14:07:47 CET 2016

A collection of heterologous networks
"""

import numpy as np

from neuron import h, gui
from Cell_builder import BCbuilder, GCbuilder

np.random.seed(10)

class GCNetwork(object):
    """
    Heterologous network consisting in excitatory and inhibitory 
    cells
    """
    def __init__(self, icells):
        """
        Creates an hetereologous network of <icells> inhibitory 
        fast-spiking neurons and 100 times more integrate-and-fire 
        excitatory neurons.

        Arguments:
        icells      (int) number of inhibitorGy neurons

        """
        scaling = 100
        self.icells = icells
        self.ecells = icells * scaling # check ModelDB: 124513

        self.PV = [BCbuilder( idx = i) for i in range(self.icells)]
        self.GC = [GCbuilder( idx = i) for i in range(self.ecells)]
        
        # synapses grouped in list of NetCon classes
        self.counter_inhibition = list() 
        self.recurrent_inhibition = list()
        self.lateral_inhibition = list()

        print('Creating %10d GC cells'%self.ecells)
        print('Creating %10d PV cells'%self.icells)

    def set_counter_inhibition_connections(self, weight = None):
        """
        Connects all inhibitory cells via inhibitory recurrent
        synapses (i.e. counter-inhibition).

        Arguments:
        weight      (float) synaptic weight in S*cm^-2
        """
        
        # clear list
        for nc in self.counter_inhibition:
            nc.weight[0] = 0.0
        
        if weight is None:
            myweight = 1e-6 # 0.001 mS*cm^-2
        else:
            myweight = weight

        for i in range( self.icells ):
            for j in np.delete( range( self.icells), i): # avoid autapses
                syn = self.PV[i].connect2target( self.PV[j].isyn, myweight )
                self.counter_inhibition.append( syn ) 

        nsyn = len( self.counter_inhibition )
        print('Adding %d recurrent inhibitory synapses'%nsyn)

    def inhibition(pEI, pRI, pLI, debug = None):
        """
        connect inhibitory and excitatory neurons in the
        network according to probabilities.
        Arguments:
        pEI     -- prob of IE connections
        pRI     -- prob of Recurrent inhibition, P(IE | EI)
        pLI     -- prob of Lateral inhibition , P(IE| not EI )

        #===============================
        # Subpopulation Indices 
        #===============================
        # ecells: total number of GC cells
        # icells: total number of PV cells
        # exc_GC_idx: GCs sending exc. to PV
        # exc_PV_idx: PV receiving exc. from GC
        # RI_PV_idx: exc_PV_idx sending inh. to exc_GC_idx
        # RI_GC_idx: exc GC_idx receiving inh. from RI_PV_idx
        # nonexc_GC_idx: GCs not sending exc. to PV
        # nonexc_PV_idx: PVs not receiving exc. from GC
        # LI_PV_idx: nonexc_PV_idx sending inh. to nonexc_GC_idx
        """
    
        #===================================================================
        # 1.- Subpopulation of cells involved in excitation 
        #===================================================================
        # 1A.- Interneuron subpopulation
        nexc_PV = int( round( self.icells * pEI ) )
        exc_PV_idx = np.random.choice( range(self.icells),  size = nexc_PV)

        # 1AA.-Subpopulation of RI interneurons (RI_PV)
        nRI_PV = int( round( nexc_PV * pRI ) )
        RI_PV_idx = np.random.choice(exc_PV_idx, size = nRI_PV)

        # 1B.- Granule cell sending excitation subpopulation
        nexc_GC = nexc_PV * scaling # scaling to have divisible integer number
        exc_GC_idx = np.random.choice( range(self.ecells), size = nexc_GC)

        # 1C.- Subpopulation of granule cell receiving inhibtion 
        nRI_GC = nRI_PV * scaling # to avoid floating points
        RI_GC_idx = np.random.choice(exc_GC_idx, size = nRI_GC)

        #===================================================================
        # 2.- Subpopulation of cells not involved in excitation
        #===================================================================
        # 2A.- Interneuron subpopulation
        nonexc_PV_idx = np.delete( range(self.icells), exc_PV_idx)

        # 2AA.-Subpopulation of LI interneurons
        nLI_PV = int( round(len(nonexc_PV_idx) * pLI ) )
        LI_PV_idx = np.random.choice(nonexc_PV_idx, size = nLI_PV)

        # 3AA. - Granule cell non-sending excitation subpopulation
        nonexc_GC_idx = np.delete( range(self.ecells), exc_GC_idx )
    
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
    
if __name__ == '__main__':
    myNetwork = GCNetwork(icells = 10)
        
        
