"""
Cell_builder.py

Authors: 
Claudia Espinoza < claumespinoza@gmail.com >
Jose Guzman < sjm.guzman@gmail.com >

Last change: Mon Sep 19 16:36:14 CEST 2016

File that contains detailed morphologies of granule cells (GC)
and basquet cells (BC) in the hippocampus

To use it:

>>> from Cell_builder import GC
>>> myGC = GC()
"""
from neuron import h 

class GC(object):
    """
    Load a hoc file containing the template of a detailed morphology
    from a granule cell imported from a neurolucida reconstruction.
    """
    def __init__(self):

        h.load_file( 'morphologies/GCTopology_template.hoc' )
        mycell = h.GCTopology() # must contain GCTopology template!
        self.soma = mycell.soma
        self.dend = mycell.dend

        self.allsec = h.SectionList()
        self.allsec.wholetree( sec = self.soma )

        for sec in self.allsec:
            sec.insert('pas')
            sec.cm = 1 # in microF/cm**2
            sec.Ra = 194 # in Ohms*c
            sec.g_pas = 1/164e3 # in Ohms*cm**2
    
   
if __name__ == '__main__':
    pass
