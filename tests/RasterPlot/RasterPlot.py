"""
RasterPlot.py

Jose Guzman, sjm.guzman@gmail.com 
Last change: Fri Sep 23 20:26:04 CEST 2016

A test with a single compartment to see the Raster plot in hoc

To execute it with IPython:
 %run RasterPlot.py

"""

import numpy as np
import matplotlib.pyplot as plt

from neuron import h
from neuron import gui # allows GUI manipulation

# Global simulation Variables
h.load_file('stdrun.hoc')
h.tstop = 500

#=========================================================================
# Single compartment cell
#=========================================================================
class SimpleCell(object):
    """
    Minimal cell example consisting of a single compartment
    with active HH conductances
    """
    def __init__(self):
        self.soma = h.Section(name = 'cell', cell = self)
        self.soma.L = self.soma.diam = 10

        Rm = 150e3 # 150 MOhms*cm^2
        self.soma.insert('hh')
        self.soma.insert('pas')

        self.soma.gnabar_hh = 0.25 
        self.soma.gl_hh = 1/Rm 
        self.soma.el_hh = h.v_init

        # hoc Vectors to record time and voltage
        self._time, self._voltage = h.Vector(), h.Vector()
        self._time.record( h._ref_t )
        self._voltage.record( self.soma(0.5)._ref_v )

        # NetCon to monitor membrane potential at the soma (i.e APs)
        self._nc = h.NetCon( self.soma(0.5)._ref_v, None, sec = self.soma )
        self._nc.threshold = 0.0

        self._spk_times = h.Vector()
        self._nc.record(self._spk_times) # spike times

    # getters
    voltage    = property(lambda self: np.array(self._voltage))
    time       = property(lambda self: np.array(self._time))
    spk_times  = property(lambda self: np.array(self._spk_times))

#=========================================================================
# Define a random synaptic current
#=========================================================================
def random_injection(nrnSection):
    """
    creates random firing on  
    """
    syn = h.ExpSyn(0.5, sec = nrnSection)
    syn.tau = 2.0 

    stim = h.NetStim()
    stim.number = 10
    stim.interval = 50
    stim.start = 50 
    stim.noise = 1

    ncstim = h.NetCon(stim, syn, sec = nrnSection)
    ncstim.weight[0] = 0.03 # enought to fire APs
    
    return(stim, syn, ncstim)


#=========================================================================
# Create a network on 100 cells
#=========================================================================

ncells = 100
cell_list = list()
x = list() # collect the mechanims, otherwise there're lost!
for _ in range(ncells):
    cell = SimpleCell()
    x.append( random_injection(nrnSection = cell.soma) )
    cell_list.append( cell )
    

# h.run()
# Plot first cell
# plt.plot(cell_list[0].time, cell_list[0].voltage)
# plt.xlabel('Time (ms)'), plt.ylabel('Potential (mV)')
# plt.show()

# return spike times
# print(cell_list[0].spk_times)

h.load_file('gRasterPlot.hoc')
