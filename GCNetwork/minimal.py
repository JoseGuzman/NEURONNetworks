"""
minimal.py 
a Minimal example of GC cell excitation
    
"""
import numpy as np

from neuron import h, gui
from Cell_builder import GCbuilder

h.tstop = 1000
h.v_init = -80

GC = GCbuilder()

# Normal distribution of and average freq = 30 Hz
# in 1000 ms 30 APs, in 100
nAPs = int( h.tstop*(30/1000) ) 
myAPs = np.random.choice(range(h.tstop), size = nAPs)


# Presynaptic stimulation

mystim = h.NetStim()
mystim.number = 1 
mystim.start = 50


ncstim = h.NetCon(mystim, GC.esyn)
ncstim.delay = 0 
ncstim.weight[0] = 1.755e-5 # one AP only

h.load_file("minimal.ses")

h.run()
"""
GC.IClamp.amp = 0.00205 #generates 1AP
GC.IClamp.dur = 2 
GC.IClamp.delay = 50
"""
