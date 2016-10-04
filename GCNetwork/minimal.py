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
nAPs = int( h.tstop*(0.03) )  # (htstop * 30)/1000 for 30 Hz
AP_times = np.random.choice(range(int(h.tstop)), size = nAPs)
print(AP_times)


# Presynaptic stimulation
def generate_spk(freq, cellobj):
    """
    generate APs in cells at a frequency given in argument

    Arguments:
    freq    -- frequency of APs (in Hz) 
    cellobj -- a cell object
    """
    
    nAPs = int( h.tstop*(freq/1000.) )  # eg.(htstop * 30)/1000 for 30 Hz
    AP_times = np.random.choice(range( int(h.tstop) ), size = nAPs)

    mystim = list()
    for time in AP_times:
    
        netstim = h.NetStim()
        netstim.number = 1 
        netstim.start  = time

        mystim.append(netstim)

    mynetcon = list()
    for st in mystim:
        ncstim = h.NetCon(st, cellobj.esyn)
        ncstim.delay = 0 
        ncstim.weight[0] = 1.755e-5 # one AP only
        
        mynetcon.append( ncstim)
    
    return(mystim, mynetcon)

h.load_file("minimal.ses")

h.run()
"""
GC.IClamp.amp = 0.00205 #generates 1AP
GC.IClamp.dur = 2 
GC.IClamp.delay = 50
"""
