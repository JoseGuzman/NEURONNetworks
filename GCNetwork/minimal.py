"""
minimal.py 

a Minimal example of GC-PV connection
    
"""
import numpy as np

from neuron import h, gui
from Cell_builder import GCbuilder, BCbuilder

h.tstop = 100 
h.v_init = -80

GC = GCbuilder()
PV = BCbuilder()


#=========================================================================
# Conductance necessary to evoke one spike in GC
#=========================================================================
GC_netstim = h.NetStim()
GC_netstim.number = 1 
GC_netstim.start  = 55 

GC_ncstim = h.NetCon(GC_netstim, GC.esyn)
GC_ncstim.delay = 0 
GC_ncstim.weight[0] = 1.753e-5 # one AP only

#=========================================================================
# Inhibitory conductance necessary to inhibit one AP in GC: 1.235e-8
#=========================================================================
#PV_GCnetcon = PV.connect2target(target = GC.isyn, weight = 1.235e-8)
#PV_GCnetcon = PV.connect2target(target = GC.isyn, weight = 1.235e-6)

#=========================================================================
# Excitatory conductance necessary to evoke one AP in GC: 1.235e-8
#=========================================================================
GC_PVnetcon = GC.connect2target(target = PV.esyn, weight = 1.753e-6)

#=========================================================================
# Conductance necessary to evoke one spike in PV 
#=========================================================================
PV_netstim = h.NetStim()
PV_netstim.number = 1 
PV_netstim.start  = 50 

PV_ncstim = h.NetCon(PV_netstim, PV.esyn)
PV_ncstim.delay = 0 
#PV_ncstim.weight[0] = 1.622e-05 # one AP only
PV_ncstim.weight[0] = 3.5e-05 # one AP only with little latency

#=========================================================================
# Conductance  GC - PV
#=========================================================================
#GC_PVnc = GC.connect2target(target = PV.esyn, weight = 1.9e-5)
        
h.load_file("gminimal.hoc")

h.run()
"""
GC.IClamp.amp = 0.00205 #generates 1AP
GC.IClamp.dur = 2 
GC.IClamp.delay = 50
"""
