"""
minimal.py

Jose Guzman, sjm.guzman@gmail.com
Last change: Tue Oct 18 18:24:48 CEST 2016

A BC connected to a PC to evoke inward inhibitory currents
"""

from neuron import h, gui
from Cell_builder import PCbuilder, BCbuilder

h.tstop = 350 
h.v_init = -70

BC = BCbuilder()
PC = PCbuilder()
PC.soma.push()
BC.IClamp.amp =2
BC.IClamp.delay =25
BC.IClamp.dur = 2

# create a weaker synapse at the soma of the PC cell
mini = h.ExpSyn(PC.soma(0.5), sec=PC.soma)
mini.tau = 10.0 # in ms
mini.e = -75.0      # in mV

# inhibitory synapse from a BC cell to a PC cell
mysyn = BC.connect2target( target = PC.isyn, weight=1e-5)
mystim = h.NetStim()
mystim.number = 1
mystim.start = 25

mync = h.NetCon(mystim, mini, sec= PC.soma ) 
mync.weight[0] = 1.e-5 

# 3 stimuli to provoke APs in the presynaptic cell
myIClamp = [h.IClamp(0.5, sec = BC.soma) for _ in range(3)]
for clamp in myIClamp:
    clamp.amp  = 2
    clamp.dur  = 2

myIClamp[0].delay = 100 
myIClamp[1].delay = 120
myIClamp[2].delay = 150
h.load_file('minimal.ses')
