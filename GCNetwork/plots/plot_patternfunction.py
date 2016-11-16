"""
plot_patternfunction.py

Claudia Espinoza, claum.espinoza@gmail.com
Jose Guzman, sjm.guzman@gmail.com

Plot the relationship between the correlation among input patterns
and output patterns of the network.
"""
import numpy as np
import matplotlib.pyplot as plt

idata = np.loadtxt('../data/rInputEI=1.txt')
odata = np.loadtxt('../data/rOutputEI=1.txt')

plt.scatter(idata, odata, color='black')
x = np.linspace(0,1,100)
plt.plot(x, x, lw=2, color='#aa0000') # linear relation

plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1 )
plt.xlabel('Input Pattern (r)')
plt.ylabel('Output Pattern (r)')
plt.grid()
plt.show()


