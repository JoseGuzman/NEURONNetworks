"""
plot_patterncompletion.py

Claudia Espinoza, claumespinoza@gmail.com
Jose Guzman, sjm.guzman@gmail.com

Last change: Mon Nov  7 20:40:01 CET 2016

Plots a color map with the correlation
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 1, 0.1)
y = np.arange(0, 1, 0.1)
X,Y = np.meshgrid(x,y)

data = np.loadtxt('similarity.txt')
data = data -1


mycmap = plt.cm.RdBu
mycmap = plt.cm.afmhot
im = plt.imshow(data, cmap=mycmap, extent=(0, 1, 0, 1))#, interpolation = 'linear')  
plt.colorbar(im)  
#plt.grid()

plt.xlabel('P(Lateral Inhibition)')
plt.ylabel('P(Recurrent Inhibition)')

plt.show()






