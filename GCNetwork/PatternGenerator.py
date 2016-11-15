"""
PatternGenerator.py

Claudia Espinoza, claumespinoza@gmail.com
Jose Guzman, sjm.guzman@gmail.com

A random pattern generator based on excitation of a network of 5000 cells
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# basic pattern with activity levesls 10%
# for 5000 cells this means that 500 cells are active
iPattern = np.concatenate( (np.ones(500), np.zeros(4500)) )

def generator(percentage = None):
    """
    degradates a patterns by a percentage %
    returns the correlation coefficient and the new degraded pattern
    """

    mysize = int ((500 * percentage)/100.)

    newPattern = iPattern.copy()
    idx = np.random.randint(low = 0, high = 500, size = mysize)
    newPattern[ idx ] = 0
    newPattern[ 500+idx ] = 1
    
    similarity = cosine_similarity(iPattern, newPattern)
    r = similarity[0][0]
    print( r )
    
    return( newPattern )
    
    
