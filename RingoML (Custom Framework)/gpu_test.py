# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 19:25:50 2021

@author: agama
"""

import numpy as np
import time

from numba import vectorize, cuda

@vectorize(['float64(float64, float64)'], target='cuda')
def VectorAdd(a, b):
    return a + b

def Add(a, b):
    return a + b

def main():

    A = np.random.randn(100000000)
    B = np.random.randn(100000000)

    start = time.time()
    Add(A, B)
    add_time = time.time() - start
    
    start = time.time()
    VectorAdd(A, B)
    vector_add_time = time.time() - start

    print ("Add took for % seconds" % add_time)
    print ("VectorAdd took for % seconds" % vector_add_time)

if __name__=='__main__':
    main()