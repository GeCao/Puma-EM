# from Numeric import *
import numpy as np
from scipy import weave
import weave
from scipy import linalg
from weave import converters
import time

def multiplyAndSum(A, B):
    code = """
           double result;
           result = sum(A*B) + sum(A+B);
           """
    weave.inline(code,
                ['A', 'B'],
                type_converters = converters.blitz,
                include_dirs = ['./MoM/'],
                library_dirs = ['./MoM/'],
                libraries = ['MoM'],
                headers = ['<iostream>','<complex>'],
                compiler = 'gcc')

if __name__=="__main__":
    N = 500
    A = np.random.rand(2, N, 2*N)
    B = np.random.rand(2, N, 2*N)
    t0 = time.time()
    multiplyAndSum(A, B)
    print(time.time() - t0, "seconds...")

    A = np.random.rand(2, 2*N*N)
    B = np.random.rand(2, 2*N*N)
    t0 = time.time()
    multiplyAndSum(A, B)
    print(time.time() - t0, "seconds...")

    A = np.random.rand(2*N*N, 2)
    B = np.random.rand(2*N*N, 2)
    t0 = time.time()
    multiplyAndSum(A, B)
    print(time.time() - t0, "seconds...")
