# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 04:57:26 2021

@author: kulka
"""
# Trying to run examples from https://stackabuse.com/tensorflow-neural-network-tutorial/
import numpy as np
from numpy.random import default_rng


def dp(p, x: np.array, y: np.array) -> float:
    k = x.size
    t1 = sum([x[i] - y[i] for i in range(k) if x[i] >= y[i]]) 
    t2 = sum([y[i] - x[i] for i in range(k) if x[i] < y[i]]) 
    pans = (t1 ** p + t2 ** p) ** (1/p)
    
    denom = sum([max(abs(x[i]), abs(y[i]), abs(x[i] - y[i])) for i in range(k)])
    return pans / denom

def generate_dp(cases, k, p):
    rng = default_rng()
    for i in range(cases):
        x = rng.uniform(-100, 100, k)
        y = rng.uniform(-100, 100, k)
        z = rng.uniform(-100, 100, k)
        ans = dp(p, x, y) + dp(p, x, z) - dp(p, x, y + z)
        if ans > 1:
            print(f'FAIL: {ans}, {x}, {y}, {z}')
            break
        # print()

def ker_sum(c, x, p):
    n = x.size
    return sum([c[i]*c[j]*(1 - dp(p, x[i], x[j])) for i in range(n) for j in range(n)])

def generate_ker_sum(cases, n, p, dim):
    rng = default_rng()
    for i in range(cases):
        x = np.array([rng.uniform(-10000, 0, dim) for j in range(n)])
        c = np.array(rng.uniform(-10000, 0, n))
        if ker_sum(c, x, p) < 0:
            print("FAIL")
            break
# x is sorted, 0 <= x <= 1
def construct_matrix(x):
    n = x.size
    return np.array([[1 - abs(x[i] - x[j]) for j in range(n)] for i in range(n)])

def compute_plm(mat):
    n = mat.shape[0]
    for i in range(2**n):
        comb = bin(i)[:1:-1]
        p = len(comb)
        ind = [k for k in range(p) if comb[k] == '1']
        d = np.array([mat[k][ind] for k in range(n)])
        d = d[ind]
        if np.linalg.det(d) < 0:
            print("FAIL")
            break
def run_tests(cases, dim):
    rng = default_rng()
    for i in range(cases):
        x = rng.uniform(0, 1, dim)
        compute_plm(construct_matrix(x))

    
run_tests(10, 10)
        
            
        
        
generate_ker_sum(1000, 3, 2, 1)
    

