import sys
from math import exp as exp
import time
import numpy as np

def sigmoid(x):
  return 1.0/(1.0+exp(-x))

def calc_y_value(A, B, w1, w2, w3, w4, w5, w6):
    C = sigmoid(w1*A+w2*B+0.5)
    D = sigmoid(w3*A+w4*B+0.5)
    Y = round(sigmoid(w5*C+w6*D+0.5))
    return Y

def determine_gate(w1, w2, w3, w4, w5, w6):
    if calc_y_value(1, 1, w1, w2, w3, w4, w5, w6) == 1:
        return "AND"
    elif calc_y_value(1, 0, w1, w2, w3, w4, w5, w6) == 1:
        return "NAND"
    else:
        return "NOR"

def benchmark_plus_opp(A, B, n=20):
    assert n > 0, "Must have one or more passes."
    
    total_time = 0.0
    min_time = float("inf")
    max_time = float("-inf")
    
    for _ in range(n):
        tik = time.time()
        sumv = A+B
        tok = time.time()
        t = tok-tik
        total_time += t
        if(max_time < t): max_time = t
        if(min_time > t): min_time = t
    avg_time = total_time/n

    print("Benchmark A+B:")
    print("Average time: " + str(avg_time))
    print("Max time: " + str(max_time))
    print("Min time: " + str(min_time))

def benchmark_add_func(A, B, n=20):
    assert n > 0, "Must have one or more passes."
    
    total_time = 0.0
    min_time = float("inf")
    max_time = float("-inf")
    
    for _ in range(n):
        tik = time.time()
        sumv = np.add(A,B)
        tok = time.time()
        t = tok-tik
        total_time += t
        if(max_time < t): max_time = t
        if(min_time > t): min_time = t
    avg_time = total_time/n

    print("Benchmark np.add(A,B)")
    print("Average time: " + str(avg_time))
    print("Max time: " + str(max_time))
    print("Min time: " + str(min_time))

test00 = False
test01 = True
if __name__ == '__main__':
    if test00:
        print(determine_gate(2, -3, 1, -4, -4, 4))
        print(determine_gate(-1, -1, -3, -2, -2, -3))
        print(determine_gate(1, 1, -5, -2, -2, 4))
        print(determine_gate(-4, -4, -2, -2, -5, -3))

    if test01:
        A = np.arange(100000000.0).reshape((10000,10000))
        benchmark_add_func(A,A, 100)
        benchmark_plus_opp(A,A, 100)