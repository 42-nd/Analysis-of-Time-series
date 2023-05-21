import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mat(t):
    matrix = np.zeros(4)
    for i in t:
        list_m=np.array([])
        list_m=np.append(list_m,3*i**(-2))
        list_m=np.append(list_m,1 * i ** (-1))
        list_m=np.append(list_m,1/3 * i )
        list_m=np.append(list_m,1/20 * i ** (3))
        matrix = np.vstack((matrix, list_m))
    return matrix[1::]

tau=[1,3,120,2880]
A=mat(tau)
B = np.array([1.6347798317034538e-20,1.5037615938121702e-21,8.965906083792707e-25,1.5933779302695733e-27])
x = np.linalg.lstsq(A,B)[0]
print(x)