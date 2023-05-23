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
