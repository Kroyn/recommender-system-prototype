# DESCRIPTION:
# Приклад реалізації у бібліотеках Python
from scipy.optimize import minimize
import numpy as np

def objective_function(x):
    return x[0]**2 + x[1]**2 + x[2]**2

x0 = np.array([1.0, 2.0, 3.0])

result = minimize(objective_function, x0, method='CG') # CG = Conjugate Gradient
print(result.x)
