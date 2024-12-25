import numpy as np
from numpy.polynomial.polynomial import Polynomial


def func(x):
    return np.abs(x)

def create_points(start_point, end_point, count_of_points, uniform_grid=True):

    if uniform_grid:
        points = np.linspace(start_point, end_point, count_of_points, dtype=np.float64)
    else:
        root = [0] * count_of_points
        root.append(1)
        points = np.polynomial.chebyshev.chebroots(root)

    return points

def mult(points, n):

    val = 1
    for i in range(n):
        val *= (points[n] - points[i])
    return val

def divide(dividers, points, i):
    for j in range(i):
        dividers[j] *= (points[j] - points[i])


def newton_interp(start_point, end_point, count_of_points, function, tolerance, uniform_grid=True):

    points = create_points(start_point, end_point, count_of_points, uniform_grid)

    f_val = np.array([function(x) for x in points])

    dividers = [1]

    res = Polynomial([f_val[0]])

    p = Polynomial([1.0])

    for i in range(1, n):

        p *= Polynomial.fromroots(points[i - 1])

        divide(dividers, points, i)

        dividers.append(mult(points, i))

        val = 0
        for j in range(i + 1):
            val += f_val[j] / dividers[j]

        res += p * val 

    corr = True

    for i in range(n):

        val = 0
        for j, x in enumerate(res):
            val += x * points[i]**j

        if np.abs(val - function(points[i])) >= tolerance:
            corr = False

    if corr:
        print("ALL GOOD")
        print(res)
    else:
        print("Can't achieve that tolerance")
        print(res)

    return res



deg = 30
a, b = -1, 1
eps = 10**(-5)
function = func

n = deg + 1

p_vals = newton_interp(a, b, n, function, eps, False)
