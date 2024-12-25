import numpy as np
from numpy.polynomial.polynomial import Polynomial


def F(x):
    return np.exp(x)

def maximize_scan(p, a, b, function, num_points=1000):
    x = np.linspace(a, b, num_points)
    y = np.array([np.abs(function(val) - p(val)) for val in x])
    max_index = np.argmax(y)
    return x[max_index]

def remez(start_point, end_point, polinom_degree, tolerance, function):
    points = np.sort(np.random.rand(deg + 2) * (b - a) + a)
    vander = np.column_stack((np.vander(points, deg + 1), np.array([(-1)**i for i in range(deg + 2)])))
    F_val = np.array([function(x) for x in points])

    err = eps * 100

    while err > eps:

        res = np.linalg.solve(vander, F_val)

        p = Polynomial(res[-2::-1])
        d = res[-1]

        new_point = maximize_scan(p, a, b, function)

        func_res = function(new_point) - p(new_point)

        ind = np.argwhere(points > new_point)

        if ind.shape[0]:
            ind = ind.min()
            if vander[ind, -1] * d * func_res < 0:
                if ind == 0:
                    vander[1:, :] = vander[:-1, :]
                    F_val[1:] = F_val[:-1]
                    vander[ind, -1] = -vander[ind, -1]
                    points[1:] = points[:-1]
                else:
                    ind -= 1
        else:
            ind = deg + 1
            if vander[ind, -1] * d * func_res < 0:
                vander[:-1, :] = vander[1:, :]
                F_val[:-1] = F_val[1:]
                vander[ind, -1] = -vander[ind, -1]
                points[:-1] = points[1:]

        vander[ind, :-1] = np.array([new_point**i for i in range(deg, -1, -1)])
        F_val[ind] = function(new_point)
        points[ind] = new_point

        err = np.abs(np.abs(func_res) - np.abs(d))
    
    return p, np.abs(func_res)

eps = 10**(-5)
deg = 10
a, b = -1, 1

p_vals , max_dev = remez(start_point=a, end_point=b, polinom_degree=deg, tolerance=eps, function=F)
print(f'maximum deviation = {max_dev}')
print(f'coef of polinom = {p_vals}')
