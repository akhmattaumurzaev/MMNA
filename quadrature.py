import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.integrate import quad
import scipy.special

def func(x):
    return np.sin(x)


def variable_change(start_point, end_point, point):
    return ((start_point + end_point) + (end_point - start_point) * point) / 2


def test(val, function, start_point, end_point, tolerance):
    result = quad(function, start_point, end_point)

    diff = np.abs(result[0] - val)

    if diff < tolerance:
        print("ALL GOOD") 
    else:
        print("Can't achieve that tolerance")

    print(f'dif={diff}')
    print(f'result={result[0]}')
    print(f'val={val}')


def quadrature_weights(points):
    n = points.shape[0]
    
    weights = np.zeros(n)

    for i in range(n):
        roots = points.copy()
        roots = np.delete(roots, i)

        w = Polynomial.fromroots(roots)
        p = (Polynomial.fromroots(roots)).integ()

        weights[i] = (p(1) - p(-1)) / w(points[i])

    return weights

def quadrature_formula(function, points, a = -1, b = 1):
    weights = quadrature_weights(points)

    func_vals = np.array([function(variable_change(a, b, point)) for point in points])

    computed_value = 0
    for i in range(points.shape[0]):
        computed_value += func_vals[i] * weights[i]
    computed_value *= (b - a) / 2

    return computed_value


def newton_cotes(a, b, points_count, function, tolerance):
    
    points = np.linspace(-1, 1, points_count, dtype=np.float64)

    computed_value = quadrature_formula(function, points, a, b)

    test(computed_value, function, a, b, tolerance)


def gauss(a, b, points_count, function, tolerance):
    
    points = scipy.special.p_roots(points_count)[0]

    computed_value = quadrature_formula(function, points, a, b)

    test(computed_value, function, a, b, tolerance)


def clenshaw_curtis(a, b, points_count, function, tolerance):

    maxim = points_count // 2
    znam = [1 - 4*k**2 for k in range(maxim)]
    computed_value = 0

    for j in range(points_count + 1):
        w = 1 / 2
        mult = 2 * j * np.pi / points_count
        ch = 0
        for k in range(1, maxim):
            ch += mult
            w += np.cos(ch) / znam[k]
        
        w *= 4 / points_count

        if not j or j == points_count:
            w /= 2
        
        computed_value += w * function(variable_change(a, b, np.cos(mult / 2)))

    computed_value *= (b - a) / 2

    test(computed_value, function, a, b, tolerance)


count_of_points = 30
a, b = -10, 10
eps = 10**(-5)
method = "clenshaw_curtis"

if method == "newton_cotes":
    newton_cotes(a, b, count_of_points, func, eps)
elif method == "gauss":
    gauss(a, b, count_of_points, func, eps)
elif method == "clenshaw_curtis":
    clenshaw_curtis(a, b, count_of_points, func, eps)
