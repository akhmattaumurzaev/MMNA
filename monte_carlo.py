import numpy as np
from scipy.integrate import quad
from scipy.stats import qmc

def func(x):
    # return 5 * x + 7
    return np.exp(x)

def test(a, b, function, tolerance, val):
    err = np.abs(quad(function, start_point, end_point)[0] - val)
    if err <= tolerance:
        print("ALL GOOD")
        print(f"Error = {err}")
        print(f"Integral = {val}")
    else:
        print("Can't achieve that tolerance")
        print(f"Error = {err}")
        print(f"Integral = {val}")

def monte_carlo(a, b, tolerance, function, sigma):

    length = b - a
    points_count = int(np.ceil( (sigma * length / tolerance)**2 / 12) )
    I = 0

    points = a + np.random.rand(points_count) * length
    for i in range(points_count):
        I += function(points[i])

    I *= length / points_count

    test(a, b, function, tolerance, I)


def geomic_monte_carlo(a, b, tolerance, function, max_val, grid, sigma):

    length = end_point - start_point
    volume = length * max_val

    points_count = int(np.ceil( (sigma * length / tolerance)**2 / 12) )

    if grid == "Uniform":
        points_x = a + np.random.rand(points_count) * length
        points_y = np.random.rand(points_count) * max_val
    elif grid == "Sobol":
        sampler = qmc.Sobol(d=2, scramble=False)
        points = sampler.random(points_count)
        points_x = [x[0] * length + a for x in points]
        points_y = [x[1] * max_val for x in points]
    else:
        sampler = qmc.Sobol(d=2, scramble=True)
        points = sampler.random(points_count)
        points_x = [x[0] * length + a for x in points]
        points_y = [x[1] * max_val for x in points]

    I = 0

    for i in range(points_count):
        val += (function(points_x[i]) >= points_y[i])

    I *= volume / points_count

    test(a, b, function, tolerance, val)

start_point = 0.1
end_point = 2
tolerance = 5 * 10**(-2)
max_val = np.exp(3) 
sigma = 3
method = "geomic_monte_carlo"
grid = "Sobol"

if method == "monte_carlo":
    monte_carlo(start_point, end_point, tolerance, func, sigma)
elif method == "geomic_monte_carlo":
    geomic_monte_carlo(start_point, end_point, tolerance, func, max_val, grid, sigma)
