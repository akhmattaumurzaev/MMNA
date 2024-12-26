import numpy as np
from numpy.polynomial.polynomial import Polynomial


def func(x, step):
    if not step % 4:
        return np.sin(x)
    elif step % 4 == 1:
        return np.cos(x)
    elif step % 4 == 2:
        return -np.sin(x)
    else:
        return -np.cos(x)


def create_points(start_point, end_point, count_of_points, uniform_grid=True):

    n = int((count_of_points + 1) / 2)

    if uniform_grid:
        points = np.linspace(start_point, end_point, n, dtype=np.float64)
    else:
        root = []
        for _ in range(n):
            root.append(0)
        root.append(1)
        points = np.polynomial.chebyshev.chebroots(root)
        points[0] = start_point
        points[-1] = end_point

    points_left = count_of_points - n

    randomized = np.random.choice(n, points_left)

    points = np.sort(np.append(points, points[randomized]))

    return points


def hermite_interpolation(points, function, tolerance):

    n = points.shape[0]
    divided_diffs = [[function(x, 0) for x in points]]

    for i in range(1, n):
        cur_order_diffs = []
        for j in range(0, n - i):
            if np.abs(points[j] - points[j + i]) < 10**(-10):
                cur_order_diffs.append(function(points[j], i) / np.math.factorial(i))
            else:
                cur_order_diffs.append((divided_diffs[i - 1][j] - divided_diffs[i - 1][j + 1]) / (points[j] - points[j + i]))
        divided_diffs.append(cur_order_diffs)

    hermite_polyn = Polynomial([divided_diffs[0][0]])

    for i in range(1, n):
        hermite_polyn += divided_diffs[i][0] * Polynomial.fromroots(points[: i])


    interp_nodes, nodes_orders = np.unique(points, return_counts=True)

    is_good = True

    for i in range(interp_nodes.shape[0]):
        for k in range(nodes_orders[i]):        
            diverg = hermite_polyn.deriv(k)

            if np.abs(diverg(interp_nodes[i]) - function(interp_nodes[i], k)) >= tolerance:
                corr = False

    if is_good:
        print("ALL GOOD")
        print(hermite_polyn)
    else:
        print("Can't achieve that tolerance")
        print(hermite_polyn)

    return hermite_polyn


# np.random.seed(42)
deg = 30
a, b = -1, 1
eps = 10**(-4)
function = func
uniform_grid = False

points_count = deg + 1

points = create_points(a, b, points_count, uniform_grid)
hermite_interpolation(points, func, eps)
