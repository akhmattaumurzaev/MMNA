import numpy as np
from scipy.linalg import eigh_tridiagonal
from itertools import combinations
from functools import reduce
from operator import mul
from numpy.polynomial.polynomial import Polynomial


def integral_1(val):
    if val % 2:
        return 0
    else:
        return 2 / (val + 1)
    
def dot_product(first_mas, second_mas, corr, integral):
    val = 0
    for ind_i, x in enumerate(first_mas):
        for ind_j, k in enumerate(second_mas):
            val += x * k * integral(ind_i + ind_j + corr)
    return val
    
def test(L, n, integral, tolerance):

    flag = True

    for i in range(0, n):
        val = dot_product(L[i, :(i + 1)], L[i, :(i + 1)], 0, integral)

        if (np.abs(np.abs(val) - 1) >= tolerance):
            flag = False
            break

        for j in range(i + 1, n):
            val = dot_product(L[i, :(i + 1)], L[j, :(j + 1)], 0, integral)

            if (np.abs(val) >= tolerance):
                flag = False

    if flag:
        print("ALL GOOD")
        print(L)
    else:
        print("Can't achive that tolerance")

def test_pol(L, n, integral, tolerance):

    flag = True

    for i in range(0, n):
        val = dot_product(L[i], L[i], 0, integral)

        if (np.abs(np.abs(val) - 1) >= tolerance):
            flag = False
            break

        for j in range(i + 1, n):
            val = dot_product(L[i], L[j], 0, integral)

            if (np.abs(val) >= tolerance):
                flag = False

    if flag:
        print("ALL GOOD")
        print(L)
    else:
        print("Can't achive that tolerance")


    
max_deg = 5
eps = 10**(-5)
integral = integral_1
method = "Eigenvalues"

n = max_deg + 1

integral = integral_1

if method == 'Gramm':

    G = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            G[i, j] = integral(i + j)
            G[j, i] = G[i, j]
        G[i, i] = integral(i + i)

    L = np.linalg.inv(np.linalg.cholesky(G))

    test(L, n, integral, eps)

elif method == "Recur":

    L = [Polynomial(np.sqrt(0.5))]

    dot_pol = Polynomial([0, 1])

    beta = 0
    
    for i in range(1, n):

        alpha = dot_product(L[-1], L[-1], 1, integral)

        if beta:
            new_pol = dot_pol * L[-1] - alpha * L[-1] - beta * L[-2]
        else:
            new_pol = dot_pol * L[-1] - alpha * L[-1]

        beta = np.sqrt(dot_product(new_pol, new_pol, 0, integral))

        new_pol /= beta

        L.append(new_pol)

    test_pol(L, n, integral, eps)

elif method == "Eigenvalues":

    L = [Polynomial(np.sqrt(0.5))]

    betta = 0

    mas_alpha = []
    mas_betta = []

    for i in range(1, n):

        alpha = dot_product(L[i - 1], L[i - 1], 1, integral)

        mas_alpha.append(alpha)

        roots = eigh_tridiagonal(np.array(mas_alpha), np.array(mas_betta))[0]

        L.append(Polynomial.fromroots(roots))

        norm = np.sqrt(dot_product(L[-1], L[-1], 0, integral))

        L[-1] /= norm

        betta = dot_product(L[-1], L[-2], 1, integral)

        mas_betta.append(betta)

    test_pol(L, n, integral, eps)
