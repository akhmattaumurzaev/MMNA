def func(x):
    return 5 * x**2 + 10 * x + 7

def func_der(x):
    return 10 * x + 10 


def test(x, function, tolerance):
    if abs(function(x)) < tolerance:
        print("ALL GOOD")
        print(function(x))
        print(x)
    else:
        print("Can't achieve that tolerance")

def newton(start_point, function, function_deriv, tolerance):
    x = start_point

    while abs(function(x) / function_deriv(x)) >= tolerance:
        x = x - function(x) / function_deriv(x)

    test(x, function, tolerance)

def grad_desc(start_point, function, function_deriv, tolerance, alpha):
    x = start_point

    first = func(x)
    second = 10 * first

    while abs(first - second) >= tolerance:
        second = first
        x = x - alpha * function_deriv(x)
        first = function(x)
        
    print(function(x))

a = 3

def test_func(x):
    return a - x*x

def test_func_der(x):
    return -2*x


start_point = 10
eps = 10**(-5)
method = "Newton"
if method == "Newton":
    newton(start_point, test_func, test_func_der, eps)
elif method == "Gradient Descent":
    alpha = 0.01
    gd(start_point, func, func_der, eps, alpha)
