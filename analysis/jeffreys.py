# -*- coding: utf-8 -*-
"""
Jeffreys prior
"""

import util

import numpy as np
import matplotlib.pyplot as plt

mu = 0
sigma = 1
u = mu
M = 1

v = (u - mu) / sigma

def w1(xi):
    return max(1 + xi * v, 0)

def w2(xi):
    m = max(1 + (xi + 1) * v, 0)
    
    return m / (xi + 1)
def w3(xi):
    m = max(1 + (2 * xi + 1) * v, 0)
    
    return m / (2 * xi + 1)
def w4(xi):
    m = max((2 * xi ** 2 + 3 * xi + 1) * v ** 2 + (4 * xi + 2) * v + 2, 0)
    
    return m / (2 * xi + 1)

def a(xi):
    return (xi + 1) ** 2 / (2 * xi + 1)

def b(xi):
    arr = [
        -w1(xi) * 2 * v,
        (xi + 1) * v ** 2,
        -1,
        2 * (xi + 1) * w1(xi) * w2(xi),
        -xi * w4(xi)]
    return sum(arr)

def c(xi):
    arr = [
        v ** 2 / (xi * w1(xi) ** 2),
        -2 * np.log(w1(xi)) / xi ** 3,
        2 * v / (xi ** 2 * w1(xi)),
        (1 / xi ** 2) * ((np.log(w1(xi)) / xi) - (1 / w1(xi))) ** 2]
    
    arr2 = [
        max(sum(arr), 0),
        (2 * max(xi + np.log(w1(xi)), 0)) / (M * xi ** 3),
        -(2 * w2(xi)) / (xi ** 2 * w1(xi)),
        w4(xi) / (xi * w1(xi) ** 2)]
    
    return sum(arr2)

def d(xi):
    return (xi + 1) * v - xi * w3(xi)

def e(xi):
    m = max(np.log(w1(xi)) - ((v * xi * (xi + 1)) / w1(xi)), 0)
    return (m / xi ** 2) - (1 / (xi + 1)) + w3(xi) / w1(xi)

def f(xi):
    m = max(np.log(w1(xi)) + ((v * xi * (xi + 1)) / w1(xi)), 0)
    return (v * m / xi ** 2) - w2(xi) + w4(xi) / w1(xi)
    
def pi(xi):
    if xi == 0:
        xi = 0.001
    arr = [
        a(xi) * (d(xi) * f(xi) - e(xi) ** 2),
        b(xi) * (c(xi) * e(xi) - b(xi) * f(xi)),
        c(xi) * (b(xi) * e(xi) - d(xi) * c(xi))]
    if sum(arr) <= 0:
        return 0
    return (1 / sigma ** 2 * w1(xi) ** 2) * sum(arr) ** 0.5

fig, ax = plt.subplots(figsize=(6, 4))

X = np.arange(-2, 2, 0.001)
Y = [pi(xi) for xi in X]
ax.plot(X, Y)
ax.axvline(0, linestyle="dashed", color="k")

ax.set_ylim([0, 100])
ax.set(
    xlabel=r"$\xi$",
    ylabel="Scaled Jeffreys prior PDF")

ax.grid(True)

if False:
    fig.tight_layout()
    plt.savefig(
        "%s/plots/jeffreys.pdf" % util.save_path,
        bbox_inches="tight")
plt.show()