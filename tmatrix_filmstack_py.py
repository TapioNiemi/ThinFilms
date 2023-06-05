
"""
Tapio Niemi, tapio.niemi@tuni.fi

This script calculates reflection, r, transmission, t, reflectance, R, and transmittance, T of a thin-film stack
using the transfer-matrix method.

The formulation follows closely the approach by Yeh and Yariv
Photonics, ISBN 13: 9780195179460, Oxford University Press
or
Optical Waves in Layered Media, ISBN-13: 978-0471731924
Wiley Series in Pure and Applied Optics

Parameters:
n1 = refractive index of incidence side material
n2 = refractive index of semi-infinite substrate
n = refractive indices of the thin-film stack [vector]
L = thicknesses of the layers of the stack [vector] [in nanometers]
lambda = wavelength [in nanometers] or in same units as wavelength
theta = incidence angle [in degrees]
polarization = 'TE' or 'TM'

Example usage of the function (Bragg mirror):
lambda = np.linspace(600, 1400, 100)
n = np.tile([1.45, 2.0], 10)  # refractive index vector of Bragg mirror
d = np.tile(1000 / np.array([1.45, 2.0]) / 4, 10)  # thickness vector
r = np.zeros_like(lambda)
t = np.zeros_like(lambda)
R = np.zeros_like(lambda)
T = np.zeros_like(lambda)
for m in range(len(lambda)):
    r[m], t[m], R[m], T[m] = tmatrix_filmstack(1, 1.5, n, d, lambda[m], 0, 'TM')
"""

import numpy as np

def tmatrix_filmstack_py(n1, n2, n, L, lambda_, theta, polarization):
    """
    Calculates reflection, r, transmission, t, reflectance, R, and transmittance, T of a thin-film stack
    using the transfer-matrix method.
    """
    # Check validity of input parameters
    if len(n) != len(L):
        raise ValueError('n and L must be of the same length!')

    # refractive index vector including surrounding materials
    n = np.array([n1] + n + [n2])

    # Number of refractive indices = length of vector n
    # We have N materials, N-1 interfaces and N-2 layers
    N = len(n)

    # Tangential  component of the incident wavevector k.
    # Important: this is the same in each of the layers!
    kt = 2 * np.pi / lambda_ * n1 * np.sin(np.deg2rad(theta))

    # Normal components of wave vectors in all layers from Pythagoras
    # Notice, this is a vector of length N
    kn = np.sqrt((2 * np.pi / lambda_ * n) ** 2 - kt ** 2)

    # Ensure exponential decay in exit material
    # This is important if total internal reflection takes place
    kn = np.real(kn) - 1j * np.abs(np.imag(kn))

    # initialize the 2x2 transfer matrix
    matrix = np.eye(2)

    # Reflection and transmission coefficients of each interface,
    # vectors of length N-1.
    # This part depends on the polarization.
    if polarization == 'TE':
        rn = (kn[:-1] - kn[1:]) / (kn[:-1] + kn[1:])
        tn = 2 * kn[:-1] / (kn[:-1] + kn[1:])
    elif polarization == 'TM':
        rn = (n[:-1] ** 2 * kn[1:] - n[1:] ** 2 * kn[:-1]) / (n[1:] ** 2 * kn[:-1] + n[:-1] ** 2 * kn[1:])
        tn = (2 * n[1:] * n[:-1] * kn[:-1]) / (n[1:] ** 2 * kn[:-1] + n[:-1] ** 2 * kn[1:])
    else:
        raise ValueError('Polarization must be TE or TM!')

    # Generate the T and P matrices for each layer/interface
    # Let's loop through all the layers, its fast enough
    for M in range(N - 2):
        Tn = 1 / tn[M] * np.array([[1, rn[M]], [rn[M], 1]])
        P = np.array([[np.exp(1j * kn[M + 1] * L[M]), 0], [0, np.exp(-1j * kn[M + 1] * L[M])]])

        # Multiply the total transfer matrix by P and T, repeat if needed
        matrix = np.dot(matrix, np.dot(Tn, P))

    # We have almost the full system matrix but are missing the final interface T
    matrix = np.dot(matrix, 1 / tn[N - 2] * np.array([[1, rn[N - 2]], [rn[N - 2], 1]]))

    # Output:
    # Reflection and transmission of fields
    r = ((matrix[1, 0] / matrix[0, 0])
    t = 1. / matrix[0, 0]

    # Reflectance and transmittance
    R = np.abs(r) ** 2
    T = np.real(kn[-1] / kn[0]) * np.abs(t) ** 2

    return r, t, R, T
