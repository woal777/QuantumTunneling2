import time

import sys
from joblib import Parallel, delayed, cpu_count
import numpy as np
from cmath import exp
from numpy.lib.scimath import sqrt
import matplotlib.pyplot as plt
from joblib.externals.loky import get_reusable_executor


class MIM:

    def __init__(self, v, fix_m=False):
        self.e = 1.602e-19  # C
        self.h = 4.135667662e-15  # eV * s
        self.hbar = self.h / (2 * np.pi)
        self.rest_mass = 0.511 * 1e+6 / 299792458 ** 2  # eV * (s / m) ** 2
        self.m_h = sqrt(self.rest_mass) / self.hbar
        self.V = v
        self.V += sys.float_info.epsilon
        self.v_calc = self.V
        if fix_m:
            self.m = .3
        else:
            self.m = np.zeros(len(self.V))
        self.kt = 300 * 8.617343e-5
        self.dx = 18e-10
        self.constants = 4. * np.pi * self.rest_mass * self.e / self.h ** 3

    def transmission(self, E_x):
        if not isinstance(self.m, float):
            for i in range(len(self.V)):
                if self.v_calc[i] - E_x < 0.1:
                    self.m[i] = 0.16869
                elif self.v_calc[i] - E_x < 0.6:
                    self.m[i] = .086 + .906 * (self.v_calc[i] - E_x) - .791 * (self.v_calc[i] - E_x) ** 2
                else:
                    self.m[i] = .3448399999999999
            self.m[0] = .65
            self.m[-1] = .65
        k = sqrt(2 * self.m * (E_x - self.v_calc)) * self.m_h * self.dx
        beta = k / self.m
        matrix = np.identity(2, dtype=np.complex)
        for j in range(len(self.V) - 1):
            T = np.zeros((2, 2), dtype=np.complex)
            T[0, 0] = (beta[j] + beta[j + 1]) / (2 * beta[j]) * exp(-1j * k[j])
            T[0, 1] = (beta[j] - beta[j + 1]) / (2 * beta[j]) * exp(-1j * k[j])
            T[1, 0] = (beta[j] - beta[j + 1]) / (2 * beta[j]) * exp(1j * k[j])
            T[1, 1] = (beta[j] + beta[j + 1]) / (2 * beta[j]) * exp(1j * k[j])
            matrix = np.dot(matrix, T)
        return 1 - abs(matrix[1][0] / matrix[0][0]) ** 2

    def gen_pot(self, v):
        if v > 0:
            self.v_calc = self.V + np.linspace(0, v, len(self.V))
        else:
            self.v_calc = self.V + np.linspace(v, 0, len(self.V))

    def current(self, V):
        # self.gen_pot(V)
        E_xm = 2
        Nx = 600
        E_rm = 2
        Nr = 600
        trans = 0.
        j = lambda n, l: trans * E_xm / Nx * \
                         (self.f(E_rm / Nr * l + E_xm / Nx * n) - self.f(
                             E_rm / Nr * l + E_xm / Nx * n - abs(V))) * E_rm / Nr

        jj = 0
        for nn in range(1, Nx + 1):
            trans = self.transmission(E_xm / Nx * nn)
            for ll in range(1, Nr + 1):
                jj += j(nn, ll)
        return self.constants * jj

    def f(self, e):
        if (e / self.kt) > 100:
            return 0
        else:
            return 1. / (1. + exp(e / self.kt))  # eV

    def iv(self, volt):
        num_cores = cpu_count()
        current = Parallel(n_jobs=num_cores)(delayed(self.current)(r) for r in volt)
        return current


if __name__ == '__main__':
    t = time.time()
    x = np.append(np.linspace(-.5, -.1, 12, endpoint=False), -np.logspace(-1, -4, 8))
    x = np.append(x, np.logspace(-4, -1, 8, endpoint=False))
    x = np.append(x, np.linspace(.1, .5, 12))
    arr = dict()
    arr['dn'] = np.genfromtxt('/home/jinho93/PycharmProjects/QuantumTunneling2/example/pt-bto-lsmo/dn', delimiter='\t')[
                :, 1]
    arr['up'] = np.genfromtxt('/home/jinho93/PycharmProjects/QuantumTunneling2/example/pt-bto-lsmo/up', delimiter='\t')[
                :, 1]
    arr['up'][-1] = 0.5645515105105274
    # for i in [2, 6 , 14]:
    #     mim = MIM(np.array([0, 1., 0]), fix_m=True)
    #     mim.dx = i * 1e-10
    #     print(sqrt(2 * mim.rest_mass) * mim.dx / mim.hbar)
    #     x = np.linspace(-1, 3, 200)
    #     y = list(map(mim.transmission, x))
    #     plt.plot(x, y)
    for i, j in arr.items():
        j += 1
        j = np.append(0, j)
        j = np.append(j, 0)
        mim = MIM(j)
        mim.dx = 8e-10
        y = mim.iv(x)
        y = np.abs(y)
        output = np.array((x, y))
        output = output.transpose()
        np.savetxt(f'{i}.dat', output, delimiter='\t')
        plt.semilogy(x, y, label=f'{i}')
    print(time.time() - t)
    get_reusable_executor(timeout=1)
    plt.legend()
    plt.show()
