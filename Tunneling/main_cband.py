from math import exp
from joblib import Parallel, delayed, cpu_count
import numpy as np
from cmath import exp as cexp
import matplotlib.pyplot as plt
from scipy.integrate import quad


class Current:

    def __init__(self, pot, cbEn=[], effMetal=1.): # Initializes the class with the given potential, and sets some other constants and variables.
        self.h = 4.135667662e-15  # eV * s
        self.__mass = 0.511 * 1e+6 / 299792458 ** 2  # eV * (s / m) ** 2
        self.__m = self.__mass
        self.e = 1.602e-19  # C
        self.hbar = self.h / (2 * np.pi)
        self.dx = 1e-10
        self.ef = 4
        self.v = np.array(pot)
        self.v[1:-1] += self.ef
        self.v_tmp = self.v
        self.kt = 300 * 8.617343e-5
        self.num_estep = 1000
        self.emax = 1.8
        self.cbEn = cbEn
        self.effMetal = effMetal

    @property
    def temperature(self):
        return self.kt / 8.617343e-5

    @temperature.setter
    def temperature(self, val):
        self.kt = val * 8.617343e-5

    @property
    def m(self):
        return self.__m / self.__mass

    @m.setter
    def m(self, new_val):
        self.__m = self.__mass * new_val

    def set_mass(self, dE=[]):
        self.__m = np.zeros(self.v.shape) * self.__mass 
        # tmp = self.__mass * self.cbEn[min(enumerate(self.cbEn[:,0]), key=lambda y: abs(y[1]-dE))[0],1]
        self.__m = [self.cbEn[min(enumerate(self.cbEn[:,0]), key=lambda y: abs(y[1]-dE[i]))[0],1] for i in range(len(self.v))]
        self.__m = np.array(self.__m) * self.__mass
            # tmp = self.__mass * 1.
        self.__m[0] = self.__mass * self.effMetal
        self.__m[-1] = self.__mass * self.effMetal


    def fermi(self, e=0.): # Computes the Fermi-Dirac statistics for a given energy.
        """
        fermi() -> fermi-dirac statistics
            Parameters
        ----------
        E : scalar
            energy"""
        if (e / self.kt) > 100:
            return 0
        else:
            return 1. / (1. + exp(e / self.kt))  # eV

    def gen_pot(self, v): # Generates a new potential by adding an offset to the original potential.
        self.v = self.v_tmp + np.linspace(0, -v, len(self.v))

    def density(self, e_x, v): # Computes the density of states at a given energy, e_x, and potential, v.
        a = quad(self.fermi, e_x, np.inf)
        b = quad(self.fermi, e_x + v, np.inf)
        return a[0], -b[0]

    def transmission(self, energy): # Computes the transmission probability for a given energy, using the transfer matrix method.
        if len(self.cbEn) > 0:
            self.set_mass(self.v - energy - self.ef)
        k = np.sqrt(2 * self.__m * (complex(energy + self.ef)- self.v)) / self.hbar * self.dx
        matrix = np.identity(2, dtype=np.complex)
        for n in range(0, len(self.v) - 1):
            if k[n] == 0:
                continue
            t = np.zeros((2, 2), dtype=np.complex)
            t[0, 0] = (k[n] + k[n + 1]) / 2 / k[n] * cexp(-1j * k[n])
            t[0, 1] = (k[n] - k[n + 1]) / 2 / k[n] * cexp(-1j * k[n])
            t[1, 0] = (k[n] - k[n + 1]) / 2 / k[n] * cexp(1j * k[n])
            t[1, 1] = (k[n] + k[n + 1]) / 2 / k[n] * cexp(1j * k[n])
            matrix = np.dot(matrix, t)
        return 1 - abs(matrix[1][0] / matrix[0][0]) ** 2, 1 - abs(matrix[0][1] / matrix[0][0]) ** 2

    def current(self, volt): # Computes the current flowing through the system under a given voltage.
        # A/m^2
        self.gen_pot(volt)
        constants = 4. * np.pi * self.__mass * self.e / self.h ** 3
        e_max = self.emax
        e_min = -self.emax
        if volt < 0:
            e_max -= volt
        else:
            e_min -= volt
        erange = np.linspace(e_min, e_max, self.num_estep)
        de = erange[1] - erange[0]

        def di(e_x):
            return np.dot(self.density(e_x, volt), self.transmission(e_x)) * de

        di = [di(ex) for ex in erange]
        i_tot = sum(di)
        return constants * i_tot


if __name__ == '__main__':
    fig = plt.figure()
    x = np.append(np.linspace(-1.5, -.2, 8), -np.logspace(-1, -3, 8))
    x = np.append(x, np.logspace(-3, -1, 8))
    x = np.append(x, np.linspace(.2, 1.5, 8))
    arr = dict()
    arr['dn'] = np.genfromtxt('/home/jinho93/PycharmProjects/QuantumTunneling2/example/pt-bto-lsmo/dn', delimiter='\t')[:, 1]
    arr['up'] = np.genfromtxt('/home/jinho93/PycharmProjects/QuantumTunneling2/example/pt-bto-lsmo/up', delimiter='\t')[:, 1]

    for i, j in arr.items():
        j += 1
        print(j)
        mim = Current(j)
        mim.dx = 18e-10
        y = [mim.current(r) for r in x]
        y = np.abs(y)
        print(y)
        output = np.array((x, y))
        output = output.transpose()
        np.savetxt(f'{i}.dat', output, delimiter='\t')
        plt.semilogy(x, y, label=f'{mim.dx}')
    plt.legend()
    plt.show()
