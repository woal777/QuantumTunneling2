#%%
import shutil
import sys

from matplotlib.axes import Axes
import os

from Tunneling.main import Current, plt, np

a = 20
b = 20
barrier = [0, *[3] * b, *[2] * a, 1]
plt.plot(barrier)
#%%
# barrier[5:16] = [0] * 11
c = Current(barrier)
c.ef = float(0.1)
x_en = np.linspace(0, 1, 1001)
c.m = 0.3
c.dx = 1e-10

x = np.linspace(0, len(barrier) * c.dx, len(barrier)) * 1e+10

c.temperature = 300

x_bias = np.linspace(0, 9, 4)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))
import matplotlib.cm as cm
for n, i in enumerate(x_bias):
    color = cm.autumn_r(n / len(x_bias))
    c.gen_pot(i)
    y = np.array([c.transmission(complex(r))[0] for r in x_en])
    ax1.plot(x, c.v, label='barrier at ' + str(int(i)) + ' eV', color=color)
    ax2.semilogy(x_en, y, label='TX at ' + str(int(i)) + ' eV', color=color)

ax1.set_ylabel('Height (eV)', fontsize=14)
ax2.set_ylabel('Transmission', fontsize=14)
ax1.set_xlabel('Distance (A)', fontsize=14)
ax2.set_xlabel('energy (eV)', fontsize=14)


ax2.set_ylim([1e-11, 1])
ax2.set_yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1])
ax2.set_yticklabels(['1e-10', '1e-8', '1e-6', '1e-4', '1e-2', '1'])
ax1.legend(loc='lower left', fontsize=14)
ax2.legend(loc='lower right', fontsize=14)

ax1.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)
# Add title to the figure
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)

plt.show()
# %%
