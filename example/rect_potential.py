from tunneling.main import Current, plt, np

c = Current([0, 2, -1])
x = np.linspace(1e-19, 3, 5222)
c.m = .1

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
for i in [1, 3, 7]:
    c.dx = i * 6e-10
    y = np.array([c.transmission(r + 0j) for r in x])
    ax1.plot(x, y[:,0])
    ax2.plot(x, y[:,1])

plt.show()