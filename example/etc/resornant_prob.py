from tunneling.main import Current, plt, np

pot = np.linspace(3.4, .0, 10)
pot = np.append(pot, np.linspace(0,3.4, 10))
c = Current([0, *pot, 0])
x = np.linspace(1e-19, 3, 2222)
c.m = .2
c.dx = 2 * 1e-10
for i, j in enumerate([[0, *pot, 0],
#                       [0, *pot, -2],
#                       [0, *pot, 2],
                       ]):
    c.v_tmp = j
    c.v = j
    y = np.array([c.transmission(r + 0j) for r in x])
    plt.plot(x, y[:,0], label=i)
    print(j)
plt.legend()
plt.show()
