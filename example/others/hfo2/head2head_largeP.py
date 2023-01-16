from tunneling.main import Current, plt, np
import pandas as pd

arr = [0, *np.linspace(0.975675, .3, 5), *np.linspace(.3, 1.160592, 5), 0]
arr2 = [0, *np.linspace(1.549519, 2, 5), *np.linspace(2, 1.549519, 5), 0]
c = Current(arr)
x = np.append(np.linspace(-.5, -.05, 16), -np.logspace(-1.5, -3, 8))
x = np.append(x, np.logspace(-3, -1.5, 8))
x = np.append(x, np.linspace(.05, .5, 16))
c.m = .2
c.dx = 2.2e-9 / 9

y = [c.current(r) for r in x]
plt.plot(x, y)
data = np.array([x, y])
df = pd.DataFrame(data.transpose())
df.to_excel('data.xlsx')
plt.show()
