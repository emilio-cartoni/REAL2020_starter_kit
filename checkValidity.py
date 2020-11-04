import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

t = np.load('transitions_915141_2500.npy', allow_pickle=True)

x_pos = [s[0][1]['cube'][0]   for s in t]
y_pos = [s[0][1]['cube'][1]   for s in t]
z_pos = [s[0][1]['cube'][2]   for s in t]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_pos, y_pos, z_pos)

plt.figure()
plt.plot(x_pos)
plt.figure()
plt.plot(y_pos)
plt.figure()
plt.plot(z_pos)


plt.show()
