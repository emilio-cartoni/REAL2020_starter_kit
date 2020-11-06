import matplotlib.pyplot as plt
import numpy as np
import sys

res = np.load(sys.argv[1], allow_pickle=True)
plt.plot(range(50), res.tolist()['detailed_scores']['2D'])
plt.ylim([0, 1])
plt.show()

