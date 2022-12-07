import numpy as np
import matplotlib.pyplot as plt

a = np.random.normal(size=1000)

b = np.where(np.abs(a) > 0.92, 6, a)

plt.hist(b, bins=50)
plt.show()
