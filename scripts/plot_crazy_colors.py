import matplotlib.pyplot as plt
import numpy as np

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

data = np.random.rand(10,20)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(data)
ax.set_xlabel('xlabel')
ax.set_aspect(2)
fig.savefig('equal.png')
ax.set_aspect('auto')
fig.savefig('auto.png')
forceAspect(ax,aspect=1)
fig.savefig('force.png')

plt.show()