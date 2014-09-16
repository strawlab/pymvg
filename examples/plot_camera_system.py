from pymvg.multi_camera_system import build_example_system
from pymvg.plot_utils import plot_system
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

n=6
z=5.0
system = build_example_system(n=n,z=z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot_system( ax, system, scale=z/5.0 )

if 1:
    # put some points to force mpl's view dimensions
    pts = np.array([[0,0,0],
                    [2*n, 2*z, 2*z]])
    ax.plot( pts[:,0], pts[:,1], pts[:,2], 'k.')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
