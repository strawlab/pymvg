import roslib; roslib.load_manifest('camera_model')

from camera_model.plot_utils import plot_camera
import camera_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
R2D = 180./np.pi

base = camera_model.load_default_camera()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n=6
x = np.linspace(0, 2*n, n)
theta = np.linspace(0, 2*np.pi, n)
dim = 5.0
for i in range(n):
    center = np.array( (x[i], 0.0, dim) )
    lookat = center + np.array( (0,1,0))
    up = -np.sin(theta[i]), 0, np.cos(theta[i])

    cam = base.get_view_camera(center,lookat,up)
    cam.name = 'theta: %.0f'%( theta[i]*R2D )
    plot_camera( ax, cam, scale = dim/5.0 )

if 1:
    # put some points to force mpl's view dimensions
    pts = np.array([[0,0,0],
                    [2*n, 2*dim, 2*dim]])
    ax.plot( pts[:,0], pts[:,1], pts[:,2], 'k.')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
