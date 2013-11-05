from pymvg import CameraModel, MultiCameraSystem
from pymvg.plot_utils import plot_system

import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fname = os.path.join('..','pymvg_camsystem_example.json')
system = MultiCameraSystem.from_pymvg_file( fname )

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
plot_system( ax, system )
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_xlim(-0.8,0.8); ax.set_ylim(-0.8,0.8); ax.set_zlim(-0.8,0.8)
plt.show()

