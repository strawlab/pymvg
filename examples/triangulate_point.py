from __future__ import print_function
from pymvg.multi_camera_system import build_example_system
import numpy as np

system = build_example_system()
point_3d_expected = np.array([1.0, 2.0, 3.0])
print('Original: %r'%(point_3d_expected,))

# find perfect 2D projection of this original 3D point
data = []
for camera_name in system.get_names():
    this_pt2d = system.find2d(camera_name,point_3d_expected)
    data.append( (camera_name,this_pt2d) )
    print('%r: %r'%(camera_name,this_pt2d))

# now triangulate 3D point
point_3d_actual = system.find3d(data)
print('Result ----> %r'%(point_3d_actual,))

