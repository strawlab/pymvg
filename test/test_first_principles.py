#!/usr/bin/env python

import numpy as np
from utils import _build_points_3d, make_pmat
import os

# ROS imports
import roslib; roslib.load_manifest('camera_model')
import geometry_msgs
import sensor_msgs

import camera_model.camera_model as camera_model

DRAW=int(os.environ.get('DRAW','0'))
if DRAW:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from camera_model.plot_utils import plot_camera

def test_simple_projection():

    # get some 3D points
    pts_3d = _build_points_3d()

    if DRAW:
        fig = plt.figure(figsize=(8,12))
        ax1 = fig.add_subplot(3,1,1, projection='3d')
        ax1.scatter( pts_3d[:,0], pts_3d[:,1], pts_3d[:,2])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

    # build a camera calibration matrix
    focal_length = 1200
    width, height = 640,480
    R = np.eye(3) # look at +Z
    c = np.array( (9.99, 19.99, 20) )
    pmat = make_pmat( focal_length, width, height, R, c)['pmat']

    # now, project these 3D points into our image plane
    pts_3d_H = np.vstack( (pts_3d.T, np.ones( (1,len(pts_3d))))) # make homog.
    undist_rst_simple = np.dot(pmat, pts_3d_H) # multiply
    undist_simple = undist_rst_simple[:2,:]/undist_rst_simple[2,:] # project

    if DRAW:
        ax2 = fig.add_subplot(3,1,2)
        ax2.plot( undist_simple[0,:], undist_simple[1,:], 'b.')
        ax2.set_xlim(0,width)
        ax2.set_ylim(height,0)
        ax2.set_title('matrix multiply')

    # build a camera model from our pmat and project onto image plane
    cam = camera_model.load_camera_from_pmat( pmat, width=width, height=height )
    undist_full = cam.project_3d_to_pixel(pts_3d).T

    if DRAW:
        plot_camera( ax1, cam, scale=10, axes_size=5.0 )
        sz = 20
        x = 5
        y = 8
        z = 19
        ax1.auto_scale_xyz( [x,x+sz], [y,y+sz], [z,z+sz] )

        ax3 = fig.add_subplot(3,1,3)
        ax3.plot( undist_full[0,:], undist_full[1,:], 'b.')
        ax3.set_xlim(0,width)
        ax3.set_ylim(height,0)
        ax3.set_title('camera_model')

    if DRAW:
        plt.show()

    assert np.allclose( undist_full, undist_simple )

if __name__=='__main__':
    test_simple_projection()
