#!/usr/bin/env python

import numpy as np
from pymvg.test.utils import _build_points_3d, make_M
import os

from pymvg.util import normalize
from pymvg.camera_model import CameraModel

DRAW=int(os.environ.get('DRAW','0'))
if DRAW:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from pymvg.plot_utils import plot_camera

def test_lookat():

    dist = 5.0

    # build camera
    center_expected = np.array( [10, 5, 20] )
    lookat_expected = center_expected + np.array( [dist, 0, 0] ) # looking in +X
    up_expected     = np.array( [0,  0,  1] )

    f = 300.0 # focal length
    width, height = 640, 480
    cx, cy = width/2.0, height/2.0

    M = np.array( [[ f, 0, cx, 0],
                      [ 0, f, cy, 0],
                      [ 0, 0,   1, 0]])
    cam1 = CameraModel.load_camera_from_M( M, width=width, height=height)
    cam = cam1.get_view_camera(center_expected, lookat_expected, up_expected)
    del cam1

    # check that the extrinsic parameters were what we expected
    (center_actual,lookat_actual,up_actual) = cam.get_view()

    lookdir_expected = normalize( lookat_expected - center_expected )
    lookdir_actual   = normalize( lookat_actual   - center_actual   )

    assert np.allclose( center_actual,  center_expected  )
    assert np.allclose( lookdir_actual, lookdir_expected )
    assert np.allclose( up_actual,      up_expected      )

    # check that the extrinsics work as expected
    pts = np.array([lookat_expected,
                    lookat_expected+up_expected])
    eye_actual = cam.project_3d_to_camera_frame( pts )

    eye_expected = [[0, 0, dist], # camera looks at +Z
                    [0,-1, dist], # with -Y as up
                    ]
    assert np.allclose( eye_actual,      eye_expected      )

    # now check some basics of the projection
    pix_actual = cam.project_3d_to_pixel( pts )

    pix_expected = [[cx,cy], # center pixel on the camera
                    [cx,cy-(f/dist)]]
    assert np.allclose( pix_actual,      pix_expected      )

def test_flip():
    for distortion in (False,True):
        yield check_flip, distortion

def check_flip(distortion=False):
    if distortion:
        d = [0.1, 0.2, 0.3, 0.4, 0.5]
    else:
        d = None

    # build camera
    center_expected = np.array( [10, 5, 20] )
    lookat_expected = center_expected + np.array( [1, 2, 0] )
    up_expected     = np.array( [0,  0,  1] )

    width, height = 640, 480

    M = np.array( [[ 300.0,     0,  321, 0],
                      [ 0,     298.0,  240, 0],
                      [ 0,         0,   1,  0]])
    cam1 = CameraModel.load_camera_from_M( M, width=width, height=height,
                                              distortion_coefficients=d )
    cam = cam1.get_view_camera(center_expected, lookat_expected, up_expected)
    del cam1

    pts = np.array([lookat_expected,
                    lookat_expected+up_expected,
                    [1,2,3],
                    [4,5,6]])
    pix_actual = cam.project_3d_to_pixel( pts )

    # Flipped camera gives same 3D->2D transform but different look direction.
    cf = cam.get_flipped_camera()
    assert not np.allclose( cam.get_lookat(), cf.get_lookat() )

    pix_actual_flipped = cf.project_3d_to_pixel( pts )
    assert np.allclose( pix_actual,      pix_actual_flipped )

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
    M = make_M( focal_length, width, height, R, c)['M']

    # now, project these 3D points into our image plane
    pts_3d_H = np.vstack( (pts_3d.T, np.ones( (1,len(pts_3d))))) # make homog.
    undist_rst_simple = np.dot(M, pts_3d_H) # multiply
    undist_simple = undist_rst_simple[:2,:]/undist_rst_simple[2,:] # project

    if DRAW:
        ax2 = fig.add_subplot(3,1,2)
        ax2.plot( undist_simple[0,:], undist_simple[1,:], 'b.')
        ax2.set_xlim(0,width)
        ax2.set_ylim(height,0)
        ax2.set_title('matrix multiply')

    # build a camera model from our M and project onto image plane
    cam = CameraModel.load_camera_from_M( M, width=width, height=height )
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
        ax3.set_title('pymvg')

    if DRAW:
        plt.show()

    assert np.allclose( undist_full, undist_simple )

if __name__=='__main__':
    test_simple_projection()
    test_lookat()
