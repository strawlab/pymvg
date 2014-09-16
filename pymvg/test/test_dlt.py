#!/usr/bin/env python
import numpy as np

import pymvg.calibration
import pymvg.camera_model

def test_dlt_old_data():
    # some old real data
    X3d = np.array([[ 304.8,    0. ,    0. ],
                    [ 304.8,  152.4,    0. ],
                    [ 304.8,  152.4,  152.4],
                    [ 304.8,    0. ,  152.4],
                    [ 178. ,   85. ,   86. ],
                    [ 178. ,   85. ,   63. ]])
    x2d_orig = np.array([[ 120. ,  475. ],
                         [ 522. ,  460. ],
                         [ 497. ,   69.6],
                         [ 120. ,   76.2],
                         [ 344. ,  200. ],
                         [ 349. ,  298. ]])

    results = pymvg.calibration.DLT(X3d, x2d_orig)
    cam=results['cam']
    x2d_reproj = cam.project_3d_to_pixel(X3d)

    # calculate point-by-point reprojection error
    err = np.sqrt(np.sum( (x2d_orig - x2d_reproj)**2, axis=1 ))

    # find mean reprojection error across all pixels
    mean_err = np.mean(err)
    assert mean_err < 3.0

def test_dlt_roundtrip():
    eye = (10,20,30)
    lookat = (11,20,30)
    up = (1,-1,0)
    # must be unit length for below to work

    n_pts = 20
    theta = np.linspace(0,2*np.pi*3,n_pts)
    z = np.linspace(-1,1,n_pts)
    scale = 0.5
    pts_3d = np.array( [ scale*np.cos(theta),
                         scale*np.sin(theta),
                         scale*z ] ).T
    pts_3d += np.array(lookat)

    cam_orig = pymvg.camera_model.CameraModel.load_camera_default()
    cam = cam_orig.get_view_camera(eye, lookat, up)

    x2d_orig = cam.project_3d_to_pixel(pts_3d)

    results = pymvg.calibration.DLT(pts_3d,x2d_orig, width=cam.width, height=cam.height)

    cam_new=results['cam']
    x2d_reproj = cam_new.project_3d_to_pixel(pts_3d)

    # calculate point-by-point reprojection error
    err = np.sqrt(np.sum( (x2d_orig - x2d_reproj)**2, axis=1 ))

    # find mean reprojection error across all pixels
    mean_err = np.mean(err)
    assert mean_err < 0.001
