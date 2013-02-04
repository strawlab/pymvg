#!/usr/bin/env python

import numpy as np
import cv # ubuntu: apt-get install python-opencv

# ROS imports
import roslib; roslib.load_manifest('camera_model')
import geometry_msgs
import sensor_msgs

from utils import _build_test_camera, _build_points_3d, get_default_options
from camera_model.npcv import numpy2opencv_image, numpy2opencv_pointmat, \
     opencv_pointmat2numpy

def test_undistortion():
    all_options = get_default_options()
    for opts in all_options:
        yield check_undistortion, opts

def check_undistortion(cam_opts):
    cam = _build_test_camera(**cam_opts)

    step = 5
    border = 65

    distorteds = []
    for row in range(border, cam.height-border, step):
        for col in range(border, cam.width-border, step):
            distorted = [col, row]
            distorteds.append(distorted)
    npdistorted = np.array(distorteds,dtype=np.float)

    src = numpy2opencv_pointmat(npdistorted)
    dst = cv.CloneMat(src)
    cv.UndistortPoints(src, dst,
                       numpy2opencv_image(cam.get_K()),
                       numpy2opencv_image(cam.get_D()),
                       R = numpy2opencv_image(cam.get_rect()),
                       P = numpy2opencv_image(cam.get_P()))
    undistorted_cv = opencv_pointmat2numpy(dst)

    undistorted_np = cam.undistort( npdistorted )
    assert undistorted_cv.shape == undistorted_np.shape
    if cam.is_opencv_compatible():
        assert np.allclose(undistorted_cv, undistorted_np)

def test_projection():
    all_options = get_default_options()
    for opts in all_options:
        for distorted in (True,False):
            yield check_projection, opts, distorted

def check_projection(cam_opts,distorted=True):
    cam = _build_test_camera(**cam_opts)
    R = cam.get_rect()
    if not np.allclose(R, np.eye(3)):
        # opencv's ProjectPoints2 does not take into account
        # rectifciation matrix, thus we skip this test.
        from nose.plugins.skip import SkipTest
        raise SkipTest("Test %s is skipped: %s" %(
            check_projection.__name__,
            'cannot check if rectification matrix is not unity'))

    pts3D = _build_points_3d()
    n_pts = len(pts3D)

    src = numpy2opencv_image(pts3D)
    dst = numpy2opencv_pointmat(np.empty( (n_pts, 2) ))

    t = np.array(cam.get_translation(),copy=True)
    t.shape = 3,1

    R = cam.get_rotation()
    rvec = numpy2opencv_image(np.empty( (1,3) ))
    cv.Rodrigues2(numpy2opencv_image(R), rvec)

    if distorted:
        cv_distortion = numpy2opencv_image(cam.get_D())
    else:
        cv_distortion = numpy2opencv_image(np.zeros((5,1)))

    Pleft = cam.get_P()[:3,:3]
    cv.ProjectPoints2(src,
                      rvec,
                      numpy2opencv_image(t),
                      numpy2opencv_image(Pleft),
                      cv_distortion,
                      dst)
    result_cv = opencv_pointmat2numpy(dst)

    result_np = cam.project_3d_to_pixel( pts3D, distorted=distorted )
    assert result_cv.shape == result_np.shape
    if cam.is_opencv_compatible():
        assert np.allclose(result_cv, result_np)
