#!/usr/bin/env python

import numpy as np
import cv # ubuntu: apt-get install python-opencv

# ROS imports
import roslib; roslib.load_manifest('camera_model')
import geometry_msgs
import sensor_msgs

from test_utils import _build_test_camera, _build_points_3d, get_default_options

# helper functions ---------------

def numpy2opencv_image(arr):
    arr = np.array(arr)
    assert arr.ndim==2
    if arr.dtype in [np.float32]:
        result = cv.CreateMat( arr.shape[0], arr.shape[1], cv.CV_32FC1)
    elif arr.dtype in [np.float64, np.float]:
        result = cv.CreateMat( arr.shape[0], arr.shape[1], cv.CV_64FC1)
    elif arr.dtype in [np.uint8]:
        result = cv.CreateMat( arr.shape[0], arr.shape[1], cv.CV_8UC1)
    else:
        raise ValueError('unknown numpy dtype "%s"'%arr.dtype)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            result[i,j] = arr[i,j]
    return result

def opencv_image2numpy( cvimage ):
    pyobj = np.asarray(cvimage)
    if pyobj.ndim == 2:
        # new OpenCV version
        result = pyobj
    else:
        # old OpenCV, so hack this
        width = cvimage.width
        height = cvimage.height
        assert cvimage.channels == 1
        assert cvimage.nChannels == 1
        assert cvimage.depth == 32
        assert cvimage.origin == 0
        result = np.empty( (height,width), dtype=np.float )
        for i in range(height):
            for j in range(width):
                result[i,j] = cvimage[i,j]
    return result

def numpy2opencv_pointmat(npdistorted):
    src = cv.CreateMat( npdistorted.shape[0], 1, cv.CV_64FC2)
    for i in range(npdistorted.shape[0]):
        src[i,0] = npdistorted[i,0], npdistorted[i,1]
    return src

def opencv_pointmat2numpy(dst):
    assert dst.width==1
    assert dst.channels == 2
    r = np.empty( (dst.height,2) )
    for i in range(dst.height):
        a,b = dst[i,0]
        r[i,:] = a,b
    return r

# --------------------- testing -----------------------------

def test_roundtrip_image():
    orig = np.array( [[100.0,100],
                      [100,200],
                      [100,300],
                      [100,400]] )
    testcv = numpy2opencv_image(orig)
    test = opencv_image2numpy( testcv )
    assert orig.shape==test.shape
    assert np.allclose( orig, test )

def test_roundtrip_pointmat():
    orig = np.array( [[100.0,100],
                      [100,200],
                      [100,300],
                      [100,400]] )
    testcv = numpy2opencv_pointmat(orig)
    test = opencv_pointmat2numpy( testcv )
    assert orig.shape==test.shape
    assert np.allclose( orig, test )

def test_undistortion_compared_to_opencv():
    all_options = get_default_options()
    for opts in all_options:
        yield check_undistortion_compared_to_opencv, opts

def check_undistortion_compared_to_opencv(cam_opts):
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

def test_projection_compared_to_opencv():
    all_options = get_default_options()
    for opts in all_options:
        for distorted in (True,False):
            yield check_projection_compared_to_opencv, opts, distorted

def check_projection_compared_to_opencv(cam_opts,distorted=True):
    cam = _build_test_camera(**cam_opts)
    R = cam.get_rect()
    if not np.allclose(R, np.eye(3)):
        # opencv's ProjectPoints2 does not take into account
        # rectifciation matrix, thus we skip this test.
        return

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

    cv.ProjectPoints2(src,
                      rvec,
                      numpy2opencv_image(t),
                      numpy2opencv_image(cam.get_K()),
                      cv_distortion,
                      dst)
    result_cv = opencv_pointmat2numpy(dst)

    result_np = cam.project_3d_to_pixel( pts3D, distorted=distorted )
    assert result_cv.shape == result_np.shape
    if cam.is_opencv_compatible():
        assert np.allclose(result_cv, result_np)
