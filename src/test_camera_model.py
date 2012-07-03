#!/usr/bin/env python

# This testing stuff is in its own module so we don't have to import
# OpenCV to use camera_model.

# ROS imports
import roslib; roslib.load_manifest('camera_model')
import geometry_msgs
import sensor_msgs

import numpy as np

from camera_model import CameraModel
from camera_model.camera_model import point_msg_to_tuple, parse_rotation_msg
import camera_model

import cv

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

def _build_test_camera(from_pmat=False,at_origin=False,ROS_test_data=False,get_input_data=False,flipped=False):
    if from_pmat:
        d = {'width': 848,
             'name': 'camera',
             'height': 480}
        pmat =  np.array([[ -1.70677031e+03,  -4.10373295e+03,  -3.88568028e+02, 6.89034515e+02],
                          [ -6.19019195e+02,  -1.01292091e+03,  -2.67534989e+03, 4.51847857e+02],
                          [ -4.52548832e+00,  -3.78900498e+00,  -7.35860226e-01, 1.00000000e+00]])
        cam = camera_model.load_camera_from_pmat( pmat, **d)
        if flipped:
            cam = cam.get_flipped_camera()
        if get_input_data:
            return dict(cam=cam)
        return cam
    if not at_origin:
        translation = geometry_msgs.msg.Point()
        translation.x = 0.273485679077
        translation.y = 0.0707310128808
        translation.z = 0.0877802104531

        rotation = geometry_msgs.msg.Quaternion()
        rotation.x = 0.309377331102
        rotation.y = 0.600893485738
        rotation.z = 0.644637681813
        rotation.w = 0.357288321925
    else:
        translation = geometry_msgs.msg.Point()
        translation.x = 0.0
        translation.y = 0.0
        translation.z = 0.0

        rotation = geometry_msgs.msg.Quaternion()
        rotation.x = 0.0
        rotation.y = 0.0
        rotation.z = 0.0
        rotation.w = 1.0

    if ROS_test_data:
        i = sensor_msgs.msg.CameraInfo()
        # these are from image_geometry ROS package in the utest.cpp file
        i.height = 480
        i.width = 640
        i.distortion_model = 'plumb_bob'
        i.D = [-0.363528858080088, 0.16117037733986861, -8.1109585007538829e-05, -0.00044776712298447841, 0.0]
        i.K = [430.15433020105519,                0.0, 311.71339830549732,
                                     0.0, 430.60920415473657, 221.06824942698509,
                                     0.0,                0.0,                1.0]
        i.R = [0.99806560714807102, 0.0068562422224214027, 0.061790256276695904,
                      -0.0067522959054715113, 0.99997541519165112, -0.0018909025066874664,
                      -0.061801701660692349, 0.0014700186639396652, 0.99808736527268516]
        i.P = [295.53402059708782, 0.0, 285.55760765075684, 0.0,
                      0.0, 295.53402059708782, 223.29617881774902, 0.0,
                      0.0, 0.0, 1.0, 0.0]
    else:
        i = sensor_msgs.msg.CameraInfo()
        i.height = 494
        i.width = 659
        i.distortion_model = 'plumb_bob'
        i.D = [-0.34146457767225, 0.196070795764995, 0.000548988393912233, 0.000657058395082583, -0.0828776806503243]
        i.K = [516.881868241566, 0.0, 333.090936517613, 0.0, 517.201263180996, 231.526036849886, 0.0, 0.0, 1.0]
        i.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        i.P = [442.17529296875, 0.0, 334.589001099812, 0.0, 0.0, 474.757141113281, 228.646131377705, 0.0, 0.0, 0.0, 1.0, 0.0]


    cam = CameraModel(translation=point_msg_to_tuple(translation),
                      rotation=parse_rotation_msg(rotation),
                      intrinsics=i,
                      name='cam',
                      )
    if flipped:
        cam = cam.get_flipped_camera()
    if get_input_data:
        return dict(cam=cam,
                    translation=translation,
                    rotation=rotation,
                    )
    return cam

def _build_points_3d():
    n_pts = 100
    x,y,z = 10.001, 20.001, 30.001
    theta = np.linspace(0,10,n_pts)%(2*np.pi)
    h = np.linspace(0,10,n_pts)
    r = 0.05
    pts3D = np.vstack( (r*np.cos(theta)+x, r*np.sin(theta)+y, h+z )).T
    return pts3D

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

def get_default_options():
    result = []
    for at_origin in (True,False):
        for ROS_test_data in (True,False):
            #for flipped in (True,False):
            for flipped in (False,):
                opts = dict(at_origin=at_origin,ROS_test_data=ROS_test_data,flipped=flipped)
                result.append(opts)
    #result.append( dict(from_pmat=True) )
    return result

def test_undistortion():
    all_options = get_default_options()
    for opts in all_options:
        yield check_undistortion, opts

def check_undistortion(cam_opts):
    cam = _build_test_camera(**cam_opts)

    npdistorted = np.array( [[100.0,100],
                             [100,200],
                             [100,300],
                             [100,400]] )
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

def test_projection_to_distorted():
    all_options = get_default_options()
    for opts in all_options:
        yield check_projection_to_distorted, opts

def check_projection_to_distorted(cam_opts):
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

    cv.ProjectPoints2(src,
                      rvec,
                      numpy2opencv_image(t),
                      numpy2opencv_image(cam.get_K()),
                      numpy2opencv_image(cam.get_D()),
                      dst)
    distorted_cv = opencv_pointmat2numpy(dst)

    distorted_np = cam.project_3d_to_pixel( pts3D, distorted=True )
    assert distorted_cv.shape == distorted_np.shape
    if cam.is_opencv_compatible():
        assert np.allclose(distorted_cv, distorted_np)

def test_projection_to_undistorted1():
    at_origin=True # this test mathematically only makes sense of camera at origin
    for ROS_test_data in (True,False):
        opts = dict(at_origin=at_origin,ROS_test_data=ROS_test_data)
        yield check_projection_to_undistorted1, opts

def check_projection_to_undistorted1(cam_opts):
    cam = _build_test_camera(**cam_opts)
    for z in np.linspace(0.1, 10, 20):
        pt = np.array([[0,0,z]])
        result = cam.project_3d_to_pixel( pt, distorted=False )
        u,v = result[0]

        assert np.allclose(u, cam.cx())
        assert np.allclose(v, cam.cy())

def test_camera_distortion_roundtrip():
    all_options = get_default_options()
    for opts in all_options:
        yield check_camera_distortion_roundtrip, opts

def check_camera_distortion_roundtrip(cam_opts):
    cam = _build_test_camera(**cam_opts)
    step = 5
    border = 65

    uv_raws = []
    for row in range(border, cam.height-border, 10):
        for col in range(border, cam.width-border, 10):
            uv_raw = [col, row] # Bug in utest.cpp line 130: these are flipped at svn r29350
            uv_raws.append(uv_raw)
    uv_raw = np.array(uv_raws)
    uv_rect = cam.undistort( uv_raw )
    uv_unrect = cam.distort( uv_rect )
    assert uv_raw.shape == uv_unrect.shape
    assert np.allclose(uv_raw, uv_unrect, atol=1.0) # within one pixel

def test_camera_projection_roundtrip():
    all_options = get_default_options()
    for distorted in (True,False):
        for opts in all_options:
            yield check_camera_projection_roundtrip, opts, distorted

def check_camera_projection_roundtrip(cam_opts,distorted=False):
    cam = _build_test_camera(**cam_opts)
    step = 5
    border = 65

    uv_raws = []
    for row in range(border, cam.height-border, 10):
        for col in range(border, cam.width-border, 10):
            uv_raw = [col, row] # Bug in utest.cpp line 130: these are flipped at svn r29350
            uv_raws.append(uv_raw)
    uv_raw = np.array(uv_raws)
    pts3d = cam.project_pixel_to_3d_ray( uv_raw, distorted=distorted )
    uv_unrect = cam.project_3d_to_pixel( pts3d, distorted=distorted )
    assert uv_raw.shape == uv_unrect.shape
    print uv_raw
    print
    print uv_unrect
    assert np.allclose(uv_raw, uv_unrect, atol=1.0) # within one pixel

def test_extrinsic_msg():
    all_options = get_default_options()
    for opts in all_options:
        yield check_extrinsic_msg, opts

def check_extrinsic_msg(cam_opts):
    cam_opts['get_input_data']=True
    r = _build_test_camera(**cam_opts)
    cam = r['cam']
    tfm = cam.get_extrinsics_as_msg()
    if 'translation' in r:
        assert np.allclose(point_msg_to_tuple(tfm.translation), point_msg_to_tuple(r['translation']))
    if 'rotation' in r:
        assert np.allclose(parse_rotation_msg(tfm.rotation,force_matrix=True),
                           parse_rotation_msg(r['rotation'],force_matrix=True))

def test_build_from_pmat():
    all_options = get_default_options()
    for opts in all_options:
        yield check_built_from_pmat, opts

def check_built_from_pmat(cam_opts):
    cam_orig = _build_test_camera(**cam_opts)
    pmat_orig = cam_orig.pmat
    cam = camera_model.load_camera_from_pmat( cam_orig.pmat, width=cam_orig.width, height=cam_orig.height)
    assert np.allclose( cam.pmat, cam_orig.pmat)

# def test_problem_pmat():
#     # This pmat (found by the DLT method) was causing me problems.
#     if 0:
#         pmat = np.array([[-13770.75567,  -11668.5582,      -64.229267,    812.000266],
#                          [ -7075.226893,  -5992.91884,     -27.953958,    416.965691],
#                          [   -16.958163,    -14.375729,     -0.077775,      1.      ]])
#         cam = camera_model.load_camera_from_pmat( pmat )

#     elif 1:
#        d = {'width': 848,
#              'name': 'camera',
#              'height': 480}
#        pmat =  np.array([[ -1.70677031e+03,  -4.10373295e+03,  -3.88568028e+02, 6.89034515e+02],
#                          [ -6.19019195e+02,  -1.01292091e+03,  -2.67534989e+03, 4.51847857e+02],
#                          [ -4.52548832e+00,  -3.78900498e+00,  -7.35860226e-01, 1.00000000e+00]])
#        cam = camera_model.load_camera_from_pmat( pmat, **d)

#     #assert np.allclose( cam.pmat, pmat) # we don't expect this since the intrinsic matrix may not be scaled

#     verts = np.array([[ 0.042306,  0.015338,  0.036328, 1.0],
#                       [ 0.03323,   0.030344,  0.041542, 1.0],
#                       [ 0.036396,  0.026464,  0.052408, 1.0]])

#     actual = cam.project_3d_to_pixel(verts[:,:3])

#     expectedh = np.dot( pmat, verts.T )
#     expected = (expectedh[:2]/expectedh[2]).T
#     assert np.allclose( expected, actual )

def test_bagfile_roundtrip():
    all_options = get_default_options()
    for opts in all_options:
        yield check_bagfile_roundtrip, opts

def check_bagfile_roundtrip(cam_opts):
    cam = _build_test_camera(**cam_opts)
    fname = '/tmp/cam-model-rosbag-test.bag'
    with open(fname,mode='wb') as fd:
        cam.save_to_bagfile(fd)
        fd.close()

    with open(fname,mode='r') as fd:
        cam2 = camera_model.load_camera_from_bagfile( fd )

        verts = np.array([[ 0.042306,  0.015338,  0.036328],
                          [ 0.03323,   0.030344,  0.041542],
                          [ 0.036396,  0.026464,  0.052408]])

        expected =  cam.project_3d_to_pixel(verts)
        actual   = cam2.project_3d_to_pixel(verts)
        assert np.allclose( expected, actual )

def test_camera_mirror_projection_roundtrip():
    all_options = get_default_options()
    for distorted in (True,False):
        for opts in all_options:
            yield check_camera_mirror_projection_roundtrip, opts, distorted

def check_camera_mirror_projection_roundtrip(cam_opts,distorted=False):
    cam_orig = _build_test_camera(**cam_opts)
    cam_mirror = cam_orig.get_mirror_camera()
    step = 5
    border = 65

    uv_raws = []
    for row in range(border, cam_orig.height-border, 10):
        for col in range(border, cam_orig.width-border, 10):
            uv_raw = [col, row] # Bug in utest.cpp line 130: these are flipped at svn r29350
            uv_raws.append(uv_raw)
    uv_raw = np.array(uv_raws)
    # Get a collection of 3D points for which we know the pixel location of
    pts3d = cam_orig.project_pixel_to_3d_ray( uv_raw, distorted=distorted )
    # Now project that through our mirror-image camera.
    uv_mirror = cam_mirror.project_3d_to_pixel( pts3d, distorted=distorted )
    # Which points should be xnew = (width-x)
    expected = np.array(uv_raw)
    expected[:,0] = cam_orig.width - uv_raw[:,0]
    assert expected.shape == uv_mirror.shape
    assert np.allclose(expected, uv_mirror, atol=1.0) # within one pixel

# def test_flip():
#     all_options = get_default_options()
#     for opts in all_options:
#         yield check_flip, opts

# def check_flip(cam_opts):
#     cam_orig = _build_test_camera(**cam_opts)
#     cam_flip = cam_orig.get_flipped_camera()

#     print 'cam_orig.get_camcenter()',cam_orig.get_camcenter()
#     print 'cam_orig.get_rotation_quat()',cam_orig.get_rotation_quat()
#     # They have different orientation (but same position) in space,
#     assert not np.allclose( cam_orig.get_rotation(), cam_flip.get_rotation())
#     assert np.allclose( cam_orig.get_camcenter(), cam_flip.get_camcenter())

#     eye, lookat, up = cam_orig.get_view()
#     eye2, lookat2, up2 = cam_flip.get_view()

#     print 'lookat, eye, lookat2', lookat, eye, lookat2
#     d1 = eye-lookat
#     d2 = lookat2-eye
#     print 'd1,d2',d1,d2

#     # but they project 3D points to same pixel locations
#     verts = np.array([[ 0.042306,  0.015338,  0.036328],
#                       [ 1.03323,   2.030344,  3.041542],
#                       [ 0.03323,   0.030344,  0.041542],
#                       [ 0.036396,  0.026464,  0.052408]])

#     expected = cam_orig.project_3d_to_pixel(verts)
#     actual   = cam_flip.project_3d_to_pixel(verts)
#     assert np.allclose( expected, actual )

def test_view():
    all_options = get_default_options()
    for opts in all_options:
        yield check_view, opts

def check_view(cam_opts):

    # This is not a very good test. (Should maybe check more eye
    # positions, more lookat directions, and more up vectors.)

    cam_orig = _build_test_camera(**cam_opts)
    eye = (10,20,30)
    lookat = (11,20,30) # must be unit length for below to work
    up = (0,-1,0)
    cam_new = cam_orig.get_view_camera(eye, lookat, up)
    eye2, lookat2, up2 = cam_new.get_view()
    assert np.allclose( eye, eye2)
    assert np.allclose( lookat, lookat2 )
    assert np.allclose( up, up2 )

def test_camcenter():
    all_options = get_default_options()
    for opts in all_options:
        cam = _build_test_camera(**opts)
        assert np.allclose( cam.get_camcenter(), cam.t_inv.T )
