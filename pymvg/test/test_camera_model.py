#!/usr/bin/env python

import numpy as np
from nose.plugins.skip import SkipTest
import os, tempfile
import pickle

from pymvg.camera_model import CameraModel
from pymvg.util import point_msg_to_tuple, parse_rotation_msg
import pymvg.align as mcsc_align

from pymvg.test.utils import _build_test_camera, get_default_options

# --------------------- testing -----------------------------

def _generate_uv_raw(width,height):
    step = 5
    border = 65

    uv_raws = []
    for row in range(border, height-border, step):
        for col in range(border, width-border, step):
            uv_raw = [col, row]
            uv_raws.append(uv_raw)
    return np.array(uv_raws)

def test_dict_roundtrip():
    all_options = get_default_options()
    for opts in all_options:
        yield check_dict_roundtrip, opts

def check_dict_roundtrip(cam_opts):
    cam = _build_test_camera(**cam_opts)
    d = cam.to_dict()
    cam2 = CameraModel.from_dict(d)
    assert cam==cam2

def test_projection_to_undistorted1():
    at_origin=True # this test mathematically only makes sense of camera at origin
    for ROS_test_data in (True,False):
        opts = dict(at_origin=at_origin,ROS_test_data=ROS_test_data)
        yield check_projection_to_undistorted1, opts

def check_projection_to_undistorted1(cam_opts):
    """check that points along optical axis are imaged onto principal point"""
    cam = _build_test_camera(**cam_opts)
    for z in np.linspace(0.1, 10, 20):
        pt = np.array([[0,0,z]])
        result = cam.project_3d_to_pixel( pt, distorted=False )
        u,v = result[0]

        assert np.allclose(u, cam.P[0,2])
        assert np.allclose(v, cam.P[1,2])

def test_camera_distortion_roundtrip():
    all_options = get_default_options()
    for opts in all_options:
        yield check_camera_distortion_roundtrip, opts

def check_camera_distortion_roundtrip(cam_opts):
    """check that uv == distort( undistort( uv ))"""
    cam = _build_test_camera(**cam_opts)
    uv_raw = _generate_uv_raw(cam.width, cam.height)
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
    """check that uv == project_to_2d( project_to_3d( uv ))"""
    cam = _build_test_camera(**cam_opts)
    uv_raw = _generate_uv_raw(cam.width, cam.height)

    pts3d = cam.project_pixel_to_3d_ray( uv_raw, distorted=distorted )
    uv_unrect = cam.project_3d_to_pixel( pts3d, distorted=distorted )
    assert uv_raw.shape == uv_unrect.shape
    assert np.allclose(uv_raw, uv_unrect, atol=1.0) # within one pixel

def test_extrinsic_msg():
    all_options = get_default_options()
    for opts in all_options:
        yield check_extrinsic_msg, opts

def check_extrinsic_msg(cam_opts):
    """check that ROS message contains actual camera extrinsic parameters"""
    cam_opts = cam_opts.copy()
    cam_opts['get_input_data']=True
    r = _build_test_camera(**cam_opts)
    cam = r['cam']
    tfm = cam.get_extrinsics_as_bunch()
    if 'translation' in r:
        assert np.allclose(point_msg_to_tuple(tfm.translation), point_msg_to_tuple(r['translation']))
    if 'rotation' in r:
        assert np.allclose(parse_rotation_msg(tfm.rotation,force_matrix=True),
                           parse_rotation_msg(r['rotation'],force_matrix=True))

def test_build_from_M():
    all_options = get_default_options()
    for opts in all_options:
        yield check_built_from_M, opts

def check_built_from_M(cam_opts):
    """check that M is preserved in load_camera_from_M() factory"""
    cam_orig = _build_test_camera(**cam_opts)
    if cam_orig.is_distorted_and_skewed():
        raise SkipTest('do not expect that skewed camera passes this test')
    M_orig = cam_orig.M
    cam = CameraModel.load_camera_from_M( M_orig )
    assert np.allclose( cam.M, M_orig)

def test_align():
    all_options = get_default_options()
    for opts in all_options:
        yield check_align, opts

def check_align(cam_opts):

    cam_orig = _build_test_camera(**cam_opts)
    M_orig = cam_orig.M
    cam_orig = CameraModel.load_camera_from_M( M_orig )
    R1 = np.eye(3)
    R2 = np.zeros((3,3))
    R2[0,1] = 1
    R2[1,0] = 1
    R2[2,2] = -1
    t1 = np.array( (0.0, 0.0, 0.0) )
    t2 = np.array( (0.0, 0.0, 0.1) )
    t3 = np.array( (0.1, 0.0, 0.0) )
    for s in [1.0, 2.0]:
        for R in [R1, R2]:
            for t in [t1, t2, t3]:
                cam_actual = cam_orig.get_aligned_camera( s, R, t )
                M_expected = mcsc_align.align_M( s,R,t, M_orig )
                cam_expected = CameraModel.load_camera_from_M( M_expected )
                assert cam_actual==cam_expected

def test_problem_M():
    """check a particular M which previously caused problems"""
    # This M (found by the DLT method) was causing me problems.
    d = {'width': 848,
         'name': 'camera',
         'height': 480}
    M =  np.array([[ -1.70677031e+03,  -4.10373295e+03,  -3.88568028e+02, 6.89034515e+02],
                   [ -6.19019195e+02,  -1.01292091e+03,  -2.67534989e+03, 4.51847857e+02],
                   [ -4.52548832e+00,  -3.78900498e+00,  -7.35860226e-01, 1.00000000e+00]])
    cam = CameraModel.load_camera_from_M( M, **d)

    #assert np.allclose( cam.M, M) # we don't expect this since the intrinsic matrix may not be scaled

    verts = np.array([[ 0.042306,  0.015338,  0.036328, 1.0],
                      [ 0.03323,   0.030344,  0.041542, 1.0],
                      [ 0.036396,  0.026464,  0.052408, 1.0]])

    actual = cam.project_3d_to_pixel(verts[:,:3])

    expectedh = np.dot( M, verts.T )
    expected = (expectedh[:2]/expectedh[2]).T
    assert np.allclose( expected, actual )

def test_distortion_yamlfile_roundtrip():
    all_options = get_default_options()
    for opts in all_options:
        yield check_distortion_yamlfile_roundtrip, opts

def check_distortion_yamlfile_roundtrip(cam_opts):
    """check that roundtrip of camera model to/from a yaml file works"""
    cam = _build_test_camera(**cam_opts)
    fname = tempfile.mktemp(suffix='.yaml')
    cam.save_intrinsics_to_yamlfile(fname)
    try:
        cam2 = CameraModel.load_camera_from_file( fname, extrinsics_required=False )
    finally:
        os.unlink(fname)

    distorted = np.array( [[100.0,100],
                           [100,200],
                           [100,300],
                           [100,400]] )
    orig_undistorted = cam.undistort( distorted )
    reloaded_undistorted = cam2.undistort( distorted )
    assert np.allclose( orig_undistorted, reloaded_undistorted )

def test_camera_mirror_projection_roundtrip():
    all_options = get_default_options()
    for axis in ('lr','ud'):
        for distorted in (True,False):
            for opts in all_options:
                yield check_camera_mirror_projection_roundtrip, opts, distorted, axis

def check_camera_mirror_projection_roundtrip(cam_opts,distorted=False,axis='lr'):
    """check that a mirrored camera gives reflected pixel coords"""
    cam_orig = _build_test_camera(**cam_opts)
    try:
        cam_mirror = cam_orig.get_mirror_camera(axis=axis)
    except NotImplementedError as err:
        raise SkipTest(str(err))
    uv_raw = _generate_uv_raw(cam_orig.width, cam_orig.height)

    # Get a collection of 3D points for which we know the pixel location of
    pts3d = cam_orig.project_pixel_to_3d_ray( uv_raw, distorted=distorted )
    # Now project that through our mirror-image camera.
    uv_mirror = cam_mirror.project_3d_to_pixel( pts3d, distorted=distorted )
    # Which points should be xnew = (width-x)
    expected = np.array(uv_raw)
    if axis=='lr':
        expected[:,0] = cam_orig.width - uv_raw[:,0]
    else:
        expected[:,1] = cam_orig.height - uv_raw[:,1]
    assert expected.shape == uv_mirror.shape
    assert np.allclose(expected, uv_mirror, atol=1.0) # within one pixel

def test_flip():
    all_options = get_default_options()
    for opts in all_options:
        yield check_flip, opts

def check_flip(cam_opts):
    cam_orig = _build_test_camera(**cam_opts)
    try:
        cam_flip = cam_orig.get_flipped_camera()
    except NotImplementedError as err:
        raise SkipTest(str(err))

    # They have different orientation (but same position) in space,
    assert not np.allclose( cam_orig.get_rotation(), cam_flip.get_rotation())
    assert np.allclose( cam_orig.get_camcenter(), cam_flip.get_camcenter())

    eye, lookat, up = cam_orig.get_view()
    eye2, lookat2, up2 = cam_flip.get_view()

    assert not np.allclose( lookat, lookat2 )

    # but they project 3D points to same pixel locations
    verts = np.array([[ 0.042306,  0.015338,  0.036328],
                      [ 1.03323,   2.030344,  3.041542],
                      [ 0.03323,   0.030344,  0.041542],
                      [ 0.036396,  0.026464,  0.052408]])

    expected = cam_orig.project_3d_to_pixel(verts)
    actual   = cam_flip.project_3d_to_pixel(verts)
    assert np.allclose( expected, actual )

def test_view():
    all_options = get_default_options()
    for opts in all_options:
        yield check_view, opts

def check_view(cam_opts):
    """check that we can reset camera extrinsic parameters"""

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

    # check a case that was previously failing
    n=6
    x = np.linspace(0, 2*n, n)
    theta = np.linspace(0, 2*np.pi, n)
    dim = 5.0
    for i in range(n):
        center = np.array( (x[i], 0.0, dim) )
        lookat = center + np.array( (0,1,0))
        up = -np.sin(theta[i]), 0, np.cos(theta[i])

        cam_new2 = cam_orig.get_view_camera(eye=center, lookat=lookat)

    # check a pathological case
    center= [ 0.,  0.,  5.]
    lookat= [ 0.,  1.,  5.]
    up = [0,-1,0]
    try:
        cam_new3 = cam_orig.get_view_camera(eye=center, lookat=lookat, up=up)
    except AssertionError as err:
        # we should get this exception
        pass
    else:
        assert 1==0, "did not fail test"

def test_camcenter():
    """check that our idea of camera center is theoretically expected value"""
    all_options = get_default_options()
    for opts in all_options:
        cam = _build_test_camera(**opts)
        assert np.allclose( cam.get_camcenter(), cam.t_inv.T )

def test_stages():
    all_options = get_default_options()
    for distorted in (True,False):
        for opts in all_options:
            yield check_stages, opts, distorted

def check_stages(cam_opts, distorted=False):
    """check the sum of all stages equals all stages summed"""
    cam = _build_test_camera(**cam_opts)

    uv_raw = _generate_uv_raw(cam.width, cam.height)
    pts3d = cam.project_pixel_to_3d_ray( uv_raw, distorted=distorted )

    # case 1: direct projection to pixels
    direct = cam.project_3d_to_pixel( pts3d, distorted=distorted )

    # case 2: project to camera frame, then to pixels
    cam_frame = cam.project_3d_to_camera_frame(pts3d)
    indirect = cam.project_camera_frame_to_pixel(cam_frame, distorted=distorted)

    assert np.allclose(direct, indirect)

def test_simple_camera():
    center = np.array( (0, 0.0, 5) )
    lookat = center + np.array( (0,1,0))
    cam = CameraModel.load_camera_simple(fov_x_degrees=90,
                                         eye=center,
                                         lookat=lookat)

def test_equality():
    center = np.array( (0, 0.0, 5) )
    lookat = center + np.array( (0,1,0))
    cam_apple1 = CameraModel.load_camera_simple(fov_x_degrees=90,
                                                eye=center,
                                                lookat=lookat,
                                                name='apple')
    cam_apple2 = CameraModel.load_camera_simple(fov_x_degrees=90,
                                                eye=center,
                                                lookat=lookat,
                                                name='apple')
    cam_orange = CameraModel.load_camera_simple(fov_x_degrees=30,
                                                eye=center,
                                                lookat=lookat,
                                                name='orange')
    assert cam_apple1==cam_apple2
    assert cam_apple1!=cam_orange
    assert not cam_apple1==cam_orange

def test_pickle_roundtrip():
    all_options = get_default_options()
    for opts in all_options:
        yield check_pickle_roundtrip, opts

def check_pickle_roundtrip(cam_opts):
    cam = _build_test_camera(**cam_opts)
    buf = pickle.dumps(cam)
    cam2 = pickle.loads(buf)
    assert cam==cam2

def test_camcenter_like():
    all_options = get_default_options()
    for opts in all_options:
        yield check_camcenter_like, opts

def check_camcenter_like(cam_opts):
    cam = _build_test_camera(**cam_opts)
    cc_expected = cam.get_camcenter()
    for n in range(4):
        nparr = np.zeros( (n,3), dtype=np.float )
        cc = cam.camcenter_like( nparr )
        for i in range(n):
            this_cc = cc[i]
            assert np.allclose(cc_expected,this_cc)
