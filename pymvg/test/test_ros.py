import os, tempfile
import numpy as np

from pymvg import CameraModel
from pymvg.test.utils import _build_test_camera, get_default_options

PYMVG_EMULATE_ROS = int(os.environ.get('PYMVG_EMULATE_ROS','0'))

if PYMVG_EMULATE_ROS:
    from ..ros_compat import sensor_msgs, geometry_msgs, rosbag, roslib
else:
    import roslib # set environment variable PYMVG_EMULATE_ROS to emulate ROS
    roslib.load_manifest('sensor_msgs')
    roslib.load_manifest('geometry_msgs')
    roslib.load_manifest('rosbag')
    import sensor_msgs.msg
    import geometry_msgs
    import rosbag

def test_bagfile_roundtrip():
    all_options = get_default_options()
    for opts in all_options:
        yield check_bagfile_roundtrip, opts

def check_bagfile_roundtrip(cam_opts):
    """check that roundtrip of camera model to/from a bagfile works"""
    cam = _build_test_camera(**cam_opts)
    fname = tempfile.mktemp(suffix='.bag')
    try:
        with open(fname,mode='wb') as fd:
            cam.save_to_bagfile(fd)

        with open(fname,mode='r') as fd:
            cam2 = CameraModel.load_camera_from_bagfile( fd )
    finally:
        os.unlink(fname)

    verts = np.array([[ 0.042306,  0.015338,  0.036328],
                      [ 0.03323,   0.030344,  0.041542],
                      [ 0.03323,   0.030344,  0.041542],
                      [ 0.03323,   0.030344,  0.041542],
                      [ 0.036396,  0.026464,  0.052408]])

    expected =  cam.project_3d_to_pixel(verts)
    actual   = cam2.project_3d_to_pixel(verts)
    assert np.allclose( expected, actual )


