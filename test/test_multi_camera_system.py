import numpy as np
from pymvg import CameraModel, MultiCameraSystem

def make_default_system():
    '''helper function to generate an instance of MultiCameraSystem'''
    lookat = np.array( (0.0, 0.0, 0.0))

    center1 = np.array( (0.0, 0.0, 5.0) )
    cam1 = CameraModel.load_camera_simple(name='cam1',
                                          fov_x_degrees=90,
                                          eye=center1,
                                          lookat=lookat,
                                          )

    center2 = np.array( (0.5, 0.0, 0.0) )
    cam2 = CameraModel.load_camera_simple(name='cam2',
                                          fov_x_degrees=90,
                                          eye=center2,
                                          lookat=lookat,
                                          )

    center3 = np.array( (0.5, 0.5, 0.0) )
    cam3 = CameraModel.load_camera_simple(name='cam3',
                                          fov_x_degrees=90,
                                          eye=center3,
                                          lookat=lookat,
                                          )

    center4 = np.array( (0.5, 0.0, 0.5) )
    cam4 = CameraModel.load_camera_simple(name='cam4',
                                          fov_x_degrees=90,
                                          eye=center4,
                                          lookat=lookat,
                                          )

    cameras = [cam1,cam2,cam3,cam4]
    system = MultiCameraSystem(cameras)
    return system

def get_default_points():
    pts_3d = [ (0.0, 0.0, 0.0),
               (0.1, 0.0, 0.0),
               (0.0, 0.1, 0.0),
               (0.0, 0.0, 0.1),
               (0.1, 0.1, 0.0),
               (0.1, 0.0, 0.1),
               (0.0, 0.1, 0.1),
               ]
    return np.array(pts_3d)

def test_roundtrip():
    system = make_default_system()
    pts_3d = get_default_points()
    for expected in pts_3d:
        pts = []
        for name in system.get_names():
            pt2d = system.find2d( name, expected )
            tup = (name, pt2d)
            pts.append(tup)

        actual = system.find3d( pts )
        assert np.allclose( expected, actual )

def test_single_and_multiple_points_find2d():
    '''ensure that find2d works with single points and multiple points'''
    system = make_default_system()
    pts_3d = get_default_points()

    name = system.get_names()[0]
    pt_3d = pts_3d[0]

    single_2d = system.find2d( name, pt_3d )
    assert single_2d.ndim==1

    multiple_2d = system.find2d( name, pts_3d )
    assert multiple_2d.ndim==2
    assert multiple_2d.shape[0]==2
    assert multiple_2d.shape[1]==pts_3d.shape[0]

def test_roundtrip_to_dict():
    system1 = make_default_system()
    system2 = MultiCameraSystem.from_dict( system1.to_dict() )
    assert system1==system2

def test_align():
    system1 = make_default_system()
    system2 = system1.get_aligned_copy( system1 ) # This should be a no-op.
    assert system1==system2
