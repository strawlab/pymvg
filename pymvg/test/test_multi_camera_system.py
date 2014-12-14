import numpy as np
import pymvg
from pymvg.camera_model import CameraModel
from pymvg.multi_camera_system import MultiCameraSystem, build_example_system
import tempfile, os

import nose.tools

def make_default_system():
    '''helper function to generate an instance of MultiCameraSystem'''
    lookat = np.array( (0.0, 0.0, 0.0))

    center1 = np.array( (0.0, 0.0, 5.0) )
    distortion1 = np.array( [0.2, 0.3, 0.1, 0.1, 0.1] )
    cam1 = CameraModel.load_camera_simple(name='cam1',
                                          fov_x_degrees=90,
                                          eye=center1,
                                          lookat=lookat,
                                          distortion_coefficients=distortion1,
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

def test_no_duplicate_names():
    cam1a = CameraModel.load_camera_simple(name='cam1')
    cam1b = CameraModel.load_camera_simple(name='cam1')
    cams = [cam1a,cam1b]
    nose.tools.assert_raises(ValueError, MultiCameraSystem, cams)

def test_equals():
    system = make_default_system()
    assert system != 1234

    system2 = MultiCameraSystem([CameraModel.load_camera_simple(name='cam%d'%i) for i in range(2)])
    system3 = MultiCameraSystem([CameraModel.load_camera_simple(name='cam%d'%i) for i in range(3)])
    assert system2 != system3

    system4 = make_default_system()
    d = system4.to_dict()
    d['camera_system'][0]['width'] += 1
    system5 = MultiCameraSystem.from_dict( d )
    assert system4 != system5

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

def test_getters():
    system1 = make_default_system()
    d = system1.to_dict()
    names1 = list(system1.get_camera_dict().keys())
    names2 = [cd['name'] for cd in d['camera_system']]
    assert set(names1)==set(names2)

    for idx in range(len(names1)):
        cam1 = system1.get_camera( names1[idx] )
        cam2 = CameraModel.from_dict(d['camera_system'][idx])
        assert cam1==cam2

def test_roundtrip_to_pymvg_file():
    system1 = make_default_system()
    fname = tempfile.mktemp(suffix='.json')
    system1.save_to_pymvg_file( fname )
    try:
        system2 = MultiCameraSystem.from_pymvg_file( fname )
        assert system1==system2
    finally:
        os.unlink( fname )

def test_pymvg_file_in_docs():
    pymvg_src_path = pymvg.__file__
    pymvg_base = os.path.split(pymvg_src_path)[0]
    pymvg_src_dir = os.path.join(pymvg_base,'..')
    fname =  os.path.join( pymvg_src_dir, 'docs', 'source', 'pymvg_camsystem_example.json')
    system = MultiCameraSystem.from_pymvg_file( fname )

def test_roundtrip_to_str():
    system1 = make_default_system()
    buf = system1.get_pymvg_str()
    system2 = MultiCameraSystem.from_pymvg_str( buf )
    assert system1==system2

def test_align():
    system1 = make_default_system()
    system2 = system1.get_aligned_copy( system1 ) # This should be a no-op.
    assert system1==system2

def test_align():
    system1 = make_default_system()
    system2 = system1.get_aligned_copy( system1 ) # This should be a no-op.
    assert system1==system2

    system3 = MultiCameraSystem([CameraModel.load_camera_simple(name='cam%d'%i) for i in range(2)])
    nose.tools.assert_raises(ValueError, system3.get_aligned_copy, system1)

def test_build_example_system():
    for n in range(2,100,5):
        system = build_example_system(n=n)
        assert n==len(system.get_names())

def test_load_mcsc():
    mydir = os.path.dirname(__file__)
    mcsc_dir = os.path.join(mydir,'external','mcsc')
    mcsc_dirname = os.path.join(mcsc_dir,'mcsc_output_20130726')
    cam_system = MultiCameraSystem.from_mcsc( mcsc_dirname )
