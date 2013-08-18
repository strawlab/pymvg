#!/usr/bin/env python
from pymvg import CameraModel, MultiCameraSystem
import os
import tempfile
import shutil
from collections import defaultdict
import numpy as np

def make_default_system(with_separate_distorions=False):
    '''helper function to generate an instance of MultiCameraSystem'''
    if with_separate_distorions:
        raise NotImplementedError
    camxs = np.linspace(-2,2,5)
    camzs = np.linspace(-2,2,5).tolist()
    camzs.pop(2)
    cameras = []
    lookat = np.array( (0.0,0,0) )
    up = np.array( (0.0,0,1) )

    for enum,camx in enumerate(camxs):
        center = np.array( (camx, -5, 0) )
        name = 'camx_%d'%(enum+1,)
        cam = CameraModel.load_camera_simple(name=name,
                                             fov_x_degrees=45,
                                             eye=center,
                                             lookat=lookat,
                                             up=up,
                                             )
        cameras.append(cam)


    for enum,camz in enumerate(camzs):
        center = np.array( (0, -5, camz) )
        name = 'camz_%d'%(enum+1,)
        cam = CameraModel.load_camera_simple(name=name,
                                             fov_x_degrees=45,
                                             eye=center,
                                             lookat=lookat,
                                             up=up,
                                             )
        cameras.append(cam)

    system = MultiCameraSystem(cameras)
    return system

def test_load_mcsc():
    mydir = os.path.dirname(__file__)
    mcsc_dirname = os.path.join(mydir,'mcsc_output_20130726')
    cam_system = MultiCameraSystem.from_mcsc( mcsc_dirname, max_skew_ratio=10 )

def get_default_points():
    N = 500
    pts_3d = 0.1*np.random.randn(N,3)
    return pts_3d

def test_mcsc_roundtrip():
    for with_rad_files in [False]:#,True]:
        yield check_mcsc_roundtrip, with_rad_files

def check_mcsc_roundtrip(with_rad_files=False):
    import multicamselfcal.execute as mcsce

    np.random.seed(3)
    mcscdir = os.path.join( os.path.dirname(mcsce.__file__),
                            '..', '..', 'MultiCamSelfCal' )
    mcscdir = os.path.abspath(mcscdir)
    out_dirname = tempfile.mkdtemp()
    try:
        mcsc = mcsce.MultiCamSelfCal(out_dirname=out_dirname,mcscdir=mcscdir)
        if with_rad_files:
            system = make_default_system(with_separate_distorions=True)
        else:
            system = make_default_system()

        cam_resolutions = dict(  (n['name'],  (n['width'],n['height']))
                               for n in system.to_dict()['camera_system'])

        pts_3d = get_default_points()

        if 0:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            from pymvg.plot_utils import plot_camera
            np.set_printoptions(precision=3, suppress=True)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for name,cam in system.get_camera_dict().iteritems():
                print
                print name,'-'*80
                print '  center:',cam.get_camcenter()
                print '  lookat:',cam.get_lookat()
                print '  up:',cam.get_up()
                print '  P'
                print cam.P
                print
                plot_camera( ax, cam)#, scale = dim/5.0 )
            ax.plot( pts_3d[:,0], pts_3d[:,1], pts_3d[:,2], 'k.')
            plt.show()

        cam_points=defaultdict(list)

        nan_tup = (np.nan, np.nan)
        for pt_3d in pts_3d:
            valid = True
            for name in system.get_names():
                w,h = cam_resolutions[name]

                x,y = system.find2d( name, pt_3d )

                if 1:
                    xn, yn = 1.0*np.random.randn( 2 )

                    x += xn
                    y += yn

                if (0 <= x) and (x < w) and (0 <= y) and (y < h):
                    cam_points[name].append((x,y))
                else:
                    cam_points[name].append(nan_tup)
                    valid = False
            if not valid:
                for name in system.get_names():
                    cam_points[name].pop()

        cam_ids=system.get_names()
        if with_rad_files:
            cam_calibrations = {} # dictionary of .yaml filenames with ROS yaml format
            raise NotImplementedError
        else:
            cam_calibrations = {}
        mcsc.create_from_cams(cam_ids=cam_ids,
                              cam_resolutions=cam_resolutions,
                              cam_points=cam_points,
                              cam_calibrations=cam_calibrations,
                              )

        result = mcsc.execute(silent=True)
        # FIXME TODO: finish this test!
    finally:
        shutil.rmtree(out_dirname)

