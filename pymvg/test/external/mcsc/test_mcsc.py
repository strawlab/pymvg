#!/usr/bin/env python
from __future__ import print_function
from pymvg.camera_model import CameraModel
from pymvg.multi_camera_system import MultiCameraSystem
import os
import tempfile
import shutil
from collections import defaultdict
import numpy as np
import multicamselfcal.execute as mcsce

def make_default_system(with_separate_distorions=False):
    '''helper function to generate an instance of MultiCameraSystem'''
    if with_separate_distorions:
        raise NotImplementedError
    camxs = np.linspace(-2,2,3)
    camzs = np.linspace(-2,2,3).tolist()
    camzs.pop(1)
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

def get_default_points():
    N = 500
    pts_3d = 1.0*np.random.randn(N,3)
    return pts_3d

def is_close(sys1,sys2,pts_3d):
    names = sys1.get_names()
    names2 = sys2.get_names()
    assert names==names2

    for pt_3d in pts_3d:
        for name in names:
            c1 = sys1.find2d( name, pt_3d )
            c2 = sys2.find2d( name, pt_3d )
            print('%s: %s %s' % (name, c1, c2))
            if not np.allclose(c1,c2):
                return False
    return True

def test_mcsc_roundtrip():
    if 1:
        # mcsc python uses logging module. This sends logs to stdout.
        import logging, sys

        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        rootLogger = logging.getLogger()

        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)

    for with_rad_files in [False]:#,True]:
        for align_existing in [False]:#True,False]:
            yield check_mcsc_roundtrip, with_rad_files, align_existing

def check_mcsc_roundtrip(with_rad_files=False,align_existing=False):
    np.random.seed(3)
    mcscdir = os.path.join( os.path.dirname(mcsce.__file__),
                            '..', '..', 'MultiCamSelfCal' )
    mcscdir = os.path.abspath(mcscdir)
    out_dirname = tempfile.mkdtemp()
    try:
        mcsc = mcsce.MultiCamSelfCal(out_dirname=out_dirname,mcscdir=mcscdir)
        if with_rad_files:
            orig_cams = make_default_system(with_separate_distorions=True)
        else:
            orig_cams = make_default_system()

        cam_resolutions = dict(  (n['name'],  (n['width'],n['height']))
                               for n in orig_cams.to_dict()['camera_system'])

        pts_3d = get_default_points()

        if 0:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            from pymvg.plot_utils import plot_camera
            np.set_printoptions(precision=3, suppress=True)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for name,cam in orig_cams.get_camera_dict().iteritems():
                print()
                print(name,'-'*80)
                print('  center:',cam.get_camcenter())
                print('  lookat:',cam.get_lookat())
                print('  up:',cam.get_up())
                print('  P')
                print(cam.P)
                print()
                plot_camera( ax, cam)#, scale = dim/5.0 )
            ax.plot( pts_3d[:,0], pts_3d[:,1], pts_3d[:,2], 'k.')
            plt.show()

        cam_points=defaultdict(list)

        nan_tup = (np.nan, np.nan)
        for pt_3d in pts_3d:
            valid = True
            for name in orig_cams.get_names():
                w,h = cam_resolutions[name]

                x,y = orig_cams.find2d( name, pt_3d )

                if 1:
                    # add noise to the 2D images
                    xn, yn = 0.01*np.random.randn( 2 )

                    x += xn
                    y += yn

                if (0 <= x) and (x < w) and (0 <= y) and (y < h):
                    cam_points[name].append((x,y))
                else:
                    cam_points[name].append(nan_tup)
                    valid = False
            if not valid:
                for name in orig_cams.get_names():
                    cam_points[name].pop()

        cam_ids=orig_cams.get_names()
        if with_rad_files:
            cam_calibrations = {} # dictionary of .yaml filenames with ROS yaml format
            raise NotImplementedError
        else:
            cam_calibrations = {}

        if align_existing:
            cam_centers = np.array([ orig_cams.get_cam(n).get_camcenter() for n in orig_cams.get_names() ]).T
        else:
            cam_centers = []

        mcsc.create_from_cams(cam_ids=cam_ids,
                              cam_resolutions=cam_resolutions,
                              cam_points=cam_points,
                              cam_calibrations=cam_calibrations,
                              cam_centers=cam_centers,
                              )

        result = mcsc.execute(silent=True)
        raw_cams = MultiCameraSystem.from_mcsc( result )
        if align_existing:
            aligned_cams = raw_cams
        else:
            aligned_cams = raw_cams.get_aligned_copy( orig_cams )

        assert is_close(orig_cams,aligned_cams,pts_3d)
    finally:
        shutil.rmtree(out_dirname)

