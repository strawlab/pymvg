import flydra.reconstruct
import os
import numpy as np

def _get_cal_sources():
    test_dir = os.path.split( __file__ )[0]
    result = [os.path.join(test_dir,f) for f in ['LaSelvaCal2_2-28.xml',
                                                 'no_skew.xml', # fake data
                                                 ]]
    return result

def test_it():
    for cal_source in _get_cal_sources():
        yield check_it, cal_source

def check_it(cal_source):
    flydra_cams = flydra.reconstruct.Reconstructor(cal_source)
    pymvg_cams = flydra_cams.convert_to_pymvg()

    for method_name in ['distort',
                        'undistort']:
        for pt in [ (320.0, 240.0), # float
                    (320, 240), # int
                    (0, 0), # in corner
                    ]:
            for cam_id in flydra_cams.cam_ids[:1]:
                flydra_coords = getattr(flydra_cams,method_name)(cam_id, pt)
                pymvg_coords  = getattr(pymvg_cams.get_camera(cam_id),method_name)([pt])
                if 0:
                    #if not np.allclose( flydra_coords, pymvg_coords, atol=1.0 ):
                    print 'cam_id',cam_id
                    print 'cal_source',cal_source
                    print 'flydra_coords',flydra_coords
                    print 'pymvg_coords',pymvg_coords
                    print method_name
                    print
                assert np.allclose( flydra_coords, pymvg_coords, atol=1.0 )

    # center = (10,20,30)
    # lookat = (11,20,30)
    # up = (0,-1,0)

    # for i,(name,cam) in enumerate(base.get_camera_dict().iteritems()):
    #     cam2 = cam.get_view_camera(center,lookat,up)
