import numpy as np
from pymvg.camera_model import CameraModel
from pymvg.multi_camera_system import MultiCameraSystem
import sys

def make_default_system():
    '''helper function to generate an instance of MultiCameraSystem'''
    lookat = np.array( (0.0, 0.0, 0.0))

    center1 = np.array( (0.0, 0.0, 0.9) )
    distortion1 = np.array( [0.2, 0.3, 0.1, 0.1, 0.1] )
    cam1 = CameraModel.load_camera_simple(name='cam1',
                                          fov_x_degrees=90,
                                          eye=center1,
                                          lookat=lookat,
                                          distortion_coefficients=distortion1,
                                          )

    center2 = np.array( (0.5, -0.8, 0.0) )
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

    system = MultiCameraSystem([cam1,cam2,cam3])
    return system

def main():
    out_fname = sys.argv[1]
    system1 = make_default_system()
    system1.save_to_pymvg_file( out_fname )

    # just make sure we can actually read it!
    system2 = MultiCameraSystem.from_pymvg_file( out_fname )
    assert system1==system2

if __name__=='__main__':
    main()
