#!/usr/bin/env python

import numpy as np
from utils import _build_points_3d, make_pmat
import fill_polygon
import scipy.misc
import tarfile, time, os, StringIO
import subprocess
import cv # ubuntu: apt-get install python-opencv

DRAW=int(os.environ.get('DRAW','0'))
if DRAW:
    import matplotlib.pyplot as plt

D2R = np.pi/180.0
R2D = 1/D2R

# ROS imports
import roslib; roslib.load_manifest('camera_model')
import tf.transformations
# import geometry_msgs
# import sensor_msgs

from camera_model.camera_model import get_rotation_matrix_and_quaternion
import camera_model

roslib.load_manifest('camera_calibration')
import camera_calibration.calibrator

def get_np_array_as_png_buf(im):
    output = StringIO.StringIO()
    pil_im = scipy.misc.toimage( im )
    pil_im.save( output, format='PNG')
    return output.getvalue()

def png_buf_to_opencv(filedata):
    imagefiledata = cv.CreateMat(1, len(filedata), cv.CV_8UC1)
    cv.SetData(imagefiledata, filedata, len(filedata))
    return cv.DecodeImageM(imagefiledata)

def np_image_to_opencv(im):
    return png_buf_to_opencv(get_np_array_as_png_buf(im))

def draw_checkerboard(check_pixels,cw,ch,imw,imh):
    assert len(check_pixels)==(cw*ch)
    x = check_pixels[:,0]
    y = check_pixels[:,1]
    assert np.alltrue( (0<=x) & (x<imw) ), 'fail: %f %f'%(np.min(x), np.max(x))
    assert np.alltrue( (0<=y) & (y<imh) ), 'fail: %f %f'%(np.min(y), np.max(y))
    canvas = 0.5*np.ones( (imh,imw) )
    for col in range(cw-1):
        for row in range(ch-1):

            if (row%2):
                color = (col%2)
            else:
                color = (col+1)%2

            llidx = (row*cw) + col
            lridx = llidx+1
            ulidx = llidx+cw
            uridx = ulidx+1
            ll = check_pixels[llidx]
            lr = check_pixels[lridx]
            ul = check_pixels[ulidx]
            ur = check_pixels[uridx]

            pts = [ ll, ul, ur, lr]
            fill_polygon.fill_polygon(pts,canvas,fill_value=color)
    return canvas

class ROSPipelineMimic:
    def generate_camera(self):
        (width,height)=(self.width,self.height)=(640,480)
        center = 1,2,3
        rot_axis = np.array((4,5,6.7))
        rot_axis = rot_axis / np.sum(rot_axis**2)
        rquat = tf.transformations.quaternion_about_axis(0.1, (rot_axis.tolist()))
        rmat,_ = get_rotation_matrix_and_quaternion(rquat)

        parts = make_pmat( 1234.56, width, height,
                           rmat, center)
        self.cal_parts = parts
        self.cam = camera_model.load_camera_from_pmat(parts['pmat'],
                                                      width=width,height=height)

    def generate_images(self):
        """make checkerboard images in camera view"""
        max_theta = 100.0*D2R
        axis = (0,1,0)
        self.check_w = 8
        self.check_h = 6
        checkerboard_w = self.check_w+2
        checkerboard_h = self.check_h+2
        self.check_size = 0.024

        base_cc_x=(np.arange(checkerboard_w)-checkerboard_w/2.0)*self.check_size
        base_cc_y=(np.arange(checkerboard_h)-checkerboard_h/2.0)*self.check_size
        base_cc = []
        save_idx = []
        for i,y in enumerate(base_cc_y):
            for j,x in enumerate(base_cc_x):
                if (i>0) and (i<checkerboard_h-1):
                    if (j>0) and (j<checkerboard_w-1):
                        # save indices of actual checkerboard corners
                        save_idx.append(len(base_cc))
                base_cc.append( (x,y,0) )
        save_idx = np.array(save_idx)

        base_cc = np.array(base_cc).T
        self.db = []

        center_pix = (self.cam.width/2.0, self.cam.height/2.0)
        n_images = 20
        for i in range(n_images):
            dist = 0.9 + 0.1*(i%3)
            theta = i/float(n_images-1)*max_theta - max_theta*0.5

            rquat = tf.transformations.quaternion_about_axis(theta, axis)
            rmat,_ = get_rotation_matrix_and_quaternion(rquat)

            this_cc = np.dot(rmat,base_cc)# + np.array([(0,0,-dist)]).T

            first_pixel = np.array( center_pix, copy=True )
            atmp = i*np.pi/2.
            dir_offset = np.array((np.cos(atmp), np.sin(atmp)))
            offset = dir_offset*40.0
            first_pixel += offset
            first_pixel.shape = (1,2)
            first_3d = self.cam.project_pixel_to_3d_ray(first_pixel,
                                                        distorted=True,
                                                        distance=dist )
            check_3d = this_cc.T + first_3d

            check_pixels = self.cam.project_3d_to_pixel(check_3d,distorted=True)
            im = draw_checkerboard(check_pixels,checkerboard_w,checkerboard_h,
                                   self.cam.width,self.cam.height)
            imsave = np.empty( (self.cam.height, self.cam.width, 3),
                               dtype=np.uint8)
            for chan in range(3):
                imsave[:,:,chan] = (im*255).astype(np.uint8)

            wcs3d = check_3d[save_idx] # world coords
            ccs3d = np.dot( self.cam.rot, wcs3d.T ).T + self.cam.translation
            ccs2d = check_pixels[save_idx] # pixel coords
            if DRAW:
                scipy.misc.imsave( 'im%03d.png'%i, imsave )
            self.db.append( {'wc':wcs3d, 'cc':ccs3d, 'pix':ccs2d, 'im':imsave })

    def save_tarball(self,tarball_fname):
        def taradd(name, buf):
            s = StringIO.StringIO(buf)
            ti = tarfile.TarInfo(name)
            ti.size = len(s.buf)
            ti.uname = 'calibrator'
            ti.mtime = int(time.time())
            tarf.addfile(tarinfo=ti, fileobj=s)

        tarf = tarfile.open(tarball_fname, 'w:gz')
        for i,imd in enumerate(self.db):
            name = "left-%04d.png" % i
            buf = get_np_array_as_png_buf(imd['im'])
            taradd(name, buf)

    def run_ros_calibrator_subprocess(self,tar_fname):
        cmd = ('rosrun camera_calibration tarfile_calibration.py %s '
               '--mono --size=%dx%d --square=%f'%(tar_fname,
                                                  self.check_w,self.check_h,
                                                  self.check_size))
        subprocess.check_call( cmd, shell=True)

    def run_ros_calibrator(self):
        info = camera_calibration.calibrator.ChessboardInfo()
        info.dim = self.check_size
        info.n_cols = self.check_w
        info.n_rows = self.check_h
        boards=[info]
        cal = camera_calibration.calibrator.MonoCalibrator(boards)
        cal.size = (self.width,self.height)
        perfectcal = camera_calibration.calibrator.MonoCalibrator(boards)
        perfectcal.size = (self.width,self.height)

        goodcorners = []
        perfectcorners = []
        for imd in self.db:
            ok, corners, board = cal.get_corners(np_image_to_opencv(imd['im']))
            if not ok:
                continue
            cnp = np.array(corners)
            myp = imd['pix']
            dist = np.sqrt(np.sum((cnp-myp)**2,axis=1))
            mean_dist = np.mean(dist)
            if mean_dist > 2:
                raise ValueError('checkboard corner localization failed')
            if DRAW:
                f = plt.figure()
                ax = f.add_subplot(111)
                ax.plot(cnp[:,0],cnp[:,1],'r+',label='cv')
                ax.plot(myp[:,0],myp[:,1],'bx',label='truth')
            goodcorners.append( (corners,board) )
            perfectcorners.append( ([(x,y) for x,y in imd['pix']], board) )

        cal.cal_fromcorners(goodcorners)
        msg = cal.as_message()

        perfectcal.cal_fromcorners(perfectcorners)
        msg2 = perfectcal.as_message()
        return {'good':msg, 'perfect':msg2}

    def calc_mean_reproj_error(self,msg):
        ros_cam = camera_model.CameraModel(intrinsics=msg)
        all_ims = []
        for imd in self.db:
            ros_pix  = ros_cam.project_3d_to_pixel(imd['cc'], distorted=True)
            d = (ros_pix-imd['pix'])**2
            drows = np.sqrt(np.sum(d, axis=1))
            mean_d = np.mean(drows)
            all_ims.append(mean_d)
        mean_err = np.mean(all_ims)
        return mean_err

def test_ros_pipeline():
    pm = ROSPipelineMimic()
    pm.generate_camera()
    pm.generate_images()
    #pm.save_tarball('/tmp/pipeline-mimic.tar.gz') # optional
    cals = pm.run_ros_calibrator()
    assert pm.calc_mean_reproj_error(cals['perfect']) < 1.0
    assert pm.calc_mean_reproj_error(cals['good']) < 5.0

    if DRAW:
        plt.show()


if __name__=='__main__':
    test_ros_pipeline()
