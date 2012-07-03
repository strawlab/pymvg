#!/usr/bin/env python

# ROS imports
import roslib; roslib.load_manifest('camera_model')
import rosbag
import tf.transformations
import sensor_msgs.msg
import geometry_msgs.msg

import numpy as np

import yaml

import warnings

# helper functions ---------------

def point_msg_to_tuple(d):
    return d.x, d.y, d.z

def normalize(vec):
    mag = np.sqrt(np.sum(vec**2))
    return vec/mag

def parse_rotation_msg(rotation, force_matrix=False):
    # rotation could either be a quaternion or a 3x3 matrix

    if hasattr(rotation,'x') and hasattr(rotation,'y') and hasattr(rotation,'z') and hasattr(rotation,'w'):
        # convert quaternion message to tuple
        rotation = quaternion_msg_to_tuple(rotation)

    if len(rotation)==4:
        if force_matrix:
            rotation = tf.transformations.quaternion_matrix(rotation)[:3,:3]
        return rotation

    if len(rotation) != 9:
        raise ValueError('expected rotation to be a quaternion or 3x3 matrix')
    rotation = np.array( rotation )
    rotation.shape = 3,3
    return rotation

def quaternion_msg_to_tuple(d):
    return d.x, d.y, d.z, d.w

def _undistort( xd, yd, D):
    # See OpenCV modules/imgprc/src/undistort.cpp
    x = np.array(xd,copy=True)
    y = np.array(yd,copy=True)

    k1, k2, t1, t2, k3 = D[:5]
    k = list(D)
    if len(k)==5:
        k = k + [0,0,0]

    for i in range(5):
        r2 = x*x + y*y
        icdist = (1 + ((k[7]*r2 + k[6])*r2 + k[5])*r2)/(1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2)
        delta_x = 2.0 * (t1)*x*y + (t2)*(r2 + 2.0*x*x)
        delta_y = (t1) * (r2 + 2.0*y*y)+2.0*(t2)*x*y
        x = (xd-delta_x)*icdist
        y = (yd-delta_y)*icdist
    return x,y


def my_rq(M):
    """RQ decomposition, ensures diagonal of R is positive"""
    import scipy.linalg
    R,K = scipy.linalg.rq(M)
    n = R.shape[0]
    for i in range(n):
        if R[i,i]<0:
            # I checked this with Mathematica. Works if R is upper-triangular.
            R[:,i] = -R[:,i]
            K[i,:] = -K[i,:]
    return R,K

def center(P):
    orig_determinant = np.linalg.det
    def determinant( A ):
        return orig_determinant( np.asarray( A ) )
    # camera center
    X = determinant( [ P[:,1], P[:,2], P[:,3] ] )
    Y = -determinant( [ P[:,0], P[:,2], P[:,3] ] )
    Z = determinant( [ P[:,0], P[:,1], P[:,3] ] )
    T = -determinant( [ P[:,0], P[:,1], P[:,2] ] )

    C_ = np.array( [[ X/T, Y/T, Z/T ]] ).T
    return C_

def is_rotation_matrix(R):
    # check if rotation matrix is really a pure rotation matrix

    # test: inverse is transpose
    testI = np.dot(R.T,R)
    if not np.allclose( testI, np.eye(len(R)) ):
        return False

    # test: determinant is unity
    dr = abs(np.linalg.det(R))
    if not np.allclose(dr,1):
        return False

    # test: has eigenvalue of unity
    l, W = np.linalg.eig(R.T)
    eps = 1e-8
    i = np.where(abs(np.real(l) - 1.0) < eps)[0]
    if not len(i):
        return False
    return True

def get_rotation_matrix_and_quaternion(rotation):
    rotation = np.array(rotation)
    if rotation.ndim==2:
        assert rotation.shape==(3,3)
        if not np.alltrue(np.isnan( rotation )):
            assert is_rotation_matrix(rotation)

        rmat = rotation

        rnew = np.eye(4)
        rnew[:3,:3] = rmat
        rquat = tf.transformations.quaternion_from_matrix(rnew)
        if not np.alltrue(np.isnan( rquat )):
            R2 = tf.transformations.quaternion_matrix(rquat)[:3,:3]
            assert np.allclose(rmat,R2)
    else:
        assert rotation.ndim==1
        assert rotation.shape==(4,)
        rquat = rotation
        rmat = tf.transformations.quaternion_matrix(rquat)[:3,:3]

        if not np.alltrue(np.isnan( rmat )):
            assert is_rotation_matrix(rmat)

    return rmat, rquat


# main class

class CameraModel(object):
    """an implementation of the Camera Model used by ROS

    See http://www.ros.org/wiki/image_pipeline/CameraInfo for a
    discussion of the coordinate system used here.

    """
    __slots__ = [
        # basic properties
        'name', # a string with the camera name
        'width','height', # the size of the image

        # extrinsic parameters
        '_rquat', # the rotation quaternion, np.array with shape (4,)
        '_camcenter', # the center of the camera, np.array with shape (3,)

        # intrinsic parameters
        '_opencv_compatible',
        # these intrinsic parameters specified like OpenCV
        'P', # used for undistorted<->normalized, np.array with shape (3,4)

        # the distortion model
        'K', # (distortion params) used for distorted<->normalized, np.array with shape (3,3)
        'distortion', # (distortion params) the distortion, np.array with shape (5,1) or (8,1)
        'rect', # (distortion params) the rectification, None or np.array with shape (3,3)
        ]
    AXIS_FORWARD = np.array((0,0,1),dtype=np.float)
    AXIS_UP = np.array((0,-1,0),dtype=np.float)
    AXIS_RIGHT = np.array((1,0,0),dtype=np.float)
    def __init__(self,
                 translation=None,
                 rotation=None,
                 intrinsics=None,
                 name=None,
                 ):
        """Instantiate a Camera Model.

        params
        ------
        translation : converted to np.array with shape (3,)
          the translational position of the camera (note: not the camera center)
        rotation : converted to np.array with shape (4,) or (3,3)
          the camera orientation as a quaternion or a 3x3 rotation vector
        intrinsics : a ROS CameraInfo message
          the intrinsic camera calibration
        name : string
          the name of the camera
        """
        if translation is None:
            translation = (0,0,0)
        if rotation is None:
            rotation = np.eye(3)
        if name is None:
            name = 'camera'

        rmat, rquat = get_rotation_matrix_and_quaternion(rotation)

        t = np.array(translation)
        t.shape = 3,1
        self._camcenter = -np.dot( rmat.T, t )[:,0]
        del t

        self._rquat = rquat

        if 1:
            # Initialize the camera calibration from a CameraInfo message.
            msg = intrinsics
            self.width = msg.width
            self.height = msg.height
            shape = (msg.height, msg.width)

            self.P = np.array(msg.P,dtype=np.float)
            self.P.shape = (3,4)
            if not np.allclose(self.P[:,3], np.zeros((3,))):
                raise NotImplementedError('not tested when 4th column of P is nonzero')

            self.K = np.array( msg.K, dtype=np.float)
            self.K.shape = (3,3)
            assert self.K.ndim == 2

            self.distortion = np.array(msg.D, dtype=np.float)
            if len(self.distortion) == 5:
                self.distortion.shape = (5,1)
            elif len(self.distortion) == 8:
                self.distortion.shape = (8,1)
            else:
                raise ValueError('distortion can have only 5 or 8 entries')

            assert self.distortion.ndim==2

            self.rect = np.array( msg.R, dtype=np.float )
            self.rect.shape = (3,3)
            if np.allclose(self.rect,np.eye(3)):
                self.rect = None

        #self.translation=np.array(translation,copy=True)
        self.name = name

        K = self.P[:3,:3]
        self._opencv_compatible = (K[0,1]==0)

        # And a final check
        if K[0,1] != 0.0:
            if np.sum(abs(self.distortion)) != 0.0:
                raise NotImplementedError('distortion/undistortion for skewed pixels not implemented')

    # -------------------------------------------------
    # properties / getters

    def get_rot(self):
        R = tf.transformations.quaternion_matrix(self._rquat)[:3,:3]
        return R
    rot = property(get_rot)

    def get_Rt(self):
        t = np.array(self.translation)
        t.shape = 3,1
        Rt = np.hstack((self.rot,t))
        return Rt
    Rt = property(get_Rt)

    def get_pmat(self):
        K = self.P[:3,:3]
        pmat = np.dot( K, self.Rt )
        return pmat
    pmat = property(get_pmat)

    def get_translation(self):
        C = np.array(self._camcenter)
        C.shape = (3,1)
        t = -np.dot(self.rot, C)[:,0]
        return t
    translation = property(get_translation)

    def get_rot_inv(self):
        return np.linalg.pinv(self.rot)
    rot_inv = property(get_rot_inv)

    def get_t_inv(self):
        ti = np.array(self._camcenter)
        ti.shape = 3,1
        return ti
    t_inv = property(get_t_inv)

    # -------------------------------------------------
    # other getters

    def is_opencv_compatible(self):
        """True iff there is no skew"""
        return self._opencv_compatible

    def get_name(self):
        return self.name

    def get_extrinsics_as_msg(self):
        msg = geometry_msgs.msg.Transform()
        for i in range(3):
            setattr(msg.translation,'xyz'[i], self.translation[i] )
        for i in range(4):
            setattr(msg.rotation,'xyzw'[i], self._rquat[i] )
        return msg

    def get_intrinsics_as_msg(self):
        i = sensor_msgs.msg.CameraInfo()
        # these are from image_geometry ROS package in the utest.cpp file
        i.height = self.height
        i.width = self.width
        i.distortion_model = 'plumb_bob'
        i.D = list(self.distortion.flatten())
        i.K = list(self.K.flatten())
        i.R = list(self.get_rect().flatten())
        i.P = list(self.P.flatten())
        return i

    def get_camcenter(self):
        return self.t_inv[:,0] # drop dimension

    def get_lookat(self,distance=1.0):
        world_coords = self.project_camera_frame_to_3d( [distance*self.AXIS_FORWARD] )
        world_coords.shape = (3,) # drop dimension
        return world_coords

    def get_up(self,distance=1.0):
        world_coords = self.project_camera_frame_to_3d( [distance*self.AXIS_UP] )
        world_coords.shape = (3,) # drop dimension
        return world_coords-self._camcenter

    def get_right(self,distance=1.0):
        cam_coords = np.array([[distance,0,0]])
        world_coords = self.project_camera_frame_to_3d( [distance*self.AXIS_RIGHT] )
        world_coords.shape = (3,) # drop dimension
        return world_coords-self._camcenter

    def get_view(self):
        return self.get_camcenter(), self.get_lookat(), self.get_up()

    def get_rotation_quat(self):
        return np.array(self._rquat)

    def get_rotation(self):
        return self.rot

    def get_K(self):
        return self.K

    def get_D(self):
        return self.distortion

    def get_rect(self):
        if self.rect is None:
            return np.eye(3)
        else:
            return self.rect

    def get_P(self):
        return self.P

    def fx(self):
        return self.P[0,0]

    def fy(self):
        return self.P[1,1]

    def cx(self):
        return self.P[0,2]

    def cy(self):
        return self.P[1,2]

    def Tx(self):
        return self.P[0,3]

    def Ty(self):
        return self.P[1,3]

    def save_to_bagfile(self,fname):
        bagout = rosbag.Bag(fname, 'w')
        topic = self.name + '/tf'
        extrinsics = self.get_extrinsics_as_msg()
        bagout.write(topic, extrinsics)
        topic = self.name + '/camera_info'
        intrinsics = self.get_intrinsics_as_msg()
        bagout.write(topic, intrinsics)
        bagout.close()

    def get_mirror_camera(self):
        """return a copy of this camera whose x coordinate is (image_width-x)"""
        if 0:

            # Method 1: flip the extrinsic coordinates to a LH
            # system. (Adjust camera center for distortion.)

            # Implementation note: I guess this should work, but it is
            # not quite right for some reason.

            flipr = np.eye(3)
            flipr[0,0] = -1
            rnew = np.dot(flipr,self.rot)
            C = self.get_camcenter()
            tnew = -np.dot(rnew, C)
            i = self.get_intrinsics_as_msg()
            i.K[2] = (self.width-i.K[2])
            i.P[2] = (self.width-i.P[2])
            camnew = CameraModel( translation = tnew,
                                  rotation = rnew,
                                  intrinsics = i,
                                  name = self.name + '_mirror',
                                  )
            return camnew
        elif 1:

            # Method 2: keep extrinsic coordinates, but flip intrinsic
            # parameter so that a mirror image is rendered.

            i = self.get_intrinsics_as_msg()
            i.K[0] = -i.K[0]
            i.P[0] = -i.P[0]

            # Now, do we flip about optical center or just the X
            # coordinate?

            if 1:

                # This flips the X coordinate but preserves the
                # optical center.

                i.K[2] = (self.width-i.K[2])
                i.P[2] = (self.width-i.P[2])

            camnew = CameraModel( translation = self.translation,
                                  rotation = self.rot,
                                  intrinsics = i,
                                  name = self.name + '_mirror',
                                  )
            return camnew

    # def get_flipped_camera(self):
    #     """return a copy of this camera looking in the opposite direction

    #     The returned camera has the same 3D->2D projection. (The
    #     2D->3D projection results in a vector in the opposite
    #     direction.)
    #     """

    #     #rold = self.get_rotation()
    #     #q = tf.transformations.quaternion_from_matrix(rold)
    #     q = self.get_rotation_quat()
    #     cosa2 = q[3]
    #     theta=2*np.arccos(cosa2)
    #     axis = q[:3]

    #     eps = 1e16

    #     if 1:
    #         newtheta = theta+np.pi
    #         axlen = np.sqrt( np.sum(axis**2 ))
    #         if axlen > eps:
    #             newaxis = axis
    #         else:
    #             newaxis = np.array((1,0,0)) # arbirtrary direction
    #     else:
    #         if theta > eps:
    #             newtheta = theta
    #             newaxis = -axis
    #         else:
    #             0

    #     qnew = tf.transformations.quaternion_about_axis(newtheta, newaxis)
    #     print 'theta, axis, q', theta, axis, q
    #     print 'newtheta, newaxis, qnew', newtheta, newaxis, qnew
    #     #qnew = tf.transformations.quaternion_from_matrix(rold)
    #     rnew = tf.transformations.quaternion_matrix(qnew)[:3,:3]
    #     if not is_rotation_matrix(rnew):
    #         print 'impossible?',theta, newaxis
    #         assert 1==0

    #     # flip = -np.eye(3)
    #     # rnew = np.dot( flip, rnew)
    #     i = self.get_intrinsics_as_msg()

    #     C = self.get_camcenter()
    #     tnew = -np.dot(rnew, C)
    #     camnew = CameraModel( translation = tnew,
    #                           rotation = rnew,
    #                           intrinsics = i,
    #                           name = self.name + '_flip',
    #                           )
    #     return camnew

    def get_view_camera(self, eye, lookat, up=None):
        """return a copy of this camera with new extrinsic coordinates"""
        eye = np.array(eye); eye.shape=(3,)
        lookat = np.array(lookat); lookat.shape=(3,)
        if up is None:
            up = np.array((0,-1,0))
        lv = lookat - eye
        f = normalize(lv)
        s = normalize( np.cross( f, up ))
        u = normalize( np.cross( f, s ))

        R = np.array( [[ s[0], u[0], f[0]],
                       [ s[1], u[1], f[1]],
                       [ s[2], u[2], f[2]]]).T

        eye.shape = (3,1)
        t = -np.dot(R,eye)

        result = CameraModel(translation=t,
                             rotation=R,
                             intrinsics=self.get_intrinsics_as_msg(),
                             name=self.name,
                             )
        return result

    # --------------------------------------------------
    # image coordinate operations

    def undistort(self, nparr):
        # See http://opencv.willowgarage.com/documentation/cpp/camera_calibration_and_3d_reconstruction.html#cv-undistortpoints

        # Parse inputs
        nparr = np.array(nparr,copy=False)
        assert nparr.ndim==2
        assert nparr.shape[1]==2

        if np.sum(abs(self.distortion)) == 0.0:
            # no distortion necessary, just copy inputs
            return np.array(nparr,copy=True)

        u = nparr[:,0]
        v = nparr[:,1]

        # prepare parameters
        K = self.get_K()

        fx = K[0,0]
        cx = K[0,2]
        fy = K[1,1]
        cy = K[1,2]

        # P=[fx' 0 cx' tx; 0 fy' cy' ty; 0 0 1 tz]

        P = self.get_P()
        fxp = P[0,0]
        cxp = P[0,2]
        fyp = P[1,1]
        cyp = P[1,2]

        # Apply intrinsic parameters to get normalized, distorted coordinates
        xpp = (u-cx)/fx
        ypp = (v-cy)/fy

        # Undistort
        (xp,yp) = _undistort( xpp, ypp, self.get_D() )

        # Now rectify
        R = self.rect
        if R is None:
            x = xp
            y = yp
        else:
            assert R.shape==(3,3)
            uh = np.vstack( (xp,yp,np.ones_like(xp)) )
            XYWt = np.dot(R, uh)
            X = XYWt[0,:]
            Y = XYWt[1,:]
            W = XYWt[2,:]
            x = X/W
            y = Y/W

        # Finally, get (undistorted) pixel coordinates
        up = x*fxp + cxp
        vp = y*fyp + cyp

        return np.vstack( (up,vp) ).T

    def distort(self, nparr):
        # See http://opencv.willowgarage.com/documentation/cpp/camera_calibration_and_3d_reconstruction.html#cv-undistortpoints

        # Based on code in pinhole_camera_model.cpp of ROS image_geometry package.

        # Parse inputs
        nparr = np.array(nparr,copy=False)
        assert nparr.ndim==2
        assert nparr.shape[1]==2

        if np.sum(abs(self.distortion)) == 0.0:
            # no distortion necessary, just copy inputs
            return np.array(nparr,copy=True)

        uv_rect_x = nparr[:,0]
        uv_rect_y = nparr[:,1]

        # prepare parameters
        P = self.get_P()

        fx = P[0,0]
        cx = P[0,2]
        Tx = P[0,3]
        fy = P[1,1]
        cy = P[1,2]
        Ty = P[1,3]

        x = (uv_rect_x - cx - Tx)/fx
        y = (uv_rect_y - cy - Ty)/fy

        if self.rect is not None:
            R = self.rect.T
            xy1 = np.vstack((x,y,np.ones_like(x)))
            X,Y,W = np.dot(R, xy1)
            xp = X/W
            yp = Y/W
        else:
            xp = x
            yp = y
        r2 = xp*xp + yp*yp
        r4 = r2*r2
        r6 = r4*r2
        a1 = 2*xp*yp
        D = self.distortion
        k1 = D[0]; k2=D[1]; p1=D[2]; p2=D[3]; k3=D[4]
        barrel = 1 + k1*r2 + k2*r4 + k3*r6
        if len(D)==8:
            barrel /= (1.0 + D[5]*r2 + D[6]*r4 + D[7]*r6)
        xpp = xp*barrel + p1*a1 + p2*(r2+2*(xp*xp))
        ypp = yp*barrel + p1*(r2+2*(yp*yp)) + p2*a1;

        K = self.get_K()
        u = xpp*K[0,0] + K[0,2]
        v = ypp*K[1,1] + K[1,2]
        return np.vstack( (u,v) ).T

    # --------------------------------------------------
    # 3D <-> image coordinate operations

    def project_pixel_to_camera_frame(self, nparr, distorted=True, distance=1.0 ):
        if distorted:
            nparr = self.undistort(nparr)
        # now nparr is undistorted (aka rectified) 2d point data

        # Parse inputs
        nparr = np.array(nparr,copy=False)
        assert nparr.ndim==2
        assert nparr.shape[1]==2
        uv_rect_x = nparr[:,0]
        uv_rect_y = nparr[:,1]

        # transform to 3D point in camera frame
        x = (uv_rect_x - self.cx() - self.Tx()) / self.fx()
        y = (uv_rect_y - self.cy() - self.Ty()) / self.fy()
        z = np.ones_like(x)
        ray_cam = np.vstack((x,y,z))
        rl = np.sqrt(np.sum(ray_cam**2,axis=0))
        ray_cam = distance*(ray_cam/rl) # normalize then scale
        return ray_cam.T

    def project_pixel_to_3d_ray(self, nparr, distorted=True, distance=1.0 ):
        ray_cam = self.project_pixel_to_camera_frame( nparr, distorted=distorted, distance=distance )
        # transform to world frame
        return self.project_camera_frame_to_3d( ray_cam )

    def project_3d_to_pixel(self, pts3d, distorted=True):
        pts3d = np.array(pts3d,copy=False)
        assert pts3d.ndim==2
        assert pts3d.shape[1]==3

        # homogeneous and transposed
        pts3d_h = np.empty( (4,pts3d.shape[0]) )
        pts3d_h[:3,:] = pts3d.T
        pts3d_h[3] = 1

        # undistorted homogeneous image coords
        cc = np.dot(self.pmat, pts3d_h)

        # project
        pc = cc[:2]/cc[2]
        u, v = pc

        if distorted:
            # distort (the currently undistorted) image coordinates
            nparr = np.vstack((u,v)).T
            u,v = self.distort( nparr ).T
        return np.vstack((u,v)).T

    def project_camera_frame_to_3d(self, pts3d):
        """take 3D coordinates in camera frame and convert to world frame"""
        cam_coords = np.array(pts3d).T
        t = self.get_translation()
        t.shape = (3,1)
        world_coords = np.dot(self.rot_inv, cam_coords - t)
        return world_coords.T

    def project_3d_to_camera_frame(self, pts3d):
        """take 3D coordinates in world frame and convert to camera frame"""
        pts3d = np.array(pts3d)
        assert pts3d.ndim==2
        assert pts3d.shape[1]==3

        # homogeneous and transposed
        pts3d_h = np.empty( (4,pts3d.shape[0]) )
        pts3d_h[:3,:] = pts3d.T
        pts3d_h[3] = 1

        # undistorted homogeneous image coords
        cc = np.dot(self.Rt, pts3d_h)

        return cc.T

    # --------------------------------------------
    # misc. helpers

    def camcenter_like(self,nparr):
        nparr = np.array(nparr,copy=False)
        assert nparr.ndim==2
        assert nparr.shape[1]==3
        return np.zeros( nparr.shape ) + self.t_inv.T

# factory functions
def load_camera_from_dict(d, extrinsics_required=True ):
    #only needs w,h,P,K,D,R
    c = sensor_msgs.msg.CameraInfo(
            height=d['image_height'],
            width=d['image_width'],
            P=d['projection_matrix']['data'],
            K=d['camera_matrix']['data'],
            D=d['distortion_coefficients']['data'],
            R=d['rectification_matrix']['data'])
    
    result = CameraModel(translation=None,  #or nan???
                         rotation=None,
                         intrinsics=c,
                         name=d['camera_name'])

    return result

SUPPORTED_FILE_TYPES = ('.bag','.yaml')
def load_camera_from_file( fname, extrinsics_required=True ):
    if fname.endswith('.bag'):
        return load_camera_from_bagfile(fname, extrinsics_required)
    elif fname.endswith('.yaml'):
        with open(fname,'r') as f:
            return load_camera_from_dict(yaml.load(f), extrinsics_required)
    else:
        raise Exception("Only supports .bag and .yaml file loading")

def load_camera_from_bagfile( bag_fname, extrinsics_required=True ):
    """factory function for class CameraModel"""
    bag = rosbag.Bag(bag_fname, 'r')
    camera_name = None
    translation = None
    rotation = None
    intrinsics = None

    for topic, msg, t in bag.read_messages():
        if 1:
            parts = topic.split('/')
            if parts[0]=='':
                parts = parts[1:]
            topic = parts[-1]
            parts = parts[:-1]
            if len(parts)>1:
                this_camera_name = '/'.join(parts)
            else:
                this_camera_name = parts[0]
            # empty, this_camera_name, topic = parts
            # assert empty==''
        if camera_name is None:
            camera_name = this_camera_name
        else:
            assert this_camera_name == camera_name

        if topic == 'tf':
            translation = msg.translation
            rotation = msg.rotation # quaternion
        elif topic == 'matrix_tf':
            translation = msg.translation
            rotation = msg.rotation # matrix
        elif topic == 'camera_info':
            intrinsics = msg
        else:
            print 'skipping message',topic
            continue

    bag.close()

    if translation is None or rotation is None:
        if extrinsics_required:
            raise ValueError('no extrinsic parameters in bag file')
        else:
            translation = (np.nan, np.nan, np.nan)
            rotation = (np.nan, np.nan, np.nan, np.nan)
    else:
        translation = point_msg_to_tuple(translation)
        rotation = parse_rotation_msg(rotation)

    if intrinsics is None:
        raise ValueError('no intrinsic parameters in bag file')

    result = CameraModel(translation=translation,
                         rotation=rotation,
                         intrinsics=intrinsics,
                         name=camera_name,
                         )
    return result


def load_camera_from_pmat( pmat, width=None, height=None, name='cam', _depth=0 ):
    pmat = np.array(pmat)
    assert pmat.shape==(3,4)
    c = center(pmat)
    M = pmat[:,:3]
    K,R = my_rq(M)
    a = K[2,2]
    if a==0:
        warnings.warn('ill-conditioned intrinsic camera parameters')
    else:
        if a != 1.0:
            if _depth > 0:
                raise ValueError('cannot scale this pmat: %s'%( repr(pmat,)))
            new_pmat = pmat/a
            cam = load_camera_from_pmat( new_pmat, width=width, height=height, name=name, _depth=_depth+1)
            return cam

    t = -np.dot(R,c)

    P = np.zeros( (3,4) )
    P[:3,:3]=K

    i = sensor_msgs.msg.CameraInfo()
    i.width = width
    i.height = height
    i.D = [0,0,0,0,0]
    i.K = list(K.flatten())
    i.R = list(np.eye(3).flatten())
    i.P = list(P.flatten())
    result = CameraModel(translation = t,
                         rotation = R,
                         intrinsics = i,
                         name=name)
    return result

def load_default_camera( ):
    pmat = np.array( [[ 300,   0, 320, 0],
                      [   0, 300, 240, 0],
                      [   0,   0,   1, 0]])
    return load_camera_from_pmat( pmat, width=640, height=480, name='cam')
