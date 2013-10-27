#!/usr/bin/env python
import numpy as np
import os, re
import json

from .ros_compat import tf, sensor_msgs, geometry_msgs, rosbag, roslib
from .align import estsimt

import warnings

D2R = np.pi/180.0

# helper functions ---------------

def point_msg_to_tuple(d):
    return d.x, d.y, d.z

def normalize(vec):
    mag = np.sqrt(np.sum(vec**2))
    return vec/mag

def parse_rotation_msg(rotation, force_matrix=False):
    # rotation could either be a quaternion or a 3x3 matrix

    if (hasattr(rotation,'x') and
        hasattr(rotation,'y') and
        hasattr(rotation,'z') and
        hasattr(rotation,'w')):
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

def np2plain(arr):
    '''convert numpy array to plain python (for serializing to yaml or json)'''
    arr = np.array(arr)
    if arr.ndim==1:
        result = plain_vec(arr)
    elif arr.ndim==2:
        result = [ plain_vec(row) for row in arr ]
    else:
        raise NotImplementedError
    return result

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
        icdist = (1 + ((k[7]*r2 + k[6])*r2 + k[5])*r2)/ \
                 (1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2)
        delta_x = 2.0 * (t1)*x*y + (t2)*(r2 + 2.0*x*x)
        delta_y = (t1) * (r2 + 2.0*y*y)+2.0*(t2)*x*y
        x = (xd-delta_x)*icdist
        y = (yd-delta_y)*icdist
    return x,y


def rq(A):
    # see first comment at
    # http://leohart.wordpress.com/2010/07/23/rq-decomposition-from-qr-decomposition/
    from numpy.linalg import qr
    from numpy import flipud
    Q,R = qr(flipud(A).T)
    R = flipud(R.T)
    Q = Q.T
    return R[:,::-1],Q[::-1,:]

def my_rq(M):
    """RQ decomposition, ensures diagonal of R is positive"""
    R,K = rq(M)
    n = R.shape[0]
    for i in range(n):
        if R[i,i]<0:
            # I checked this with Mathematica. Works if R is upper-triangular.
            R[:,i] = -R[:,i]
            K[i,:] = -K[i,:]
    return R,K

def center(P,eps=1e-8):
    orig_determinant = np.linalg.det
    def determinant( A ):
        return orig_determinant( np.asarray( A ) )
    # camera center
    X = determinant( [ P[:,1], P[:,2], P[:,3] ] )
    Y = -determinant( [ P[:,0], P[:,2], P[:,3] ] )
    Z = determinant( [ P[:,0], P[:,1], P[:,3] ] )
    T = -determinant( [ P[:,0], P[:,1], P[:,2] ] )

    assert abs(T)>eps, "cannot calculate 3D camera center: camera at infinity"
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

    # test: has one eigenvalue of unity
    l, W = np.linalg.eig(R)
    eps = 1e-8
    i = np.where(abs(np.real(l) - 1.0) < eps)[0]
    if not len(i):
        return False
    return True

def get_rotation_matrix_and_quaternion(rotation):
    rotation_orig = rotation
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
    elif rotation.ndim==0:
        assert rotation.dtype == object
        rotation = (rotation_orig.x,
                    rotation_orig.y,
                    rotation_orig.z,
                    rotation_orig.w)
        return get_rotation_matrix_and_quaternion(rotation)
    else:
        assert rotation.ndim==1
        assert rotation.shape==(4,)
        rquat = rotation
        rmat = tf.transformations.quaternion_matrix(rquat)[:3,:3]

        if not np.alltrue(np.isnan( rmat )):
            assert is_rotation_matrix(rmat)

    return rmat, rquat

def get_vec_str(vec):
    assert vec.ndim==1
    # use numpy for printing (suppresses small values when others are large)
    tmp = np.array_repr(vec, precision=5, suppress_small=True)
    assert tmp.startswith('array([')
    tmp = tmp[7:]
    assert tmp.endswith('])')
    tmp = tmp[:-2]
    tmp = tmp.strip()
    tmps = [t.strip() for t in tmp.split(',')]

    # convert -0 to 0
    tmps2 = []
    for t in tmps:
        if t=='-0.':
            tmps2.append('0.')
        else:
            tmps2.append(t)
    tmps = tmps2

    tmps = ['% 8s'%(t,) for t in tmps ]
    result = ', '.join( tmps )
    return result

def plain_vec(vec):
    '''make a list of plain types'''
    if hasattr( vec, 'dtype' ):
        # assume it's a simple numpy array
        # TODO: FIXME: could make this much better
        result = [ float(el) for el in vec ]
    else:
        # no change
        result = vec
    return result

def normalize_M(pmat,eps=1e-6):
    pmat_orig = pmat
    M = pmat[:,:3]
    t = pmat[:,3,np.newaxis]
    K,R = my_rq(M)
    if abs(K[2,2]-1.0)>eps:
        pmat = pmat/K[2,2]
    assert np.allclose(center(pmat_orig),center(pmat))
    return pmat

def parse_radfile(filename):
    result = {}
    regex = re.compile(r'^(?P<key>[_a-zA-Z][a-zA-Z0-9_.]*)\s*=\s*(?P<value>.*)$')
    with open(filename,mode='r') as fd:
        for line in fd.readlines():
            line = line[:line.find('#')] # strip comments
            line = line.strip() # strip whitespace
            if len(line)==0:
                # discard empty lines
                continue
            matchobj = regex.match(line)
            assert matchobj is not None
            d = matchobj.groupdict()
            result[ d['key'] ] = float(d['value'])

    K = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            K[i,j] = result[ 'K%d%d'%(i+1, j+1) ]

    distortion = np.array(( result['kc1'],
                            result['kc2'],
                            result['kc3'],
                            result['kc4'],
                            result.get('kc5',0.0) ))
    return K, distortion

# main class
class CameraModel(object):
    """an implementation of the Camera Model used by ROS and OpenCV

    Tranformations: We can think about the overall projection to 2D in
    two steps. Step 1 takes 3D world coordinates and, with a simple
    matrix multiplication and perspective division, projects them to
    undistorted 2D coordinates. Step 2 takes these undistorted 2D
    coordinates and distorts them so they are 'distorted' and match up
    with a real camera with radial distortion, for example.

    3D world --(step1)----> undistorted 2D ---(step2)----> distorted 2D

    Step 1 is accomplished by making the world coordinates a
    homogeneous vector of length 4, multiplying by a 3x4 matrix M
    (built from P, R and t) to get values [r,s,t] in which the
    undistorted 2D coordinates are [r/t, s/t]. (The implementation is
    vectorized so that in fact many points at once can be
    transformed.)

    Step 2 is somewhat complicated in that it allows a separate focal
    length and camera center to be used for distortion. Undistorted 2D
    coordinates are transformed first to uncorrected normalized image
    coordinates using parameters from P, then corrected using a
    rectification matrix. These corrected normalized image coordinates
    are then used in conjunction with the distortion model to create
    distorted normalized pixels which are finally transformed to
    distorted image pixels by K.

    Coordinate system: the camera is looking at +Z, with +X rightward
    and +Y down. For more information, see
    http://www.ros.org/wiki/image_pipeline/CameraInfo

    As noted on the link above, this differs from the coordinate
    system of Harley and Zisserman, which has Z forward, Y up, and X
    to the left (looking towards +Z).'

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
        'K', # used for distorted<->normalized, a scaled version of P[:3,:3], np.array with shape (3,3)

        # (The scaling of K, with the default alpha=0, is such that
        # every pixel in the undistorted image is valid, thus throwing
        # away some pixels. With alpha=1, P==K and all pixels in the
        # original image are in the undistorted image.)

        # the distortion model
        'distortion', # (distortion params) the distortion, np.array with shape (5,1) or (8,1)
        'rect', # (distortion params) the rectification, None or np.array with shape (3,3)
        ]
    AXIS_FORWARD = np.array((0,0,1),dtype=np.float)
    AXIS_UP = np.array((0,-1,0),dtype=np.float)
    AXIS_RIGHT = np.array((1,0,0),dtype=np.float)

    # --- start of CameraModel constructors ------------------------------------

    def __init__(self, name, width, height, _rquat, _camcenter, P, K, distortion, rect):
        self.name = name
        self.width = width
        self.height = height
        self._rquat = _rquat
        self._camcenter = _camcenter
        self.P = P
        self.K = K
        self.distortion = distortion
        self.rect = rect

        self._opencv_compatible = (self.P[0,1]==0)

    @classmethod
    def from_ros_like(cls,
                      translation=None,
                      rotation=None,
                      intrinsics=None,
                      name=None,
                      max_skew_ratio=1e15,
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
        _camcenter = -np.dot( rmat.T, t )[:,0]
        del t

        _rquat = rquat

        if 1:
            # Initialize the camera calibration from a CameraInfo message.
            msg = intrinsics
            width = msg.width
            height = msg.height
            shape = (msg.height, msg.width)

            P = np.array(msg.P,dtype=np.float)
            P.shape = (3,4)
            if not np.allclose(P[:,3], np.zeros((3,))):
                raise NotImplementedError('not tested when 4th column of P is nonzero')

            K = np.array( msg.K, dtype=np.float)
            K.shape = (3,3)
            assert K.ndim == 2

            distortion = np.array(msg.D, dtype=np.float)
            if len(distortion) == 5:
                distortion.shape = (5,)
            elif len(distortion) == 8:
                distortion.shape = (8,)
            else:
                raise ValueError('distortion can have only 5 or 8 entries')

            assert distortion.ndim==1

            if msg.R is None:
                rect = None
            else:
                rect = np.array( msg.R, dtype=np.float )
                rect.shape = (3,3)
                if np.allclose(rect,np.eye(3)):
                    rect = None

        K_ = P[:3,:3]

        # If skew is 15 orders of magnitude less than focal length, ignore it.
        if abs(K_[0,1]) > (abs(K_[0,0])/max_skew_ratio):
            if np.sum(abs(distortion)) != 0.0:
                skew = K_[0,1]
                fx = K_[0,0]
                raise NotImplementedError('distortion/undistortion for skewed pixels not implemented (skew: %s, fx: %s)'%(skew,fx))
        result = cls(name, width, height, _rquat, _camcenter, P, K, distortion, rect)
        return result

    @classmethod
    def from_dict(cls, d, extrinsics_required=True, max_skew_ratio=1e15 ):
        translation = None
        rotation = None

        if 'image_height' in d:
            # format saved in ~/.ros/camera_info/<camera_name>.yaml
            #only needs w,h,P,K,D,R
            c = sensor_msgs.msg.CameraInfo(
                height=d['image_height'],
                width=d['image_width'],
                P=d['projection_matrix']['data'],
                K=d['camera_matrix']['data'],
                D=d['distortion_coefficients']['data'],
                R=d['rectification_matrix']['data'])
            name = d['camera_name']
        else:
            # format saved by roslib.message.strify_message( sensor_msgs.msg.CameraInfo() )
            c = sensor_msgs.msg.CameraInfo(
                height = d['height'],
                width = d['width'],
                P=d['P'],
                K=d['K'],
                D=d['D'],
                R=d['R'])
            name = d.get('name',None)
            translation = d.get('translation',None)
            rotation = d.get('rotation',None)

        if translation is None or rotation is None:
            if extrinsics_required:
                raise ValueError('extrinsic parameters are required, but not provided')

        result = cls.from_ros_like(translation=translation,
                                   rotation=rotation,
                                   intrinsics=c,
                                   name=name,
                                   max_skew_ratio = max_skew_ratio,
                                   )
        return result

    @classmethod
    def load_camera_from_file( cls, fname, extrinsics_required=True ):
        if fname.endswith('.bag'):
            return cls.load_camera_from_bagfile(fname, extrinsics_required=extrinsics_required)
        elif (fname.endswith('.yaml') or
              fname.endswith('.json')):
            if fname.endswith('.yaml'):
                import yaml
                with open(fname,'r') as f:
                    d = yaml.safe_load(f)
            else:
                assert fname.endswith('.json')
                with open(fname,'r') as f:
                    d = json.load(f)
            return cls.from_dict(d, extrinsics_required=extrinsics_required)
        else:
            raise ValueError("only supports: .bag .yaml .json")

    @classmethod
    def load_camera_from_bagfile( cls, bag_fname, extrinsics_required=True ):
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
                warnings.warn('skipping message topic %r'%topic)
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

        result = cls.from_ros_like(translation=translation,
                                   rotation=rotation,
                                   intrinsics=intrinsics,
                                   name=camera_name,
                                   )
        return result


    @classmethod
    def load_camera_from_M( cls, pmat, width=None, height=None, name='cam',
                            distortion_coefficients=None,
                            _depth=0, eps=1e-15 ):
        """create CameraModel instance from a camera matrix M"""
        pmat = np.array(pmat)
        assert pmat.shape==(3,4)
        M = pmat[:,:3]
        K,R = my_rq(M)
        if not is_rotation_matrix(R):
            # RQ may return left-handed rotation matrix. Make right-handed.
            R2 = -R
            K2 = -K
            assert np.allclose(np.dot(K2,R2), np.dot(K,R))
            K,R = K2,R2
        a = K[2,2]
        if a==0:
            warnings.warn('ill-conditioned intrinsic camera parameters')
        else:
            if abs(a-1.0) > eps:
                if _depth > 0:
                    raise ValueError('cannot scale this pmat: %s'%( repr(pmat,)))
                new_pmat = pmat/a
                cam = cls.load_camera_from_M( new_pmat, width=width, height=height, name=name, _depth=_depth+1)
                return cam

        c = center(pmat)
        t = -np.dot(R,c)

        P = np.zeros( (3,4) )
        P[:3,:3]=K

        if distortion_coefficients is None:
            distortion_coefficients = np.zeros((5,))
        else:
            distortion_coefficients = np.array(distortion_coefficients)
            assert distortion_coefficients.shape == (5,)

        i = sensor_msgs.msg.CameraInfo()
        i.width = width
        i.height = height
        i.D = [float(val) for val in distortion_coefficients]
        i.K = list(K.flatten())
        i.R = list(np.eye(3).flatten())
        i.P = list(P.flatten())
        result = cls.from_ros_like(translation = t,
                                   rotation = R,
                                   intrinsics = i,
                                   name=name)
        return result

    @classmethod
    def load_camera_default(cls):
        pmat = np.array( [[ 300,   0, 320, 0],
                          [   0, 300, 240, 0],
                          [   0,   0,   1, 0]])
        return cls.load_camera_from_M( pmat, width=640, height=480, name='cam')

    @classmethod
    def load_camera_from_ROS_tf( cls,
                                 translation=None,
                                 rotation=None,
                                 **kwargs):
        rmat, rquat = get_rotation_matrix_and_quaternion(rotation)
        if hasattr(translation,'x'):
            translation = (translation.x, translation.y, translation.z)
        C = np.array(translation)
        C.shape = 3,1

        r2 =  np.linalg.pinv(rmat)
        rmat2, rquat2 = get_rotation_matrix_and_quaternion(r2)

        t = -np.dot( rmat2, C)[:,0]

        return cls.from_ros_like(translation=t, rotation=rquat2, **kwargs)

    @classmethod
    def load_camera_simple( cls,
                            fov_x_degrees=30.0,
                            width=640, height=480,
                            eye=(0,0,0),
                            lookat=(0,0,-1),
                            up=None,
                            name='simple',
                            distortion_coefficients=None,
                            ):
        aspect = float(width)/float(height)
        fov_y_degrees = fov_x_degrees/aspect
        f = (width/2.0) / np.tan(fov_x_degrees*D2R/2.0)
        cx = width/2.0
        cy = height/2.0
        M = np.array( [[ f, 0, cx, 0],
                       [ 0, f, cy, 0],
                       [ 0, 0,  1, 0]])
        c1 = cls.load_camera_from_M( M, width=width, height=height, name=name,
                                     distortion_coefficients=distortion_coefficients)
        c2 = c1.get_view_camera( eye=eye, lookat=lookat, up=up)
        return c2

    # --- end of CameraModel constructors --------------------------------------

    def __str__(self):
        template = '''camera {name!r}{size_str}:
   extrinsic parameters:
        center  : {center}
        look at : {lookat}
        up      : {up}
   intrinsic parameters:
        P       : [[{P0}],
                   [{P1}],
                   [{P2}]]
        K       : [[{K0}],
                   [{K1}],
                   [{K2}]]
        distortion : {D}
        rectification : {rect}
'''
        if self.width is not None:
            size_str = ' (%dx%d)'%(self.width,self.height)
        else:
            size_str = ''
        center, lookat, up = map(get_vec_str,self.get_view())

        P = self.P[:3,:3]
        P0,P1,P2 = [get_vec_str(P[i]) for i in range(3)]

        K = self.get_K()[:3,:3]
        K0,K1,K2 = [get_vec_str(K[i]) for i in range(3)]

        D = get_vec_str(self.distortion.flatten())

        if self.rect is None:
            rect = 'None'
        else:
            rtmp = '''[[{r0}],
                   [{r1}],
                   [{r2}]]
'''
            r = self.rect
            r0,r1,r2 = [get_vec_str(r[i]) for i in range(3)]
            d = dict()
            d.update(locals())
            rect = rtmp.format( **d )

        d = dict(name=self.name)
        d.update(locals())
        return template.format( **d )

    def __eq__(self,other):
        assert isinstance( self, CameraModel )
        if not isinstance( other, CameraModel ):
            return False
        # hmm, could do better than comparing strings...
        c1s = str(self)
        c2s = str(other)
        return c1s==c2s

    def __ne__(self,other):
        return not (self==other)

    def to_dict(self):
        d = {}
        d['name'] =self.name
        d['height'] = self.height
        d['width'] = self.width
        d['P'] = np2plain(self.P)
        d['K'] = np2plain(self.K)
        d['D'] = np2plain(self.distortion[:])
        if self.rect is None:
            d['R'] = np2plain(np.eye(3))
        else:
            d['R'] = np2plain(self.rect)
        d['translation']=np2plain(self.translation)
        d['rotation']=np2plain(self.get_rot())
        return d

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

    def get_M(self):
        P33 = self.P[:3,:3]
        M = np.dot( P33, self.Rt )
        return M
    M = property(get_M)

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

    def get_ROS_tf(self):
        rmat = self.get_rot_inv()
        rmat2, rquat2 = get_rotation_matrix_and_quaternion(rmat)
        return self.get_camcenter(), rquat2

    def get_intrinsics_as_msg(self):
        i = sensor_msgs.msg.CameraInfo()
        # these are from image_geometry ROS package in the utest.cpp file
        i.height = self.height
        i.width = self.width
        assert len(self.distortion) == 5
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

    def save_intrinsics_to_yamlfile(self,fname):
        msg = self.get_intrinsics_as_msg()
        buf = roslib.message.strify_message(msg)
        with open(fname,'w') as fd:
            fd.write( buf )

    def get_mirror_camera(self,axis='lr',hold_center=False):
        """return a copy of this camera whose x coordinate is (image_width-x)"""
        assert axis in ['lr','ud']
        # Keep extrinsic coordinates, but flip intrinsic
        # parameter so that a mirror image is rendered.

        i = self.get_intrinsics_as_msg()
        if axis=='lr':
            i.K[0] = -i.K[0]
            i.P[0] = -i.P[0]

            if not hold_center:
                # This flips the X coordinate but preserves the
                # optical center.

                i.K[2] = (self.width-i.K[2])
                i.P[2] = (self.width-i.P[2])
        else:
            # axis=='ud'
            i.K[4] = -i.K[4]
            i.P[5] = -i.P[5]

            if not hold_center:

                # This flips the Y coordinate but preserves the
                # optical center.

                i.K[5] = (self.height-i.K[5])
                i.P[6] = (self.height-i.P[6])

        camnew = CameraModel.from_ros_like(
                              translation = self.translation,
                              rotation = self.rot,
                              intrinsics = i,
                              name = self.name + '_mirror',
                              )
        return camnew

    def get_flipped_camera(self):
        """return a copy of this camera looking in the opposite direction

        The returned camera has the same 3D->2D projection. (The
        2D->3D projection results in a vector in the opposite
        direction.)
        """
        cc, la, up = self.get_view()
        lv = la-cc # look vector

        lv2 = -lv
        up2 = -up
        la2 = cc+lv2

        camnew = self.get_view_camera(cc, la2, up2).get_mirror_camera(hold_center=True)
        camnew.distortion[3] = -self.distortion[3]

        if camnew.rect is not None:
            raise NotImplementedError('No support for flipping cameras '
                                      'that require rectifcation')
        return camnew

    def get_view_camera(self, eye, lookat, up=None):
        """return a copy of this camera with new extrinsic coordinates"""
        eye = np.array(eye); eye.shape=(3,)
        lookat = np.array(lookat); lookat.shape=(3,)
        gen_up = False
        if up is None:
            up = np.array((0,-1,0))
            gen_up = True
        lv = lookat - eye
        f = normalize(lv)
        old_settings = np.seterr(invalid='ignore')
        s = normalize( np.cross( f, up ))
        np.seterr(**old_settings)
        if np.isnan(s[0]) and gen_up:
            up = np.array((0,0,1))
            s = normalize( np.cross( f, up ))
        assert not np.isnan(s[0]), 'invalid up vector'
        u = normalize( np.cross( f, s ))
        R = np.array( [[ s[0], u[0], f[0]],
                       [ s[1], u[1], f[1]],
                       [ s[2], u[2], f[2]]]).T

        eye.shape = (3,1)
        t = -np.dot(R,eye)

        result = CameraModel.from_ros_like(
                             translation=t,
                             rotation=R,
                             intrinsics=self.get_intrinsics_as_msg(),
                             name=self.name,
                             )
        return result

    def get_aligned_camera(self, scale, rotation, translation):
        """return a copy of this camera with new extrinsic coordinates"""
        s,R,t = scale, rotation, translation
        cc, la, up = self.get_view()
        f = la-cc

        X = np.linalg.inv( R )

        fa = np.dot( f, X )
        cca0 = cc*s
        cca = np.dot( cca0, X )
        laa = cca+fa
        up2 = np.dot( up, X )

        cca2 = cca+t
        laa2 = laa+t

        return self.get_view_camera(cca2, laa2, up2)

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

    def project_camera_frame_to_pixel(self, pts3d, distorted=True):
        pts3d = np.array(pts3d,copy=False)
        assert pts3d.ndim==2
        assert pts3d.shape[1]==3

        # homogeneous and transposed
        pts3d_h = np.empty( (4,pts3d.shape[0]) )
        pts3d_h[:3,:] = pts3d.T
        pts3d_h[3] = 1

        # undistorted homogeneous image coords
        cc = np.dot(self.P, pts3d_h)

        # project
        pc = cc[:2]/cc[2]
        u, v = pc

        if distorted:
            # distort (the currently undistorted) image coordinates
            nparr = np.vstack((u,v)).T
            u,v = self.distort( nparr ).T
        return np.vstack((u,v)).T

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
        cc = np.dot(self.M, pts3d_h)

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

class MultiCameraSystem:
    def __init__(self,cameras):
        self._cameras={}
        for camera in cameras:
            assert isinstance(camera, CameraModel)
            name = camera.name
            if name in self._cameras:
                raise ValueError('Cannot create MultiCameraSystem with '
                                 'multiple identically-named cameras.')
            self._cameras[name] = camera

    @classmethod
    def from_dict(cls, d):
        cam_dict_list = d['camera_system']
        cams = [CameraModel.from_dict(cd) for cd in cam_dict_list]
        return MultiCameraSystem( cameras=cams )

    @classmethod
    def from_mcsc(cls, dirname, max_skew_ratio=10 ):
        '''create MultiCameraSystem from output directory of MultiCamSelfCal'''

        # FIXME: This is a bit convoluted because it's been converted
        # from multiple layers of internal code. It should really be
        # simplified and cleaned up.

        do_normalize_pmat=True

        all_Pmat = {}
        all_Res = {}
        all_K = {}
        all_distortion = {}

        opj = os.path.join

        with open(opj(dirname,'camera_order.txt'),mode='r') as fd:
            cam_ids = fd.read().strip().split('\n')

        with open(os.path.join(dirname,'Res.dat'),'r') as res_fd:
            for i, cam_id in enumerate(cam_ids):
                fname = 'camera%d.Pmat.cal'%(i+1)
                pmat = np.loadtxt(opj(dirname,fname)) # 3 rows x 4 columns
                if do_normalize_pmat:
                    pmat_orig = pmat
                    pmat = normalize_M(pmat)
                all_Pmat[cam_id] = pmat
                all_Res[cam_id] = map(int,res_fd.readline().split())

        # load non linear parameters
        rad_files = [ f for f in os.listdir(dirname) if f.endswith('.rad') ]
        for cam_id_enum, cam_id in enumerate(cam_ids):
            filename = os.path.join(dirname,
                                    'basename%d.rad'%(cam_id_enum+1,))
            if os.path.exists(filename):
                K, distortion = parse_radfile(filename)
                all_K[cam_id] = K
                all_distortion[cam_id] = distortion
            else:
                if len(rad_files):
                    raise RuntimeError(
                        '.rad files present but none named "%s"'%filename)
                warnings.warn('no non-linear data (e.g. radial distortion) '
                              'in calibration for %s'%cam_id)
                all_K[cam_id] = None
                all_distortion[cam_id] = None

        cameras = []
        for cam_id in cam_ids:
            w,h = all_Res[cam_id]
            Pmat = all_Pmat[cam_id]
            M = Pmat[:,:3]
            K,R = my_rq(M)
            if not is_rotation_matrix(R):
                # RQ may return left-handed rotation matrix. Make right-handed.
                R2 = -R
                K2 = -K
                assert np.allclose(np.dot(K2,R2), np.dot(K,R))
                K,R = K2,R2

            P = np.zeros((3,4))
            P[:3,:3] = K
            KK = all_K[cam_id] # from rad file or None
            distortion = all_distortion[cam_id]

            # (ab)use PyMVG's rectification to do coordinate transform
            # for MCSC's undistortion.

            # The intrinsic parameters used for 3D -> 2D.
            ex = P[0,0]
            bx = P[0,2]
            Sx = P[0,3]
            ey = P[1,1]
            by = P[1,2]
            Sy = P[1,3]

            if KK is None:
                rect = np.eye(3)
                KK = P[:,:3]
            else:
                # Parameters used to define undistortion coordinates.
                fx = KK[0,0]
                fy = KK[1,1]
                cx = KK[0,2]
                cy = KK[1,2]

                rect = np.array([[ ex/fx,     0, (bx+Sx-cx)/fx ],
                                 [     0, ey/fy, (by+Sy-cy)/fy ],
                                 [     0,     0,       1       ]]).T

            if distortion is None:
                distortion = np.zeros((5,))

            C = center(Pmat)
            rot = R
            t = -np.dot(rot, C)[:,0]

            d = {'width':w,
                 'height':h,
                 'P':P,
                 'K':KK,
                 'R':rect,
                 'translation':t,
                 'rotation':rot,
                 'D':distortion,
                 'name':cam_id,
                 }
            cam = CameraModel.from_dict(d, max_skew_ratio=max_skew_ratio)
            cameras.append( cam )
        return MultiCameraSystem( cameras=cameras )

    def __eq__(self, other):
        assert isinstance( self, MultiCameraSystem )
        if not isinstance( other, MultiCameraSystem ):
            return False
        if len(self.get_names()) != len(other.get_names()):
            return False
        for name in self.get_names():
            if self._cameras[name] != other._cameras[name]:
                return False
        return True

    def __ne__(self,other):
        return not (self==other)

    def get_names(self):
        result = list(self._cameras.keys())
        result.sort()
        return result

    def get_camera_dict(self):
        return self._cameras

    def to_dict(self):
        return {'camera_system':
                [self._cameras[name].to_dict() for name in self._cameras]}

    def find3d(self,pts,undistort=True):
        """Find 3D coordinate using all data given

        Implements a linear triangulation method to find a 3D
        point. For example, see Hartley & Zisserman section 12.2
        (p.312). Also, 12.8 for intersecting lines.

        By default, this function will undistort 2D points before
        finding a 3D point.
        """
        # for info on SVD, see Hartley & Zisserman (2003) p. 593 (see
        # also p. 587)
        # Construct matrices
        A=[]
        P=[]
        for name,xy in pts:
            cam = self._cameras[name]
            if undistort:
                xy = cam.undistort( [xy] )
            Pmat = cam.get_M() # Pmat is 3 rows x 4 columns
            row2 = Pmat[2,:]
            x,y = xy[0,:]
            A.append( x*row2 - Pmat[0,:] )
            A.append( y*row2 - Pmat[1,:] )

        # Calculate best point
        A=np.array(A)
        u,d,vt=np.linalg.svd(A)
        X = vt[-1,0:3]/vt[-1,3] # normalize
        return X

    def find2d(self,camera_name,xyz,distorted=True):
        cam = self._cameras[camera_name]

        xyz = np.array(xyz)
        rank1 = xyz.ndim==1

        xyz = np.atleast_2d(xyz)
        pix = cam.project_3d_to_pixel( xyz, distorted=distorted ).T

        if rank1:
            # convert back to rank1
            pix = pix[:,0]
        return pix

    def get_aligned_copy(self, other):
        """return copy of self that is scaled, translated, and rotated to best match other"""
        assert isinstance( other, MultiCameraSystem)

        orig_names = self.get_names()
        new_names = other.get_names()
        names = set(orig_names).intersection( new_names )
        if len(names) < 3:
            raise ValueError('need 3 or more cameras in common to align.')
        orig_points = np.array([ self._cameras[name].get_camcenter() for name in names ]).T
        new_points = np.array([ other._cameras[name].get_camcenter() for name in names ]).T

        s,R,t = estsimt(orig_points,new_points)
        assert is_rotation_matrix(R)

        new_cams = []
        for name in self.get_names():
            orig_cam = self._cameras[name]
            new_cam = orig_cam.get_aligned_camera(s,R,t)
            new_cams.append( new_cam )
        result = MultiCameraSystem(new_cams)
        return result
