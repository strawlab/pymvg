#!/usr/bin/env python
import numpy as np
import os, re
import json
import warnings

from .quaternions import quaternion_matrix, quaternion_from_matrix

# helper class

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

# helper functions ---------------

def is_string(value):
    try:
        # Python 2
        return isinstance(value,basestring)
    except:
        # Python 3
        return isinstance(value,str)

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
            rotation = quaternion_matrix(rotation)[:3,:3]
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

    t1,t2 = D[2:4]

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
    # note the side-effects this has:
    # http://ksimek.github.io/2012/08/14/decompose/ (TODO: check for
    # these effects in PyMVG.)
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

def is_rotation_matrix(R,eps=1e-8):
    # check if rotation matrix is really a pure rotation matrix

    # test: inverse is transpose
    testI = np.dot(R.T,R)
    if not np.allclose( testI, np.eye(len(R)) ):
        return False

    # test: determinant is unity
    dr = np.linalg.det(R)
    if not (abs(dr-1.0)<eps):
        return False

    # test: has one eigenvalue of unity
    l, W = np.linalg.eig(R)
    i = np.where(abs(np.real(l) - 1.0) < eps)[0] # XXX do we need to check for complex part?
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
        rquat = quaternion_from_matrix(rnew)
        if not np.alltrue(np.isnan( rquat )):
            R2 = quaternion_matrix(rquat)[:3,:3]
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
        rmat = quaternion_matrix(rquat)[:3,:3]

        if not np.alltrue(np.isnan( rmat )):
            assert is_rotation_matrix(rmat)

    return rmat, rquat

def plain_vec(vec,eps=1e-15):
    '''make a list of plain types'''
    if hasattr( vec, 'dtype' ):
        # assume it's a simple numpy array
        # TODO: FIXME: could make this much better
        result = [ float(el) for el in vec ]
    else:
        # no change
        result = vec
    r2=[]
    for r in result:
        if abs(r)<eps:
            r2.append(0)
        else:
            r2.append(float(r))
    return r2

def normalize_M(pmat,eps=1e-6):
    pmat_orig = pmat
    M = pmat[:,:3]
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

# JSON compatible pretty printing ----------------

def _pretty_vec(row):
    els = [ json.dumps(el) for el in row ]
    rowstr = '[ ' + ', '.join(els) + ' ]'
    return rowstr

def _pretty_arr(arr,indent=11):
    rowstrs = []
    for row in arr:
        rowstr = _pretty_vec(row)
        rowstrs.append( rowstr )
    sep = ',\n' + ' '*indent
    buf = '[' + sep.join(rowstrs) + ']'
    return buf

def _cam_str(cam):
    buf = '''{"name": "%s",
     "width": %d,
     "height": %d,
     "P": %s,
     "K": %s,
     "D": %s,
     "R": %s,
     "Q": %s,
     "translation": %s
    }'''%(cam['name'], cam['width'], cam['height'], _pretty_arr(cam['P']),
          _pretty_arr(cam['K']),
          _pretty_vec(cam['D']),
          _pretty_arr(cam['R']),
          _pretty_arr(cam['Q']),
          _pretty_vec(cam['translation'])
          )
    return buf

def pretty_json_dump(d):
    keys = list(d.keys())
    assert len(keys)==2
    assert d['__pymvg_file_version__']=='1.0'
    cams = d['camera_system']
    cam_strs = [_cam_str(cam) for cam in cams]
    cam_strs = ',\n    '.join(cam_strs)
    buf = '''{ "__pymvg_file_version__": "1.0",
  "camera_system": [
    %s
  ]
}''' % cam_strs
    return buf

# end pretty printing ----------
