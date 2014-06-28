"""utilities to emulate parts of ROS

pymvg was originally written as a [ROS](http://ros.org) package. This
module allows pymvg to run without ROS.
"""
import math
import numpy
import numpy as np
import json
import io
import warnings

class FakeMessage(object):
    """abstract base class"""
    _sub_msgs = []
    _sub_attrs = []
    def __init__(self,**kwargs):
        for key in self._sub_msgs:
            setattr(self,key,self._sub_msgs[key]())
        for key in kwargs:
            setattr(self,key,kwargs[key])
    def get_dict(self):
        result = {}
        result['__json_message__'] = self.get_message_type()
        for key in self._sub_msgs:
            value = getattr(self,key)
            val_simple = value.get_dict()
            result[key] = val_simple
        for key in self._sub_attrs:
            value = getattr(self,key)
            result[key] = value
        return result
    def get_message_type(self):
        return self._msg_type

def make_list(vec):
    avec = np.array(vec)
    assert avec.ndim==1
    result = []
    for el in avec:
        newel = float(el)
        assert newel==el
        result.append(newel)
    return result

class CameraInfo(FakeMessage):
    _sub_msgs = []
    _sub_attrs = ['D','P','K','R','width','height']
    _msg_type = 'CameraInfo'
    def _get_simple_dict(self):
        result = {}
        for key in ['D','P','K','R']:
            result[key] = make_list(getattr(self,key))
        result['width'] = self.width
        result['height'] = self.height
        return result
    def __str__(self):
        d = self._get_simple_dict()
        result = json.dumps(d,sort_keys=True,indent=4)
        return result

class Point(FakeMessage):
    _msg_type = 'Point'
    _sub_attrs = ['x','y','z']

class Quaternion(FakeMessage):
    _msg_type = 'Quaternion'
    _sub_attrs = ['x','y','z','w']

class Vector3(FakeMessage):
    _msg_type = 'Vector3'
    _sub_attrs = ['x','y','z']

class Transform(FakeMessage):
    _msg_type = 'Transform'
    _sub_msgs = {'translation':Vector3,'rotation':Quaternion}

def make_registry(mlist):
    registry = {}
    for m in mlist:
        registry[m._msg_type] = m
    return registry

registry = make_registry( [Quaternion, Vector3, Transform, CameraInfo] )

def strify_message(msg):
    assert isinstance(msg,FakeMessage)
    return str(msg)

def fake_message_encapsulate(topic,value):
    result = {}
    result['__json_toplevel__'] = True
    result['topic'] = topic
    result['value'] = value.get_dict()
    return result

def fake_message_writer( messages, fd ):
    msg_list = []
    for topic,value in messages:
        assert isinstance(value,FakeMessage), ('%r not instance of FakeMessage' % (value,))
        sd = fake_message_encapsulate( topic, value )
        msg_list.append(sd)
    buf = json.dumps(msg_list, sort_keys=True, indent=4)
    try:
        # Python 2
        str_type = unicode
    except NameError:
        # Python 3
        str_type = bytes
    utf8 = str_type(buf.encode('UTF-8'))
    fd.write(utf8)

def parse_json_schema(m):
    typ=m['__json_message__']
    klass = registry[ typ ]
    result = klass()
    for attr in result._sub_attrs:
        setattr(result,attr,m[attr])
    for attr in result._sub_msgs:
        sub_msg = parse_json_schema( m[attr] )
        setattr(result,attr,sub_msg)
    return result


def fake_message_loader( fd ):
    buf = fd.read()
    msg_list = json.loads(buf)
    result = []
    for m in msg_list:
        assert m['__json_toplevel__']
        topic = m['topic']
        valuebuf = m['value']
        value = parse_json_schema(valuebuf)
        result.append( (topic, value) )
    return result

class Bag(object):
    def __init__(self, file, mode):
        assert mode in ['r','w']
        self.mode=mode
        if hasattr(file,'write'):
            self.fd = file
        else:
            self.fd = io.open(file,mode=mode,encoding='UTF-8')
        if mode=='w':
            warnings.warn('pymvg.ros_compat.Bag is writing a file, but this '
                          'is not a real bag file.')
            self.messages = []
        else:
            self.messages = fake_message_loader( self.fd )
        self.closed=False
    def __del__(self):
        self.close()
    def write(self,topic,value):
        self.messages.append( (topic,value) )
    def close(self):
        if self.closed:
            return
        if self.mode=='w':
            fake_message_writer( self.messages, self.fd )
        self.fd.close()
        self.closed = True

    def read_messages(self):
        for topic,value in self.messages:
            t = 0.0
            yield (topic,value,t)

_EPS = numpy.finfo(float).eps * 4.0


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    True

    """
    q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
    n = numpy.dot(q, q)
    if n < _EPS:
        return numpy.identity(4)
    q *= math.sqrt(2.0 / n)
    q = numpy.outer(q, q)
    return numpy.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])


def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True

    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:4, :4]
    if isprecise:
        q = numpy.empty((4, ))
        t = numpy.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = numpy.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = numpy.linalg.eigh(K)
        q = V[[3, 0, 1, 2], numpy.argmax(w)]
    if q[0] < 0.0:
        numpy.negative(q, q)
    return q

class Bunch(object):
    pass

def _get_tf():
    tf = Bunch()
    tf.transformations = Bunch()
    tf.transformations.quaternion_from_matrix = quaternion_from_matrix
    tf.transformations.quaternion_matrix = quaternion_matrix
    return tf

def _get_sensor_msgs():
    sensor_msgs = Bunch()
    sensor_msgs.msg = Bunch()
    sensor_msgs.msg.CameraInfo = CameraInfo
    return sensor_msgs

def _get_geometry_msgs():
    geometry_msgs = Bunch()
    geometry_msgs.msg = Bunch()
    geometry_msgs.msg.Point = Point
    geometry_msgs.msg.Quaternion = Quaternion
    geometry_msgs.msg.Transform = Transform
    return geometry_msgs

def _get_rosbag():
    rosbag = Bunch()
    rosbag.Bag = Bag
    return rosbag

def _get_roslib():
    roslib = Bunch()
    roslib.message = Bunch()
    roslib.message.strify_message = strify_message
    return roslib

tf = _get_tf()
sensor_msgs = _get_sensor_msgs()
geometry_msgs = _get_geometry_msgs()
rosbag = _get_rosbag()
roslib = _get_roslib()
