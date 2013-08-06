import math
import numpy
import numpy as np
import pickle
import yaml

class FakeMessage(object):
    def __init__(self,**kwargs):
        for key in kwargs:
            setattr(self,key,kwargs[key])
    def __getattr__(self,key):
        # normal name-resolution failed, return a sub-message
        result = FakeMessage()
        setattr(self,key,result)
        return result
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)
    # def _get_clean_dict(self):
    #     '''attempt to remove all tuples and numpy arrays'''
    #     result = {}
    #     print 'get clean'
    #     for key in self.__dict__:
    #         print 'key',key
    #         val = getattr(self,key)
    #         print 'val',val
    #         print 'val.__type__',type(val)
    #         val = self._make_safe( val )
    #         result[key] = val
    #     return result
    # def _make_safe( self, val ):
    #     if isinstance(val,FakeMessage):
    #         print 'fake---'
    #         val = val._get_clean_dict()
    #     # elif isinstance(val,np.ndarray):
    #     #     2/0
    #     # elif isinstance(val,np.core.multiarray):
    #     #     3/0
    #     elif isinstance(val, np.ndarray):
    #         print 'np'
    #         4/0
    #     elif isinstance(val, np.generic):
    #         print 'np2'
    #         print 'val',val
    #         testval = float(val)
    #         if testval==float(val):
    #             val = testval
    #         else:
    #             raise NotImplementedError
    #     elif isinstance(val,tuple):
    #         print 'tup'
    #         val = list(val)

    #     if hasattr(val,'__len__'):
    #         print 'list'
    #         val = [ self._make_safe( element ) for element in val ]
    #     return val

    # def __str__(self):
    #     d = self._get_clean_dict()
    #     result = yaml.dump( d )
    #     print result
    #     1/0
    #     return result
    def __str__(self):
        d = self._get_simple_dict()
        result = yaml.dump(d)
        return result

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
    def _get_simple_dict(self):
        result = {}
        for key in ['D','P','K','R']:
            result[key] = make_list(getattr(self,key))
        result['width'] = self.width
        result['height'] = self.height
        return result

class Point(FakeMessage):
    pass

class Quaternion(FakeMessage):
    pass

class Transform(FakeMessage):
    pass

def strify_message(msg):
    assert isinstance(msg,FakeMessage)
    return str(msg)

class Bag(object):
    def __init__(self, file, mode):
        assert mode in ['r','w']
        self.mode=mode
        if hasattr(file,'write'):
            self.fd = file
        else:
            self.fd = open(file,mode=mode)
        if mode=='w':
            self.messages = []
        else:
            self.messages = pickle.load( self.fd )
        self.closed=False
    def __del__(self):
        self.close()
    def write(self,topic,value):
        self.messages.append( (topic,value) )
    def close(self):
        if self.closed:
            return
        if self.mode=='w':
            pickle.dump( self.messages, self.fd )
        self.fd.close()
        self.closed = True

    def read_messages(self):
        for topic,value in self.messages:
            t = 0.0
            yield (topic,value,t)

_EPS = numpy.finfo(float).eps * 4.0

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """
    q = numpy.array(quaternion[:4], dtype=numpy.float64, copy=True)
    nq = numpy.dot(q, q)
    if nq < _EPS:
        return numpy.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = numpy.outer(q, q)
    return numpy.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=numpy.float64)

def quaternion_from_matrix(matrix):
    """Return quaternion from rotation matrix.

    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
    True

    """
    q = numpy.empty((4, ), dtype=numpy.float64)
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:4, :4]
    t = numpy.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
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
