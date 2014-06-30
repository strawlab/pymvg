"""utilities to emulate parts of ROS

pymvg was originally written as a [ROS](http://ros.org) package. This
module allows pymvg to run without ROS.
"""
import numpy as np
import json
import io
import warnings
from .quaternions import quaternion_matrix as qquaternion_matrix
from .quaternions import quaternion_from_matrix as qquaternion_from_matrix

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

class Bunch(object):
    pass

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

sensor_msgs = _get_sensor_msgs()
geometry_msgs = _get_geometry_msgs()
