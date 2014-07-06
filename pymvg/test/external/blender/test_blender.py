import subprocess
import tempfile
from pymvg.test.utils import make_M, _build_test_camera, get_default_options

# See http://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex

def blend(src):
    with tempfile.NamedTemporaryFile(suffix='.py') as fd:
        name = fd.name
        fd.write(src)
        fd.flush()

        cmd = '/usr/bin/blender','--background','--python',(name)
        subprocess.check_call(cmd)#,shell=True)

def test_call_blender():
    src = """print('hello')
    """
    blend(src)

def test_save_camera():
    all_options = get_default_options()
    for opts in all_options:
        yield check_save_camera, opts

def check_save_camera(cam_opts):
    cam1 = _build_test_camera(**cam_opts)
    src = cam1.get_blender_py_src()
    blend(src)
