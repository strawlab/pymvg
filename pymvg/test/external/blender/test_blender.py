import subprocess
import tempfile

# See http://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex

def blend(src):
    with tempfile.NamedTemporaryFile(suffix='.py') as fd:
        name = fd.name
        fd.write(src)
        fd.flush()

        cmd = '/usr/bin/blender','--background','--python',(name)
        subprocess.check_call(cmd)#,shell=True)

def test_blender():
    src = """print('hello')
    """
    blend(src)
