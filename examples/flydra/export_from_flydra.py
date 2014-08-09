import sys
import flydra.reconstruct

cal_source=sys.argv[1]
r = flydra.reconstruct.Reconstructor(cal_source)
base = r.convert_to_pymvg(ignore_water=True)

center = (10,20,30)
lookat = (11,20,30)
up = (0,-1,0)

for i,(name,cam) in enumerate(base.get_camera_dict().iteritems()):
    cam2 = cam.get_view_camera(center,lookat,up)
