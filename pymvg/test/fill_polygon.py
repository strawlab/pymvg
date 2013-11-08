import numpy as np

def posint(x,maxval=np.inf):
    x = int(x)
    if x < 0:
        x = 0
    if x>maxval:
        return maxval
    return x

def fill_polygon(pts,image,fill_value=255):
    if len(pts)>=3:
        height, width = image.shape[:2]
        pts = [ (posint(y,height-1),posint(x,width-1)) for (x,y) in pts]
        if image.ndim == 3:
            _fill_polygon(pts, image[:,:,0], color=fill_value)
        else:
            _fill_polygon(pts, image[:,:], color=fill_value)

# from https://raw.github.com/luispedro/mahotas/master/mahotas/polygon.py
def _fill_polygon(polygon, canvas, color=1):
    '''
    fill_polygon([(y0,x0), (y1,x1),...], canvas, color=1)

    Draw a filled polygon in canvas

    Parameters
    ----------
    polygon : list of pairs
        a list of (y,x) points
    canvas : ndarray
        where to draw, will be modified in place
    color : integer, optional
        which colour to use (default: 1)
    '''
# algorithm adapted from: http://www.alienryderflex.com/polygon_fill/
    if not polygon:
        return
    min_y = min(y for y,x in polygon)
    max_y = max(y for y,x in polygon)
    polygon = [(float(y),float(x)) for y,x in polygon]
    for y in xrange(min_y, max_y+1):
        nodes = []
        j = -1
        for i,p in enumerate(polygon):
            pj = polygon[j]
            if p[0] < y and pj[0] >= y or pj[0] < y and p[0] >= y:
                dy = pj[0] - p[0]
                if dy:
                    nodes.append( (p[1] + (y-p[0])/(pj[0]-p[0])*(pj[1]-p[1])) )
                elif p[0] == y:
                    nodes.append(p[1])
            j = i
        nodes.sort()
        for n,nn in zip(nodes[::2],nodes[1::2]):
            nn += 1
            canvas[y,n:nn] = color

def test_fill():
    poly = [ [1,1],
             [1, 3],
             [3, 2],
             ]
    shape = (5,5,3)
    results = []
    for dtype in [np.float, np.int, np.uint8]:
        image = np.zeros( shape, dtype=dtype )
        fill_polygon( poly, image, fill_value=255 )
        results.append( image )

    for i in range(len(results)-1):
        i1 = results[i]
        i2 = results[i+1]
        assert np.allclose(i1,i2)
