import numpy as np
import matplotlib
# matplotlib.use('Agg')
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.collections import PatchCollection

import matplotlib.patches as patches
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from descartes import PolygonPatch

import matplotlib.pyplot as plt  # matplotlib.use('Agg') # TkAgg

import numpy as np
import math

def lengthSquare(X, Y):
    xDiff = X[0] - Y[0]
    yDiff = X[1] - Y[1]
    return xDiff * xDiff + yDiff * yDiff

def sidesMinAngle(iface):
    #iface : faces0[i,:]
    p1 = iface[1]
    p2 = iface[2]
    p3 = iface[3]
    A = points[p1]
    B = points[p2]
    C = points[p3]
    # Square of lengths be a2, b2, c2
    a2 = lengthSquare(B, C)
    b2 = lengthSquare(A, C)
    c2 = lengthSquare(A, B)

    # length of sides be a, b, c
    a = math.sqrt(a2);
    b = math.sqrt(b2);
    c = math.sqrt(c2);

    # From Cosine law
    alpha = math.acos((b2 + c2 - a2) /
                         (2 * b * c));
    betta = math.acos((a2 + c2 - b2) /
                         (2 * a * c));
    gamma = math.acos((a2 + b2 - c2) /
                         (2 * a * b));

    angle = min (alpha, betta, gamma)
    if angle <0.01:
      if angle == alpha:
        side1 = [p1, p2]
        side1.sort()
        side2 = [p1, p3]
        side2.sort()
        return [tuple(side1), tuple(side2)]
      if angle == betta:
        side1 = [p1, p2]
        side1.sort()
        side2 = [p2, p3]
        side2.sort()
        return [tuple(side1), tuple(side2)]
      if angle == gamma:
        side1 = [p3, p2]
        side1.sort()
        side2 = [p1, p3]
        side2.sort()
        return [tuple(side1), tuple(side2)]
    return False

def det_3pts(points, dict_pts):
    coord = []
    mdet = 0
    m = np.ones((3, 3))
    for i in range(0, len(points)):
        coord = dict_pts[points[i]]
        m[i, 1] = coord[0]
        m[i, 2] = coord[1]
    mdet = round(np.linalg.det(m), 5)
    return mdet


def det_4pts(points, dict_pts):
    def z(x, y):
        f = round(x**2 + y**2, 5)
        return f

    coord = []
    mdet = 0
    m = np.ones((4, 4))
    for i in range(0, len(points)):
        coord = dict_pts[points[i]]
        m[i, 1] = coord[0]
        m[i, 2] = coord[1]
        m[i, 3] = z(coord[0], coord[1])
    mdet = round(np.linalg.det(m), 5)
    return mdet


def add_edges(ax, verts, edges, alpha=0.1, c='b'):
    segs
    segs[:, :, 1] = ys
    segs[:, :, 0] = x
    line_segments = LineCollection(segs, linewidths=(0.5, 1, 1.5, 2),
                               colors=colors, linestyle='solid')

def add_mesh(ax, verts, faces, alpha=0.1, c='b'):
    mesh = PolyCollection(verts[faces], alpha=alpha)
    if c == 'b':
        face_color = (141 / 255, 184 / 255, 226 / 255)
    elif c == 'r':
        face_color = (226 / 255, 184 / 255, 141 / 255)
    edge_color = (50 / 255, 50 / 255, 50 / 255)
    mesh.set_edgecolor(edge_color)
    mesh.set_facecolor(face_color)
    ax.add_collection(mesh)

if __name__ == '__main__':
    with open("/home/dragon/Documents/yyp/Q1-input1.txt") as f:
        lines1 = f.readlines()

    lines = []
    for i in range(0, len(lines1)):
        if lines1[i] != '\n':
            line = lines1[i].rstrip().split()
            lines.append(line)

    a = lines.index(['Points'])
    c = lines.index(['Faces'])
    b = lines.index(['Edges'])
    print(a)
    print(b)
    print(c)
    points0 = np.array(lines[a + 1:(b - 1)])
    edges0 = np.array(lines[(b + 1):(c - 1)], int)
    faces0 = np.array(lines[(c + 1):], int)

    points = {}
    for i in range(0, len(points0)):
        points[int(points0[i, 0])] = [float(points0[i, 1]), float(points0[i, 2])]

    dict_eid = {}
    for i in range(0, len(edges0)):
        edges0[i, 1:].sort()
        dict_eid[(edges0[i, 1], edges0[i, 2])] = edges0[i, 0]

    stack = []


    f = {}
    for i in range(0, len(faces0)):
        faces0[i, 1:].sort()
        judge = sidesMinAngle(faces0[i])
        print(judge)
        if judge:
          angle_eid1 = dict_eid[judge[0]]
          angle_eid2 = dict_eid[judge[1]]
          stack.append(angle_eid1)
          stack.append(angle_eid2)

        key1 = str(faces0[i, 1]) + '-' + str(faces0[i, 2])
        key2 = str(faces0[i, 1]) + '-' + str(faces0[i, 3])
        key3 = str(faces0[i, 2]) + '-' + str(faces0[i, 3])

        if key1 not in f:
            f[key1] = [faces0[i, 0]]
        else:
            f[key1].append(faces0[i, 0])

        if key2 not in f:
            f[key2] = [faces0[i, 0]]
        else:
            f[key2].append(faces0[i, 0])

        if key3 not in f:
            f[key3] = [faces0[i, 0]]
        else:
            f[key3].append(faces0[i, 0])

    iterate = 0
    while len(stack) != 0:
        print(stack)
        print(len(stack))
        uv = stack.pop()
        t = []
        other_edges = []
        u = edges0[uv][1]
        v = edges0[uv][2]
        id = f[str(u) + '-' + str(v)]
        print(id)
        if len(id) > 1:
            t.append(faces0[id[0], 1:])
            t.append(faces0[id[1], 1:])
            #p:id[0]|q:id[1]
            print(t)
            print(list(set(t[0]) - set(t[1])))
            p = list(set(t[0]) - set(t[1]))[0]
            q = list(set(t[1]) - set(t[0]))[0]
            #find other edges sorting in an increasing order and put to a list
            o1 = [u, p]
            o1.sort()
            o2 = [u, q]
            o2.sort()
            o3 = [v, p]
            o3.sort()
            o4 = [v, q]
            o4.sort()
            #up uq vp vq

            other_edges.append(tuple(o1))
            other_edges.append(tuple(o2))
            other_edges.append(tuple(o3))
            other_edges.append(tuple(o4))

            # construct condition to test if edge uv is locally delaunay
            points3 = [u, v, p]
            points4 = [u, v, p, q]
            delta = det_3pts(points3, points)
            gamma = det_4pts(points4, points)
            #above is all about find the coordinate or four points constructing 2 triangles
            if delta * gamma < 0:
                iterate += 1
                #print('change edge')
                #update edges0 list
                sbst_e = [p, q]
                sbst_e.sort()
                edges0[uv, 1:] = np.array(sbst_e)
                #update dict_eid
                del dict_eid[(u, v)]
                dict_eid[(sbst_e[0], sbst_e[1])] = uv
                #id = face id
                #update faces0 list
                sbst_face1 = [u, p, q]
                sbst_face1.sort()
                sbst_face2 = [v, p, q]
                sbst_face2.sort()
                faces0[id[0], 1:] = np.array(sbst_face1)
                faces0[id[1], 1:] = np.array(sbst_face2)
                #update dict['edges': [faceID]]
                del f[str(u) + '-' + str(v)]
                f[str(sbst_e[0]) + '-' + str(sbst_e[1])] = id
                #update faceID for edges who belong to another triangle after flipped
                oe2 = str(o2[0]) + '-' + str(o2[1])
                oe3 = str(o3[0]) + '-' + str(o3[1])
                # uq
                f[oe2].remove(id[1])
                f[oe2].append(id[0])
                #vp
                f[oe3].remove(id[0])
                f[oe3].append(id[1])

    print(iterate)

    pps = np.array(points0[:,1:])
    ffs = np.array(faces0[:,1:])
    print(pps.shape, ffs.shape)

    fig = plt.figure()
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)  # get rid of margins

    ax = fig.add_subplot(111)
    ax.axis('off')

    # We will convert this into a single mesh, and then use add_group_meshs to plot it in 3D
    verts = np.array([[0, 0],
                        [1, 1],
                        [0, 1],
                        [0.5, 0.5],
                        [0.8, 1.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 4]])
    edges = np.array([[0, 1],[1, 2],[0, 2],[0, 4]])
    # add_mesh(ax, verts=pps, faces=ffs, alpha=0.1, c='b')
    add_edges(ax, verts=pps, edges=ffs, alpha=0.1, c='b')
    plt.show()
