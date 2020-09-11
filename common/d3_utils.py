import numpy as np
from math import pi ,sin, cos
import itertools
from pyquaternion import Quaternion
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# from He et. al
def get_3d_bbox(scale, shift = 0):
    """
    Input:
        scale: [3] or scalar
        shift: [3] or scalar
    Return
        bbox_3d: [3, N]

    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                  [scale / 2, +scale / 2, -scale / 2],
                  [-scale / 2, +scale / 2, scale / 2],
                  [-scale / 2, +scale / 2, -scale / 2],
                  [+scale / 2, -scale / 2, scale / 2],
                  [+scale / 2, -scale / 2, -scale / 2],
                  [-scale / 2, -scale / 2, scale / 2],
                  [-scale / 2, -scale / 2, -scale / 2]]) +shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d

def pts_inside_box(pts, bbox):
    # pts: N x 3
    # bbox: 8 x 3 (-1, 1, 1), (1, 1, 1), (1, -1, 1), (-1, -1, 1), (-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)
    u1 = bbox[5, :] - bbox[4, :]
    u2 = bbox[7, :] - bbox[4, :]
    u3 = bbox[0, :] - bbox[4, :]

    up = pts - np.reshape(bbox[4, :], (1, 3))
    p1 = np.matmul(up, u1.reshape((3, 1)))
    p2 = np.matmul(up, u2.reshape((3, 1)))
    p3 = np.matmul(up, u3.reshape((3, 1)))
    p1 = np.logical_and(p1>0, p1<np.dot(u1, u1))
    p2 = np.logical_and(p2>0, p2<np.dot(u2, u2))
    p3 = np.logical_and(p3>0, p3<np.dot(u3, u3))
    return np.logical_and(np.logical_and(p1, p2), p3)

def point_in_hull_slow(point, hull, tolerance=1e-12):
    """
    Check if a point lies in a convex hull.
    :param point: nd.array (1 x d); d: point dimension
    :param hull: The scipy ConvexHull object
    :param tolerance: Used to compare to a small positive constant because of issues of numerical precision
    (otherwise, you may find that a vertex of the convex hull is not in the convex hull)
    """
    return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations)

# def point_in_hull_fast(points: np.array, bounding_box: Box):
#     """
#     Check if a point lies in a bounding box. We first rotate the bounding box to align with axis. Meanwhile, we
#     also rotate the whole point cloud. Finally, we just check the membership with the aid of aligned axis.
#     :param points: nd.array (N x d); N: the number of points, d: point dimension
#     :param bounding_box: the Box object
#     return: The membership of points within the bounding box
#     """
#     # Make sure it is a unit quaternion
#     bounding_box.orientation = bounding_box.orientation.normalised

#     # Rotate the point clouds
#     pc = bounding_box.orientation.inverse.rotation_matrix @ points.T
#     pc = pc.T

#     orientation_backup = Quaternion(bounding_box.orientation)  # Deep clone it
#     bounding_box.rotate(bounding_box.orientation.inverse)

#     corners = bounding_box.corners()

#     # Test if the points are in the bounding box
#     idx = np.where((corners[0, 7] <= pc[:, 0]) & (pc[:, 0] <= corners[0, 0]) &
#                    (corners[1, 1] <= pc[:, 1]) & (pc[:, 1] <= corners[1, 0]) &
#                    (corners[2, 2] <= pc[:, 2]) & (pc[:, 2] <= corners[2, 0]))[0]

#     # recover
#     bounding_box.rotate(orientation_backup)

#     return idx

def iou_3d(bbox1, bbox2, nres=50):
    bmin = np.min(np.concatenate((bbox1, bbox2), 0), 0)
    bmax = np.max(np.concatenate((bbox1, bbox2), 0), 0)
    xs = np.linspace(bmin[0], bmax[0], nres)
    ys = np.linspace(bmin[1], bmax[1], nres)
    zs = np.linspace(bmin[2], bmax[2], nres)
    pts = np.array([x for x in itertools.product(xs, ys, zs)])
    flag1 = pts_inside_box(pts, bbox1)
    flag2 = pts_inside_box(pts, bbox2)
    intersect = np.sum(np.logical_and(flag1, flag2))
    union = np.sum(np.logical_or(flag1, flag2))
    if union==0:
        return 1
    else:
        return intersect/float(union)

# def scale_iou(sample_annotation: EvalBox, sample_result: EvalBox) -> float:
#     """
#     This method compares predictions to the ground truth in terms of scale.
#     It is equivalent to intersection over union (IOU) between the two boxes in 3D,
#     if we assume that the boxes are aligned, i.e. translation and rotation are considered identical.
#     :param sample_annotation: GT annotation sample.
#     :param sample_result: Predicted sample.
#     :return: Scale IOU.
#     """
#     # Validate inputs.
#     sa_size = np.array(sample_annotation.size)
#     sr_size = np.array(sample_result.size)
#     assert all(sa_size > 0), 'Error: sample_annotation sizes must be >0.'
#     assert all(sr_size > 0), 'Error: sample_result sizes must be >0.'

#     # Compute IOU.
#     min_wlh = np.minimum(sa_size, sr_size)
#     volume_annotation = np.prod(sa_size)
#     volume_result = np.prod(sr_size)
#     intersection = np.prod(min_wlh)  # type: float
#     union = volume_annotation + volume_result - intersection  # type: float
#     iou = intersection / union

#     return iou

def transform_coordinates_3d(coordinates, RT):
    """
    Input:
        coordinates: [3, N]
        RT: [4, 4]
    Return
        new_coordinates: [3, N]

    """
    if coordinates.shape[0] != 3 and coordinates.shape[1]==3:
        # print('transpose box channels')
        coordinates = coordinates.transpose()
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :]/new_coordinates[3, :]
    return new_coordinates

def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input:
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates


def compute_RT_distances(RT_1, RT_2):
    '''
    :param RT_1: [4, 4]. homogeneous affine transformation
    :param RT_2: [4, 4]. homogeneous affine transformation
    :return: theta: angle difference of R in degree, shift: l2 difference of T in centimeter
    '''
    #print(RT_1[3, :], RT_2[3, :])
    ## make sure the last row is [0, 0, 0, 1]
    if RT_1 is None or RT_2 is None:
        return -1

    try:
        assert np.array_equal(RT_1[3, :], RT_2[3, :])
        assert np.array_equal(RT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(RT_1[3, :], RT_2[3, :])

    R1 = RT_1[:3, :3]/np.cbrt(np.linalg.det(RT_1[:3, :3]))
    T1 = RT_1[:3, 3]

    R2 = RT_2[:3, :3]/np.cbrt(np.linalg.det(RT_2[:3, :3]))
    T2 = RT_2[:3, 3]

    R = R1 @ R2.transpose()
    theta = np.arccos((np.trace(R) - 1)/2) * 180/np.pi
    shift = np.linalg.norm(T1-T2) * 100
    # print(theta, shift)

    if theta < 5 and shift < 5:
        return 10 - theta - shift
    else:
        return -1

def axis_diff_degree(v1, v2):
    v1 = v1.reshape(-1)
    v2 = v2.reshape(-1)
    r_diff = np.arccos(np.sum(v1*v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))) * 180 / np.pi
    # print(r_diff)
    return min(r_diff, 180-r_diff)

def rot_diff_degree(rot1, rot2):
    return rot_diff_rad(rot1, rot2) / np.pi * 180

def rot_diff_rad(rot1, rot2):
    return np.arccos( ( np.trace(np.matmul(rot1, rot2.T)) - 1 ) / 2 ) % (2*np.pi)

def rotate_points_with_rotvec(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def dist_between_3d_lines(p1, e1, p2, e2):
    p1 = p1.reshape(-1)
    p2 = p2.reshape(-1)
    e1 = e1.reshape(-1)
    e2 = e2.reshape(-1)
    orth_vect = np.cross(e1, e2)
    product = np.sum(orth_vect * (p1 - p2))
    dist = product / np.linalg.norm(orth_vect)

    return np.abs(dist)

def project3d(pcloud_target, projMat, height=512, width=512):
    pcloud_projected = np.dot(pcloud_target, projMat.T)
    pcloud_projected_ndc = pcloud_projected/pcloud_projected[:, 3:4]
    img_coord = (pcloud_projected_ndc[:, 0:2] + 1)/(1/256)
    print('transformed image coordinates:\n', img_coord.shape)
    u = img_coord[:, 0]
    v = img_coord[:, 1]
    u = u.astype(np.int16)
    v = v.astype(np.int16)
    v = 512 - v
    print('u0, v0:\n', u[0], v[0])
    # rgb_raw[v, u]   = 250*np.array([0, 0, 1])              #rgb_raw[u, v] +

    return u, v # x, y in cv coords


def point_3d_offset_joint(joint, point):
    """
    joint: [x, y, z] or [[x, y, z] + [rx, ry, rz]]
    point: N * 3
    """
    if len(joint) == 2:
        P0 = np.array(joint[0])
        P  = np.array(point)
        l  = np.array(joint[1]).reshape(1, 3)
        P0P= P - P0
        PP = np.dot(P0P, l.T) * l / np.linalg.norm(l)**2  - P0P
    return PP


def rotate_pts(source, target):
    '''
    func: compute rotation between source: [N x 3], target: [N x 3]
    '''
    # pre-centering
    source = source - np.mean(source, 0, keepdims=True)
    target = target - np.mean(target, 0, keepdims=True)
    M = np.matmul(target.T, source)
    U, D, Vh = np.linalg.svd(M, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]
    R = np.matmul(U, Vh)
    return R


def transform_pts(source, target):
    # source: [N x 3], target: [N x 3]
    # pre-centering and compute rotation
    source_centered = source - np.mean(source, 0, keepdims=True)
    target_centered = target - np.mean(target, 0, keepdims=True)
    rotation = rotate_pts(source_centered, target_centered)

    scale = scale_pts(source_centered, target_centered)

    # compute translation
    translation = np.mean(target.T-scale*np.matmul(rotation, source.T), 1)
    return rotation, scale, translation


def scale_pts(source, target):
    '''
    func: compute scaling factor between source: [N x 3], target: [N x 3]
    '''
    pdist_s = source.reshape(source.shape[0], 1, 3) - source.reshape(1, source.shape[0], 3)
    A = np.sqrt(np.sum(pdist_s**2, 2)).reshape(-1)
    pdist_t = target.reshape(target.shape[0], 1, 3) - target.reshape(1, target.shape[0], 3)
    b = np.sqrt(np.sum(pdist_t**2, 2)).reshape(-1)
    scale = np.dot(A, b) / (np.dot(A, A)+1e-6)
    return scale

def compute_3d_rotation_axis(pts_0, pts_1, rt, orientation=None, line_pts=None, methods='H-L', item='eyeglasses', viz=False):
    """
    pts_0: points in NOCS space of cannonical status(scaled)
    pts_1: points in camera space retrieved from depth image;
    rt: rotation + translation in 4 * 4
    """
    num_parts = len(rt)
    print('we have {} parts'.format(num_parts))

    chained_pts = [None] * num_parts
    delta_Ps = [None] * num_parts
    chained_pts[0] = np.dot( np.concatenate([ pts_0[0], np.ones((pts_0[0].shape[0], 1)) ], axis=1), rt[0].T )
    axis_list = []
    angle_list= []
    if item == 'eyeglasses':
        for j in range(1, num_parts):
            chained_pts[j] = np.dot(np.concatenate([ pts_0[j], np.ones((pts_0[j].shape[0], 1)) ], axis=1), rt[0].T)

            if methods == 'H-L':
                RandIdx = np.random.randint(chained_pts[j].shape[1], size=5)
                orient, position= estimate_joint_HL(chained_pts[j][RandIdx, 0:3], pts_1[j][RandIdx, 0:3])
                joint_axis = {}
                joint_axis['orient']   = orient
                joint_axis['position'] = position
                source_offset_arr= point_3d_offset_joint([position.reshape(1, 3), orient], chained_pts[j][RandIdx, 0:3])
                rotated_offset_arr= point_3d_offset_joint([position.reshape(1, 3), orient.reshape(1, 3)], pts_1[j][RandIdx, 0:3])
                angle = []
                for m in range(RandIdx.shape[0]):
                    modulus_0 = np.linalg.norm(source_offset_arr[m, :])
                    modulus_1 = np.linalg.norm(rotated_offset_arr[m, :])
                    cos_angle = np.dot(source_offset_arr[m, :].reshape(1, 3), rotated_offset_arr[m, :].reshape(3, 1))/(modulus_0 * modulus_1)
                    angle_per_pair = np.arccos(cos_angle)
                    angle.append(angle_per_pair)
                print('angle per pair from multiple pairs: {}', angle)
                angle_list.append(sum(angle)/len(angle))

            axis_list.append(joint_axis)
            angle_list.append(angle)

    return axis_list, angle_list

def point_rotate_about_axis(pts, anchor, unitvec, theta):
    a, b, c = anchor.reshape(3)
    u, v, w = unitvec.reshape(3)
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    ss =  u*x + v*y + w*z
    x_rotated = (a*(v**2 + w**2) - u*(b*v + c*w - ss)) * (1 - cos(theta)) + x * cos(theta) + (-c*v + b*w - w*y + v*z) * sin(theta)
    y_rotated = (b*(u**2 + w**2) - v*(a*u + c*w - ss)) * (1 - cos(theta)) + y * cos(theta) + (c*u - a*w + w*x - u*z) * sin(theta)
    z_rotated = (c*(u**2 + v**2) - w*(a*u + b*v - ss)) * (1 - cos(theta)) + z * cos(theta) + (-b*u + a*v - v*x + u*y) * sin(theta)
    rotated_pts = np.zeros_like(pts)
    rotated_pts[:, 0] = x_rotated
    rotated_pts[:, 1] = y_rotated
    rotated_pts[:, 2] = z_rotated

    return rotated_pts

# def yaw_diff(gt_box: EvalBox, eval_box: EvalBox, period: float = 2*np.pi) -> float:
#     """
#     Returns the yaw angle difference between the orientation of two boxes.
#     :param gt_box: Ground truth box.
#     :param eval_box: Predicted box.
#     :param period: Periodicity in radians for assessing angle difference.
#     :return: Yaw angle difference in radians in [0, pi].
#     """
#     yaw_gt = quaternion_yaw(Quaternion(gt_box.rotation))
#     yaw_est = quaternion_yaw(Quaternion(eval_box.rotation))

#     return abs(angle_diff(yaw_gt, yaw_est, period))


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw

def estimate_joint_HL(source_pts, rotated_pts):
    # estimate offsets
    delta_P = rotated_pts - source_pts
    assert delta_P.shape[1] == 3, 'points coordinates dimension is wrong, current is {}'.format(delta_P.shape)
    mid_pts = (source_pts + rotated_pts)/2
    CC      = np.zeros((3, 3), dtype=np.float32)
    BB      = np.zeros((delta_P.shape[0], 1), dtype=np.float32)
    for j in range(0, delta_P.shape[0]):
        CC += np.dot(delta_P[j, :].reshape(3, 1), delta_P[j, :].reshape(1, 3))
        BB[j] = np.dot(delta_P[j, :].reshape(1, 3), mid_pts[j, :].reshape((3, 1)))
    w, v   = np.linalg.eig(CC)
    print('eigen vectors are: \n', v)
    print('eigne values are: \n', w)
    orient = v[:, np.argmin(np.squeeze(w))].reshape(3, 1)

    # we already decouple the orient & position
    mat_1 = np.linalg.pinv( np.dot(delta_P.T, delta_P) )

    position = np.dot( np.dot(mat_1, delta_P.T), BB)
    print('orient has shape {}, position has shape {}'.format(orient.shape, position.shape))

    return orient, position


# def calc_displace_vector(points: np.array, curr_box: Box, next_box: Box):
#     """
#     Calculate the displacement vectors for the input points.
#     This is achieved by comparing the current and next bounding boxes. Specifically, we first rotate
#     the input points according to the delta rotation angle, and then translate them. Finally we compute the
#     displacement between the transformed points and the input points.
#     :param points: The input points, (N x d). Note that these points should be inside the current bounding box.
#     :param curr_box: Current bounding box.
#     :param next_box: The future next bounding box in the temporal sequence.
#     :return: Displacement vectors for the points.
#     """
#     assert points.shape[1] == 3, "The input points should have dimension 3."

#     # Make sure the quaternions are normalized
#     curr_box.orientation = curr_box.orientation.normalised
#     next_box.orientation = next_box.orientation.normalised

#     delta_rotation = curr_box.orientation.inverse * next_box.orientation
#     rotated_pc = (delta_rotation.rotation_matrix @ points.T).T
#     rotated_curr_center = np.dot(delta_rotation.rotation_matrix, curr_box.center)
#     delta_center = next_box.center - rotated_curr_center

#     rotated_tranlated_pc = rotated_pc + delta_center

#     pc_displace_vectors = rotated_tranlated_pc - points

#     return pc_displace_vectors

def voxelize(pts, voxel_size, extents=None, num_T=35, seed: float = None):
    """
    Voxelize the input point cloud. Code modified from https://github.com/Yc174/voxelnet
    Voxels are 3D grids that represent occupancy info.

    The input for the voxelization is expected to be a PointCloud
    with N points in 4 dimension (x,y,z,i). Voxel size is the quantization size for the voxel grid.

    voxel_size: I.e. if voxel size is 1 m, the voxel space will be
    divided up within 1m x 1m x 1m space. This space will be -1 if free/occluded and 1 otherwise.
    min_voxel_coord: coordinates of the minimum on each axis for the voxel grid
    max_voxel_coord: coordinates of the maximum on each axis for the voxel grid
    num_divisions: number of grids in each axis
    leaf_layout: the voxel grid of size (numDivisions) that contain -1 for free, 0 for occupied

    :param pts: Point cloud as N x [x, y, z, i]
    :param voxel_size: Quantization size for the grid, vd, vh, vw
    :param extents: Optional, specifies the full extents of the point cloud.
                    Used for creating same sized voxel grids. Shape (3, 2)
    :param num_T: Number of points in each voxel after sampling/padding
    :param seed: The random seed for fixing the data generation.
    """
    # Check if points are 3D, otherwise early exit
    if pts.shape[1] < 3 or pts.shape[1] > 4:
        raise ValueError("Points have the wrong shape: {}".format(pts.shape))

    if extents is not None:
        if extents.shape != (3, 2):
            raise ValueError("Extents are the wrong shape {}".format(extents.shape))

        filter_idx = np.where((extents[0, 0] < pts[:, 0]) & (pts[:, 0] < extents[0, 1]) &
                              (extents[1, 0] < pts[:, 1]) & (pts[:, 1] < extents[1, 1]) &
                              (extents[2, 0] < pts[:, 2]) & (pts[:, 2] < extents[2, 1]))[0]
        pts = pts[filter_idx]

    # Discretize voxel coordinates to given quantization size
    discrete_pts = np.floor(pts[:, :3] / voxel_size).astype(np.int32)

    # Use Lex Sort, sort by x, then y, then z
    x_col = discrete_pts[:, 0]
    y_col = discrete_pts[:, 1]
    z_col = discrete_pts[:, 2]
    sorted_order = np.lexsort((z_col, y_col, x_col))

    # Save original points in sorted order
    points = pts[sorted_order]
    discrete_pts = discrete_pts[sorted_order]

    # Format the array to c-contiguous array for unique function
    contiguous_array = np.ascontiguousarray(discrete_pts).view(
        np.dtype((np.void, discrete_pts.dtype.itemsize * discrete_pts.shape[1])))

    # The new coordinates are the discretized array with its unique indexes
    _, unique_indices = np.unique(contiguous_array, return_index=True)

    # Sort unique indices to preserve order
    unique_indices.sort()

    voxel_coords = discrete_pts[unique_indices]

    # Number of points per voxel, last voxel calculated separately
    num_points_in_voxel = np.diff(unique_indices)
    num_points_in_voxel = np.append(num_points_in_voxel, discrete_pts.shape[0] - unique_indices[-1])

    # Compute the minimum and maximum voxel coordinates
    if extents is not None:
        min_voxel_coord = np.floor(extents.T[0] / voxel_size)
        max_voxel_coord = np.ceil(extents.T[1] / voxel_size) - 1
    else:
        min_voxel_coord = np.amin(voxel_coords, axis=0)
        max_voxel_coord = np.amax(voxel_coords, axis=0)

    # Get the voxel grid dimensions
    num_divisions = ((max_voxel_coord - min_voxel_coord) + 1).astype(np.int32)

    # Bring the min voxel to the origin
    voxel_indices = (voxel_coords - min_voxel_coord).astype(int)

    # Padding the points within each voxel
    padded_voxel_points = np.zeros([unique_indices.shape[0], num_T, pts.shape[1] + 3], dtype=np.float32)
    padded_voxel_points = padding_voxel(padded_voxel_points, unique_indices, num_points_in_voxel, points, num_T, seed)

    return padded_voxel_points, voxel_indices, num_divisions
    
if __name__ == '__main__':
    #>>>>>>>>> 3D IOU compuatation
    from scipy.spatial.transform import Rotation
    bbox1 = np.array([[-1, 1, 1], [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, -1], [1, 1, -1], [1, -1, -1], [-1, -1, -1]])
    print('bbox1.shape: ', bbox1.shape)
    rotmatrix = Rotation.from_rotvec(np.pi/4 * np.array([np.sqrt(3)/3, np.sqrt(3)/3, np.sqrt(3)/3])).as_dcm()
    bbox2 = np.matmul(bbox1, rotmatrix.T)
    bbox3 = bbox1 + np.array([[1, 0, 0]])
    rotmatrix2 = Rotation.from_rotvec(np.pi/4 * np.array([0, 0, 1])).as_dcm()
    bbox4 = np.matmul(bbox1, rotmatrix2.T)
    bbox5 = bbox1 + np.array([[2, 0, 0]])
    print(iou_3d(bbox1, bbox1))
    print(iou_3d(bbox1, bbox2))
    print(iou_3d(bbox1, bbox3))
    print(iou_3d(bbox1, bbox4))
    print(iou_3d(bbox1, bbox5))
    #>>>>>>>>> test for joint parameters fitting 
    source_pts  = np.array([[5, 1, 5], [0, 0, 1], [0.5,0.5,0.5], [2, 0, 1], [3, 3, 5]])
    p1 = np.array([0,0,0])
    p2 = np.array([1,1,1])
    unitvec = (p2 - p1) / np.linalg.norm(p2 - p1)
    anchor  = p1
    rotated_pts = point_rotate_about_axis(source_pts, anchor, unitvec, pi)
    joint_axis, position = estimate_joint_HL(source_pts, rotated_pts)
    print(joint_axis, position)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(source_pts[:, 0], source_pts[:, 1], source_pts[:, 2], c='r',
               marker='o', label='source pts')
    ax.scatter(rotated_pts[:, 0], rotated_pts[:, 1], rotated_pts[:, 2], c='b',
               marker='o', label='rotated pts')
    linepts = unitvec * np.mgrid[-5:5:2j][:, np.newaxis] + np.array(p1).reshape(1, 3)
    ax.plot3D(*linepts.T, linewidth=5, c='green')
    ax.legend(loc='lower left')
    plt.show()
