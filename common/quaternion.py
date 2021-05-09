import torch
import numpy as np
import numpy.testing as npt
# PyTorch-backed implementations

def normalize(q):
    assert q.shape[-1] == 4
    norm = q.norm(dim=-1, keepdim=True)
    return q.div(norm)


def assert_normalized(q, atol=1e-3):
    assert q.shape[-1] == 4
    norm = q.norm(dim=-1)
    #norm = norm.cpu()
    norm_check =  (norm - 1.0).abs()
    try:
        assert torch.max(norm_check) < atol
    except:
        print("normalization failure: {}.".format(torch.max(norm_check)))
        return -1
    return 0
    #npt.assert_allclose(norm_check, np.zeros_like(norm_check), rtol=0, atol=atol)


def multiply(q, r):
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    assert q.shape == r.shape
    
    original_shape = q.shape

    real1, im1 = q.split([1, 3], dim=-1)
    real2, im2 = r.split([1, 3], dim=-1)

    real = real1*real2 -  torch.sum(im1*im2, dim=-1, keepdim=True) 
    im = real1*im2 + real2*im1 + im1.cross(im2)
    return torch.cat((real, im), dim=-1)


def conjugate(q):
    assert q.shape[-1] == 4
    w, xyz = q.split([1, 3], dim=-1)
    return torch.cat((w, -xyz), dim=-1)


def rotate(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    #print('inside rotate: ', q.shape, q.dtype, v.shape, v.dtype)
    assert q.shape[:-1] == v.shape[:-1]
    
    original_shape = list(v.shape)
    assert_normalized(q)
    
    zeros = torch.zeros(original_shape[:-1]+[1])
    qv = torch.cat((zeros, v), dim=-1)

    result = multiply(multiply(q, qv), conjugate(q))
    _, xyz = result.split([1, 3], dim=-1)
    return xyz.view(original_shape)


def relative_angle(q, r):
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    assert q.shape == r.shape

    assert_normalized(q)
    assert_normalized(r)

    dot_product = torch.sum(q*r, dim=-1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    angle = 2*torch.acos(dot_product.abs())

    return angle

def unit_quaternion_to_matrix(q):
    assert_normalized(q)
    w, x, y, z= torch.unbind(q, dim=-1)
    matrix = torch.stack(( 1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y* w,
                        2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w,  
                        2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x -2*y*y),
                        dim=-1)
    matrix_shape = list(matrix.shape)[:-1]+[3,3]
    return matrix.view(matrix_shape).contiguous()

def matrix_to_unit_quaternion(matrix):
    assert matrix.shape[-1] == matrix.shape[-2] == 3
    matrix = torch.tensor(matrix)

    r = torch.sqrt(1 + matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2])
    s = 1.0/(2*r + 1e-7)
    w = 0.5*r
    x = (matrix[..., 2, 1] - matrix[..., 1, 2])*s
    y = (matrix[..., 0, 2] - matrix[..., 2, 0])*s
    z = (matrix[..., 1, 0] - matrix[..., 0, 1])*s
    q = torch.cat((w, x, y, z), dim=-1) if w.dim() >= 1 else torch.stack((w, x, y, z))

    return normalize(q) 

def qeuler(q, order, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    
    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)
    
    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]
    
    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1+epsilon, 1-epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1+epsilon, 1-epsilon))
    elif order == 'zxy':
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1+epsilon, 1-epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1+epsilon, 1-epsilon))
    elif order == 'yxz':
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1+epsilon, 1-epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2*(q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1+epsilon, 1-epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    else:
        raise

    return torch.stack((x, y, z), dim=1).view(original_shape)

# Numpy-backed implementations

def qmul_np(q, r):
    q = torch.from_numpy(q).contiguous()
    r = torch.from_numpy(r).contiguous()
    return qmul(q, r).numpy()

def qrot_np(q, v):
    q = torch.from_numpy(q).contiguous()
    v = torch.from_numpy(v).contiguous()
    return qrot(q, v).numpy()

def qeuler_np(q, order, epsilon=0, use_gpu=False):
    if use_gpu:
        q = torch.from_numpy(q).cuda()
        return qeuler(q, order, epsilon).cpu().numpy()
    else:
        q = torch.from_numpy(q).contiguous()
        return qeuler(q, order, epsilon).numpy()

def qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.
    
    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4
    
    result = q.copy()
    dot_products = np.sum(q[1:]*q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0)%2).astype(bool)
    result[1:][mask] *= -1
    return result

def expmap_to_quaternion(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3
    
    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)

    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5*theta).reshape(-1, 1)
    xyz = 0.5*np.sinc(0.5*theta/np.pi)*e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)

def euler_to_quaternion(e, order):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3
    
    original_shape = list(e.shape)
    original_shape[-1] = 4
    
    e = e.reshape(-1, 3)
    
    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]
    
    rx = np.stack((np.cos(x/2), np.sin(x/2), np.zeros_like(x), np.zeros_like(x)), axis=1)
    ry = np.stack((np.cos(y/2), np.zeros_like(y), np.sin(y/2), np.zeros_like(y)), axis=1)
    rz = np.stack((np.cos(z/2), np.zeros_like(z), np.zeros_like(z), np.sin(z/2)), axis=1)

    result = None
    for coord in order:
        if coord == 'x':
            r = rx
        elif coord == 'y':
            r = ry
        elif coord == 'z':
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul_np(result, r)
            
    # Reverse antipodal representation to have a non-negative "w"
    if order in ['xyz', 'yzx', 'zxy']:
        result *= -1
    
    return result.reshape(original_shape)

def generate_random_quaternion(shape):
    rand_norm = torch.randn(shape)
    rand_q = normalize(rand_norm)
    return rand_q
