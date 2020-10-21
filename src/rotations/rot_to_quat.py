import torch
if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.getcwd())
    sys.path.append(os.path.join(os.getcwd() + '/src'))
    sys.path.append(os.path.join(os.getcwd() + '/lib'))

from helper import re_quat
from rotations import norm_quat
def _copysign(a, b):
    """ From PyTorch3D see def _copysign(a, b)
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)

def rot_to_quat(matrix, conv='wxyz'):

    """From PyTorch3D see def matrix_to_quaternion(matrix)
    Args:
        rot ([type]): [description]
        conv (str, optional): [description]. Defaults to 'wxyz'.
    """

    if matrix.shape == (3, 3):
        matrix = matrix.reshape((1, 3, 3))
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")

    zero = matrix.new_zeros((1,))
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * torch.sqrt(torch.max(zero, 1 + m00 + m11 + m22))
    x = 0.5 * torch.sqrt(torch.max(zero, 1 + m00 - m11 - m22))
    y = 0.5 * torch.sqrt(torch.max(zero, 1 - m00 + m11 - m22))
    z = 0.5 * torch.sqrt(torch.max(zero, 1 - m00 - m11 + m22))
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])

    if conv == 'xyzw':
        return norm_quat(torch.stack((o1, o2, o3, o0), -1))
    elif conv == 'wxyz':
        return norm_quat(torch.stack((o0, o1, o2, o3), -1))
    else:
        raise Exception('undefined quaternion convention')

def test_rot_to_quat():
    from scipy.spatial.transform import Rotation as R
    import numpy as np
    from scipy.stats import special_ortho_group
    from rotations import RearangeQuat
    import time
    bs = 1000
    re_q = RearangeQuat(bs)
    mat = special_ortho_group.rvs(dim=3, size=bs)

    quat = R.from_matrix(mat).as_quat()
    q_test = rot_to_quat(torch.tensor(mat), conv='wxyz')

    print(quat,'\n \n ', q_test)
    m = q_test[:,0] > 0

    mat2 = R.from_quat(  q_test.numpy() ).as_matrix()

    print("Fiff", torch.sum(torch.norm( torch.tensor(mat-mat2), dim=(1,2) ), dim=0))
   
    #print( "DIF", torch.sum(torch.norm( torch.tensor(quat[m]) - q_test[m], dim=1 ), dim=0))
    
    # q = torch.from_numpy(quat.astype(np.float32)).cuda()
    # re_q(q, input_format='xyzw')

    # mat2 = special_ortho_group.rvs(dim=3, size=bs)
    # quat2 = R.from_matrix(mat2).as_quat()
    # q2 = torch.from_numpy(quat2.astype(np.float32)).cuda()
    # re_q(q2, input_format='xyzw')

    # r1 = R.from_matrix(mat)
    # R_out = r1 * R.from_matrix(mat2)

    # print(f'scipy xyzw {R_out.as_quat()}')

    # st = time.time()
    # for i in range(0, 1000):
    #     out = compose_quat(q, q2)
    # print(f'torch wxyz { compose_quat(q, q2) } ')
    # print(f'took for 1000 iterations of {bs} bs {time.time()-st}s')


if __name__ == "__main__":
    test_rot_to_quat()
    pass