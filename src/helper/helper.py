# import yagmail
from sklearn.neighbors import NearestNeighbors
import yaml
import numpy as np
import collections

import torch
import copy

# orig
# def batched_index_select(t, inds, dim=1):
#     """index batch tensor

#     Args:
#         t ([torch.Tensor]): BS x select_dim x Features
#         dim ([int]): select_dim = 1
#         inds ([torch.Tensor]): BS x select_dim

#     Returns:
#         [type]: [description]
#     """
#     dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
#     out = t.gather(dim, dummy)  # b x e x f
#     return out


def batched_index_select(t, inds, dim=1):
    """index batch tensor

    Args:
        t ([torch.Tensor]): BS x select_dim x Features or BS x select_dim x Feat1 x Feat2 
        dim ([int]): select_dim = 1
        inds ([torch.Tensor]): BS x select_dim

    Returns:
        [type]: [description]
    """
    if len(t.shape) == 3:
        dummy = inds.unsqueeze(2).expand(
            inds.size(0), inds.size(1), t.size(2))
    elif len(t.shape) == 4:
        dummy = inds.unsqueeze(2).unsqueeze(3).expand(
            inds.size(0), inds.size(1), t.size(2), t.size(3))
    elif len(t.shape) == 5:
        dummy = inds[:, :, None, None, None].expand(
            inds.size(0), inds.size(1), t.size(2), t.size(3), t.size(4))
    out = t.gather(dim, dummy)  # b x e x f
    return out


def quat_to_rot(rot, conv='wxyz', device='cpu'):
    """converts quat into rotation matrix
    Args:
        rot ([type]): [description]
        conv (str, optional): [description]. Defaults to 'wxyz'.

    Raises:
        Exception: [description]

    Returns:
        [type]: [description]
    """
    if conv == 'wxyz':
        w = rot[:, 0]
        x = rot[:, 1]
        y = rot[:, 2]
        z = rot[:, 3]
    elif conv == 'xyzw':
        y = rot[:, 1]
        z = rot[:, 2]
        w = rot[:, 3]
        x = rot[:, 0]
    else:
        raise Exception('undefined quaternion convention')

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    num_rotations = rot.shape[0]
    matrix = torch.empty((num_rotations, 3, 3), device=device)

    matrix[:, 0, 0] = x2 - y2 - z2 + w2
    matrix[:, 1, 0] = 2 * (xy + zw)
    matrix[:, 2, 0] = 2 * (xz - yw)

    matrix[:, 0, 1] = 2 * (xy - zw)
    matrix[:, 1, 1] = - x2 + y2 - z2 + w2
    matrix[:, 2, 1] = 2 * (yz + xw)

    matrix[:, 0, 2] = 2 * (xz + yw)
    matrix[:, 1, 2] = 2 * (yz - xw)
    matrix[:, 2, 2] = - x2 - y2 + z2 + w2

    return matrix


def flatten_list(d, parent_key='', sep='_'):
    items = []
    for num, element in enumerate(d):
        new_key = parent_key + sep + str(num) if parent_key else str(num)

        if isinstance(element, collections.MutableMapping):
            items.extend(flatten_dict(element, new_key, sep=sep).items())
        else:
            if isinstance(element, list):
                if isinstance(element[0], dict):
                    items.extend(flatten_list(element, new_key, sep=sep))
                    continue
            items.append((new_key, element))
    return items


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            if isinstance(v, list):
                if isinstance(v[0], dict):
                    items.extend(flatten_list(v, new_key, sep=sep))
                    continue
            items.append((new_key, v))
    return dict(items)


def norm_quat(q):
    # ToDo raise type and dim error
    return q / torch.sqrt(torch.sum(q * q))


def pad(s, sym='-', p='l', length=80):
    if len(s) > length:
        return s
    else:
        if p == 'c':
            front = int((length - len(s)) / 2)
            s = sym * front + s
            back = int(length - len(s))
            s = s + sym * back
        if p == 'l':
            back = int(length - len(s))
            s = s + sym * back
        return s


def re_quat(q, input_format):

    if input_format == 'xyzw':
        if isinstance(q, torch.Tensor):
            v0 = q[0].clone()
        else:
            v0 = copy.deepcopy(q[0])

        q[0] = q[3]
        q[3] = q[2]
        q[2] = q[1]
        q[1] = v0
        return q
    elif input_format == 'wxyz':
        if isinstance(q, torch.Tensor):
            v0 = q[0].clone()
        else:
            v0 = copy.deepcopy(q[0])

        q[0] = q[1]
        q[1] = q[2]
        q[2] = q[3]
        q[3] = v0
        return q

# check if Tres is valid
def anal_tensor(T, name, print_on_error = False):
    if torch.any( torch.isnan(T) ) or torch.any( torch.isinf(T) ) and print_on_error:
        print( f'{name} Analyze Tensor {name}')
        print( f'       Shape  : {T.shape}')
        print( f'       Min    : {torch.min(T)}')
        print( f'       Max    : {torch.max(T)}')
        print( f'       NAN    : {torch.any( torch.isnan(T) ) }')
        print( f'       INF    : {torch.any( torch.isinf(T) )}')
        print( f'       Device : {T.device}')
        print( f'       DType  : {T.dtype}')
    return torch.any( torch.isnan(T) ) or torch.any( torch.isinf(T) )
    
def send_email(text):
    pass

def rotation_angle(q, device):
    # in radians
    q = norm_quat(q)
    unit_r = torch.t(torch.tensor(
        [[0, 0, 0, 1]], dtype=torch.float32, device=device))
    return torch.asin(torch.mm(q, unit_r)) * 2


def nearest_neighbor(src, dst):
    assert src.shape[1] == dst.shape[1]

    neigh = NearestNeighbors(n_neighbors=1, n_jobs=8)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def replace_item(obj, key, replace_value):
    for k, v in obj.items():
        if isinstance(v, dict):
            obj[k] = replace_item(v, key, replace_value)
    if key in obj:
        obj[key] = replace_value
    return obj


def generate_unique_idx(num, max_idx):
    a = random.sample(range(0, max_idx), k=min(num, max_idx))
    while len(a) < num:
        a = a + random.sample(
            range(0, max_idx), k=min(max_idx, num - len(a)))
    return a


def get_bbox_480_640(label):
    border_list = [-1, 40, 80, 120, 160, 200, 240, 280,
                   320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
    img_width = 480
    img_length = 640

    # print(type(label))
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


if __name__ == "__main__":
    from scipy.spatial.transform import Rotation as R
    import torch
    import numpy as np

    from scipy.stats import special_ortho_group
    nr = 1000
    x = special_ortho_group.rvs(dim=3, size=nr)
    rot_scipy = R.from_matrix(x)
    rot_mat_com = quat_to_rot(torch.from_numpy(
        rot_scipy.as_quat()), conv='xyzw')

    print('Average difference between rotation matrixes:',
          torch.sum(rot_mat_com - torch.from_numpy(x)) / nr)
