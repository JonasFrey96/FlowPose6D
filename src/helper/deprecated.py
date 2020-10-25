
def get_ref_ite(exp):
    # Hyper parameters that should  be moved to config
    refine_iterations = exp.get(
        'training', {}).get('refine_iterations', 1)
    rand = exp.get(
        'training', {}).get('refine_iterations_range', 0)
    # uniform distributions of refine iterations +- refine_iterations_range
    # default: refine_iterations = refine_iterations
    if rand > 0:
        refine_iterations = random.randrange(
            refine_iterations - rand)
    return refine_iterations

def get_inital(mode, gt_rot_wxyz, gt_trans, pred_r_current, pred_t_current, cfg={}, d='cpu'):
    if mode == 'DenseFusionInit':
        pred_rot_wxyz = pred_r_current
        pred_trans = pred_t_current
    elif mode == 'TransNoise':
        n = 0
        m = cfg.get('translation_noise_inital', 0.01)
    elif mode == 'RotTransNoise':
        n = cfg.get('rot_noise_deg_inital', 10)
        m = cfg.get('translation_noise_inital', 0.01)
    elif mode == 'RotNoise':
        n = cfg.get('rot_noise_deg_inital', 10)
        m = 0
    else:
        raise AssertionError

    if not mode == 'DenseFusionInit':

        r = R.from_euler('zyx', np.random.normal(
            0, n, (gt_trans.shape[0], 3)), degrees=True)
        a = RearangeQuat(gt_trans.shape[0])
        tn = a(torch.tensor(r.as_quat(), device=d), 'xyzw')

        pred_rot_wxyz = compose_quat(
            gt_rot_wxyz, tn)
        pred_trans = torch.normal(mean=gt_trans, std=m)
    return pred_rot_wxyz, pred_trans


def ret_cropped_image(img):
    test = img.nonzero(as_tuple=False)
    a = torch.max(test[:, 0]) + 1
    b = torch.max(test[:, 1]) + 1
    c = torch.max(test[:, 2]) + 1
    return img[:a, :b, :c]

# TODO Move if finalized to other files


def padded_cat(list_of_images, device):
    """returns torch.tensor of concatenated images with dim = max size of image padded with zeros

    Args:
        list_of_images ([type]): List of Images Channels x Heigh x Width

    Returns:
        padded_cat [type]: Tensor of concatination result len(list_of_images) x Channels x max(Height) x max(Width)
        valid_indexe: len(list_of_images) x 2
    """
    c = list_of_images[0].shape[0]
    h = [x.shape[1] for x in list_of_images]
    w = [x.shape[2] for x in list_of_images]
    max_h = max(h)
    max_w = max(w)
    padded_cat = torch.zeros(
        (len(list_of_images), c, max_h, max_w), device=device)
    for i, img in enumerate(list_of_images):
        padded_cat[i, :, :h[i], :w[i]] = img

    valid_indexes = torch.tensor([h, w], device=device)
    return padded_cat, valid_indexes
