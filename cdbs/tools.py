from .pkgs import *



def seed_worker(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    np.random.seed(worker_info.seed % 2**32)


def align_number(raw_number, expected_num_digits):
    string_number = str(raw_number)
    ori_num_digits = len(string_number)
    assert ori_num_digits <= expected_num_digits
    return (expected_num_digits - ori_num_digits) * '0' + string_number


def load_pc(load_path, dlmt):
    pc = np.loadtxt(fname=load_path, dtype=np.float32, delimiter=dlmt)
    return pc


def save_pc(pc, save_path, dlmt):
    assert pc.ndim == 2
    assert type(pc) in [torch.Tensor, np.ndarray]
    if type(pc) == torch.Tensor:
        pc = pc.cpu().data.numpy()
    np.savetxt(save_path, pc, '%.6f', delimiter=dlmt)


def index_points(pc, idx):
    device = pc.device
    B = pc.shape[0]
    view_shape = list(idx.shape) 
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B).to(device).view(view_shape).repeat(repeat_shape)
    pc_selected = pc[batch_indices, idx, :]
    return pc_selected


def fps(xyz, num_sample):
    assert xyz.ndim==3 and xyz.size(2)==3
    batch_size, num_input, device = xyz.size(0), xyz.size(1), xyz.device
    batch_indices = torch.arange(batch_size, dtype=torch.long).to(device)
    fps_idx = torch.zeros(batch_size, num_sample, dtype=torch.long).to(device)
    distance = torch.ones(batch_size, num_input).to(device) * 1e10
    farthest = torch.randint(0, num_input, (batch_size,), dtype=torch.long).to(device)
    for i in range(num_sample):
        fps_idx[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, -1)
        dist = torch.sum((xyz-centroid)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return fps_idx


def min_max_normalization(x):
    x_min = x.min()
    x_max = x.max()
    x_mmn = (x - x_min) / (x_max - x_min)
    return x_mmn


def random_sampling(pc, num_sample):
    # pc: (num_points, num_channels)
    # pc_sampled: # [num_sample, num_channels]
    num_points, num_channels = pc.shape
    assert num_sample < num_points
    selected_indices = np.random.choice(num_points, num_sample, replace=False) # (num_sample,)
    pc_sampled = pc[selected_indices, :]
    return pc_sampled


def farthest_point_sampling(pc, num_sample):
    # pc: (num_points, num_channels)
    # pc_sampled: [num_sample, num_channels]
    num_points, num_channels = pc.shape
    assert num_sample < num_points
    xyz = pc[:, 0:3] # sampling is based on spatial distance
    centroids = np.zeros((num_sample,))
    distance = np.ones((num_points,)) * 1e10
    farthest = np.random.randint(0, num_points)
    for i in range(num_sample):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    pc_sampled = pc[centroids.astype(np.int32)]
    return pc_sampled


def axis_rotation(pc, angle, axis):
    # pc: (num_points, num_channels=3/6)
    # angle: [0, 2*pi]
    # axis: 'x', 'y', 'z'
    # pc_rotated: (num_points, num_channels=3/6)
    num_points, num_channels = pc.shape
    assert num_channels in [3, 6]
    assert angle>=0 and angle <= (2*np.pi)
    assert axis in ['x', 'y', 'z']
    # generate the rotation matrix
    c = np.cos(angle).astype(np.float32)
    s = np.sin(angle).astype(np.float32)
    if axis == 'x':
        rot_mat = np.array([ [1, 0, 0], [0, c, -s], [0, s, c] ]).astype(np.float32)
    if axis == 'y':
        rot_mat = np.array([ [c, 0, s], [0, 1, 0], [-s, 0, c] ]).astype(np.float32)
    if axis == 'z':
        rot_mat = np.array([ [c, -s, 0], [s, c, 0], [0, 0, 1] ]).astype(np.float32)
    # apply the rotation matrix
    if num_channels == 3:
        pc_rotated = np.matmul(pc, rot_mat) # (num_points, 3)
    if num_channels == 6:
        pc_rotated = np.concatenate((np.matmul(pc[:, 0:3], rot_mat), np.matmul(pc[:, 3:6], rot_mat)), axis=1) # (num_points, 6)
    return pc_rotated


def random_axis_rotation(pc, axis, return_angle=False):
    # pc: (num_points, num_channels=3/6)
    # axis: 'x', 'y', 'z'
    # pc_rotated: (num_points, num_channels=3/6)
    num_points, num_channels = pc.shape
    assert num_channels in [3, 6]
    assert axis in ['x', 'y', 'z']
    # generate a random rotation matrix
    angle = np.random.uniform() * 2 * np.pi
    c = np.cos(angle).astype(np.float32)
    s = np.sin(angle).astype(np.float32)
    if axis == 'x':
        rot_mat = np.array([ [1, 0, 0], [0, c, -s], [0, s, c] ]).astype(np.float32)
    if axis == 'y':
        rot_mat = np.array([ [c, 0, s], [0, 1, 0], [-s, 0, c] ]).astype(np.float32)
    if axis == 'z':
        rot_mat = np.array([ [c, -s, 0], [s, c, 0], [0, 0, 1] ]).astype(np.float32)
    # apply the rotation matrix
    if num_channels == 3:
        pc_rotated = np.matmul(pc, rot_mat) # (num_points, 3)
    if num_channels == 6:
        pc_rotated = np.concatenate((np.matmul(pc[:, 0:3], rot_mat), np.matmul(pc[:, 3:6], rot_mat)), axis=1) # (num_points, 6)
    if not return_angle:
        return pc_rotated
    else:
        return pc_rotated, angle
    
    
def random_rotation(pc, return_angle=False):
    # pc: (num_points, num_channels=3/6)
    # pc_rotated: (num_points, num_channels=3/6)
    num_points, num_channels = pc.shape
    assert num_channels in [3, 6]
    rot_mat = scipy_R.random().as_matrix().astype(np.float32) # (3, 3)
    if num_channels == 3:
        pc_rotated = np.matmul(pc, rot_mat) # (num_points, 3)
    if num_channels == 6:
        pc_rotated = np.concatenate((np.matmul(pc[:, 0:3], rot_mat), np.matmul(pc[:, 3:6], rot_mat)), axis=1) # (num_points, 6)
    if not return_angle:
        return pc_rotated
    else:
        rot_ang = scipy_R.from_matrix(np.transpose(rot_mat)).as_euler('xyz', degrees=True).astype(np.float32) # (3,)
        for aid in range(3):
            if rot_ang[aid] < 0:
                rot_ang[aid] = 360.0 + rot_ang[aid]
        return pc_rotated, rot_ang
    
    
def random_jittering(pc, sigma, bound):
    # pc: (num_points, num_channels)
    # sigma: standard deviation of zero-mean Gaussian noise
    # bound: clip noise values
    # pc_jittered: [num_points, num_channels]
    num_points, num_channels = pc.shape
    assert sigma > 0
    assert bound > 0
    gaussian_noises = np.random.normal(0, sigma, size=(num_points, 3)).astype(np.float32) # (num_points, 3)
    bounded_gaussian_noises = np.clip(gaussian_noises, -bound, bound).astype(np.float32) # (num_points, 3)
    if num_channels == 3:
        pc_jittered = pc + bounded_gaussian_noises
    if num_channels > 3:
        xyz = pc[:, 0:3]
        attr = pc[:, 3:]
        pc_jittered = np.concatenate((xyz + bounded_gaussian_noises, attr), axis=1)
    return pc_jittered
    
    
def random_dropout(pc, min_dp_ratio, max_dp_ratio, return_num_dropped=False):
    # pc: (num_points, num_channels)
    # max_dp_ratio: (0, 1)
    # pc_dropped: [num_points, num_channels]
    num_points, num_channels = pc.shape
    assert min_dp_ratio>=0 and min_dp_ratio<=1
    assert max_dp_ratio>=0 and max_dp_ratio<=1
    assert min_dp_ratio <= max_dp_ratio
    dp_ratio = np.random.random() * (max_dp_ratio-min_dp_ratio) + min_dp_ratio
    num_dropped = int(num_points * dp_ratio)
    pc_dropped = pc.copy()
    if num_dropped > 0:
        dp_indices = np.random.choice(num_points, num_dropped, replace=False)
        pc_dropped[dp_indices, :] = pc_dropped[0, :] # all replaced by the first row of "pc"
    if not return_num_dropped:
        return pc_dropped
    else:
        return pc_dropped, num_dropped
    
    
def random_isotropic_scaling(pc, min_s_ratio, max_s_ratio, return_iso_scaling_ratio=False):
    # pc: (num_points, num_channels)
    # pc_iso_scaled: [num_points, num_channels]
    num_points, num_channels = pc.shape
    assert min_s_ratio > 0 and min_s_ratio <= 1
    assert max_s_ratio >= 1
    iso_scaling_ratio = np.random.random() * (max_s_ratio - min_s_ratio) + min_s_ratio
    if num_channels == 3:
        pc_iso_scaled = pc * iso_scaling_ratio
    if num_channels > 3:
        xyz = pc[:, 0:3]
        attr = pc[:, 3:]
        pc_iso_scaled = np.concatenate((xyz * iso_scaling_ratio, attr), axis=1)
    if not return_iso_scaling_ratio:
        return pc_iso_scaled
    else:
        return pc_iso_scaled, iso_scaling_ratio
    
    
def random_anisotropic_scaling(pc, min_s_ratio, max_s_ratio, return_aniso_scaling_ratio=False):
    # pc: (num_points, num_channels)
    # pc_aniso_scaled: [num_points, num_channels]
    num_points, num_channels = pc.shape
    assert min_s_ratio > 0 and min_s_ratio <= 1
    assert max_s_ratio >= 1
    aniso_scaling_ratio = (np.random.random(3) * (max_s_ratio - min_s_ratio) + min_s_ratio).astype('float32')
    pc_aniso_scaled = pc.copy()
    pc_aniso_scaled[:, 0] *= aniso_scaling_ratio[0]
    pc_aniso_scaled[:, 1] *= aniso_scaling_ratio[1]
    pc_aniso_scaled[:, 2] *= aniso_scaling_ratio[2]
    if not return_aniso_scaling_ratio:
        return pc_aniso_scaled
    else:
        return pc_aniso_scaled, aniso_scaling_ratio


def random_translation(pc, max_offset, return_offset=False):
    # pc: (num_points, num_channels)
    # pc_translated: [num_points, num_channels]
    num_points, num_channels = pc.shape
    assert max_offset > 0
    offset = np.random.uniform(low=-max_offset, high=max_offset, size=[3]).astype('float32')
    pc_translated = pc.copy()
    pc_translated[:, 0] += offset[0]
    pc_translated[:, 1] += offset[1]
    pc_translated[:, 2] += offset[2]
    if not return_offset:
        return pc_translated
    else:
        return pc_translated, offset
    
    
def centroid_normalization(pc):
    # pc: (num_points, num_channels)
    # pc_normalized: (num_points, num_channels)
    num_points, num_channels = pc.shape
    xyz = pc[:, 0:3]
    attr = pc[:, 3:]
    xyz = xyz - np.mean(xyz, axis=0)
    max_d = np.max(np.sqrt(np.abs(np.sum(xyz**2, axis=1)))) # a scalar
    xyz_normalized = xyz / max_d
    pc_normalized = np.concatenate((xyz_normalized, attr), axis=1)
    return pc_normalized
    
    
def bounding_box_normalization(pc):
    # pc: (num_points, num_channels)
    # pc_normalized: (num_points, num_channels)
    num_points, num_channels = pc.shape
    xyz = pc[:, 0:3]
    attr = pc[:, 3:]
    xyz = xyz - (np.min(xyz, axis=0) + np.max(xyz, axis=0))/2
    max_d = np.max(np.sqrt(np.abs(np.sum(xyz**2, axis=1)))) # a scalar
    xyz_normalized = xyz / max_d
    pc_normalized = np.concatenate((xyz_normalized, attr), axis=1)
    return pc_normalized


def random_shuffling(pc):
    # pc: (num_points, num_channels)
    # pc_shuffled: (num_points, num_channels)
    num_points, num_channels = pc.shape
    idx_shuffled = np.arange(num_points)
    np.random.shuffle(idx_shuffled)
    pc_shuffled = pc[idx_shuffled]
    return pc_shuffled


