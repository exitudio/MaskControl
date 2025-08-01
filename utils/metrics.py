import numpy as np
from scipy import linalg
import torch


def calculate_mpjpe(gt_joints, pred_joints):
    """
    gt_joints: num_poses x num_joints(22) x 3
    pred_joints: num_poses x num_joints(22) x 3
    (obtained from recover_from_ric())
    """
    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"

    # Align by root (pelvis)
    pelvis = gt_joints[:, [0]].mean(1)
    gt_joints = gt_joints - torch.unsqueeze(pelvis, dim=1)
    pelvis = pred_joints[:, [0]].mean(1)
    pred_joints = pred_joints - torch.unsqueeze(pelvis, dim=1)

    # Compute MPJPE
    mpjpe = torch.linalg.norm(pred_joints - gt_joints, dim=-1) # num_poses x num_joints=22
    mpjpe_seq = mpjpe.mean(-1) # num_poses

    return mpjpe_seq

# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists

def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0)
    else:
        return top_k_mat


def calculate_matching_score(embedding1, embedding2, sum_all=False):
    assert len(embedding1.shape) == 2
    assert embedding1.shape[0] == embedding2.shape[0]
    assert embedding1.shape[1] == embedding2.shape[1]

    dist = linalg.norm(embedding1 - embedding2, axis=1)
    if sum_all:
        return dist.sum(axis=0)
    else:
        return dist



def calculate_activation_statistics(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


# def compute_kps_error(cur_motion, gt_skel_motions, sampled_keyframes):
#     '''
#     cur_motion [32, 22, 3, 196]
#     gt_skel_motions [32, 22, 3, 196]
#     sampled_keyframes [32, 5]
#     '''
#     batch_size = cur_motion.shape[0]
#     dist_err = torch.zeros_like(sampled_keyframes, dtype=torch.float)
#     sampled_keyframes = sampled_keyframes.long()
#     for ii in range(batch_size):
#         cur_keyframes = sampled_keyframes[ii]
#         motion_xz = cur_motion[ii, 0, [0,1, 2]]
#         gt_xz = gt_skel_motions[ii, 0, [0,1, 2]]  # [2, 196]
#         # This should be location error for each key frames in meter
#         cur_err = torch.linalg.norm(motion_xz[:, cur_keyframes] - gt_xz[:, cur_keyframes], dim=0)
#         dist_err[ii, :] = cur_err[:]
#     return dist_err

def cross_combination_joints():
    controllable_joints = {
        "pelvis": 0,
        "l_foot": 10,
        "r_foot": 11,
        "head": 15,
        "left_wrist": 20,
        "right_wrist": 21,
    }
    choose_combination = [
        [0],
        [10],
        [11],
        [15],
        [20],
        [21],
        [0, 10],
        [0, 11],
        [0, 15],
        [0, 20],
        [0, 21],
        [10, 11],
        [10, 15],
        [10, 20],
        [10, 21],
        [11, 15],
        [11, 20],
        [11, 21],
        [15, 20],
        [15, 21],
        [20, 21],
        [0, 10, 11],
        [0, 10, 15],
        [0, 10, 20],
        [0, 10, 21],
        [0, 11, 15],
        [0, 11, 20],
        [0, 11, 21],
        [0, 15, 20],
        [0, 15, 21],
        [0, 20, 21],
        [10, 11, 15],
        [10, 11, 20],
        [10, 11, 21],
        [10, 15, 20],
        [10, 15, 21],
        [10, 20, 21],
        [11, 15, 20],
        [11, 15, 21],
        [11, 20, 21],
        [15, 20, 21],
        [0, 10, 11, 15],
        [0, 10, 11, 20],
        [0, 10, 11, 21],
        [0, 10, 15, 20],
        [0, 10, 15, 21],
        [0, 10, 20, 21],
        [0, 11, 15, 20],
        [0, 11, 15, 21],
        [0, 11, 20, 21],
        [0, 15, 20, 21],
        [10, 11, 15, 20],
        [10, 11, 15, 21],
        [10, 11, 20, 21],
        [10, 15, 20, 21],
        [11, 15, 20, 21],
        [0, 10, 11, 15, 20],
        [0, 10, 11, 15, 21],
        [0, 10, 11, 20, 21],
        [0, 10, 15, 20, 21],
        [0, 11, 15, 20, 21],
        [10, 11, 15, 20, 21],
        [0, 10, 11, 15, 20, 21],
    ]
    return choose_combination

control_joint_ids = [0, 10, 11, 15, 20, 21]



def random_mask_cross(joints, n_joints=22, density=1):
    cross_joints = cross_combination_joints()
    choose = np.random.choice(len(cross_joints), 1).item()
    choose_joint = cross_joints[choose]

    length = joints.shape[0]
    choose_seq_num = np.random.choice(length - 1, 1) + 1
    if density in [1, 2, 5]:
        choose_seq_num = density
    else:
        choose_seq_num = int(length * density / 100)
    print('length:', length)
    print('choose_seq_num:', choose_seq_num)
    choose_seq = np.random.choice(length, choose_seq_num, replace=False)
    choose_seq.sort()
    mask_seq = np.zeros((length, n_joints, 3)).astype(np.bool)

    print('choose_seq:', choose_seq)
    print('choose_joint:', choose_joint)
    for cj in choose_joint:
        mask_seq[choose_seq, cj] = True
    return mask_seq


joints_by_part = {
    'lower': [0, 10, 11],
    'upper': [15, 20, 21],

    "pelvis": [0],
    "l_foot": [10],
    "r_foot": [11],
    "head": [15],
    "left_wrist": [20],
    "right_wrist": [21],

    #### For STMC ##########
    "left arm": [20, 18],  # [20, 18, 16, 13]
    "right arm": [21, 19],  # [21, 19, 17, 14] wrist, elbow, shoulder, Clavicle
    "legs": [10, 4,
             11, 5, ],
            #  [10, 7, 4, 1,
            #  11, 8, 5, 2],
    # "head": [15], # repeated 
    "spine": [0] # 0, 3, 6, 9
}