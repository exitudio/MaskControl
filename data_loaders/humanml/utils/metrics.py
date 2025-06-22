import numpy as np
from scipy import linalg
from scipy.ndimage import uniform_filter1d
import torch

def compute_kps_error(cur_motion, gt_skel_motions, original_mask):
    '''
    cur_motion [bs, 22, 3, seq_len]
    gt_skel_motions [bs, 22, 3, seq_len]
    mask [bs, 22, 3, seq_len]
    '''
    mask = original_mask.permute(0, 3, 1, 2).bool().float().sum(dim=-1).clamp(0,1)  # [bs, 22, 3, seq_len] -> [bs, seq_len, 22, 3] -> [bs, seq_len, 22]
    joint_error = (cur_motion.permute(0, 3, 1, 2) - gt_skel_motions.permute(0, 3, 1, 2)) 
    assert joint_error.shape[-1] == 3
    dist_err = torch.linalg.norm(joint_error, dim=-1) * mask.float() # [bs, seq_len, 22]  joint error norm in xyz dim
    mask = mask.sum(dim=-1) #[bs, seq_len]
    dist_err = dist_err.sum(dim=-1) # [bs, seq_len]
    dist_err = torch.where(mask > 0, dist_err / mask, torch.tensor(0.0, dtype=torch.float32, device=dist_err.device)) # [bs, seq_len]   joint error average on 22 joints
    mask = mask.clamp(0,1)  # [bs, seq_len]
    return dist_err.detach().cpu().numpy(), mask.detach().cpu().numpy()

def calculate_trajectory_error(dist_error, mask, strict=True):
    ''' dist_error and mask shape [bs, seq_len]: error for each kps in metre
      Two threshold: 20 cm and 50 cm.
    If mean error in sequence is more then the threshold, fails
    return: traj_fail(0.2), traj_fail(0.5), all_kps_fail(0.2), all_kps_fail(0.5), all_mean_err.
        Every metrics are already averaged.

    '''
    assert dist_error.shape == mask.shape
    bs, seqlen = dist_error.shape
    assert np.all(np.logical_or(mask == 0, mask == 1)), "mask array contains values other than 0 or 1"
    mask_sum = mask.sum(-1)  # [bs]
    mean_err_traj = np.where(mask_sum > 1e-5, (dist_error * mask.astype(float)).sum(-1)/ mask_sum, 0.0)

    if strict:
        # Traj fails if any of the key frame fails
        traj_fail_02 = 1.0 - (dist_error * mask.astype(float) <= 0.2).all(1)
        traj_fail_05 = 1.0 - (dist_error * mask.astype(float) <= 0.5).all(1)
    else:
        # Traj fails if the mean error of all keyframes more than the threshold
        traj_fail_02 = (mean_err_traj > 0.2).astype(float)
        traj_fail_05 = (mean_err_traj > 0.5).astype(float)
    all_fail_02 = np.where(mask_sum > 1e-5, (dist_error * mask.astype(float) > 0.2).sum(-1) / mask_sum, 0.0)
    all_fail_05 = np.where(mask_sum > 1e-5, (dist_error * mask.astype(float) > 0.5).sum(-1) / mask_sum, 0.0)
    result = np.concatenate([traj_fail_02[..., None], traj_fail_05[..., None], all_fail_02[..., None], all_fail_05[..., None], mean_err_traj[..., None]], axis=1)
     
    assert np.all(~np.isnan(result)), "result array contains NaN values"
    assert result.shape == (bs, 5)
    return result


def calculate_trajectory_diversity(trajectories, lengths):
    ''' Standard diviation of point locations in the trajectories
    Args:
        trajectories: [bs, rep, 196, 2]
        lengths: [bs]
    '''
    # [32, 2, 196, 2 (xz)]
    # mean_trajs = trajectories.mean(1, keepdims=True)
    # dist_to_mean = np.linalg.norm(trajectories - mean_trajs, axis=3)
    def traj_div(traj, length):
        # traj [rep, 196, 2]
        # length (int)
        traj = traj[:, :length, :]
        # point_var = traj.var(axis=0, keepdims=True).mean()
        # point_var = np.sqrt(point_var)
        # return point_var

        mean_traj = traj.mean(axis=0, keepdims=True)
        dist = np.sqrt(((traj - mean_traj)**2).sum(axis=2))
        rms_dist = np.sqrt((dist**2).mean())
        return rms_dist
        
    div = []
    for i in range(len(trajectories)):
        div.append(traj_div(trajectories[i], lengths[i]))
    return np.array(div).mean()

def calculate_skating_ratio(motions):
    thresh_height = 0.05 # 10
    fps = 20.0
    thresh_vel = 0.50 # 20 cm /s 
    avg_window = 5 # frames

    batch_size = motions.shape[0]
    # 10 left, 11 right foot. XZ plane, y up
    # motions [bs, 22, 3, max_len]
    verts_feet = motions[:, [10, 11], :, :].detach().cpu().numpy()  # [bs, 2, 3, max_len]
    verts_feet_plane_vel = np.linalg.norm(verts_feet[:, :, [0, 2], 1:] - verts_feet[:, :, [0, 2], :-1],  axis=2) * fps  # [bs, 2, max_len-1]
    # [bs, 2, max_len-1]
    vel_avg = uniform_filter1d(verts_feet_plane_vel, axis=-1, size=avg_window, mode='constant', origin=0)

    verts_feet_height = verts_feet[:, :, 1, :]  # [bs, 2, max_len]
    # If feet touch ground in agjecent frames
    feet_contact = np.logical_and((verts_feet_height[:, :, :-1] < thresh_height), (verts_feet_height[:, :, 1:] < thresh_height))  # [bs, 2, max_len - 1]
    # skate velocity
    skate_vel = feet_contact * vel_avg

    # it must both skating in the current frame
    skating = np.logical_and(feet_contact, (verts_feet_plane_vel > thresh_vel))
    # and also skate in the windows of frames
    skating = np.logical_and(skating, (vel_avg > thresh_vel))

    # Both feet slide
    skating = np.logical_or(skating[:, 0, :], skating[:, 1, :]) # [bs, max_len -1]
    skating_ratio = np.sum(skating, axis=1) / skating.shape[1]
    
    return skating_ratio, skate_vel