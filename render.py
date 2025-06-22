from visualize import vis_utils
import numpy as np
from utils.motion_process import recover_from_ric
import torch


path = f'/home/epinyoan/git/momask-TTA/output/stmc/Render/STMC'
folder = path
motions = np.load(f'{path}/motion.npy')

# folder = '2024-11-25-19-46-29__a-person-is-casually-walking-straight-and-then-pivots-to-their-left.'
# path = f'/nfs-gs/epinyoan/git/momask-TTA/output/abitary_obj/emb_editing'
# motions = np.load(f'{path}/avoid1_trim.npy')

if motions.shape[1] > 22:
    moment = np.load('study/moment.npy', allow_pickle=True)
    npy2obj = vis_utils.npy2obj(motions, path, mean=moment[0], std=moment[1], skip=1)
else:
    npy2obj = vis_utils.npy2obj(motions, folder, skip=1)

# omnicontrol / GMD
# path = f'/nfs-gs/epinyoan/git/guided-motion-diffusion/save/unet_adazero_xl_x0_abs_proj10_fp16_clipwd_224/samples_000500000_seed10_a_person_walks_forward_and_waves_his_hands'
# aa = np.load(path + '/results.npy', allow_pickle=True)
# npy2obj = vis_utils.npy2obj(aa.item()['motion'].transpose((0, 3, 1, 2))[0], path)


# path = f'/nfs-gs/epinyoan/git/momask-TTA/output/MotionLCM/fly/'
# motions = np.load(path + 'a person crosses their arms for chest fly.npy', allow_pickle=True)
# moment = np.load('study/moment.npy', allow_pickle=True)
# npy2obj = vis_utils.npy2obj(motions, path, mean=moment[0], std=moment[1], skip=1)