import json
import argparse
from generation.load_model import get_models

import numpy as np
import torch
from utils.trajectory_plot import draw_circle_with_waves, draw_circle_with_waves2
from exit.utils import visualize_2motions
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--iter_each', type=int, default=100, help='number of logits optimization at each unmask step') 
parser.add_argument('--iter_last', type=int, default=600, help='number of logits optimization at the last unmask step') 
parser.add_argument('--path_name', type=str, default='./output/test')
parser.add_argument('--show', action='store_true', help='auto run html') 
args = parser.parse_args()
os.makedirs(f'{args.path_name}/source', mode=os.umask(0), exist_ok=False)

with open("./generation/args.json", "r") as f:
    opt = json.load(f)
opt = argparse.Namespace(**opt)

ct2m_transformer, vq_model, res_model, moment = get_models(opt)
k = 0
# clip_text = ['the man walks forward in a straight line.']
clip_text = ['the man walks forward in a straight line.']
cond = torch.tensor([[195, 0]]) # (f, j)
m_length = torch.tensor([196]).cuda()

global_joint = torch.zeros((m_length.shape[0], 196, 22, 3), device=m_length.device)
# [y, z, x] of plotly
# [side , height, front]
global_joint[k, 195, 0] = torch.tensor([0,  1.0229,  5.])
global_joint_mask = (global_joint.sum(-1) != 0)

avoid_points = torch.tensor([[0.014, 1.991, 2.981, 1],
                             [2.014, 1.991, 2.981, 1],
                             [1.014, 1.991, 2.981, 1],
                             [-1.014, 1.991, 2.981, 1],
                             [-2.014, 1.991, 2.981, 1]]).cuda()

print(' Optimizing...')
import timeit
start = timeit.default_timer()

def abitary_func(pred):
    cond = avoid_points
    loss_colli = 0.0
    # batch SDF
    joint = 15
    if len(cond.shape) == 2:
        from einops import repeat
        cond = repeat(cond, 'o four -> b f o four', b=pred.shape[0], f=pred.shape[1])
        pred = repeat(pred, 'b f j d -> b f j o d', o=cond.shape[0])
    
    dist = torch.norm(pred[:, :, joint] - cond[..., :3], dim=-1)
    dist = torch.clamp(cond[..., 3] - dist, min=0.0)
    loss_colli = dist[cond[..., 3]>0].mean()
    return loss_colli

pred_motions_denorm, pred_motions = ct2m_transformer.generate_with_control(clip_text, m_length, time_steps=10, cond_scale=4,
                                                                        temperature=1, topkr=.9,
                                                                        force_mask=opt.force_mask, 
                                                                        vq_model=vq_model, 
                                                                        global_joint=global_joint, 
                                                                        global_joint_mask=global_joint_mask,
                                                                        _mean=torch.tensor(moment[0]).cuda(),
                                                                        _std=torch.tensor(moment[1]).cuda(),
                                                                        res_cond_scale=5,
                                                                        res_model=None,
                                                                        control_opt = {
                                                                            'each_lr': 6e-2,
                                                                            'each_iter': args.iter_each,
                                                                            'lr': 6e-2,
                                                                            'iter': args.iter_last,
                                                                        },
                                                                        abitary_func=abitary_func)
stop = timeit.default_timer()
print('Time: ', stop - start)  
# path 1
r_pos = pred_motions_denorm[k, :m_length[k], 0]
root_path = r_pos.detach().cpu().numpy()

# path 2
root_path2 = pred_motions_denorm[k, :, 0, :m_length[k]].detach().cpu().numpy()

visualize_2motions(pred_motions[k].detach().cpu().numpy(), 
                moment[1], 
                moment[0], 
                't2m', 
                m_length[k], 
                # pred_motions[k].detach().cpu().numpy(),
                # root_path=root_path,
                # root_path2=avoid_points.detach().cpu().numpy(),
                ocpc_points=avoid_points[..., :3].unsqueeze(0).detach().cpu().numpy(),
                save_path=f'{args.path_name}/generation.html'
                )
np.save(args.path_name+'/generation.npy', pred_motions[k, :m_length[0]].detach().cpu().numpy())
np.save(args.path_name+'/avoid.npy', avoid_points.detach().cpu().numpy())

fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
ax.set_xlim((-7, 7))
ax.set_ylim((0, 10))
for o in avoid_points.detach().cpu().numpy():
    c = plt.Circle((o[0], o[2]), o[3], color='r')
    ax.add_patch(c)

r_pos = pred_motions_denorm[k, :m_length[k], 0]
root_path = r_pos.detach().cpu().numpy()
for o in root_path:
    c = plt.Circle((o[0], o[2]), .01, color='b')
    ax.add_patch(c)

fig.savefig(args.path_name+'/avoid.png', dpi=fig.dpi)
print('Done.')