import json
import argparse
from generation.load_model import get_models

import numpy as np
import torch
from utils.trajectory_plot import draw_circle_with_waves, draw_circle_with_waves2
from exit.utils import visualize_2motions
import os

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
traj1 = draw_circle_with_waves()
traj2 = draw_circle_with_waves2()

clip_text = ['a person walks in a circle counter-clockwise']
m_length = torch.tensor([196]).cuda()

k=0
global_joint = torch.zeros((m_length.shape[0], 196, 22, 3), device=m_length.device)
global_joint[k, :, 0] = traj1
global_joint[k, :, 20] = traj2
global_joint_mask = (global_joint.sum(-1) != 0)

print(' Optimizing...')
pred_motions_denorm, pred_motions = ct2m_transformer.generate_with_control(clip_text, m_length, time_steps=10, cond_scale=4,
                                                                        temperature=1, topkr=.9,
                                                                        force_mask=opt.force_mask, 
                                                                        vq_model=vq_model, 
                                                                        global_joint=global_joint, 
                                                                        global_joint_mask=global_joint_mask,
                                                                        _mean=torch.tensor(moment[0]).cuda(),
                                                                        _std=torch.tensor(moment[1]).cuda(),
                                                                        res_cond_scale=5,
                                                                        res_model=res_model,
                                                                        control_opt = {
                                                                            'each_lr': 6e-2,
                                                                            'each_iter': args.iter_each,
                                                                            'lr': 6e-2,
                                                                            'iter': args.iter_last,
                                                                        })
print('Done.')


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
                root_path=traj1.detach().cpu().numpy(),
                root_path2=traj2.detach().cpu().numpy(),
                save_path=f'{args.path_name}/generation.html',
                show=args.show
                )
np.save(f'{args.path_name}/generation.npy', pred_motions[k, :m_length[0]].detach().cpu().numpy())
np.save(f'{args.path_name}/trj_cond.npy', global_joint[k, :m_length[0]].detach().cpu().numpy())