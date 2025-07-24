import os
from os.path import join as pjoin

import torch

from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper

from utils.fixseed import fixseed

import numpy as np
from models.mask_transformer.control_transformer import ControlTransformer
from exit.utils import visualize_2motions, animate3d
import matplotlib.pyplot as plt

def load_vq_model(vq_opt):
    # opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    dim_pose = 251 if vq_opt.dataset_name == 'kit' else 263

    vq_model = RVQVAE(vq_opt,
                dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.output_emb_width,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'), map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_opt.name} Completed!')
    return vq_model, vq_opt

def load_trans_model(model_opt, which_model):
    t2m_transformer = MaskTransformer(code_dim=model_opt.code_dim,
                                      cond_mode='text',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                      map_location=opt.device)
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    # print(ckpt.keys())
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Mask Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer

def load_ctrltrans_model(model_opt, opt, which_model, clip_version, vq_model, moment):
    ct2m_transformer = ControlTransformer(code_dim=model_opt.code_dim,
                                      cond_mode='text',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=model_opt,
                                      mean=torch.tensor(moment[0]).cuda(),
                                      std=torch.tensor(moment[1]).cuda(),
                                    #   trans_path='./checkpoints/t2m/1_mtrans_lossAllMaskNoMask/model/latest.tar',
                                      vq_model=vq_model,
                                      control=opt.control)
    ct2m_transformer.cuda()
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, opt.ctrl_name, 'model', which_model),
                      map_location='cpu')
    missing_keys, unexpected_keys = ct2m_transformer.load_state_dict(ckpt['ct2m_transformer'], strict=False)
    return ct2m_transformer

def load_res_model(res_opt, vq_opt, clip_version):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
                                            cond_mode='text',
                                            latent_dim=res_opt.latent_dim,
                                            ff_size=res_opt.ff_size,
                                            num_layers=res_opt.n_layers,
                                            num_heads=res_opt.n_heads,
                                            dropout=res_opt.dropout,
                                            clip_dim=512,
                                            shared_codebook=vq_opt.shared_codebook,
                                            cond_drop_prob=res_opt.cond_drop_prob,
                                            # codebook=vq_model.quantizer.codebooks[0] if opt.fix_token_emb else None,
                                            share_weight=res_opt.share_weight,
                                            clip_version=clip_version,
                                            opt=res_opt)
    res_transformer.cuda()
    ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', 'net_best_fid.tar'),
                      map_location='cpu')
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    return res_transformer

def get_models(opt):
    moment = np.load('./generation/moment.npy', allow_pickle=True)

    opt.res_name = 'tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw'
    opt.ctrl_name = 'z2024-08-27-21-07-55_CtrlNet_randCond1-196_l1.5XEnt.5TTT__cross'
    opt.name = '1_mtrans_lossAllMaskNoMask'
    opt.control = 'random'

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    # out_dir = pjoin(opt.check)
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.ctrl_name)
    model_dir = pjoin(root_dir, 'model')
    out_dir = pjoin(root_dir, 'eval')
    os.makedirs(out_dir, exist_ok=True)

    out_path = pjoin(out_dir, "%s.log"%opt.ext)

    f = open(pjoin(out_path), 'w')
    print(opt, file=f, flush=True)

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)
    clip_version = 'ViT-B/32'

    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_model = load_res_model(res_opt, vq_opt, clip_version)

    assert res_opt.vq_name == model_opt.vq_name


    ##### ---- Dataloader ---- #####
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22


    file = 'latest.tar'
    ct2m_transformer = load_ctrltrans_model(model_opt, opt, file, clip_version, vq_model, moment)
    ct2m_transformer.res_model = res_model
    ct2m_transformer.res_model.process_embed_proj_weight()
    ct2m_transformer.ctrl_eval()

    TTT = True
    ct2m_transformer.TTT =TTT
    ct2m_transformer.vq_model = vq_model
    return ct2m_transformer, vq_model, res_model, moment

if __name__ == '__main__':
    
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    import json
    # with open("args.json", "w") as f:
    #     json.dump(vars(opt), f)
    import argparse
    with open("/home/epinyoan/git/momask-TTA/output/timeline/args.json", "r") as f:
        opt = json.load(f)
    opt = argparse.Namespace(**opt)

    ct2m_transformer, vq_model, res_model, moment = get_models(opt)

    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if opt.dataset_name == 'kit' \
        else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    k = 0
    import datetime
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # date = '-'.join(date.split('-')[2:])
    path_name = 'STMC_iter300_last300_resOnlyBP_bpTimeline_pad5'
    path_name = '-'.join(path_name.split(' '))
    path_name = f'./output/gen/{date}__{path_name}'
    os.makedirs(f'{path_name}/source', exist_ok=False)
    os.chmod(path_name, 0o777)
    from exit.utils import copyComplete
    copyComplete('generate.py', f'{path_name}/source/')
    copyComplete('./models/mask_transformer/transformer.py', f'{path_name}/source/')
    # copyComplete('./utils/trajectory_plot.py', f'{path_name}/source/')

    mode = 'STMC_eval' # AnyJoint, Timeline, Avoidance, STMC, STMC_eval
    import timeit
    start = timeit.default_timer()
    if mode == 'AnyJoint':
        from utils.trajectory_plot import get_sine, get_sine_updown, get_sine_updown2, draw_circle_with_waves, draw_circle_with_waves2, draw_straight_line, get_spiral, draw_face_1
        # traj1 = draw_circle_with_waves()
        # traj2 = draw_circle_with_waves2()
        _data = np.load('pivots_left.npy', allow_pickle=True)
        motion = _data[1][:_data[3]]
        traj1 = torch.tensor(motion[:, 20]).cuda()
        traj1[:150] = 0
        traj1[150:, 2] += 1
        traj1[150:, 0] -= 1
        
        clip_text = ['A person walks forward, casually greeting others with a wave or hello']
        # cond = torch.tensor([[195, 0]]) # (f, j)
        m_length = torch.tensor([120]).cuda()

        # run user rating 
        # data = np.load('user_rate.npy', allow_pickle=True)
        # for i in range(10):
        #     clip_text = data[i][0] # text
        #     data[i][1] # motion (length, 22, 3)
        #     data[i][2] # motion raw (length, 263)
        #     m_length = torch.tensor(data[i][3]).cuda().unsqueeze(0) # motion length 
        #     traj1 = torch.tensor(data[i][1][:, 0]).cuda()

        #     global_joint = torch.zeros((m_length.shape[0], 196, 22, 3), device=m_length.device)
        #     global_joint[k, :, 0] = traj1
        #     # global_joint[k, :, 21] = traj2
        #     global_joint_mask = (global_joint.sum(-1) != 0)

        global_joint = torch.zeros((m_length.shape[0], 196, 22, 3), device=m_length.device)
        # global_joint[k, :, 21] = traj1
        global_joint[k, 47, 20] = torch.Tensor([0.6395, 1.5145, 1.9326]).cuda()
        global_joint_mask = (global_joint.sum(-1) != 0)
        # tensor([0.9832, 0.8975, 3.1048], device='cuda:1')

        ct2m_transformer.ctrl_net  = None
        pred_motions_denorm, pred_motions = ct2m_transformer.generate_with_control(clip_text, m_length, time_steps=10, cond_scale=4,
                                                                                temperature=1, topkr=.9,
                                                                                force_mask=opt.force_mask, 
                                                                                vq_model=vq_model, 
                                                                                global_joint=global_joint, 
                                                                                global_joint_mask=global_joint_mask,
                                                                                _mean=torch.tensor(moment[0]).cuda(),
                                                                                _std=torch.tensor(moment[1]).cuda(),
                                                                                TTT=True,
                                                                                res_cond_scale=5,
                                                                                res_model=res_model,
                                                                                control_opt = {
                                                                                    'each_lr': 6e-2,
                                                                                    'each_iter': 0,
                                                                                    'lr': 6e-2,
                                                                                    'iter': 0,
                                                                                })
        # path 1
        r_pos = pred_motions_denorm[k, :m_length[k], 0]
        root_path = r_pos.detach().cpu().numpy()

        # path 2
        root_path2 = pred_motions_denorm[k, :, 0, :m_length[k]].detach().cpu().numpy()

        visualize_2motions(pred_motions[k].detach().cpu().numpy(), 
                        moment[1], 
                        moment[0], 
                        't2m', 
                        72, 
                        # pred_motions[k].detach().cpu().numpy(),
                        # root_path=root_path,
                        root_path2=global_joint[k, 47:48, 20].detach().cpu().numpy(),
                        save_path=f'{path_name}/generation.html'
                        )
        np.save(path_name+'/generation.npy', pred_motions[k, :m_length[0]].detach().cpu().numpy())
        np.save(path_name+'/trj_cond.npy', global_joint[k, :m_length[0]].detach().cpu().numpy())
        
    elif mode == 'Timeline':
        timeline = [#
            ['l_foot', 'a person kicks left legs.', [0, 60]],
            ['lower', 'a person jumps forward.', [60, 120]],
             ['upper', 'a person puts hands in the air.', [0, 120]]
            ]
        m_length = torch.tensor([120]).cuda()
        pred_motions_denorm, pred_motions = ct2m_transformer.timeline_control(timeline=timeline, m_length=m_length,
                                                                                vq_model=vq_model, 
                                                                                _mean=torch.tensor(moment[0]).cuda(),
                                                                                _std=torch.tensor(moment[1]).cuda(),
                                                                                res_model=res_model)
        k = 0
        visualize_2motions(pred_motions[k].detach().cpu().numpy(), 
                        moment[1], 
                        moment[0], 
                        't2m', 
                        m_length[k], 
                        save_path=path_name+'/generate.html'
                        )
        np.save(path_name+'/timeline.npy', pred_motions[k].detach().cpu().numpy())
    elif mode == 'STMC_eval':
        timelines, m_length, npy_paths, bp_timeline = np.load('/home/epinyoan/git/momask-TTA/output/stmc/timelines.npy', allow_pickle=True)
        m_length = torch.from_numpy(np.array(m_length, dtype=np.int64)).cuda()

        batch_size = 100
        total_len = len(timelines)

        os.makedirs(f'{path_name}/npy', exist_ok=True)
        from tqdm import tqdm
        for i in tqdm(range(0, total_len, batch_size), desc="Processing batches"):
            # Slice the batch
            timelines_batch = timelines[i:i + batch_size]
            m_length_batch = m_length[i:i + batch_size]
            npy_paths_batch = npy_paths[i:i + batch_size]
            bp_timeline_batch = bp_timeline[i:i + batch_size]

            pred_motions_denorm, pred_motions = ct2m_transformer.timeline_STMC(timelines=timelines_batch, m_length=m_length_batch,
                                                                                    vq_model=vq_model, 
                                                                                    _mean=torch.tensor(moment[0]).cuda(),
                                                                                    _std=torch.tensor(moment[1]).cuda(),
                                                                                    res_model=res_model,
                                                                                    bp_timeline=bp_timeline_batch)
            for idx, (length, npy_path) in enumerate(zip(m_length_batch, npy_paths_batch)):
                unic_sample = pred_motions_denorm[idx, :length.cpu().item()]
                # shape T, 22, 3
                np.save(f'{path_name}/npy/{npy_path.split("/")[-1]}', unic_sample.cpu().detach().numpy())
        
    elif mode == 'STMC':
        timelines = [[#
            [['spine', 'legs', 'right arm'], 'pick something with the right hand', [0, 177]],
            [['legs'], 'walk backwards', [167, 295]],
            [['left arm'], 'wave with the left hand', [4, 103]],
        ]]

        bp_timeline=[
        {
            'left arm': [(0, 0, 14), (2, 4, 103), (0, 93, 177), (1, 167, 295)], 
            'right arm': [(0, 0, 177), (1, 167, 295)], 
            'legs': [(0, 0, 177), (1, 167, 295)], 
            'head': [(0, 0, 177), (1, 167, 295)], 
            'spine': [(0, 0, 177), (1, 167, 295)]
        }
        ]
        m_length = torch.tensor([196*2]).cuda()
        pred_motions_denorm, pred_motions = ct2m_transformer.timeline_STMC(timelines=timelines, m_length=m_length,
                                                                                vq_model=vq_model, 
                                                                                _mean=torch.tensor(moment[0]).cuda(),
                                                                                _std=torch.tensor(moment[1]).cuda(),
                                                                                res_model=res_model,
                                                                                bp_timeline=bp_timeline)
        k = 0
        visualize_2motions(pred_motions[k].detach().cpu().numpy(), 
                        moment[1], 
                        moment[0], 
                        't2m', 
                        m_length[k], 
                        save_path=path_name+'/generate.html'
                        )
    elif mode == 'Avoidance':
        k = 0
        # clip_text = ['the man walks forward in a straight line.']
        clip_text = ['the man walks zigzag.']
        cond = torch.tensor([[195, 0]]) # (f, j)
        m_length = torch.tensor([196]).cuda()

        global_joint = torch.zeros((m_length.shape[0], 196, 22, 3), device=m_length.device)
        # [y, z, x] of plotly
        # [side , height, front]
        global_joint[k, 195, 0] = torch.tensor([.5,  1.0229,  6.5])
        global_joint_mask = (global_joint.sum(-1) != 0)

        avoid_points = torch.tensor([
                                [.5, 0., 1.5, 1],
                                 [-1.5, 0., 4, 2],
                                 [1.5, 0., 5.5, 1],

                                 [3, 0., 3, 2],
                                 ]).cuda()
        # ct2m_transformer.ctrl_net = False
        pred_motions_denorm, pred_motions = ct2m_transformer.generate_with_control(clip_text, m_length, time_steps=10, cond_scale=4,
                                                                        temperature=1, topkr=.9,
                                                                        force_mask=opt.force_mask, 
                                                                        vq_model=vq_model, 
                                                                        global_joint=global_joint, 
                                                                        global_joint_mask=global_joint_mask,
                                                                        _mean=torch.tensor(moment[0]).cuda(),
                                                                        _std=torch.tensor(moment[1]).cuda(),
                                                                        TTT=True,
                                                                        res_cond_scale=5,
                                                                        res_model=None,
                                                                        control_opt = {
                                                                            'each_lr': 6e-2,
                                                                            'each_iter': 0,
                                                                            'lr': 6e-2,
                                                                            'iter': 0,
                                                                        },
                                                                        avoid_points=avoid_points,
                                                                        abitary_func=None)
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
                        save_path=f'{path_name}/generation.html'
                        )
        np.save(path_name+'/generation.npy', pred_motions[k, :m_length[0]].detach().cpu().numpy())
        np.save('./avoid1.npy', pred_motions[k].detach().cpu().numpy())

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

        fig.savefig('avoid.png', dpi=fig.dpi)

    print('------------ DONE ------------------')
    # np.save('./timeline2.npy', pred_motions[k].detach().cpu().numpy())

# python eval_t2m_trans.py --name t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_vq --dataset_name t2m --gpu_id 3 --cond_scale 4 --time_steps 18 --temperature 1 --topkr 0.9 --gumbel_sample --ext cs4_ts18_tau1_topkr0.9_gs