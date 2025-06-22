import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from os.path import join as pjoin

from models.mask_transformer.transformer import MaskTransformer
from models.mask_transformer.transformer_trainer import MaskTransformerTrainer
from models.vq.model import RVQVAE

from options.train_option import TrainT2MOptions

from utils.plot_script import plot_3d_motion
from utils.motion_process import recover_from_ric
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils.paramUtil import t2m_kinematic_chain, kit_kinematic_chain

from data.t2m_dataset import Text2MotionDataset
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper
import utils.eval_t2m as eval_t2m


def plot_t2m(data, save_dir, captions, m_lengths):
    return
    data = train_dataset.inv_transform(data)

    # print(ep_curves.shape)
    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        joint_data = joint_data[:m_lengths[i]]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4'%i)
        # print(joint.shape)
        plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20)

def load_vq_model():
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, opt.device)
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
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {opt.vq_name}')
    return vq_model, vq_opt

import torch
from collections import defaultdict
import torch.optim as optim
# import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from utils.utils import *
from os.path import join as pjoin
from utils.eval_t2m import evaluation_mask_transformer, evaluation_res_transformer
from models.mask_transformer.tools import *

from einops import rearrange, repeat

def def_value():
    return 0.0

class CtrlNetTrainer:
    def __init__(self, args, ct2m_transformer, vq_model):
        self.opt = args
        self.ct2m_transformer = ct2m_transformer
        self.ct2m_transformer.vq_model = vq_model
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_t2m_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr


    def forward(self, batch_data):

        conds, motion, m_lens = batch_data
        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        # (b, n, q)
        # code_idx, _ = self.vq_model.encode(motion)
        # m_lens = m_lens // 4

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        # loss_dict = {}
        # self.pred_ids = []
        # self.acc = []

        loss_emb, ce_loss, _pred_ids, _acc, loss_tta = self.ct2m_transformer(conds, m_lens, motion)
        # TODO return TTT loss
        return loss_emb, ce_loss, loss_tta, _acc

    def update(self, batch_data):
        loss_emb, ce_loss, loss_TTT, acc = self.forward(batch_data)

        self.opt_t2m_transformer.zero_grad()
        (0*loss_emb+.5*ce_loss+.5*loss_TTT).backward()
        self.opt_t2m_transformer.step()
        self.scheduler.step()

        return loss_emb.item(), ce_loss.item(), loss_TTT.item(), acc

    def save(self, file_name, ep, total_it):
        t2m_trans_state_dict = self.ct2m_transformer.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            'ct2m_transformer': t2m_trans_state_dict,
            'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.ct2m_transformer.load_state_dict(checkpoint['ct2m_transformer'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_t2m_transformer.load_state_dict(checkpoint['opt_t2m_transformer']) # Optimizer

            self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except:
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval):
        TTT = True
        self.ct2m_transformer.to(self.device)
        self.vq_model.to(self.device)

        # temp_vq = self.ct2m_transformer.vq_model
        # self.ct2m_transformer.vq_model = None
        # self.opt_t2m_transformer = optim.AdamW(list(self.ct2m_transformer.parameters()), 
        #                                        betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        # self.ct2m_transformer.vq_model = temp_vq
        self.opt_t2m_transformer = optim.AdamW(list(list(self.ct2m_transformer.seqTransEncoder_control.parameters()) + \
                                                    list(self.ct2m_transformer.encoder_control.parameters()) + \
                                                    list(self.ct2m_transformer.first_zero_linear.parameters()) + \
                                                    list(self.ct2m_transformer.mid_zero_linear.parameters()) ), 
                                               betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt_t2m_transformer,
                                                        milestones=self.opt.milestones,
                                                        gamma=self.opt.gamma)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        ##### mock eval info #####
        i = 0
        opt.time_steps = 10
        opt.cond_scale = 4
        opt.temperature = 1
        opt.topkr = .9
        opt.force_mask = False
        self.ct2m_transformer.TTT = TTT
        opt.which_epoch = 'latest'

        best_kps_mean = float('inf')
        best_fid, best_div, Rprecision, best_matching, best_skate_ratio, best_mm, traj_err, _avoid_dist, kps_mean = \
                eval_t2m.evaluation_mask_transformer_test_plus_res(eval_val_loader, vq_model, None, ct2m_transformer, None,
                                                                    i, eval_wrapper=eval_wrapper,
                                                        time_steps=opt.time_steps, cond_scale=opt.cond_scale,
                                                        temperature=opt.temperature, topkr=opt.topkr,
                                                                    force_mask=opt.force_mask, cal_mm=True, f=None, TTT=TTT, pred_num_batch=16, logger=self.logger, epoch=epoch,
                                                                    control=opt.control, 
                                                                    density=-1, opt=opt)
        best_acc = 0.

        while epoch < self.opt.max_epoch:
            self.ct2m_transformer.ctrl_train()
            self.vq_model.eval()

            for i, batch in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss_emb, loss, loss_TTT, acc = self.update(batch_data=batch)
                logs['loss_emb'] += loss_emb
                logs['loss'] += loss
                logs['loss_TTT'] += loss_TTT
                logs['acc'] += acc
                logs['lr'] += self.opt_t2m_transformer.param_groups[0]['lr']

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    # self.logger.add_scalar('val_loss', val_loss, it)
                    # self.l
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            epoch += 1

            print('Validation time:')
            self.vq_model.eval()
            self.ct2m_transformer.ctrl_eval()

            val_loss_emb = []
            val_loss = []
            val_loss_TTT = []
            val_acc = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss_emb, loss, loss_TTT, acc = self.forward(batch_data)
                    val_loss_emb.append(loss_emb.item())
                    val_loss.append(loss.item())
                    val_loss_TTT.append(loss_TTT.item())
                    val_acc.append(acc)

            print(f"Validation loss:{np.mean(val_loss):.3f}, accuracy:{np.mean(val_acc):.3f}")

            self.logger.add_scalar('Val/loss_emb', np.mean(val_loss_emb), epoch)
            self.logger.add_scalar('Val/loss', np.mean(val_loss), epoch)
            self.logger.add_scalar('Val/loss_TTT', np.mean(val_loss_TTT), epoch)
            self.logger.add_scalar('Val/acc', np.mean(val_acc), epoch)

            if np.mean(val_acc) > best_acc:
                print(f"Improved accuracy from {best_acc:.02f} to {np.mean(val_acc)}!!!")
                self.save(pjoin(self.opt.model_dir, 'net_best_acc.tar'), epoch, it)
                best_acc = np.mean(val_acc)
            if epoch % 5 == 0:
                best_fid, best_div, Rprecision, best_matching, best_skate_ratio, best_mm, traj_err_key, _avoid_dist, _ = \
                eval_t2m.evaluation_mask_transformer_test_plus_res(eval_val_loader, vq_model, None, ct2m_transformer, None,
                                                                    i, eval_wrapper=eval_wrapper,
                                                        time_steps=opt.time_steps, cond_scale=opt.cond_scale,
                                                        temperature=opt.temperature, topkr=opt.topkr,
                                                                    force_mask=opt.force_mask, cal_mm=True, f=None, TTT=TTT, pred_num_batch=16, logger=self.logger, epoch=epoch,
                                                                    control=opt.control, 
                                                                    density=-1, opt=opt)
                if best_kps_mean > kps_mean:
                    self.save(pjoin(self.opt.model_dir, 'best_kps.tar'), epoch, it)
                

if __name__ == '__main__':
    parser = TrainT2MOptions()
    opt = parser.parse()

    ### ADDED for CtrlNet EVAL ###
    opt.ctrl_net = True
    opt.each_lr = 6e-2
    opt.each_iter = 0
    opt.last_lr = 6e-2
    opt.last_iter = 0
    ############################

    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    if opt.name != 'TEMP':
        from exit.utils import init_save_folder
        init_save_folder(opt.save_root)
    opt.model_dir = pjoin(opt.save_root, 'model')
    # opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./checkpoints/', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    # os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 22
        opt.max_motion_len = 55
        dim_pose = 263
        radius = 4
        fps = 20
        kinematic_chain = t2m_kinematic_chain
        dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    elif opt.dataset_name == 'kit': #TODO
        opt.data_root = './dataset/KIT-ML'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_len = 55
        kinematic_chain = kit_kinematic_chain
        dataset_opt_path = './checkpoints/kit/Comp_v6_KLD005/opt.txt'

    else:
        raise KeyError('Dataset Does Not Exist')

    opt.text_dir = pjoin(opt.data_root, 'texts')

    vq_model, vq_opt = load_vq_model()

    clip_version = 'ViT-B/32'

    opt.num_tokens = vq_opt.nb_code

    

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'meta', 'std.npy'))

    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')

    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file)
    val_dataset = Text2MotionDataset(opt, mean, std, val_split_file)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)

    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'val', device=opt.device)

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    from models.mask_transformer.control_transformer import ControlTransformer
    ct2m_transformer = ControlTransformer(code_dim=vq_opt.code_dim,
                                        cond_mode='text',
                                        latent_dim=opt.latent_dim,
                                        ff_size=opt.ff_size,
                                        num_layers=opt.n_layers,
                                        num_heads=opt.n_heads,
                                        dropout=opt.dropout,
                                        clip_dim=512,
                                        cond_drop_prob=opt.cond_drop_prob,
                                        clip_version=clip_version,
                                        opt=opt,
                                        mean=torch.tensor(eval_val_loader.dataset.mean, requires_grad=False).cuda(),
                                        std=torch.tensor(eval_val_loader.dataset.std, requires_grad=False).cuda(),
                                        trans_path=f'./checkpoints/{opt.dataset_name}/{opt.trans_name}/model/latest.tar',
                                        vq_model=vq_model,
                                        control=opt.control)

    # if opt.fix_token_emb:
    #     ct2m_transformer.load_and_freeze_token_emb(vq_model.quantizer.codebooks[0])

    all_params = 0
    pc_transformer = sum(param.numel() for param in ct2m_transformer.parameters_wo_clip())

    # print(ct2m_transformer)
    # print("Total parameters of ct2m_transformer net: {:.2f}M".format(pc_transformer / 1000_000))
    all_params += pc_transformer

    print('Total parameters of all models: {:.2f}M'.format(all_params / 1000_000))

    trainer = CtrlNetTrainer(opt, ct2m_transformer, vq_model)

    trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper=eval_wrapper, plot_eval=plot_t2m)