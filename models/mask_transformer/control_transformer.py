import torch
import torch.nn as nn
from models.mask_transformer.transformer import MaskTransformer
from utils.motion_process import recover_from_ric
from models.vq.encdec import Encoder, Decoder
from models.mask_transformer.tools import *
import copy
from models.mask_transformer.transformer import OutputProcess_Bert
from random import random 
from einops import rearrange, repeat
from utils.metrics import control_joint_ids
from utils.motion_process import recover_from_ric
from utils.metrics import control_joint_ids, joints_by_part
from torch.distributions.categorical import Categorical

def return_all_layers(trans, src, mask=None, src_key_padding_mask=None):
    output = src
    all_layers = []
    for mod in trans.layers:
        output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        all_layers.append(output)
    if trans.norm is not None:
        output = trans.norm(output)
    return all_layers

def forward_with_condition(trans, src, list_of_controlnet_output, mask=None, src_key_padding_mask=None):
    output = src
    for mod, control_feat in zip(trans.layers, list_of_controlnet_output):
        output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        output = output + control_feat
    if trans.norm is not None:
        output = trans.norm(output)
    return output

def freeze_block(block):
    block.eval()
    for p in block.parameters():
        p.requires_grad = False

def unfreeze_block(block):
    block.train()
    for p in block.parameters():
        p.requires_grad = True


class ControlTransformer(MaskTransformer):

    def __init__(self, code_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.1, clip_dim=512, cond_drop_prob=0.1,
                 clip_version=None, opt=None, mean=None, std=None, trans_path='', vq_model=None, control=None, **kargs):

        super().__init__(code_dim=code_dim, cond_mode=cond_mode, latent_dim=latent_dim, ff_size=ff_size, num_layers=num_layers,
                 num_heads=num_heads, dropout=dropout, clip_dim=clip_dim, cond_drop_prob=cond_drop_prob,
                 clip_version=clip_version, opt=opt, **kargs)
        self.num_layers = num_layers
        self.mean = mean
        self.std = std

        if trans_path != '':
            ckpt = torch.load(trans_path, map_location=opt.device)
            model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
            # print(ckpt.keys())
            missing_keys, unexpected_keys = self.load_state_dict(ckpt[model_key], strict=False)
            freeze_block(self)
        ################# CntrlNet ##################

        # linear layers init with zeros
        self.first_zero_linear = nn.Linear(self.latent_dim, self.latent_dim)
        self.mid_zero_linear = nn.ModuleList(
            [nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.num_layers)])


        # seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
        #                                                   nhead=num_heads,
        #                                                   dim_feedforward=ff_size,
        #                                                   dropout=dropout,
        #                                                   activation='gelu')
        # self.seqTransEncoder_control = nn.TransformerEncoder(seqTransEncoderLayer,
        #                                                 num_layers=self.num_layers)
        # self.init_weights(self.seqTransEncoder_control)                                        
        # print('self.__init_weights:', super(ControlTransformer, self).__init_weights)
        # TODO should trans be a copy???
        self.seqTransEncoder_control = copy.deepcopy(self.seqTransEncoder)

        # TODO make them dynamic params
        self.control = control
        input_emb_width = 6 if self.control == 'trajectory' else len(control_joint_ids)*2*3 # num j x [relative dif+ abs] x 3 dim
        self.encoder_control = Encoder(input_emb_width=input_emb_width, output_emb_width=self.latent_dim, down_t=2, stride_t=2, width=512, depth=3,
                               dilation_growth_rate=3, activation='relu', norm=None)
        # self.output_process_control = OutputProcess_Bert(out_feats=opt.num_tokens, latent_dim=latent_dim)
        
        def init_zero_conv(m):
            if isinstance(m, (torch.nn.Conv1d, nn.Linear)):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        self.first_zero_linear.apply(init_zero_conv)
        # self.encoder_control.apply(init_zero_conv)
        self.mid_zero_linear.apply(init_zero_conv)
        # self.output_process_control.apply(init_zero_conv)
        self.ctrl_train()

        self.vq_model = vq_model
        self.mask_emb_vq = vq_model.quantizer.layers[0].codebook.mean(0)
        self.ctrl_net = True
        

    def ctrl_eval(self):
        freeze_block(self)

    def ctrl_train(self):
        # unfreeze_block(self)
        # if hasattr(self, 'vq_model'):
        #     freeze_block(self.vq_model)
        freeze_block(self)
        unfreeze_block(self.seqTransEncoder_control)
        unfreeze_block(self.encoder_control)
        unfreeze_block(self.first_zero_linear)
        unfreeze_block(self.mid_zero_linear)

    def forward(self, y, m_length, pose):
        # code_idx, _ = self.vq_model.encode(pose)
        x_encoder = self.vq_model.encoder(pose.permute(0, 2, 1))
        code_idx = self.vq_model.quantizer.quantize(x_encoder)
        ids = code_idx[..., 0]
        # KIT is broken here.
        assert m_length.min() > 30 # make sure this is raw range, not downsampling 4
        m_lens = m_length//4


        # joint_cond_pos = recover_from_ric((pose * _std + _mean).float(), self.opt.joints_num)
        global_joint = recover_from_ric(self.inv_transform(pose).float(), self.opt.joints_num)
        bs, ntokens = ids.shape
        device = ids.device
        ########### 1. GMD - 5 frames (can be repeated too)
        # n_keyframe = 5
        # sampled_keyframes = torch.rand(pose.shape[0], n_keyframe).cuda() * m_length.unsqueeze(-1)
        # sampled_keyframes = torch.floor(sampled_keyframes).int().sort()[0]
        # batch_arange = torch.arange(sampled_keyframes.size(0)).unsqueeze(1)
        # ctrlNet_cond = torch.zeros_like(joint_cond_pos)
        # ctrlNet_cond[batch_arange, sampled_keyframes] = joint_cond_pos[batch_arange, sampled_keyframes]

        # cond_mask = torch.zeros_like(pose[..., 0], dtype=bool)
        # cond_mask[batch_arange, sampled_keyframes] = True

        ########### 2. Random 50-100% KF ########################################
        # non_pad_mask_raw = lengths_to_mask(m_length, pose.shape[1]) #(b, n)
        # cond_mask = .5 + torch.rand(non_pad_mask_raw.size(), device=non_pad_mask_raw.device) * .5
        # cond_mask = torch.bernoulli(cond_mask).bool()
        # cond_mask = (non_pad_mask_raw * cond_mask)
        # ctrlNet_cond = cond_mask.unsqueeze(-1) * joint_cond_pos 
        ########### 3. 100% KF ######################################
        # global_joint_mask = lengths_to_mask(m_length, pose.shape[1]) #(b, n)

        ########### 4. Rand 0-100% Cond ###########################
        if self.control == 'trajectory' or self.control == 'random':
            rand_length = torch.rand(m_length.shape).cuda() * 196 # m_length.cuda()
            rand_length = rand_length.round().clamp(min=1, max=196)

            all_len_mask = lengths_to_mask(m_length, pose.shape[1]) #(b, n)
            batch_randperm = torch.rand((bs, pose.shape[1]), device=pose.device)
            batch_randperm[~all_len_mask] = 1
            batch_randperm = batch_randperm.argsort(dim=-1)
            global_joint_mask = batch_randperm < rand_length.unsqueeze(-1)
            global_joint_mask = global_joint_mask * all_len_mask

        ######################################
            if self.control == 'trajectory':
                global_joint_mask = repeat(global_joint_mask, 'b f -> b f j', j=self.opt.joints_num).clone()
                global_joint_mask[..., 1:] = False
            elif self.control == 'random':
                _global_joint_mask = global_joint_mask
                global_joint_mask = torch.zeros((*global_joint_mask.shape, self.opt.joints_num), device=_global_joint_mask.device, dtype=bool)
                control_joints = torch.tensor(control_joint_ids, device=pose.device)
                rand_indx = torch.randint(len(control_joints), (_global_joint_mask.shape[0],)) # random index (bs,)
                global_joint_mask[torch.arange(global_joint_mask.shape[0]),:, 
                                control_joints[rand_indx]] = _global_joint_mask # set idx of joint to frames mask
        elif self.control == 'cross':
            rand_length = torch.rand(m_length.shape, device=m_length.device) * m_length * len(control_joint_ids)
            rand_length = rand_length.cuda().round().clamp(min=1)

            all_len_mask = lengths_to_mask(m_length, pose.shape[1]) #(b, n)
            batch_randperm = torch.rand((*all_len_mask.shape, len(control_joint_ids)), device=pose.device)
            batch_randperm[~all_len_mask] = 1
            batch_randperm = batch_randperm.reshape((all_len_mask.shape[0], -1) )
            batch_randperm = batch_randperm.argsort(dim=-1)
            batch_randperm = batch_randperm.reshape((*all_len_mask.shape, -1) )
            _global_joint_mask = batch_randperm < rand_length.unsqueeze(-1).unsqueeze(-1)
            _global_joint_mask = _global_joint_mask * all_len_mask.unsqueeze(-1)

            global_joint_mask = torch.zeros((*pose.shape[:2], self.opt.joints_num), device=pose.device, dtype=bool)
            global_joint_mask[..., control_joint_ids] = _global_joint_mask
        else:
            raise Exception(f'{self.control} is not implemented yet!!!')
            
        ######################################



        # Positions that are PADDED are ALL FALSE
        non_pad_mask = lengths_to_mask(m_lens, ntokens) #(b, n)
        ids = torch.where(non_pad_mask, ids, self.pad_id)

        with torch.no_grad():
            cond_vector = self.encode_text(y)

        rand_time = uniform((bs,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)

        batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1)
        # Positions to be MASKED are ALL TRUE
        mask = batch_randperm < num_token_masked.unsqueeze(-1)

        # Positions to be MASKED must also be NON-PADDED
        mask &= non_pad_mask

        # Note this is our training target, not input
        labels = torch.where(non_pad_mask, ids, self.mask_id)
        # labels = torch.where(mask, ids, self.mask_id)

        x_ids = ids.clone()

        # Further Apply Bert Masking Scheme
        # Step 1: 10% replace with an incorrect token
        mask_rid = get_mask_subset_prob(mask, 0.1)
        rand_id = torch.randint_like(x_ids, high=self.opt.num_tokens)
        x_ids = torch.where(mask_rid, rand_id, x_ids)
        # Step 2: 90% x 10% replace with correct token, and 90% x 88% replace with mask token
        mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)

        # mask_mid = mask

        x_ids = torch.where(mask_mid, self.mask_id, x_ids)

        ### Calculate difference from current tokens (mased) with target pose ###
        _x_ids = x_ids.clone()
        id_overflow = _x_ids>=self.opt.num_tokens # set mask emb to average of all codebooks
        _x_ids[id_overflow] = 0

        ######### w/o logit noise ######
        x_d = self.vq_model.quantizer.get_codes_from_indices(_x_ids.unsqueeze(-1))
        x = x_d.sum(dim=0)
        ######### logit noise ##########
        # std = .1
        # num_cb = self.vq_model.quantizer.codebooks[0].shape[0]
        # one_hot = F.one_hot(_x_ids, num_classes=num_cb).float()
        # noise_hot = torch.normal(torch.zeros_like(one_hot), torch.ones_like(one_hot)*std)
        # noise_hot = one_hot+noise_hot
        # x = (noise_hot @ self.vq_model.quantizer.codebooks[0])
        ###################################

        x[id_overflow] = self.mask_emb_vq
        padding_mask = ~non_pad_mask
        # x = x.masked_fill(padding_mask.unsqueeze(-1), 0.)
        x = x.permute(0, 2, 1) # [64, 49, 512] => [64, 512, 49]
        pred_motions = self.vq_model.decoder(x) # [64, 196, 263]
        pred_motions_denorm = pred_motions * self.std + self.mean
        pred_motions_denorm = recover_from_ric(pred_motions_denorm.float(), self.opt.joints_num)

        ctrlNet_cond = (global_joint - pred_motions_denorm) * global_joint_mask.unsqueeze(-1)
        ctrlNet_cond2 = global_joint * global_joint_mask.unsqueeze(-1)
        if self.control == 'trajectory':
            ctrlNet_cond = ctrlNet_cond[:, :, 0]
            ctrlNet_cond2 = ctrlNet_cond2[:, :, 0]
        else:
            ctrlNet_cond = ctrlNet_cond[..., control_joint_ids, :]
            ctrlNet_cond = ctrlNet_cond.reshape((*ctrlNet_cond.shape[:2], -1))
            ctrlNet_cond2 = ctrlNet_cond2[..., control_joint_ids, :]
            ctrlNet_cond2 = ctrlNet_cond2.reshape((*ctrlNet_cond2.shape[:2], -1))
        ctrlNet_cond = torch.cat([ctrlNet_cond, ctrlNet_cond2], dim=-1)
        ###########################################################################

        ########## With Logit noise ##############
        # trans_emb = noise_hot @ self.token_emb.weight[:num_cb]
        # trans_emb[id_overflow] = self.token_emb(torch.tensor(self.mask_id).to(device))
        # logits = self.trans_forward(trans_emb, cond_vector, ~non_pad_mask, force_mask=False, ctrlNet_cond=ctrlNet_cond)
        #############################################

        logits = self.trans_forward(x_ids, cond_vector, ~non_pad_mask, force_mask=False, ctrlNet_cond=ctrlNet_cond)
        ce_loss, pred_id, acc = cal_performance(logits, labels, ignore_index=self.mask_id)



        ###### Control Loss #######
        emb = F.softmax(logits.permute(0,2,1)/1, dim=-1) @ self.vq_model.quantizer.codebooks[0]
        emb = emb.masked_fill(padding_mask.unsqueeze(-1), 0.) # TODO should use average emb??
        pred_motions = self.vq_model.forward_decoder(emb)
        # emb[labels!=self.mask_id] = x_d[0, labels!=self.mask_id]
        pred_motions_denorm = pred_motions * self.std + self.mean
        pred_motions_denorm = recover_from_ric(pred_motions_denorm.float(), self.opt.joints_num)

        ###### Embedding loss ######
        loss_emb = torch.zeros(1, device=emb.device) # F.l1_loss(x_encoder.permute(0,2,1)[non_pad_mask], emb[non_pad_mask], reduction='mean')

        # batch_arange = torch.arange(sampled_keyframes.size(0)).unsqueeze(1)
        # loss_tta = F.l1_loss(pred_motions_denorm[batch_arange, sampled_keyframes][..., 0, [0, 1, 2]], 
        #                     gt_skel_motions[batch_arange, sampled_keyframes][..., 0, [0, 1, 2]], reduction='mean')

        # TODO weight by sample
        # print('____ TODO ______ weight by sample')
        loss_tta = F.l1_loss(pred_motions_denorm[global_joint_mask], 
                                        global_joint[global_joint_mask], reduction='mean') # mse_loss l1_loss

        return loss_emb, ce_loss, pred_id, acc, loss_tta
    
    def trans_forward(self, motion_ids, cond, padding_mask, force_mask=False, ctrlNet_cond=None):
        if self.ctrl_net is None or not self.ctrl_net:
            return super().trans_forward(motion_ids, cond, padding_mask, force_mask, ctrlNet_cond)
        
        cond = self.mask_cond(cond, force_mask=force_mask)

        # print(motion_ids.shape)
        if len(motion_ids.shape) == 2: # [32, 49]
            x = self.token_emb(motion_ids)
        else: # [64, 49, 512], b, f, e
            x = motion_ids
        # print(x.shape)
        # (b, seqlen, d) -> (seqlen, b, latent_dim)
        x = self.input_process(x)

        cond = self.cond_emb(cond).unsqueeze(0) #(1, b, latent_dim)
        x = self.position_enc(x)
        xseq = torch.cat([cond, x], dim=0) #(seqlen+1, b, latent_dim)
        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1) #(b, seqlen+1)

        ################ ADD Control ##################
        control_input = self.encoder_control(ctrlNet_cond.permute(0,2,1))
        control_input = control_input.permute(2, 0, 1) # [50, b, 384]
        control_input = self.first_zero_linear(control_input)
        # TODO cond should be added to CtrlNet??? 
        # torch.zeros_like(cond)
        control_input = xseq + torch.cat((torch.zeros_like(cond), control_input), axis=0)  # [seqlen+1, bs, d]
        ###############################################

        control_output_list = return_all_layers(self.seqTransEncoder_control, 
                                                control_input, 
                                                src_key_padding_mask=padding_mask)
        for i in range(self.num_layers):
            control_output_list[i] = self.mid_zero_linear[i](control_output_list[i])



        # print(xseq.shape, padding_mask.shape)

        # print(padding_mask.shape, xseq.shape)

        output = forward_with_condition(self.seqTransEncoder, 
                                        control_input, 
                                        list_of_controlnet_output=control_output_list,
                                        src_key_padding_mask=padding_mask)[1:] #(seqlen, b, e)
        logits = self.output_process(output) #(seqlen, b, e) -> (b, ntoken, seqlen)
        return logits
    

    def forward_predmotion(self, emb):
        pred_motions = self.vq_model.forward_decoder(emb)
        pred_motions_denorm = pred_motions * self.std + self.mean
        pred_motions_denorm = recover_from_ric(pred_motions_denorm.float(), self.opt.joints_num)
        return pred_motions, pred_motions_denorm
    
    def get_loss(self, pred_motions_denorm, global_joint, global_joint_mask):
        ########## Per Joint ####################
        # # Compute MSE loss without reduction
        # # loss = F.mse_loss(pred_motions_denorm, global_joint, reduction='none')  # Shape: (2048, 196, 22, 3)
        # _loss = (pred_motions_denorm - global_joint) ** 2
        # masked_loss = _loss * global_joint_mask.unsqueeze(-1)  # Masked loss, same shape
        # sum_loss = masked_loss.sum(dim=(1, 2, 3))  # Shape: (2048,)
        # valid_counts = global_joint_mask.sum(dim=(1, 2)) * 3 # 3 x,y,z #.clamp(min=1)  # Avoid division by zero
        # loss_tta = sum_loss / valid_counts  # Shape: (2048,)
        # loss_tta = loss_tta.sum()
        # return loss_tta

        ########## Per Sample ###################################
        # Expand mask to match the last dim (3D)
        expanded_mask = global_joint_mask.unsqueeze(-1).expand(-1, -1, -1, 3)  # [256, 196, 22, 3]

        # Apply mask and compute per-sample loss
        squared_error = (pred_motions_denorm - global_joint) ** 2  # [256, 196, 22, 3]
        masked_error = squared_error * expanded_mask  # zeros out the masked entries

        # Per-sample normalization
        valid_counts = expanded_mask.sum(dim=(1, 2, 3)).clamp(min=1)  # [256]
        per_sample_loss = masked_error.sum(dim=(1, 2, 3)) / valid_counts  # [256]

        # Final loss: mean across batch
        loss_tta = per_sample_loss.mean()
        return loss_tta

    # @torch.no_grad()
    # @eval_decorator
    def generate(self,
                 conds,
                 m_lens,
                 timesteps: int,
                 cond_scale: int,
                 temperature=1,
                 topk_filter_thres=0.0,
                 gsample=False,
                 force_mask=False,
                 vq_model = None,
                 global_joint=None, 
                 global_joint_mask=None,
                 _mean=None,
                _std=None,
                lr=6e-2,
                each_iter=50,
                avoid_points=None,
                abitary_func=None,
                is_relative=False
                 ):
        select_after_conf = False
        # assert len(timesteps) >= len(cond_scales) == self.opt.num_quantizers

        device = next(self.parameters()).device
        seq_len = 49 if max(m_lens) <= 49 else 49*2 # for STMC max len is 300 something
        batch_size = len(m_lens)

        if self.cond_mode == 'text':
            if conds is None:
                cond_vector = torch.zeros((m_lens.shape[0], 512), device=m_lens.device)
            else:
                with torch.no_grad():
                    cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        # print(padding_mask.shape, )

        # Start from all tokens being masked
        ids = torch.where(padding_mask, self.pad_id, self.mask_id)
        emb = self.token_emb(ids)
        mask_emb = emb[0, 0].detach().clone()
        scores = torch.where(padding_mask, 1e5, 0.)
        starting_temperature = temperature

        logits_shape = [*emb.shape[:2], vq_model.quantizer.codebooks[0].shape[0]]
        filtered_logits = torch.ones(logits_shape, device=emb.device, dtype=torch.float)

        for timestep, steps in zip(torch.linspace(0, 1, timesteps, device=device), range(timesteps)):
            # temperature = (1-timestep)*.9 + .1
            # 0 < timestep < 1
            rand_mask_prob = self.noise_schedule(timestep)  # Tensor

            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = torch.round(rand_mask_prob * m_lens).clamp(min=1)  # (b, )

            # select num_token_masked tokens with lowest scores to be masked
            sorted_indices = scores.argsort(
                dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            ids = torch.where(is_mask, self.mask_id, ids)

            # emb = self.token_emb(ids)
            emb = torch.where(is_mask.unsqueeze(-1), mask_emb, emb) 
            # emb = probs @ vq_model.quantizer.codebooks[0]

            '''
            Preparing input
            '''
            ctrlNet_cond = None
            ### Calculate difference from current tokens (mased) with target pose ###
            _emb = F.softmax(filtered_logits/temperature, dim=-1) @ vq_model.quantizer.codebooks[0]
            _emb[is_mask] = self.mask_emb_vq
            # _emb = _emb.masked_fill(padding_mask.unsqueeze(-1), 0.)
            _pred_motions = vq_model.forward_decoder(_emb)
            _pred_motions_denorm = _pred_motions * self.std + self.mean
            _pred_motions_denorm = recover_from_ric(_pred_motions_denorm.float(), self.opt.joints_num)

            ctrlNet_cond = (global_joint - _pred_motions_denorm) * global_joint_mask.unsqueeze(-1)
            ctrlNet_cond2 = global_joint * global_joint_mask.unsqueeze(-1)
            if self.control == 'trajectory':
                ctrlNet_cond = ctrlNet_cond[:, :, 0]
                ctrlNet_cond2 = ctrlNet_cond2[:, :, 0]
            else:
                ctrlNet_cond = ctrlNet_cond[..., control_joint_ids, :]
                ctrlNet_cond = ctrlNet_cond.reshape((*ctrlNet_cond.shape[:2], -1))
                ctrlNet_cond2 = ctrlNet_cond2[..., control_joint_ids, :]
                ctrlNet_cond2 = ctrlNet_cond2.reshape((*ctrlNet_cond2.shape[:2], -1))
            ctrlNet_cond = torch.cat([ctrlNet_cond, ctrlNet_cond2], dim=-1)
            ###########################################################################
            
            # (b, num_token, seqlen)
            logits = self.forward_with_cond_scale(emb, cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  cond_scale=cond_scale,
                                                  force_mask=force_mask,
                                                  ctrlNet_cond=ctrlNet_cond)

            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            # print(logits.shape, self.opt.num_tokens)
            # clean low prob token
            # logits = torch.where(is_mask.unsqueeze(-1), logits, filtered_logits) 
            filtered_logits = logits.clone().detach() # top_k(logits, topk_filter_thres, dim=-1)

            '''
            Update ids
            '''
            # if force_mask:
            temperature = starting_temperature
            # else:
            # temperature = starting_temperature * (steps_until_x0 / timesteps)
            # temperature = max(temperature, 1e-4)
            # print(filtered_logits.shape)
            # temperature is annealed, gradually reducing temperature as well as randomness
            if gsample:  # use gumbel_softmax sampling
                # print("1111")
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
            else:  # use multinomial sampling
                # print("2222")
                if each_iter != 0: # steps_until_x0 == 0:
                    # seq_len = 49 # max(m_lens)
                    padding_mask = ~lengths_to_mask(m_lens, seq_len)
                    filtered_logits.requires_grad = True

                    optimizer = torch.optim.AdamW([filtered_logits], lr=lr, betas=(0.5, 0.9), weight_decay=1e-6) #  + list(trans.parameters()) + list(vq_model.parameters())

                        
                    if each_iter > 0:
                        iter = each_iter
                    else:
                        # dynamic TTT, each_iter is negative
                        iter = (steps+1)*(-each_iter)
                    for i in range(iter):
                        # if steps < timesteps-3:
                        #     break
                        ################ Residual Layers ################
                        # num_quant_layers = 6
                        # history_sum = 0
                        # logits = torch.cat([filtered_logits, torch.zeros_like(filtered_logits[..., :1])], dim=-1)
                        # all_logits = [logits]
                        # for i in range(1, num_quant_layers):
                        #     token_embed = self.res_model.token_embed_weight[i-1]
                        #     _emb = F.softmax(logits/.1, dim=-1) @ token_embed
                        #     history_sum += _emb
                        #     logits = self.res_model.forward_with_cond_scale(history_sum, i, cond_vector, padding_mask, cond_scale=cond_scale)
                        #     logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
                        #     pred_ids = gumbel_sample(logits, temperature=temperature, dim=-1)  # (b, seqlen)
                        #     ids = torch.where(padding_mask, self.pad_id, pred_ids)
                        #     logits = torch.where(ids.unsqueeze(-1)==self.pad_id, token_embed[..., -1], logits)
                        #     all_logits.append(logits)
                        # emb = history_sum
                        ################ No Residual Layers ################
                        _probs = ((filtered_logits / max(temperature, 1e-10)) + gumbel_noise(filtered_logits))
                        emb = F.softmax(_probs, dim=-1) @ vq_model.quantizer.codebooks[0]
                        # emb = F.softmax(filtered_logits/temperature, dim=-1) @ vq_model.quantizer.codebooks[0]
                        ################################################################

                        emb = emb.masked_fill(padding_mask.unsqueeze(-1), 0.) # [320, 49, 512]
                        pred_motions, pred_motions_denorm = self.forward_predmotion(emb)

                        # print(pred_motions_denorm[batch_arange, sampled_keyframes][..., 0, [0, 2]].shape)
                        if global_joint[global_joint_mask].sum() != 0:
                            if is_relative:
                                joints = convert_pred_motions_to_joints(pred_motions)
                                loss_tta = F.mse_loss(joints[global_joint_mask], 
                                                    global_joint[global_joint_mask], reduction='mean') # mse_loss, l1_loss
                            else:
                                # loss_tta = F.mse_loss(pred_motions_denorm[global_joint_mask], 
                                #                     global_joint[global_joint_mask], reduction='mean') # mse_loss, l1_loss
                                loss_tta = self.get_loss(pred_motions_denorm, global_joint, global_joint_mask)
                            # loss_tta = loss_tta*global_joint_mask.sum()/2000
                        else:
                            loss_tta = 0
                        if avoid_points is not None:
                            loss_sdf = sdf(pred_motions_denorm, 
                                        avoid_points, m_lens*4)
                            loss_tta += loss_sdf
                        if abitary_func is not None:
                            loss_tta += abitary_func(pred_motions_denorm)
                        if loss_tta == 0:
                            break
                        # print('loss_tta:', loss_tta)
                        # loss_tta.requires_grad = True
                        optimizer.zero_grad()
                        loss_tta.backward()
                        optimizer.step()
                    # pred_motions, pred_motions_denorm = forward_predmotion()
                

                # print(temperature, starting_temperature, steps_until_x0, timesteps)
                # print(probs / temperature)
                # probs = F.softmax(filtered_logits / temperature, dim=-1)  # (b, seqlen, ntoken)
                _probs = ((filtered_logits / max(temperature, 1e-10)) + gumbel_noise(filtered_logits))
                probs = F.softmax(_probs, dim=-1)
                # print('_probs:', _probs[0,0])
                # print([0,0])
                pred_emb = probs @ self.token_emb.weight[:vq_model.quantizer.codebooks[0].shape[0]]
                if select_after_conf:
                    probs = F.softmax(logits / temperature, dim=-1)  # (b, seqlen, ntoken)
                pred_ids = Categorical(probs).sample()  # (b, seqlen)
                # pred_ids = probs.argmax(-1)  # (b, seqlen)

            # print(pred_ids.max(), pred_ids.min())
            # if pred_ids.
            ids = torch.where(is_mask, pred_ids, ids)
            emb = pred_emb # torch.where(is_mask.unsqueeze(-1), pred_emb, emb) 

            '''
            Updating scores
            '''
            if select_after_conf:
                probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
            else:
                probs_without_temperature = filtered_logits.softmax(dim=-1)  # (b, seqlen, ntoken)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
            scores = scores.squeeze(-1)  # (b, seqlen)

            # We do not want to re-mask the previously kept tokens, or pad tokens
            scores = scores.masked_fill(~is_mask, 1e5)

        ids = torch.where(padding_mask, -1, ids)
        # print("Final", ids.max(), ids.min())

        return ids, filtered_logits
    

    def generate_with_control(self, clip_text, m_length, time_steps, cond_scale,
                                temperature, topkr,
                                force_mask, 
                                vq_model, 
                                global_joint, 
                                global_joint_mask,
                                _mean,
                                _std,
                                res_cond_scale,
                                res_model,
                                control_opt = None,
                                avoid_points = None,
                                abitary_func = None,
                                is_relative=False):
        m_lens = m_length // 4
        mids, logits = self.generate(clip_text, m_lens, time_steps, cond_scale,
                                    temperature=temperature, topk_filter_thres=topkr,
                                    force_mask=force_mask, 
                                    vq_model=vq_model, 
                                    global_joint=global_joint, 
                                    global_joint_mask=global_joint_mask,
                                    _mean=_mean,
                                    _std=_std,
                                    lr=control_opt['each_lr'], each_iter=control_opt['each_iter'],
                                    avoid_points=avoid_points,
                                    abitary_func=abitary_func,
                                    is_relative=is_relative)

        # for i in range(1):
        #     logits = self.refine(logits, clip_text, m_lens, cond_scale=cond_scale, global_joint=global_joint, global_joint_mask=global_joint_mask)

        if res_model is not None:
            if hasattr(res_model, 'vq_model'):
                pred_ids, pred_logits = res_model.generate(mids, clip_text, m_lens, temperature=1, cond_scale=res_cond_scale, logits=logits, global_joint=global_joint, global_joint_mask=global_joint_mask)
            else:
                pred_ids, pred_logits = res_model.generate(mids, clip_text, m_lens, temperature=1, cond_scale=res_cond_scale, logits=logits)

        seq_len = 49 if max(m_lens) <= 49 else 49*2 # 49 # 
        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        if control_opt['iter'] != 0:
            if res_model is not None:
                emb = 0
                for i in range(pred_logits.shape[-1]):
                    if i == 0:
                        temp = temperature
                    else:
                        temp = 1e-8
                    emb += F.softmax(pred_logits[..., :-1, i]/temp, dim=-1) @ vq_model.quantizer.codebooks[i]
            else:
                emb = F.softmax(logits/1, dim=-1) @ vq_model.quantizer.codebooks[0]
            emb = emb.detach().clone()
            emb = emb.masked_fill(padding_mask.unsqueeze(-1), 0.)
            emb.requires_grad = True
            optimizer = torch.optim.AdamW([emb], lr=control_opt['lr'], betas=(0.5, 0.9), weight_decay=1e-6) #  + list(trans.parameters()) + list(vq_model.parameters())

            for i in range(control_opt['iter']):
                pred_motions, pred_motions_denorm = self.forward_predmotion(emb)

                if global_joint[global_joint_mask].sum() != 0:
                    if is_relative:
                        joints = convert_pred_motions_to_joints(pred_motions)
                        loss_tta = F.mse_loss(joints[global_joint_mask], 
                                            global_joint[global_joint_mask], reduction='mean') # mse_loss, l1_loss
                    else:
                        # loss_tta = F.mse_loss(pred_motions_denorm[global_joint_mask], 
                        #                     global_joint[global_joint_mask], reduction='mean') # mse_loss, l1_loss
                        loss_tta = self.get_loss(pred_motions_denorm, global_joint, global_joint_mask)
                    # loss_tta = loss_tta*(global_joint_mask.shape[0]/8)
                    # loss_tta = loss_tta*global_joint_mask.sum()/2000

                    
                else:
                    loss_tta = 0
                if avoid_points is not None:
                    loss_sdf = sdf(pred_motions_denorm, 
                                   avoid_points, m_length)
                    loss_tta += loss_sdf
                if abitary_func is not None:
                    loss_tta += abitary_func(pred_motions_denorm)
                if loss_tta == 0:
                    break
                
                optimizer.zero_grad()
                loss_tta.backward()
                optimizer.step()
            pred_motions, pred_motions_denorm = self.forward_predmotion(emb)
        
        else: 
            if res_model is not None:
                emb = 0
                for i in range(pred_logits.shape[-1]):
                    if i == 0:
                        temp = 1
                    else:
                        temp = 1e-8
                    emb += F.softmax(pred_logits[..., :-1, i]/temp, dim=-1) @ vq_model.quantizer.codebooks[i]
            else:
                emb = F.softmax(logits/1, dim=-1) @ vq_model.quantizer.codebooks[0]
            emb = emb.masked_fill(padding_mask.unsqueeze(-1), 0.)
            pred_motions = vq_model.forward_decoder(emb)

            pred_motions_denorm = pred_motions * _std + _mean
            pred_motions_denorm = recover_from_ric(pred_motions_denorm.float(), self.opt.joints_num)
        return pred_motions_denorm, pred_motions
    
    def timeline_control(self, timeline, m_length,
                                vq_model, 
                                _mean,
                                _std,
                                res_model,
                                is_relative=False):
        ### Fixed Param ###
        cond_scale = 4
        force_mask = False
        topkr=.9
        temperature = 1
        ####################

        pred_motions_denorm = None
        # layers, b, f, j, d
        for i in range(len(timeline)):
            global_joint_mask = torch.zeros((1, 196, 22), dtype=bool, device=m_length.device)
            print('i', i)
            if i == 0:
                # TODO need to remove CTRLNet in the first generation
                global_joint = torch.zeros((1, 196, 22, 3), dtype=torch.float, device=m_length.device)
                # TODO this is just work around
                _m_length = m_length/m_length * timeline[i][2][1] # 
                print('_m_length:', _m_length)
                ctrl_net = self.ctrl_net
                self.ctrl_net = None
                pred_motions_denorm, pred_motions = self.generate_with_control([timeline[i][1]], _m_length, time_steps=10, cond_scale=cond_scale,
                                                                            temperature=temperature, topkr=topkr,
                                                                            force_mask=force_mask, 
                                                                            vq_model=vq_model, 
                                                                            global_joint=global_joint, 
                                                                            global_joint_mask=global_joint_mask,
                                                                            _mean=_mean,
                                                                            _std=_std,
                                                                            res_cond_scale=5,
                                                                            res_model=None,
                                                                            control_opt = {
                                                                                'each_lr': 6e-2,
                                                                                'each_iter': 0,
                                                                                'lr': 6e-2,
                                                                                'iter': 0,
                                                                            })
                self.ctrl_net = ctrl_net
            elif i > 0:
                for j in range(i):
                    frames = timeline[j][-1]
                    ctrl_joints = joints_by_part[timeline[j][0]]
                    global_joint_mask[0, frames[0]:frames[1], ctrl_joints] = True
                
                if is_relative:
                    global_joint = convert_pred_motions_to_joints(pred_motions).detach().clone()
                else:
                    global_joint = pred_motions_denorm.detach().clone()
                if is_relative:
                    ctrl_net = self.ctrl_net
                    self.ctrl_net = not is_relative
                pred_motions_denorm, pred_motions = self.generate_with_control([timeline[i][1]], m_length, time_steps=10, cond_scale=cond_scale,
                                                                            temperature=temperature, topkr=topkr,
                                                                            force_mask=force_mask, 
                                                                            vq_model=vq_model, 
                                                                            global_joint=global_joint, 
                                                                            global_joint_mask=global_joint_mask,
                                                                            _mean=_mean,
                                                                            _std=_std,
                                                                            res_cond_scale=5,
                                                                            res_model=None,
                                                                            control_opt = {
                                                                                'each_lr': 6e-2,
                                                                                'each_iter': 100,
                                                                                'lr': 6e-2,
                                                                                'iter': 600,
                                                                            },
                                                                            is_relative=is_relative) # res_model if i==len(timeline)-1 else None
                if is_relative:
                    self.ctrl_net = ctrl_net
        return pred_motions_denorm, pred_motions

    def timeline_STMC(self, timelines, m_length,
                                vq_model, 
                                _mean,
                                _std,
                                res_model,
                                bp_timeline):
        from utils.metrics import joints_by_part
        ### Fixed Param ###
        cond_scale = 4
        force_mask = False
        topkr=.9
        temperature = 1
        ####################

        pred_motions_denorm = None
        # layers, b, f, j, d

        #################### prepare timeline ####################
        joints = []
        texts = []
        lengths = []
        timeline_idx = []
        count_per_timeline = []
        current_count = 0
        for i, timeline in enumerate(timelines):
            for entry in timeline:
                parts, text, (start, end) = entry
                ctrl_joints = []
                for p in parts:
                    indices = joints_by_part.get(p, [])
                    if isinstance(indices, int):
                        indices = [indices]
                    ctrl_joints.extend(indices)
                joints.append(ctrl_joints)
                texts.append(text)
                lengths.append(end - start)
                timeline_idx.append(i)
            count_per_timeline.append(current_count)
            current_count += len(timeline)
        ########################################
        lengths = torch.from_numpy(np.array(lengths)).to(m_length.device)
        rounded_length = (lengths + 3) // 4 * 4
        bodypart_joint = torch.zeros((len(joints), 196, 22, 3), dtype=torch.float, device=m_length.device)
        bodypart_joint_mask = torch.zeros((len(joints), 196, 22), dtype=bool, device=m_length.device)
        self.ctrl_net = None
        pred_motions_denorm, pred_motions = self.generate_with_control(texts, rounded_length, time_steps=10, cond_scale=cond_scale,
                                                                        temperature=temperature, topkr=topkr,
                                                                        force_mask=force_mask, 
                                                                        vq_model=vq_model, 
                                                                        global_joint=bodypart_joint, 
                                                                        global_joint_mask=bodypart_joint_mask,
                                                                        _mean=_mean,
                                                                        _std=_std,
                                                                        res_cond_scale=5,
                                                                        res_model=res_model,
                                                                        control_opt = {
                                                                            'each_lr': 6e-2,
                                                                            'each_iter': 0,
                                                                            'lr': 6e-2,
                                                                            'iter': 0,
                                                                        })
        pred_motions0 = pred_motions
        
        bodypart_relative_joint = convert_pred_motions_to_joints(pred_motions).detach().clone()
        blank_joints = torch.zeros((len(timelines), 196*2, 22, 3), dtype=torch.float, device=m_length.device)
        blank_joints_mask = torch.zeros((len(timelines), 196*2, 22), dtype=bool, device=m_length.device)
        all_gen_idx = 0
        cutoff = 180
        transition = 4

        for timeline_index, (bp_timeline_map, timeline_entries) in enumerate(zip(bp_timeline, timelines)):
            for bodypart, intervals in bp_timeline_map.items():
                for interval_idx, (seq_index, global_bp_start, global_bp_end) in enumerate(intervals):
                    # global_bp_start/ global_bp_end ==> global frame of bodypart
                    # local_start/local_end  ==> relative frame of bodypart to local generated samples
                    # global_gen_start ==> global frame of generated samples
                    if interval_idx > 0:
                        cur_start = intervals[interval_idx][1]
                        prev_end = intervals[interval_idx-1][2]
                        if cur_start <= prev_end:
                            global_bp_start += 5
                    if interval_idx < len(intervals) -1:
                        cur_end = intervals[interval_idx][2]
                        next_start = intervals[interval_idx+1][1]
                        if cur_end >= next_start:
                            global_bp_end -= 5

                    timeline_item = timeline_entries[seq_index]
                    global_gen_start, _ = timeline_item[2]
                    local_start = global_bp_start - global_gen_start
                    local_end = global_bp_end - global_gen_start

                    # in case it end less than 4 frame padding will not fit so rm them all
                    if local_end <= 0:
                        continue

                    # print(f"Timeline {timeline_index}, Seq {seq_index}, '{bodypart}', local_start: {local_start}, local_end: {local_end}, global_gen_start: {global_gen_start}, global_bp_start:{global_bp_start}, global_bp_end:{global_bp_end}")

                    ctrl_joints = joints_by_part[bodypart]
                    # print(ctrl_joints)
                    blank_joints_mask[timeline_index, global_bp_start:global_bp_end, ctrl_joints] = True
                    
                    # print('1:', blank_joints[timeline_index, global_bp_start:global_bp_end, ctrl_joints].shape)
                    # print('2:', bodypart_relative_joint[count_per_timeline[timeline_index] + seq_index, local_start:local_end, ctrl_joints].shape)
                    # print('global_bp_start:', global_bp_start, 'global_bp_end:', global_bp_end)
                    # print('local_start:', local_start, 'local_end:', local_end)
                    # print(intervals)
                    blank_joints[timeline_index, global_bp_start:global_bp_end, ctrl_joints] = \
                        bodypart_relative_joint[count_per_timeline[timeline_index] + seq_index, local_start:local_end, ctrl_joints]
                    
        
        blank_length = torch.ones(len(timelines), device=m_length.device) * 196 *2
        # texts[:len(timelines)]
        pred_motions_denorm, pred_motions = self.generate_with_control(None, blank_length, time_steps=10, cond_scale=cond_scale,
                                                                        temperature=temperature, topkr=topkr,
                                                                        force_mask=force_mask, 
                                                                        vq_model=vq_model, 
                                                                        global_joint=blank_joints, 
                                                                        global_joint_mask=blank_joints_mask,
                                                                        _mean=_mean,
                                                                        _std=_std,
                                                                        res_cond_scale=5,
                                                                        res_model=None,
                                                                        control_opt = {
                                                                            'each_lr': 6e-2,
                                                                            'each_iter': 300,
                                                                            'lr': 6e-2,
                                                                            'iter': 300,
                                                                        },
                                                                        is_relative=True)
        
        # step1_relative_joint = convert_pred_motions_to_joints(pred_motions).detach().clone()
        # blank_joints_mask[i, 1, start:_end, ctrl_joints] = True
        # blank_joints[i, 1, start:_end, ctrl_joints] = bodypart_relative_joint[bp_idx, _start:_end, ctrl_joints]
        
        
        return pred_motions_denorm, pred_motions

    def forward_refine(self, motion, y, m_length):
        # KIT is broken here.
        assert m_length.min() > 30 # make sure this is raw range, not downsampling 4
        m_lens = m_length//4

        x = motion.permute(0, 2, 1).float()
        x_encoder = self.vq_model.encoder(x)
        code_idx = self.vq_model.quantizer.quantize(x_encoder)
        pose = self.vq_model.decoder(x_encoder)

        global_joint = recover_from_ric(self.inv_transform(pose).float(), self.opt.joints_num)
        device = motion.device
        bs, ntokens = motion.shape[:2]
        ntokens = int(ntokens/4)

        ########### 4. Rand 0-100% Cond ###########################
        if self.control == 'trajectory' or self.control == 'random':
            rand_length = torch.rand(m_length.shape).cuda() * 196 # m_length.cuda()
            rand_length = rand_length.round().clamp(min=1, max=196)

            all_len_mask = lengths_to_mask(m_length, pose.shape[1]) #(b, n)
            batch_randperm = torch.rand((bs, pose.shape[1]), device=pose.device)
            batch_randperm[~all_len_mask] = 1
            batch_randperm = batch_randperm.argsort(dim=-1)
            global_joint_mask = batch_randperm < rand_length.unsqueeze(-1)
            global_joint_mask = global_joint_mask * all_len_mask

        ######################################
            if self.control == 'trajectory':
                global_joint_mask = repeat(global_joint_mask, 'b f -> b f j', j=self.opt.joints_num).clone()
                global_joint_mask[..., 1:] = False
            elif self.control == 'random':
                _global_joint_mask = global_joint_mask
                global_joint_mask = torch.zeros((*global_joint_mask.shape, self.opt.joints_num), device=_global_joint_mask.device, dtype=bool)
                control_joints = torch.tensor(control_joint_ids, device=pose.device)
                rand_indx = torch.randint(len(control_joints), (_global_joint_mask.shape[0],)) # random index (bs,)
                global_joint_mask[torch.arange(global_joint_mask.shape[0]),:, 
                                control_joints[rand_indx]] = _global_joint_mask # set idx of joint to frames mask
        elif self.control == 'cross':
            rand_length = torch.rand(m_length.shape, device=m_length.device) * m_length * len(control_joint_ids)
            rand_length = rand_length.cuda().round().clamp(min=1)

            all_len_mask = lengths_to_mask(m_length, pose.shape[1]) #(b, n)
            batch_randperm = torch.rand((*all_len_mask.shape, len(control_joint_ids)), device=pose.device)
            batch_randperm[~all_len_mask] = 1
            batch_randperm = batch_randperm.reshape((all_len_mask.shape[0], -1) )
            batch_randperm = batch_randperm.argsort(dim=-1)
            batch_randperm = batch_randperm.reshape((*all_len_mask.shape, -1) )
            _global_joint_mask = batch_randperm < rand_length.unsqueeze(-1).unsqueeze(-1)
            _global_joint_mask = _global_joint_mask * all_len_mask.unsqueeze(-1)

            global_joint_mask = torch.zeros((*pose.shape[:2], self.opt.joints_num), device=pose.device, dtype=bool)
            global_joint_mask[..., control_joint_ids] = _global_joint_mask
        else:
            raise Exception(f'{self.control} is not implemented yet!!!')
            
        ######################################

        non_pad_mask = lengths_to_mask(m_lens, ntokens) #(b, n)
        with torch.no_grad():
            cond_vector = self.encode_text(y)


        
        #### Noise ####
        b = motion.shape[0]
        f = int(motion.shape[1]/4)
        num_cb = self.vq_model.quantizer.codebooks[0].shape[0]
        flip_prob = .3
        std = .03
        replace_mask = torch.bernoulli(torch.full((b, f), flip_prob)).bool().cuda()
        random_indices = torch.randint(0, num_cb, (b, f)).cuda()

        code_flip = random_indices * replace_mask + code_idx[..., 0] * ~replace_mask
        one_hot = F.one_hot(code_flip, num_classes=num_cb).float()
        noise_hot = torch.normal(torch.zeros_like(one_hot), torch.ones_like(one_hot)*std)
        noise_hot = one_hot+noise_hot

        #### Condition ####
        vq_emb = (noise_hot @ self.vq_model.quantizer.codebooks[0]).permute(0,2,1)
        pred_motions = self.vq_model.decoder(vq_emb) # [64, 196, 263]
        pred_motions_denorm = pred_motions * self.std + self.mean
        pred_motions_denorm = recover_from_ric(pred_motions_denorm.float(), self.opt.joints_num)

        ctrlNet_cond = (global_joint - pred_motions_denorm) * global_joint_mask.unsqueeze(-1)
        ctrlNet_cond2 = global_joint * global_joint_mask.unsqueeze(-1)
        if self.control == 'trajectory':
            ctrlNet_cond = ctrlNet_cond[:, :, 0]
            ctrlNet_cond2 = ctrlNet_cond2[:, :, 0]
        else:
            ctrlNet_cond = ctrlNet_cond[..., control_joint_ids, :]
            ctrlNet_cond = ctrlNet_cond.reshape((*ctrlNet_cond.shape[:2], -1))
            ctrlNet_cond2 = ctrlNet_cond2[..., control_joint_ids, :]
            ctrlNet_cond2 = ctrlNet_cond2.reshape((*ctrlNet_cond2.shape[:2], -1))
        ctrlNet_cond = torch.cat([ctrlNet_cond, ctrlNet_cond2], dim=-1)
        ###########################################################################
        padding_mask = ~non_pad_mask

        # [64, 49, 512], b, f, e
        trans_emb = noise_hot @ self.token_emb.weight[:num_cb]
        mask_emb = self.token_emb.weight[self.mask_id]
        trans_emb[padding_mask] = mask_emb
        logits = self.trans_forward(trans_emb, cond_vector, ~non_pad_mask, force_mask=False, ctrlNet_cond=ctrlNet_cond)
    
        ###### Embeddin loss ######
        trans_emb_out = F.softmax(logits.permute(0,2,1), dim=-1) @ self.token_emb.weight[:num_cb]
        trans_emb_gt = self.token_emb.weight[code_idx[..., 0]]
        loss_emb = F.l1_loss(trans_emb_out[non_pad_mask], trans_emb_gt[non_pad_mask], reduction='mean')

        ###### Control Loss #######
        emb = F.softmax(logits.permute(0,2,1)/1, dim=-1) @ self.vq_model.quantizer.codebooks[0]
        
        emb = emb.masked_fill(padding_mask.unsqueeze(-1), 0.) # TODO should use average emb??
        pred_motions = self.vq_model.forward_decoder(emb)
        # emb[labels!=self.mask_id] = x_d[0, labels!=self.mask_id]
        pred_motions_denorm = pred_motions * self.std + self.mean
        pred_motions_denorm = recover_from_ric(pred_motions_denorm.float(), self.opt.joints_num)

        # batch_arange = torch.arange(sampled_keyframes.size(0)).unsqueeze(1)
        # loss_tta = F.l1_loss(pred_motions_denorm[batch_arange, sampled_keyframes][..., 0, [0, 1, 2]], 
        #                     gt_skel_motions[batch_arange, sampled_keyframes][..., 0, [0, 1, 2]], reduction='mean')


        # ce_loss, pred_id, acc = cal_performance(logits, labels, ignore_index=self.mask_id)


        # TODO weight by sample
        # print('____ TODO ______ weight by sample')
        loss_tta = F.l1_loss(pred_motions_denorm[global_joint_mask], 
                                        global_joint[global_joint_mask], reduction='mean') # mse_loss l1_loss
        return loss_emb, loss_tta
    
    def refine(self, logits, conds, m_lens, cond_scale: int, global_joint=None, global_joint_mask=None,):
        seq_len = 49
        with torch.no_grad():
            cond_vector = self.encode_text(conds)

        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        emb = F.softmax(logits, dim=-1) @ self.vq_model.quantizer.codebooks[0]
        emb = emb.masked_fill(padding_mask.unsqueeze(-1), 0.) # TODO should use average emb??
        _pred_motions = self.vq_model.forward_decoder(emb)
        _pred_motions_denorm = _pred_motions * self.std + self.mean
        _pred_motions_denorm = recover_from_ric(_pred_motions_denorm.float(), self.opt.joints_num)

        ctrlNet_cond = (global_joint - _pred_motions_denorm) * global_joint_mask.unsqueeze(-1)
        ctrlNet_cond2 = global_joint * global_joint_mask.unsqueeze(-1)
        if self.control == 'trajectory':
            ctrlNet_cond = ctrlNet_cond[:, :, 0]
            ctrlNet_cond2 = ctrlNet_cond2[:, :, 0]
        else:
            ctrlNet_cond = ctrlNet_cond[..., control_joint_ids, :]
            ctrlNet_cond = ctrlNet_cond.reshape((*ctrlNet_cond.shape[:2], -1))
            ctrlNet_cond2 = ctrlNet_cond2[..., control_joint_ids, :]
            ctrlNet_cond2 = ctrlNet_cond2.reshape((*ctrlNet_cond2.shape[:2], -1))
        ctrlNet_cond = torch.cat([ctrlNet_cond, ctrlNet_cond2], dim=-1)


        trans_emb = F.softmax(logits, dim=-1) @ self.token_emb.weight[:self.vq_model.quantizer.codebooks[0].shape[0]]
        logits = self.forward_with_cond_scale(trans_emb, cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  cond_scale=cond_scale,
                                                  force_mask=False,
                                                  ctrlNet_cond=ctrlNet_cond)
        return logits.permute(0, 2, 1)

    def inv_transform(self, data):
        assert self.std is not None and self.mean is not None
        return data * self.std + self.mean
    


def convert_pred_motions_to_joints(pred_motions):
    # 0 root rotation
    # 1-3 root position
    # realtive_motions = pred_motions[:, :, 4:(22-1)*3+4]
    # realtive_motions = F.pad(realtive_motions, (3, 0)) # padd zeros for first 3 so total joints=22
    realtive_motions = pred_motions[:, :, 1:(22-1)*3+4]
    realtive_motions[..., :3] += pred_motions[..., :1] # quick hack to make angle differentiable, need to move to another dim
    # realtive_motions[..., :3] += realtive_motions[..., :3] # 
    return realtive_motions.reshape(*realtive_motions.shape[:-1], 22, 3)

def sdf(pred, cond, m_length):
    # joint = 0
    # height = .25
    # dist = torch.clamp(pred[:, 53, joint][..., 1] - height, min=0.0)
    # loss_colli = dist.sum(axis=-1) * 1 / (dist>0).sum(axis=-1)
    # loss_colli = dist.mean()
    # return loss_colli



    w_colli = 5
    loss_colli = 0.0
    # batch SDF
    joint = 10
    if len(cond.shape) == 2:
        from einops import repeat
        cond = repeat(cond, 'o four -> b f o four', b=pred.shape[0], f=pred.shape[1])
        pred = repeat(pred, 'b f j d -> b f j o d', o=cond.shape[0])
    
    dist = torch.norm(pred[:, :, joint] - cond[..., :3], dim=-1)
    dist = torch.clamp(cond[..., 3] - dist, min=0.0)
    loss_colli = dist[cond[..., 3]>0].mean()
    # loss_colli = dist.sum(axis=-1) * w_colli / (dist>0).sum(axis=-1)
    # loss_colli = dist[cond[..., 3]>0] * w_colli
    # loss_colli = dist.mean()

    # single point for visualization
    # for i in range(cond.shape[0]):
    #     joint = 15 # head
    #     dist = torch.norm(pred[0, :m_length[0], joint] - cond[i, :3], dim=-1)
    #     dist = torch.clamp(cond[i, 3] - dist, min=0.0)
    #     if dist.sum() != 0 :
    #         loss_colli += dist.sum() / (dist!=0).sum() * w_colli #/ m_length[0] 
    #     else:
    #         loss_colli += dist.sum()
    return loss_colli