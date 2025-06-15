import os
from os.path import join as pjoin

import torch
import torch.nn.functional as F

from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE, LengthEstimator

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt

from utils.fixseed import fixseed
from visualization.joints2bvh import Joint2BVHConvertor
from torch.distributions.categorical import Categorical


from utils.motion_process import recover_from_ric, recover_root_rot_pos
from utils.plot_script import plot_3d_motion

from utils.paramUtil import t2m_kinematic_chain

import numpy as np
clip_version = 'ViT-B/32'

def load_vq_model(vq_opt):
    # opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_model = RVQVAE(vq_opt,
                vq_opt.dim_pose,
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
    print(f'Loading VQ Model {vq_opt.name} Completed!')
    return vq_model, vq_opt

def load_trans_model(model_opt, opt, which_model):
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
                      map_location='cpu')
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    # print(ckpt.keys())
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer

def load_res_model(res_opt):
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

    ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', 'net_best_fid.tar'),
                      map_location=opt.device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    return res_transformer

def load_len_estimator(opt):
    model = LengthEstimator(512, 50)
    ckpt = torch.load(pjoin('checkpoints', opt.dataset_name, 'length_estimator', 'model', 'finest.tar'),
                      map_location=opt.device)
    model.load_state_dict(ckpt['estimator'])
    print(f'Loading Length Estimator from epoch {ckpt["epoch"]}!')
    return model


if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse(is_eval=True)
    opt.combine=True
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    dim_pose = 251 if opt.dataset_name == 'kit' else 263

    # out_dir = pjoin(opt.check)
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    result_dir = pjoin('./generation', opt.ext)
    joints_dir = pjoin(result_dir, 'joints')
    animation_dir = pjoin(result_dir, 'animations')
    os.makedirs(joints_dir, exist_ok=True)
    os.makedirs(animation_dir,exist_ok=True)

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)


    #######################
    ######Loading RVQ######
    #######################
    vq_opt_path = pjoin('./log/vq', opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    #################################
    ######Loading R-Transformer######
    #################################
    res_opt_path = pjoin('checkpoints', opt.dataset_name, opt.res_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_model = load_res_model(res_opt)

    assert res_opt.vq_name == model_opt.vq_name

    #################################
    ######Loading M-Transformer######
    #################################
    t2m_transformer = load_trans_model(model_opt, opt, 'latest.tar')

    ##################################
    #####Loading Length Predictor#####
    ##################################
    length_estimator = load_len_estimator(model_opt)

    t2m_transformer.eval()
    vq_model.eval()
    res_model.eval()
    length_estimator.eval()

    res_model.to(opt.device)
    t2m_transformer.to(opt.device)
    vq_model.to(opt.device)
    length_estimator.to(opt.device)

    ##### ---- Dataloader ---- #####
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22

    mean = np.load(pjoin('checkpoints', opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin('checkpoints', opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))
    
    def inv_transform(data):
        return data * std + mean

    prompt_list = []
    length_list = []

    est_length = False
    if opt.text_prompt != "":
        # prompt_list.append(opt.text_prompt)
        # if opt.motion_length == 0:
        #     est_length = True
        # else:
        #     length_list.append(opt.motion_length)
        opt.text_prompt=opt.text_prompt.split('|')
        print(opt.text_prompt)
        for i in range(len(opt.text_prompt)):
            prompt_list.append(opt.text_prompt[i])
            est_length=True

        
    elif opt.text_path != "":
        with open(opt.text_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                infos = line.split('#')
                prompt_list.append(infos[0])
                if len(infos) == 1 or (not infos[1].isdigit()):
                    est_length = True
                    length_list = []
                else:
                    length_list.append(int(infos[-1]))
    else:
        raise "A text prompt, or a file a text prompts are required!!!"
    # print('loading checkpoint {}'.format(file))
    print("prompt_list",prompt_list)

    if est_length:
        print("Since no motion length are specified, we will use estimated motion lengthes!!")
        text_embedding = t2m_transformer.encode_text(prompt_list)
        pred_dis = length_estimator(text_embedding)
        probs = F.softmax(pred_dis, dim=-1)  # (b, ntoken)
        token_lens = Categorical(probs).sample()  # (b, seqlen)
        # lengths = torch.multinomial()
    else:
        token_lens = torch.LongTensor(length_list) // 4
        token_lens = token_lens.to(opt.device).long()

    m_length = token_lens * 4
    captions = prompt_list

    sample = 0
    kinematic_chain = t2m_kinematic_chain
    converter = Joint2BVHConvertor()

    for r in range(opt.repeat_times):
        print("-->Repeat %d"%r)
        with torch.no_grad():
            mids, pred_len = t2m_transformer.generate(captions, token_lens,
                                            timesteps=opt.time_steps,
                                            cond_scale=opt.cond_scale,
                                            temperature=opt.temperature,
                                            topk_filter_thres=opt.topkr,
                                            gsample=opt.gumbel_sample,
                                            is_predict_len=opt.motion_length==-1)
            token_lens = pred_len
            m_length = token_lens*4
            # print(mids)
            # print(mids.shape)
            mids = res_model.generate(mids, captions, token_lens, temperature=1, cond_scale=5)
            print(len(prompt_list),opt.combine)
            if len(prompt_list)>0 and opt.combine:
                num_transition_token = 2
                b = mids.shape[0]
                half_token_length = (pred_len/2).int()
                idx_full_len = half_token_length >= 24
                half_token_length[idx_full_len] = half_token_length[idx_full_len] - 1
                tokens = -1 * torch.ones((b-1, 50), dtype=torch.long)
                transition_train_length = []
                for i in range(b-1):
                    i_index_motion = mids[i ,: , 0]
                    i1_index_motion = mids[i+1 ,: , 0]

                    left_end = half_token_length[i]
                    right_start = left_end + num_transition_token
                    end = right_start + half_token_length[i+1]

                    tokens[i, :left_end] = i_index_motion[pred_len[i]-left_end: pred_len[i]]
                    tokens[i, left_end:right_start] = t2m_transformer.pad_id
                    tokens[i, right_start:end] = i1_index_motion[:half_token_length[i+1]]
                    transition_train_length.append(end)
                transition_train_length = torch.tensor(transition_train_length).to(mids.device).long()
                tokens = tokens.scatter_(-1, transition_train_length[..., None], t2m_transformer.end_id).long() # add end token to tell model where to stop generating
                inpainting_mask = tokens == t2m_transformer.pad_id
                tokens[tokens == -1] = t2m_transformer.pad_id
                inpaint_index = t2m_transformer.edit2(None, tokens)

                # residual 
                all_tokens = []
                for i in range(b-1):
                    all_tokens.append(mids[i, :pred_len[i]])

                    inpaint_all = inpaint_index[i, inpainting_mask[i]].unsqueeze(-1)
                    inpaint_all = torch.nn.functional.pad(inpaint_all,  (0, 5),  value = -1)
                    all_tokens.append(inpaint_all)
                all_tokens.append(mids[-1, :pred_len[-1]])
                mids = torch.cat(all_tokens).unsqueeze(0)
                # m_length=sum(pred_len) * 4 + (len(prompt_list) - 1) * 8
                

            pred_motions = vq_model.forward_decoder(mids)
            print("pred_motions",len(pred_motions))


            pred_motions = pred_motions.detach().cpu().numpy()
            print("pred_motions detach",len(pred_motions))
            data = inv_transform(pred_motions)
            print(len(data))

        

        for k, (caption, joint_data)  in enumerate(zip(captions, data)):
            print("---->Sample %d: %s %d"%(k, caption, m_length[k]))
            animation_path = pjoin(animation_dir, str(k))
            joint_path = pjoin(joints_dir, str(k))

            os.makedirs(animation_path, exist_ok=True)
            os.makedirs(joint_path, exist_ok=True)

            joint_data = joint_data if opt.combine else joint_data[:m_length[k]]
            # x, y = recover_root_rot_pos(torch.from_numpy(joint_data).float())
            # print(x.shape, y.shape)
            # a = joint_data[..., 4:(22 - 1) * 3 + 4]
            # print(a.shape)
            joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()
            # Shape: 196 x 22 x 3
            # joint[:, 0, :] => 196 x 1 x 3 (original_trans)
            # smpl

            bvh_path = pjoin(animation_path, "sample%d_repeat%d_len%d_ik.bvh"%(k, r, m_length[k]))
            _, ik_joint = converter.convert(joint, filename=bvh_path, iterations=100)

            bvh_path = pjoin(animation_path, "sample%d_repeat%d_len%d.bvh" % (k, r, m_length[k]))
            
            _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)


            save_path = pjoin(animation_path, "sample%d_repeat%d_len%d.mp4"%(k, r, m_length[k]))
            ik_save_path = pjoin(animation_path, "sample%d_repeat%d_len%d_ik.mp4"%(k, r, m_length[k]))

            plot_3d_motion(ik_save_path, kinematic_chain, ik_joint, title=caption, fps=20)
            plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20)
            np.save(pjoin(joint_path, "sample%d_repeat%d_len%d.npy"%(k, r, m_length[k])), joint)
            np.save(pjoin(joint_path, "sample%d_repeat%d_len%d_ik.npy"%(k, r, m_length[k])), ik_joint)
    
    print(f"Motion generation completed successfully! BVH files saved in {animation_dir}")
    exit(0)

    