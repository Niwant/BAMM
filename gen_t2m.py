# gen_t2m.py 									
import os
from os.path import join as pjoin
import gc
import copy
import threading

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.cuda.amp import autocast

from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE, LengthEstimator

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils.motion_process import recover_from_ric
from visualization.joints2bvh import Joint2BVHConvertor
from visualization.bvh2joints import BVH2JointConvertor
from utils.paramUtil import t2m_kinematic_chain
from utils.plot_script import plot_3d_motion
from utils.joints_to_features import joints_to_features

import numpy as np

clip_version = 'ViT-B/32'


class BVHGenerator:
    def __init__(self):
        parser = EvalT2MOptions()
        self.opt = parser.parse(is_mock=True, is_eval=True)
        self.opt.res_name = "tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw"
        self.opt.name = "2024-02-14-14-27-29_8_GPT_officialTrans_2iterPrdictEnd"
        self.opt.text_prompt = ""
        self.opt.motion_length = -1
        self.opt.repeat_times = 1
        self.opt.gpu_id = 0
        self.opt.seed = 1
        self.opt.ext = "generation_name_nopredlen"
        self.opt.combine = True

        fixseed(self.opt.seed)
        self.opt.device = torch.device("cpu" if self.opt.gpu_id == -1 else f"cuda:{self.opt.gpu_id}")
        torch.autograd.set_detect_anomaly(True)

        root_dir = pjoin(self.opt.checkpoints_dir, self.opt.dataset_name, self.opt.name)
        result_dir = pjoin('./generation', self.opt.ext)
        self.joints_dir = pjoin(result_dir, 'joints')
        self.animation_dir = pjoin(result_dir, 'animations')
        os.makedirs(self.joints_dir, exist_ok=True)
        os.makedirs(self.animation_dir, exist_ok=True)

        model_opt_path = pjoin(root_dir, 'opt.txt')
        self.model_opt = get_opt(model_opt_path, device=self.opt.device)

        vq_opt_path = pjoin('./log/vq', self.opt.dataset_name, self.model_opt.vq_name, 'opt.txt')
        self.vq_opt = get_opt(vq_opt_path, device=self.opt.device)
        self.vq_model = self.load_vq_model()

        self.model_opt.num_tokens = self.vq_opt.nb_code
        self.model_opt.num_quantizers = self.vq_opt.num_quantizers
        self.model_opt.code_dim = self.vq_opt.code_dim

        res_opt_path = pjoin('checkpoints', self.opt.dataset_name, self.opt.res_name, 'opt.txt')
        self.res_opt = get_opt(res_opt_path, device=self.opt.device)
        self.res_model = self.load_res_model()

        assert self.res_opt.vq_name == self.model_opt.vq_name

        self.t2m_transformer = self.load_trans_model('latest.tar')
        self.length_estimator = self.load_len_estimator()

        self.vq_model.eval().to(self.opt.device)
        self.res_model.eval().to(self.opt.device)
        self.t2m_transformer.eval().to(self.opt.device)
        self.length_estimator.eval().to(self.opt.device)

        self.converter = Joint2BVHConvertor()
        self.kinematic_chain = t2m_kinematic_chain
        self.bvhconverter = BVH2JointConvertor()
        self.mean = np.load(pjoin('checkpoints', self.opt.dataset_name, self.model_opt.vq_name, 'meta', 'mean.npy'))
        self.std = np.load(pjoin('checkpoints', self.opt.dataset_name, self.model_opt.vq_name, 'meta', 'std.npy'))

    def load_vq_model(self):
        model = RVQVAE(self.vq_opt, self.vq_opt.dim_pose, self.vq_opt.nb_code, self.vq_opt.code_dim,
                       self.vq_opt.output_emb_width, self.vq_opt.down_t, self.vq_opt.stride_t,
                       self.vq_opt.width, self.vq_opt.depth, self.vq_opt.dilation_growth_rate,
                       self.vq_opt.vq_act, self.vq_opt.vq_norm)
        ckpt = torch.load(pjoin(self.vq_opt.checkpoints_dir, self.vq_opt.dataset_name, self.vq_opt.name, 'model', 'net_best_fid.tar'), map_location='cpu')
        model.load_state_dict(ckpt.get('vq_model', ckpt['net']))
        print(f"Loaded VQ model: {self.vq_opt.name}")
        return model

    def load_res_model(self):
        self.res_opt.num_quantizers = self.vq_opt.num_quantizers
        self.res_opt.num_tokens = self.vq_opt.nb_code
        model = ResidualTransformer(
            code_dim=self.vq_opt.code_dim,
            cond_mode='text',
            latent_dim=self.res_opt.latent_dim,
            ff_size=self.res_opt.ff_size,
            num_layers=self.res_opt.n_layers,
            cond_drop_prob=self.res_opt.cond_drop_prob,
            num_heads=self.res_opt.n_heads,
            dropout=self.res_opt.dropout,
            clip_dim=512,
            shared_codebook=self.vq_opt.shared_codebook,
            share_weight=self.res_opt.share_weight,
            clip_version=clip_version,
            opt=self.res_opt
        )
        ckpt = torch.load(pjoin(self.res_opt.checkpoints_dir, self.res_opt.dataset_name, self.res_opt.name, 'model', 'net_best_fid.tar'), map_location=self.opt.device)
        model.load_state_dict(ckpt['res_transformer'], strict=False)
        print(f"Loaded Residual Transformer: {self.res_opt.name}")
        return model

    def load_trans_model(self, which_model):
        model = MaskTransformer(self.model_opt.code_dim, 'text', self.model_opt.latent_dim, self.model_opt.ff_size,
                                self.model_opt.n_layers, self.model_opt.n_heads, self.model_opt.dropout, 512,
                                self.model_opt.cond_drop_prob, clip_version, self.model_opt)
        ckpt = torch.load(pjoin(self.model_opt.checkpoints_dir, self.model_opt.dataset_name, self.model_opt.name, 'model', which_model), map_location='cpu')
        state_dict = ckpt.get('t2m_transformer', ckpt.get('trans'))
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded Transformer: {self.model_opt.name} from epoch {ckpt['ep']}")
        return model

    def load_len_estimator(self):
        model = LengthEstimator(512, 50)
        ckpt = torch.load(pjoin('checkpoints', self.opt.dataset_name, 'length_estimator', 'model', 'finest.tar'), map_location=self.opt.device)
        model.load_state_dict(ckpt['estimator'])
        print(f"Loaded Length Estimator from epoch {ckpt['epoch']}")
        return model

    def inv_transform(self, data):
        return data * self.std + self.mean

    def create_bvh_from_in(self, prompt_text):

        torch.cuda.empty_cache()
        gc.collect()

        if torch.cuda.is_available():
            print(f"[GPU] Before Gen | Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB | Cached: {torch.cuda.memory_reserved() / 1e6:.2f} MB")

        local_opt = copy.copy(self.opt)
        local_opt.text_prompt = prompt_text
        prompt_list = prompt_text.split('|')
        print("🔍 PROMPTS RECEIVED:", prompt_list)

        with torch.no_grad():
            with autocast():
                text_embedding = self.t2m_transformer.encode_text(prompt_list)
                print("🔍 TEXT EMBEDDINGS SHAPE:", text_embedding.shape)

                pred_dis = self.length_estimator(text_embedding)
                probs = F.softmax(pred_dis, dim=-1)
                token_lens = Categorical(probs).sample().to(local_opt.device)
                m_length = token_lens * 4

                print("🧪 COMBINE ENABLED:", local_opt.combine)

                mids, pred_len = self.t2m_transformer.generate(
                    prompt_list,
                    token_lens,
                    timesteps=local_opt.time_steps,
                    cond_scale=local_opt.cond_scale,
                    temperature=local_opt.temperature,
                    topk_filter_thres=local_opt.topkr,
                    gsample=local_opt.gumbel_sample,
                    is_predict_len=local_opt.motion_length == -1
                )

                mids = self.res_model.generate(mids, prompt_list, token_lens, temperature=1, cond_scale=5)

                combine = local_opt.combine
                if mids.shape[0] <= 1 and combine:
                    print("Only one motion segment — disabling combine mode.")
                    combine = False

                if len(prompt_list) > 0 and combine:
                    num_transition_token = 2
                    b = mids.shape[0]
                    half_token_length = (pred_len / 2).int()
                    idx_full_len = half_token_length >= 24
                    half_token_length[idx_full_len] -= 1

                    tokens = -1 * torch.ones((b - 1, 50), dtype=torch.long).to(mids.device)
                    transition_train_length = []
                    for i in range(b - 1):
                        i_index_motion = mids[i, :, 0]
                        i1_index_motion = mids[i + 1, :, 0]

                        left_end = half_token_length[i]
                        right_start = left_end + num_transition_token
                        end = right_start + half_token_length[i + 1]

                        tokens[i, :left_end] = i_index_motion[pred_len[i] - left_end: pred_len[i]]
                        tokens[i, left_end:right_start] = self.t2m_transformer.pad_id
                        tokens[i, right_start:end] = i1_index_motion[:half_token_length[i + 1]]
                        transition_train_length.append(end)

                    transition_train_length = torch.tensor(transition_train_length).to(mids.device).long()
                    tokens = tokens.scatter_(-1, transition_train_length[..., None], self.t2m_transformer.end_id).long()
                    inpainting_mask = tokens == self.t2m_transformer.pad_id
                    tokens[tokens == -1] = self.t2m_transformer.pad_id
                    inpaint_index = self.t2m_transformer.edit2(None, tokens)

                    all_tokens = []
                    for i in range(b - 1):
                        all_tokens.append(mids[i, :pred_len[i]])
                        inpaint_all = inpaint_index[i, inpainting_mask[i]].unsqueeze(-1)
                        inpaint_all = torch.nn.functional.pad(inpaint_all, (0, 5), value=-1)
                        all_tokens.append(inpaint_all)
                    all_tokens.append(mids[-1, :pred_len[-1]])
                    print("🧪 COMBINING PROMPTS INTO SINGLE MOTION")

                    mids = torch.cat(all_tokens).unsqueeze(0)

                pred_motions = self.vq_model.forward_decoder(mids).detach().cpu().numpy()
                data = self.inv_transform(pred_motions)
                print("🔍 NUM DATA SAMPLES RETURNED:", len(data))
                print("🔍 FINAL PROMPTS:", prompt_list)

        def export_bvh(k, joint_data):
            length = m_length[k]
            animation_path = pjoin(self.animation_dir, str(k))
            joint_path = pjoin(self.joints_dir, str(k))
            os.makedirs(animation_path, exist_ok=True)
            os.makedirs(joint_path, exist_ok=True)
            joint_data = joint_data if combine else joint_data[:length]
            joint = recover_from_ric(torch.from_numpy(joint_data).float().pin_memory().to(self.opt.device), 22).cpu().numpy()
            bvh_path = pjoin(animation_path, f"sample{k}_repeat0_len{length}_ik.bvh")
            self.converter.convert(joint, filename=bvh_path, iterations=50)
            bvh_path = pjoin(animation_path, f"sample{k}_repeat0_len{length}.bvh")
            self.converter.convert(joint, filename=bvh_path, iterations=50, foot_ik=False)

        threads = [threading.Thread(target=export_bvh, args=(k, joint_data)) for k, joint_data in enumerate(data)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if torch.cuda.is_available():
            print(f"[GPU] After Gen | Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB | Cached: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
        torch.cuda.empty_cache()
        gc.collect()

        return True


    def add_motion_inbetween(self , prompt_text , insert_time=2):
            torch.cuda.empty_cache()
            gc.collect()
            combine = self.opt.combine
            if torch.cuda.is_available():
                print(f"[GPU] Before Gen | Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB | Cached: {torch.cuda.memory_reserved() / 1e6:.2f} MB")

            local_opt = copy.copy(self.opt)
            local_opt.text_prompt = prompt_text
            # prompt_list = prompt_text.split('|')
            # print("🔍 PROMPTS RECEIVED:", prompt_list)
            with torch.no_grad():
                with autocast():
                    print("🔍 Generating motion inbetween for prompt:", local_opt.text_prompt)
                    joints = self.bvhconverter.convert('../web/mesh/public/sample0_repeat0_len196.bvh')
                    print("step 1: done")
                    ric_features = joints_to_features(joints)
                    print("step 2: done")
                    mean = np.load(pjoin('checkpoints', self.opt.dataset_name, self.model_opt.vq_name, 'meta', 'mean.npy'))
                    std = np.load(pjoin('checkpoints', self.opt.dataset_name, self.model_opt.vq_name, 'meta', 'std.npy'))
                    normed_joints = (ric_features - mean) / std
                    ric_tensor = torch.from_numpy(normed_joints).float().unsqueeze(0).to(self.opt.device) # (1, seq_len, 263)
                    quantized, tokens = self.vq_model.encode(ric_tensor)
                    print("step 4: done")
                    # Make these params later 
                    FPS = 20
                    t_sec=insert_time # addtion at this second
                    downsample_factor=5
                    frame_index = int(t_sec * FPS)
                    insert_frame = int(frame_index / downsample_factor)

                    prompt_list = ["the person runs in forward direction",]
                    est_length = True
                    length_list = []

                    if est_length:
                        print("Since no motion length are specified, we will use estimated motion lengthes!!")
                        text_embedding = self.t2m_transformer.encode_text(prompt_list)
                        pred_dis = self.length_estimator(text_embedding)
                        probs = F.softmax(pred_dis, dim=-1)  # (b, ntoken)
                        token_lens = Categorical(probs).sample()  # (b, seqlen)
                        # lengths = torch.multinomial()
                    else:
                        token_lens = torch.LongTensor(length_list) // 4
                        token_lens = token_lens.to(self.opt.device).long()

                    m_length = token_lens * 4
                    captions = prompt_list
                    mids, pred_len = self.t2m_transformer.generate(captions, token_lens,
                                    timesteps=self.opt.time_steps,
                                    cond_scale=self.opt.cond_scale,
                                    temperature=self.opt.temperature,
                                    topk_filter_thres=self.opt.topkr,
                                    gsample=self.opt.gumbel_sample,
                                    is_predict_len=self.opt.motion_length==-1
                                    )
                    token_lens = pred_len
                    m_length = token_lens*4
                    # print(mids)
                    # print(mids.shape)
                    mids_1 = self.res_model.generate(mids, captions, token_lens, temperature=1, cond_scale=5)
                    pred_motions = self.vq_model.forward_decoder(mids_1)
                   
                    # pred_motions_1 = pred_motions.detach().numpy()
    
                    print("step 5: done")
                    # Extract parts from quantized
                    quantized_base = quantized[0, :]  # Shape: [L, 6]
                    part1 = quantized_base[:insert_frame]        # Quantized before 10s
                    part2 = quantized_base[insert_frame:]        # Quantized after 10s


                    motion_A = part1[:, 0]        # Last tokens before 10s
                    motion_B = mids_1[0, : , 0]     # New motion to insert
                    motion_C = part2[:, 0]        # Tokens after 10s
                    # Calculate actual lengths
                    len_A = len(motion_A)
                    len_B = len(motion_B)
                    len_C = len(motion_C)
                    print(len_A, len_B, len_C)

                    # Adjust half_token_length based on minimum available length
                    # half_token_length = min(24, len_A, len_B, len_C)
                    # num_transition_token = 2
                    # Prepare transition tokens
                    tokens = -1 * torch.ones((2, 50), dtype=torch.long, device=self.opt.device)
                    # tokens = -1 * torch.ones((2, half_token_length*2 + num_transition_token), dtype=torch.long, device=self.opt.device)

                    half_token_length = 24
                    num_transition_token = 2
                    motion_C = torch.nn.functional.pad(motion_C, (0, half_token_length - len(motion_C)), value=self.t2m_transformer.pad_id)

                    # === Transition 1: part1 (A) to mids (B) ===
                    tokens[0, :half_token_length] = motion_A[-half_token_length:]
                    tokens[0, half_token_length:half_token_length + num_transition_token] = self.t2m_transformer.pad_id
                    tokens[0, half_token_length + num_transition_token:] = motion_B[:half_token_length]
                    print("step 6: done")
                    # === Transition 2: mids (B) to part2 (C) ===
                    tokens[1, :half_token_length] = motion_B[-half_token_length:]
                    print("step 7: done")
                    tokens[1, half_token_length:half_token_length + num_transition_token] = self.t2m_transformer.pad_id
                    print("step 8: done")
                    tokens[1, half_token_length + num_transition_token:] = motion_C[:half_token_length]

                    # Predict transitions
                    tokens[tokens == -1] = self.t2m_transformer.pad_id
                    inpainting_mask = tokens == self.t2m_transformer.pad_id
                    inpaint_index = self.t2m_transformer.edit2(None, tokens)

                    # Extract transitions
                    transition1 = inpaint_index[0, inpainting_mask[0]].unsqueeze(-1)
                    transition1 = torch.nn.functional.pad(transition1, (0, 5), value=-1)

                    transition2 = inpaint_index[1, inpainting_mask[1]].unsqueeze(-1)
                    transition2 = torch.nn.functional.pad(transition2, (0, 5), value=-1)

                    # Construct final motion
                    final_motion = torch.cat([
                        part1,         # original motion up to 10s
                        transition1,   # transition: original → mids
                        mids_1[0][:pred_len[0]],       # inserted new motion
                        transition2,   # transition: mids → original continuation
                        part2          # rest of original motion
                    ], dim=0).unsqueeze(0)  # Add batch dim

                    # Final result
                    quantized = final_motion
                    pred_motions = self.vq_model.forward_decoder(quantized)
                    data = self.inv_transform(pred_motions.detach().cpu().numpy())
                    print("🔍 NUM DATA SAMPLES RETURNED:", len(data))
            def export_bvh(k, joint_data):
                length = m_length[k]
                animation_path = pjoin(self.animation_dir, str(k))
                joint_path = pjoin(self.joints_dir, str(k))
                os.makedirs(animation_path, exist_ok=True)
                os.makedirs(joint_path, exist_ok=True)
                joint_data = joint_data if combine else joint_data[:length]
                joint = recover_from_ric(torch.from_numpy(joint_data).float().pin_memory().to(self.opt.device), 22).cpu().numpy()
                bvh_path = pjoin(animation_path, f"sample_between_ik.bvh")
                self.converter.convert(joint, filename=bvh_path, iterations=50)
                bvh_path = pjoin(animation_path, f"sample_between.bvh")
                self.converter.convert(joint, filename=bvh_path, iterations=50, foot_ik=False)
            
            threads = [threading.Thread(target=export_bvh, args=(k, joint_data)) for k, joint_data in enumerate(data)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            if torch.cuda.is_available():
                print(f"[GPU] After Gen | Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB | Cached: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
                torch.cuda.empty_cache()
                gc.collect()

            return True
    
