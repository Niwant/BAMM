import torch
import torch.nn as nn
import numpy as np
# from networks.layers import *
import torch.nn.functional as F
import clip
from einops import rearrange, repeat
import math
from random import random
from tqdm.auto import tqdm
from typing import Callable, Optional, List, Dict
from copy import deepcopy
from functools import partial
from models.mask_transformer.tools import *
from torch.distributions.categorical import Categorical
from models.mask_transformer.transformer_block import TransformerEncoderLayer, TransformerDecoderLayer, TransformerEncoder

class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # [bs, ntokens, input_feats]
        x = x.permute((1, 0, 2)) # [seqen, bs, input_feats]
        # print(x.shape)
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x

class PositionalEncoding(nn.Module):
    #Borrow from MDM, the same as above, but add dropout, exponential may improve precision
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) #[max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class OutputProcess_Bert(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!
        self.out_feats = out_feats

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, c, seqlen]
        return output

class OutputProcess(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, e, seqlen]
        return output


class MaskTransformer(nn.Module):
    def __init__(self, code_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.1, clip_dim=512, cond_drop_prob=0.1,
                 clip_version=None, opt=None, **kargs):
        super().__init__()
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')

        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt

        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob

        if self.cond_mode == 'action':
            assert 'num_actions' in kargs
        self.num_actions = kargs.get('num_actions', 1)

        '''
        Preparing Networks
        '''
        self.input_process = InputProcess(self.code_dim, self.latent_dim)
        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)

        if self.opt.trans == 'official':
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=num_heads,
                                                            dim_feedforward=ff_size,
                                                            dropout=dropout,
                                                            activation='gelu')

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=num_layers)
        elif self.opt.trans == 't2mgpt':
            self.seqTransEncoder = TransformerEncoder(nn.Sequential(*[TransformerEncoderLayer(embed_dim=self.latent_dim, 
                                            n_head=num_heads, 
                                            drop_out_rate=dropout, 
                                            dim_feedforward=ff_size) for _ in range(num_layers)]))
        else:
            raise Exception('Type of opt.trans '+self.opt.trans+' is not supported')
        
        self.encode_action = partial(F.one_hot, num_classes=self.num_actions)

        # if self.cond_mode != 'no_cond':
        if self.cond_mode == 'text':
            self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
        elif self.cond_mode == 'action':
            self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
        elif self.cond_mode == 'uncond':
            self.cond_emb = nn.Identity()
        else:
            raise KeyError("Unsupported condition mode!!!")


        _num_tokens = opt.num_tokens + 3  # two dummy tokens, one for masking, one for padding
        self.end_id = opt.num_tokens
        self.pad_id = opt.num_tokens + 1
        self.mask_id = opt.num_tokens + 2
        self.num_heads = num_heads

        self.output_process = OutputProcess_Bert(out_feats=opt.num_tokens + 1, latent_dim=latent_dim)

        self.token_emb = nn.Embedding(_num_tokens, self.code_dim)

        self.apply(self.__init_weights)

        '''
        Preparing frozen weights
        '''

        if self.cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)

        self.noise_schedule = cosine_schedule

    def load_and_freeze_token_emb(self, codebook):
        '''
        :param codebook: (c, d)
        :return:
        '''
        assert self.training, 'Only necessary in training mode'
        c, d = codebook.shape
        self.token_emb.weight = nn.Parameter(torch.cat([codebook, torch.zeros(size=(2, d), device=codebook.device)], dim=0)) #add two dummy tokens, 0 vectors
        self.token_emb.requires_grad_(False)
        # self.token_emb.weight.requires_grad = False
        # self.token_emb_ready = True
        print("Token embedding initialized!")

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        # Cannot run on cpu
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16
        # Date 0707: It's necessary, only unecessary when load directly to gpu. Disable if need to run on cpu

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).to(torch.float32)  # (b, clip_dim)
        return feat_clip_text

    def mask_cond(self, cond, force_mask=False):
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def trans_forward(self, motion_ids, cond, padding_mask, force_mask=None, cond_idx=None):
        '''
        :param motion_ids: (b, seqlen)
        :padding_mask: (b, seqlen), all pad positions are TRUE else FALSE
        :param cond: (b, embed_dim) for text, (b, num_actions) for action
        :param force_mask: boolean
        :return:
            -logits: (b, num_token, seqlen)
        '''
        if self.training or force_mask==False:
            cond = self.mask_cond(cond, force_mask=False)
        elif force_mask:
            cond = self.mask_cond(cond, force_mask=True)
        else:
            cond1 = self.mask_cond(cond, force_mask=False)
            cond2 = self.mask_cond(cond, force_mask=True)
            cond = torch.cat([cond1, cond2], dim=0)
            motion_ids = repeat(motion_ids, 'b t -> (repeat b) t', repeat=2)
            if cond_idx is not None:
                cond_idx = repeat(cond_idx, 'b t -> (repeat b) t', repeat=2)
        cond = self.cond_emb(cond).unsqueeze(0) #(1, b, latent_dim)
        if len(motion_ids) > 0:
            # print(motion_ids.shape)
            x = self.token_emb(motion_ids)
            # print(x.shape)
            # (b, seqlen, d) -> (seqlen, b, latent_dim)
            x = self.input_process(x)
            xseq = torch.cat([cond, x], dim=0) #(seqlen+1, b, latent_dim)
        else:
            xseq = cond
        xseq = self.position_enc(xseq) # Diff from MoMask, we add position emb to cond


        # padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1) #(b, seqlen+1)
        # print(xseq.shape, padding_mask.shape)

        # print(padding_mask.shape, xseq.shape)

        ################################################################
        block_size, b = xseq.shape[:2]
        attn = torch.tril(torch.ones(block_size, block_size)).to(xseq.device).bool()
        attn = repeat(attn, 'T1 T2 -> b T1 T2', b=b)

        if cond_idx is not None:
            cond_pos = cond_idx != self.pad_id
            cond_pos = torch.cat([torch.ones_like(cond_pos[:, 0:1]), cond_pos], dim=1) #(b, seqlen+1)
            
            cond_pos_horizon = repeat(cond_pos, 'b T1 -> b T1 T2',  T2=block_size)
            attn = torch.logical_and(attn, ~cond_pos_horizon)
            
            cond_pos = repeat(cond_pos, 'b T2 -> b T1 T2',  T1=block_size)
            attn = torch.logical_or(attn, cond_pos)
        
        if self.opt.trans == 'official':
            attn = torch.repeat_interleave(attn, self.num_heads, dim=0)
        elif self.opt.trans == 't2mgpt':
            attn = repeat(attn, 'b T1 T2 -> b h T1 T2',  h=self.num_heads)
        self.attn = attn

        output = self.seqTransEncoder(xseq, mask=~attn) #(seqlen, b, e)
        ################################################################
        
        
        logits = self.output_process(output) #(seqlen, b, e) -> (b, ntoken, seqlen)
        return logits

    def forward(self, ids, y, m_lens):
        '''
        :param ids: (b, n)
        :param y: raw text for cond_mode=text, (b, ) for cond_mode=action
        :m_lens: (b,)
        :return:
        '''

        bs, ntokens = ids.shape
        device = ids.device

        # Positions that are PADDED are ALL FALSE
        non_pad_mask = lengths_to_mask(m_lens, ntokens + 1) #(b, n)
        ids = torch.cat([ids, 
                         torch.ones((ids.shape[0], 1), device=ids.device)*self.end_id],
                         dim=-1).long()
        ids = torch.where(non_pad_mask, ids, self.pad_id)
        ids.scatter_(-1, m_lens[..., None], self.end_id)

        # 1. rand con
        import random
        rate_cond = .5 # probability of how often cond occur
        max_cond = .5 # proportion of number of max condition occur
        occure_r = random.random()
        cond_r = random.random() * max_cond
        r = cond_r if occure_r<rate_cond else 0
        cond_pos = torch.empty_like(ids, dtype=torch.float, device=ids.device).uniform_(0, 1) < r
        # print(cond_pos[0].sum()/cond_pos[0].shape[0])
        cond_idx = ids.clone()
        with_end_mask = lengths_to_mask(m_lens+1, ntokens + 1) #(b, n)
        cond_idx[torch.logical_or(~cond_pos, ~with_end_mask)] = self.pad_id
        
        # 2. end con
        # cond_idx = torch.where(ids==self.end_id, self.end_id, self.pad_id)
        
        # 3. GPT
        # cond_idx = None


        force_mask = False
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(y)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(y).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(bs, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")


        '''
        Prepare mask
        '''
        # rand_time = uniform((bs,), device=device)
        # rand_mask_probs = self.noise_schedule(rand_time)
        # num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)

        # batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1)
        # # Positions to be MASKED are ALL TRUE
        # mask = batch_randperm < num_token_masked.unsqueeze(-1)

        # # Positions to be MASKED must also be NON-PADDED
        # mask &= non_pad_mask

        # Note this is our training target, not input
        labels = ids #torch.where(mask, ids, self.mask_id)
        x_ids = ids.clone()

        mask_rid = torch.bernoulli(np.random.rand(1)[0]*.5 * torch.ones(x_ids.shape, device=x_ids.device)) > 0
        rand_id = torch.randint_like(x_ids, high=self.opt.num_tokens)
        before_end_mask = lengths_to_mask(m_lens, ntokens+1)
        x_ids = torch.where(mask_rid*before_end_mask, rand_id, x_ids)
        
        logits = self.trans_forward(x_ids, cond_vector, None, force_mask, cond_idx=cond_idx)[..., :-1]
        non_pad_mask_with_end = lengths_to_mask(m_lens+1, ntokens + 1)
        weigths = non_pad_mask_with_end / (non_pad_mask_with_end.sum(-1).unsqueeze(-1) * non_pad_mask_with_end.shape[0])
        logits_masked = logits.permute(0,2,1)[non_pad_mask_with_end]
        labels_masked = labels[non_pad_mask_with_end]
        weigths_masked = weigths[non_pad_mask_with_end]
        
        ce_loss = F.cross_entropy(logits_masked, labels_masked, reduction = 'none')
        ce_loss = (ce_loss * weigths_masked).sum()
        _, pred_id, acc = cal_performance(logits, labels, ignore_index=self.pad_id)

        return ce_loss, pred_id, acc

    def forward_with_cond_scale(self,
                                motion_ids,
                                cond_vector,
                                padding_mask,
                                cond_scale=3,
                                force_mask = None,
                                cond_idx=None):

        # aux_logits = self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True, cond_idx=cond_idx)
        if force_mask:
            logits = self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True, cond_idx=cond_idx)
            return logits
        logits = self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=force_mask, cond_idx=cond_idx)
        logits, aux_logits = logits[:int(logits.shape[0]/2)], logits[int(logits.shape[0]/2):]
        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return scaled_logits

    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 conds,
                 m_lens,
                 timesteps: int,
                 cond_scale: int,
                 temperature=1,
                 topk_filter_thres=0.9,
                 gsample=False,
                 force_mask=False,
                 is_predict_len=True
                 ):
        with torch.no_grad():
            cond_vector = self.encode_text(conds).to(torch.float32)
        is_softmax = True
        seq_len = 49 #max(m_lens)
        

        ids = torch.ones((len(conds), seq_len+1), dtype=torch.long, device=m_lens.device) * self.pad_id
        ids, scores = self.gen_one(ids, cond_vector, seq_len, cond_idx=None, cond_scale=cond_scale, pred_len=is_predict_len)
        if is_predict_len:
            ids, pred_len = self.pad_after_end(ids)
        else:
            pred_len = m_lens
        padding_mask = ~lengths_to_mask(pred_len, seq_len+1)
        ids = ids.scatter(-1, pred_len[..., None], self.end_id)

        MASK_STATIC = False
        if MASK_STATIC:
            is_mask = torch.ones_like(ids, dtype=torch.bool)
            is_mask[:, 1::2] = False
        else:
            scores = torch.where(padding_mask, 1e5, 0.)
            
            num_token_masked = torch.round(.5 * pred_len).clamp(min=1)
            sorted_indices = scores.argsort( dim=1)
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))

        cons_pos = ((~is_mask * ~padding_mask) + (ids==self.end_id)) > 0
        cond_idx = torch.where(cons_pos, ids, self.pad_id).clone()

        ids, scores = self.gen_one(ids, cond_vector, seq_len, cond_idx, cond_scale=3, pred_len=False)
        ids[cons_pos] = cond_idx[cons_pos]

        # scores = torch.cat([scores, torch.ones(scores.shape[0], 1, device=ids.device)*1e5], dim=1)
        # scores = scores.masked_fill(~is_mask, 1e5)
        return ids, pred_len
    
    def gen_one(self, idx, cond_vector, seq_len, cond_idx, cond_scale, pred_len=True):
        probs_all = []
        idx = idx.clone()
        for k in range(seq_len):
            logits = self.forward_with_cond_scale(idx, cond_vector=cond_vector,
                                                    padding_mask=None,
                                                    cond_scale=cond_scale,
                                                    cond_idx=cond_idx)
            # logits = top_k(logits[..., -1], topk_filter_thres, dim=-1)
            logits = logits[..., k]
            if not pred_len:
                logits = logits[..., :-1]
            probs = F.softmax(logits, dim=1)
            dist = Categorical(probs)
            current_idx = dist.sample()
            current_idx = current_idx.unsqueeze(-1)
            current_probs = torch.gather(probs, 1, current_idx)
            probs_all.append(current_probs)
            idx[:, k:k+1] = current_idx
            # if cond_idx is not None:
            #     cons_pos = cond_idx != self.pad_id
            #     idx[cons_pos] = cond_idx[cons_pos]
        return idx, torch.cat(probs_all, dim=1)

    def pad_after_end(self, xs):
        # prevent case that length is 0 by move end_id to the second index and fill up "0" in the first index
        pred_end_at_first = xs[:, 0] >= self.end_id
        xs[:, 0][pred_end_at_first] = 0
        xs[:, 1][pred_end_at_first] = self.end_id
        
        pred_len = (torch.ones(xs.shape[0], device=xs.device) * (xs.shape[1] + 1)).long()
        # From https://discuss.pytorch.org/t/first-nonzero-index/24769/3
        mask_max_values, max_indices = torch.max(xs >= self.end_id, dim=1)
        max_indices[~mask_max_values] = -1
        pred_len[max_indices>=0] = max_indices[max_indices>=0]
        motion_mask = lengths_to_mask(pred_len+1, xs.shape[1])
        xs = xs * motion_mask + self.pad_id * ~motion_mask
        return xs, pred_len

    def pad_when_end(self, xs):
        # prevent case that length is 0 by move end_id to the second index and fill up "0" in the first index
        pred_end_at_first = xs[:, 0] >= self.end_id
        xs[:, 0][pred_end_at_first] = 0
        xs[:, 1][pred_end_at_first] = self.end_id
        
        pred_len = (torch.ones(xs.shape[0], device=xs.device) * (xs.shape[1] + 1)).long()
        # From https://discuss.pytorch.org/t/first-nonzero-index/24769/3
        mask_max_values, max_indices = torch.max(xs >= self.end_id, dim=1)
        max_indices[~mask_max_values] = -1
        pred_len[max_indices>=0] = max_indices[max_indices>=0]
        motion_mask = lengths_to_mask(pred_len, xs.shape[1])
        xs = xs * motion_mask + -1 * ~motion_mask
        return xs, pred_len
    
    # problem with multiple end_id in the same sample, Torch use first and last inconsistently
    # def pad_when_end(self, xs):
    #     gen_len = torch.ones(xs.shape[0], device=xs.device) * (xs.shape[1] + 1)
    #     b_ids, end_seqs = ((xs == self.end_id).nonzero(as_tuple=True))
    #     # gen_len[b_ids.flip(dims=(0,))] = end_seqs.flip(dims=(0,)).float()
    #     gen_len[b_ids] = end_seqs.float()
    #     motion_mask = lengths_to_mask(gen_len, xs.shape[1])
    #     xs = xs * motion_mask + -1 * ~motion_mask
    #     return xs

    @torch.no_grad()
    @eval_decorator
    def edit2(self,
             texts,
             cond_tokens,
             ):
        device = next(self.parameters()).device
        seq_len = cond_tokens.shape[1]

            
        if texts is None:
            cond_vector = torch.zeros((cond_tokens.shape[0], 512)).to(device)
            force_mask = True
        else:
            with torch.no_grad():
                cond_vector = self.encode_text(texts)
            force_mask=None,

        # seq_len = 49 #max(m_lens)
        GT_LEN = False

        idx = cond_tokens.clone()
        cond_idx = cond_tokens.clone()
        cond_pos = cond_idx!=self.pad_id
        for k in range(seq_len):
            logits = self.forward_with_cond_scale(idx, cond_vector=cond_vector,
                                                  padding_mask=None,
                                                  cond_scale=3,
                                                  force_mask=force_mask,
                                                  cond_idx=cond_idx)
            # logits = top_k(logits[..., -1], topk_filter_thres, dim=-1)
            logits = logits[..., k]
            logits = logits[..., :-1] # don't predict end token
            if True:
                probs = F.softmax(logits, dim=1)
                dist = Categorical(probs)
                current_idx = dist.sample()
                current_idx = current_idx.unsqueeze(-1)
                idx[:, k:k+1] = current_idx
                idx[cond_pos] = cond_idx[cond_pos]
            else:
                idx = gumbel_sample(logits, temperature=temperature, dim=-1)
                idx.unsqueeze_(-1)
        return idx


    @torch.no_grad()
    @eval_decorator
    def edit(self,
             conds,
             tokens,
             m_lens,
             timesteps: int,
             cond_scale: int,
             temperature=1,
             topk_filter_thres=0.9,
             gsample=False,
             force_mask=False,
             edit_mask=None,
             padding_mask=None,
             ):

        assert edit_mask.shape == tokens.shape if edit_mask is not None else True
        device = next(self.parameters()).device
        seq_len = tokens.shape[1]

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(1, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        if padding_mask == None:
            padding_mask = ~lengths_to_mask(m_lens, seq_len)

        # Start from all tokens being masked
        if edit_mask == None:
            mask_free = True
            ids = torch.where(padding_mask, self.pad_id, tokens)
            edit_mask = torch.ones_like(padding_mask)
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            scores = torch.where(edit_mask, 0., 1e5)
        else:
            mask_free = False
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            ids = torch.where(edit_mask, self.mask_id, tokens)
            scores = torch.where(edit_mask, 0., 1e5)
        starting_temperature = temperature

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            # 0 < timestep < 1
            rand_mask_prob = 0.16 if mask_free else self.noise_schedule(timestep)  # Tensor

            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = torch.round(rand_mask_prob * edit_len).clamp(min=1)  # (b, )

            # select num_token_masked tokens with lowest scores to be masked
            sorted_indices = scores.argsort(
                dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            # is_mask = (torch.rand_like(scores) < 0.8) * ~padding_mask if mask_free else is_mask
            ids = torch.where(is_mask, self.mask_id, ids)

            '''
            Preparing input
            '''
            # (b, num_token, seqlen)
            logits = self.forward_with_cond_scale(ids, cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  cond_scale=cond_scale,
                                                  force_mask=force_mask)

            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            # print(logits.shape, self.opt.num_tokens)
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

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
                probs = F.softmax(filtered_logits, dim=-1)  # (b, seqlen, ntoken)
                # print(temperature, starting_temperature, steps_until_x0, timesteps)
                # print(probs / temperature)
                pred_ids = Categorical(probs / temperature).sample()  # (b, seqlen)

            # print(pred_ids.max(), pred_ids.min())
            # if pred_ids.
            ids = torch.where(is_mask, pred_ids, ids)

            '''
            Updating scores
            '''
            probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
            scores = scores.squeeze(-1)  # (b, seqlen)

            # We do not want to re-mask the previously kept tokens, or pad tokens
            scores = scores.masked_fill(~edit_mask, 1e5) if mask_free else scores.masked_fill(~is_mask, 1e5)

        ids = torch.where(padding_mask, -1, ids)
        # print("Final", ids.max(), ids.min())
        return ids

    @torch.no_grad()
    @eval_decorator
    def edit_beta(self,
                  conds,
                  conds_og,
                  tokens,
                  m_lens,
                  cond_scale: int,
                  force_mask=False,
                  ):

        device = next(self.parameters()).device
        seq_len = tokens.shape[1]

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
                if conds_og is not None:
                    cond_vector_og = self.encode_text(conds_og)
                else:
                    cond_vector_og = None
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
            if conds_og is not None:
                cond_vector_og = self.enc_action(conds_og).to(device)
            else:
                cond_vector_og = None
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~lengths_to_mask(m_lens, seq_len)

        # Start from all tokens being masked
        ids = torch.where(padding_mask, self.pad_id, tokens)  # Do not mask anything

        '''
        Preparing input
        '''
        # (b, num_token, seqlen)
        logits = self.forward_with_cond_scale(ids,
                                              cond_vector=cond_vector,
                                              cond_vector_neg=cond_vector_og,
                                              padding_mask=padding_mask,
                                              cond_scale=cond_scale,
                                              force_mask=force_mask)

        logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)

        '''
        Updating scores
        '''
        probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
        tokens[tokens == -1] = 0  # just to get through an error when index = -1 using gather
        og_tokens_scores = probs_without_temperature.gather(2, tokens.unsqueeze(dim=-1))  # (b, seqlen, 1)
        og_tokens_scores = og_tokens_scores.squeeze(-1)  # (b, seqlen)

        return og_tokens_scores


class ResidualTransformer(nn.Module):
    def __init__(self, code_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8, cond_drop_prob=0.1,
                 num_heads=4, dropout=0.1, clip_dim=512, shared_codebook=False, share_weight=False,
                 clip_version=None, opt=None, **kargs):
        super(ResidualTransformer, self).__init__()
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')

        # assert shared_codebook == True, "Only support shared codebook right now!"

        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt

        self.cond_mode = cond_mode
        # self.cond_drop_prob = cond_drop_prob

        if self.cond_mode == 'action':
            assert 'num_actions' in kargs
        self.num_actions = kargs.get('num_actions', 1)
        self.cond_drop_prob = cond_drop_prob

        '''
        Preparing Networks
        '''
        self.input_process = InputProcess(self.code_dim, self.latent_dim)
        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=ff_size,
                                                          dropout=dropout,
                                                          activation='gelu')

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=num_layers)

        self.encode_quant = partial(F.one_hot, num_classes=self.opt.num_quantizers)
        self.encode_action = partial(F.one_hot, num_classes=self.num_actions)

        self.quant_emb = nn.Linear(self.opt.num_quantizers, self.latent_dim)
        # if self.cond_mode != 'no_cond':
        if self.cond_mode == 'text':
            self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
        elif self.cond_mode == 'action':
            self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
        else:
            raise KeyError("Unsupported condition mode!!!")


        _num_tokens = opt.num_tokens + 1  # one dummy tokens for padding
        self.pad_id = opt.num_tokens

        # self.output_process = OutputProcess_Bert(out_feats=opt.num_tokens, latent_dim=latent_dim)
        self.output_process = OutputProcess(out_feats=code_dim, latent_dim=latent_dim)

        if shared_codebook:
            token_embed = nn.Parameter(torch.normal(mean=0, std=0.02, size=(_num_tokens, code_dim)))
            self.token_embed_weight = token_embed.expand(opt.num_quantizers-1, _num_tokens, code_dim)
            if share_weight:
                self.output_proj_weight = self.token_embed_weight
                self.output_proj_bias = None
            else:
                output_proj = nn.Parameter(torch.normal(mean=0, std=0.02, size=(_num_tokens, code_dim)))
                output_bias = nn.Parameter(torch.zeros(size=(_num_tokens,)))
                # self.output_proj_bias = 0
                self.output_proj_weight = output_proj.expand(opt.num_quantizers-1, _num_tokens, code_dim)
                self.output_proj_bias = output_bias.expand(opt.num_quantizers-1, _num_tokens)

        else:
            if share_weight:
                self.embed_proj_shared_weight = nn.Parameter(torch.normal(mean=0, std=0.02, size=(opt.num_quantizers - 2, _num_tokens, code_dim)))
                self.token_embed_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, _num_tokens, code_dim)))
                self.output_proj_weight_ = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, _num_tokens, code_dim)))
                self.output_proj_bias = None
                self.registered = False
            else:
                output_proj_weight = torch.normal(mean=0, std=0.02,
                                                  size=(opt.num_quantizers - 1, _num_tokens, code_dim))

                self.output_proj_weight = nn.Parameter(output_proj_weight)
                self.output_proj_bias = nn.Parameter(torch.zeros(size=(opt.num_quantizers, _num_tokens)))
                token_embed_weight = torch.normal(mean=0, std=0.02,
                                                  size=(opt.num_quantizers - 1, _num_tokens, code_dim))
                self.token_embed_weight = nn.Parameter(token_embed_weight)

        self.apply(self.__init_weights)
        self.shared_codebook = shared_codebook
        self.share_weight = share_weight

        if self.cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)

    # def

    def mask_cond(self, cond, force_mask=False):
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        # Cannot run on cpu
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16
        # Date 0707: It's necessary, only unecessary when load directly to gpu. Disable if need to run on cpu

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text


    def q_schedule(self, bs, low, high):
        noise = uniform((bs,), device=self.opt.device)
        schedule = 1 - cosine_schedule(noise)
        return torch.round(schedule * (high - low)) + low

    def process_embed_proj_weight(self):
        if self.share_weight and (not self.shared_codebook):
            # if not self.registered:
            self.output_proj_weight = torch.cat([self.embed_proj_shared_weight, self.output_proj_weight_], dim=0)
            self.token_embed_weight = torch.cat([self.token_embed_weight_, self.embed_proj_shared_weight], dim=0)
                # self.registered = True

    def output_project(self, logits, qids):
        '''
        :logits: (bs, code_dim, seqlen)
        :qids: (bs)

        :return:
            -logits (bs, ntoken, seqlen)
        '''
        # (num_qlayers-1, num_token, code_dim) -> (bs, ntoken, code_dim)
        output_proj_weight = self.output_proj_weight[qids]
        # (num_qlayers, ntoken) -> (bs, ntoken)
        output_proj_bias = None if self.output_proj_bias is None else self.output_proj_bias[qids]

        output = torch.einsum('bnc, bcs->bns', output_proj_weight, logits)
        if output_proj_bias is not None:
            output += output + output_proj_bias.unsqueeze(-1)
        return output



    def trans_forward(self, motion_codes, qids, cond, padding_mask, force_mask=False):
        '''
        :param motion_codes: (b, seqlen, d)
        :padding_mask: (b, seqlen), all pad positions are TRUE else FALSE
        :param qids: (b), quantizer layer ids
        :param cond: (b, embed_dim) for text, (b, num_actions) for action
        :return:
            -logits: (b, num_token, seqlen)
        '''
        cond = self.mask_cond(cond, force_mask=force_mask)

        # (b, seqlen, d) -> (seqlen, b, latent_dim)
        x = self.input_process(motion_codes)

        # (b, num_quantizer)
        q_onehot = self.encode_quant(qids).float().to(x.device)

        q_emb = self.quant_emb(q_onehot).unsqueeze(0)  # (1, b, latent_dim)
        cond = self.cond_emb(cond).unsqueeze(0)  # (1, b, latent_dim)

        x = self.position_enc(x)
        xseq = torch.cat([cond, q_emb, x], dim=0)  # (seqlen+2, b, latent_dim)

        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:2]), padding_mask], dim=1)  # (b, seqlen+2)
        output = self.seqTransEncoder(xseq, src_key_padding_mask=padding_mask)[2:]  # (seqlen, b, e)
        logits = self.output_process(output)
        return logits

    def forward_with_cond_scale(self,
                                motion_codes,
                                q_id,
                                cond_vector,
                                padding_mask,
                                cond_scale=3,
                                force_mask=False):
        bs = motion_codes.shape[0]
        # if cond_scale == 1:
        qids = torch.full((bs,), q_id, dtype=torch.long, device=motion_codes.device)
        if force_mask:
            logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask, force_mask=True)
            logits = self.output_project(logits, qids-1)
            return logits

        logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask)
        logits = self.output_project(logits, qids-1)
        if cond_scale == 1:
            return logits

        aux_logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask, force_mask=True)
        aux_logits = self.output_project(aux_logits, qids-1)

        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return scaled_logits

    def forward(self, all_indices, y, m_lens):
        '''
        :param all_indices: (b, n, q)
        :param y: raw text for cond_mode=text, (b, ) for cond_mode=action
        :m_lens: (b,)
        :return:
        '''

        self.process_embed_proj_weight()

        bs, ntokens, num_quant_layers = all_indices.shape
        device = all_indices.device

        # Positions that are PADDED are ALL FALSE
        non_pad_mask = lengths_to_mask(m_lens, ntokens)  # (b, n)

        q_non_pad_mask = repeat(non_pad_mask, 'b n -> b n q', q=num_quant_layers)
        all_indices = torch.where(q_non_pad_mask, all_indices, self.pad_id) #(b, n, q)

        # randomly sample quantization layers to work on, [1, num_q)
        active_q_layers = q_schedule(bs, low=1, high=num_quant_layers, device=device)

        # print(self.token_embed_weight.shape, all_indices.shape)
        token_embed = repeat(self.token_embed_weight, 'q c d-> b c d q', b=bs)
        gather_indices = repeat(all_indices[..., :-1], 'b n q -> b n d q', d=token_embed.shape[2])
        # print(token_embed.shape, gather_indices.shape)
        all_codes = token_embed.gather(1, gather_indices)  # (b, n, d, q-1)

        cumsum_codes = torch.cumsum(all_codes, dim=-1) #(b, n, d, q-1)

        active_indices = all_indices[torch.arange(bs), :, active_q_layers]  # (b, n)
        history_sum = cumsum_codes[torch.arange(bs), :, :, active_q_layers - 1]

        force_mask = False
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(y)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(y).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(bs, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        logits = self.trans_forward(history_sum, active_q_layers, cond_vector, ~non_pad_mask, force_mask)
        logits = self.output_project(logits, active_q_layers-1)
        ce_loss, pred_id, acc = cal_performance(logits, active_indices, ignore_index=self.pad_id)

        return ce_loss, pred_id, acc

    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 motion_ids,
                 conds,
                 m_lens,
                 temperature=1,
                 topk_filter_thres=0.9,
                 cond_scale=2,
                 num_res_layers=-1, # If it's -1, use all.
                 force_mask=False
                 ):

        # print(self.opt.num_quantizers)
        # assert len(timesteps) >= len(cond_scales) == self.opt.num_quantizers
        self.process_embed_proj_weight()

        device = next(self.parameters()).device
        seq_len = motion_ids.shape[1]
        batch_size = len(conds)

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        # token_embed = repeat(self.token_embed_weight, 'c d -> b c d', b=batch_size)
        # gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
        # history_sum = token_embed.gather(1, gathered_ids)

        # print(pa, seq_len)
        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        # print(padding_mask.shape, motion_ids.shape)
        motion_ids = torch.where(padding_mask, self.pad_id, motion_ids)
        all_indices = [motion_ids]
        history_sum = 0
        num_quant_layers = self.opt.num_quantizers if num_res_layers==-1 else num_res_layers+1

        for i in range(1, num_quant_layers):
            # print(f"--> Working on {i}-th quantizer")
            # Start from all tokens being masked
            # qids = torch.full((batch_size,), i, dtype=torch.long, device=motion_ids.device)
            token_embed = self.token_embed_weight[i-1]
            token_embed = repeat(token_embed, 'c d -> b c d', b=batch_size)
            gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
            history_sum += token_embed.gather(1, gathered_ids)

            logits = self.forward_with_cond_scale(history_sum, i, cond_vector, padding_mask, cond_scale=cond_scale, force_mask=force_mask)
            # logits = self.trans_forward(history_sum, qids, cond_vector, padding_mask)

            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)

            # probs = F.softmax(filtered_logits, dim=-1)  # (b, seqlen, ntoken)
            # # print(temperature, starting_temperature, steps_until_x0, timesteps)
            # # print(probs / temperature)
            # pred_ids = Categorical(probs / temperature).sample()  # (b, seqlen)

            ids = torch.where(padding_mask, self.pad_id, pred_ids)

            motion_ids = ids
            all_indices.append(ids)

        all_indices = torch.stack(all_indices, dim=-1)
        # padding_mask = repeat(padding_mask, 'b n -> b n q', q=all_indices.shape[-1])
        # all_indices = torch.where(padding_mask, -1, all_indices)
        all_indices = torch.where(all_indices==self.pad_id, -1, all_indices)
        # all_indices = all_indices.masked_fill()
        return all_indices

    @torch.no_grad()
    @eval_decorator
    def edit(self,
            motion_ids,
            conds,
            m_lens,
            temperature=1,
            topk_filter_thres=0.9,
            cond_scale=2
            ):

        # print(self.opt.num_quantizers)
        # assert len(timesteps) >= len(cond_scales) == self.opt.num_quantizers
        self.process_embed_proj_weight()

        device = next(self.parameters()).device
        seq_len = motion_ids.shape[1]
        batch_size = len(conds)

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        # token_embed = repeat(self.token_embed_weight, 'c d -> b c d', b=batch_size)
        # gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
        # history_sum = token_embed.gather(1, gathered_ids)

        # print(pa, seq_len)
        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        # print(padding_mask.shape, motion_ids.shape)
        motion_ids = torch.where(padding_mask, self.pad_id, motion_ids)
        all_indices = [motion_ids]
        history_sum = 0

        for i in range(1, self.opt.num_quantizers):
            # print(f"--> Working on {i}-th quantizer")
            # Start from all tokens being masked
            # qids = torch.full((batch_size,), i, dtype=torch.long, device=motion_ids.device)
            token_embed = self.token_embed_weight[i-1]
            token_embed = repeat(token_embed, 'c d -> b c d', b=batch_size)
            gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
            history_sum += token_embed.gather(1, gathered_ids)

            logits = self.forward_with_cond_scale(history_sum, i, cond_vector, padding_mask, cond_scale=cond_scale)
            # logits = self.trans_forward(history_sum, qids, cond_vector, padding_mask)

            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)

            # probs = F.softmax(filtered_logits, dim=-1)  # (b, seqlen, ntoken)
            # # print(temperature, starting_temperature, steps_until_x0, timesteps)
            # # print(probs / temperature)
            # pred_ids = Categorical(probs / temperature).sample()  # (b, seqlen)

            ids = torch.where(padding_mask, self.pad_id, pred_ids)

            motion_ids = ids
            all_indices.append(ids)

        all_indices = torch.stack(all_indices, dim=-1)
        # padding_mask = repeat(padding_mask, 'b n -> b n q', q=all_indices.shape[-1])
        # all_indices = torch.where(padding_mask, -1, all_indices)
        all_indices = torch.where(all_indices==self.pad_id, -1, all_indices)
        # all_indices = all_indices.masked_fill()
        return all_indices