# inference.py

import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from os.path import join as pjoin

from gen_t2m import load_model_bundle
from utils.motion_process import recover_from_ric
from visualization.joints2bvh import Joint2BVHConvertor
from utils.paramUtil import t2m_kinematic_chain


def generate_motion_from_prompt(model_bundle, prompt: str):
    vq_model = model_bundle["vq_model"]
    res_model = model_bundle["res_model"]
    t2m_transformer = model_bundle["t2m_transformer"]
    length_estimator = model_bundle["length_estimator"]
    device = model_bundle["device"]

    # Prepare input
    prompt_list = [prompt]
    text_embedding = t2m_transformer.encode_text(prompt_list)
    pred_dis = length_estimator(text_embedding)
    probs = F.softmax(pred_dis, dim=-1)
    token_lens = Categorical(probs).sample().to(device)
    m_length = token_lens * 4

    with torch.no_grad():
        mids, pred_len = t2m_transformer.generate(
            prompt_list,
            token_lens,
            timesteps=10,
            cond_scale=1.0,
            temperature=1.0,
            topk_filter_thres=0.9,
            gsample=False,
            is_predict_len=True
        )

        mids = res_model.generate(mids, prompt_list, pred_len, temperature=1.0, cond_scale=5.0)
        pred_motions = vq_model.forward_decoder(mids).detach().cpu().numpy()

    # De-normalize
    mean = np.load(pjoin("checkpoints", model_bundle["vq_opt"].dataset_name, model_bundle["model_opt"].vq_name, "meta", "mean.npy"))
    std = np.load(pjoin("checkpoints", model_bundle["vq_opt"].dataset_name, model_bundle["model_opt"].vq_name, "meta", "std.npy"))
    data = pred_motions * std + mean

    # Recover joints
    joint = recover_from_ric(torch.from_numpy(data[0]).float(), 22).numpy()

    # Save to BVH
    out_dir = "generation/web_output/animations/0"
    os.makedirs(out_dir, exist_ok=True)
    bvh_path = pjoin(out_dir, f"sample_output.bvh")
    converter = Joint2BVHConvertor()
    converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)

    return bvh_path


# CLI test
if _name_ == "_main_":
    model_bundle = load_model_bundle()
    path = generate_motion_from_prompt(model_bundle, "a person waves one hand then steps back")
    print("Generated:", path)