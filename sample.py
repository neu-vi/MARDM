import os
from os.path import join as pjoin
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import random
from models.AE import AE_models
from models.MARDM import MARDM_models
from models.LengthEstimator import LengthEstimator
from utils.motion_process import recover_from_ric, plot_3d_motion, kit_kinematic_chain, t2m_kinematic_chain
import argparse

import glob
from pathlib import Path

import re


def indices_from_batch(shard_index: int, n: int, shard_size: int):
    if shard_index is None or shard_index < 0:
        return set()
    lo = shard_index * shard_size
    hi = min((shard_index + 1) * shard_size - 1, n - 1)
    if lo <= hi:
        return set(range(lo, hi + 1))
    return set()


def apply_index_filter(pending, n_total: int, shard_size: int, shard_index):
    if shard_index is None or shard_size is None:
        return pending
    inc_set = indices_from_batch(shard_index, n_total, shard_size)
    return [i for i in pending if i in inc_set]


def safe_stem(s: str, max_len: int = 120) -> str:
    # Replace path separators explicitly (Linux/macOS: '/', Windows: '\')
    s = s.replace("/", "_").replace("\\", "_")

    # Replace anything not filename-friendly with underscores
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)

    # Collapse runs of underscores and trim
    s = re.sub(r"_+", "_", s).strip("._-")

    # Avoid overly long filenames (common filesystem limit is 255 bytes)
    if len(s) > max_len:
        s = s[:max_len].rstrip("._-")

    return s or "caption"


def chunked(xs: list[int], n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def load_descriptions_dir(desc_dir: Path):
    """
    Reads ./descriptions/*.txt.
    Returns:
      prompt_list: list[str] of ALL descriptions across all files (in file order, then line order)
      group_of_prompt: list[str] same length as prompt_list; folder name per prompt (base name of file)
    """
    prompt_list: list[str] = []
    group_of_prompt: list[str] = []

    for fpath in sorted(desc_dir.glob("*.txt")):
        group = fpath.stem  # base name (e.g., M000123)
        for line in fpath.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            prompt_list.append(line)
            group_of_prompt.append(group)

    return prompt_list, group_of_prompt


def main(args):
    #################################################################################
    #                                      Seed                                     #
    #################################################################################
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # setting this to true significantly increase training and sampling speed
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    #################################################################################
    #                                       Data                                    #
    #################################################################################
    dim_pose = 64 if args.dataset_name == 'kit' else 67
    nb_joints = 21 if args.dataset_name == 'kit' else 22
    data_root = f'{args.dataset_dir}/KIT-ML/' if args.dataset_name == 'kit' else f'{args.dataset_dir}/HumanML3D/'
    mean = np.load(pjoin(data_root, 'Mean.npy'))
    std = np.load(pjoin(data_root, 'Std.npy'))
    #################################################################################
    #                                      Models                                   #
    #################################################################################
    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')
    result_dir = pjoin('./generation', args.name)
    os.makedirs(result_dir, exist_ok=True)

    ae = AE_models[args.ae_model](input_width=dim_pose)
    ckpt = torch.load(pjoin(args.checkpoints_dir, args.dataset_name, args.ae_name, 'model',
                            'latest.tar' if args.dataset_name == 't2m' else 'net_best_fid.tar'), map_location='cpu')
    model_key = 'ae'
    ae.load_state_dict(ckpt[model_key])

    ema_mardm = MARDM_models[args.model](ae_dim=ae.output_emb_width, cond_mode='text')
    model_dir = pjoin(model_dir, 'latest.tar')
    checkpoint = torch.load(model_dir, map_location='cpu')
    missing_keys2, unexpected_keys2 = ema_mardm.load_state_dict(checkpoint['ema_mardm'], strict=False)
    assert len(unexpected_keys2) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys2])

    length_estimator = LengthEstimator(512, 50)
    ckpt = torch.load(pjoin(args.checkpoints_dir, args.dataset_name, 'length_estimator', 'model', 'finest.tar'),
                      map_location='cpu')
    length_estimator.load_state_dict(ckpt['estimator'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #################################################################################
    #                                     Sampling                                  #
    #################################################################################
    prompt_list = []
    group_of_prompt = []
    length_list = []

    est_length = False

    if args.descriptions:
        desc_dir = Path(args.descriptions_dir)
        prompt_list, group_of_prompt = load_descriptions_dir(desc_dir)
        if len(prompt_list) == 0:
            raise RuntimeError(f"No descriptions found under {desc_dir}/*.txt")
        est_length = True
    elif args.text_prompt != "":
        prompt_list.append(args.text_prompt)
        group_of_prompt.append("single")
        if args.motion_length == 0:
            est_length = True
        else:
            length_list.append(args.motion_length)
    elif args.text_path != "":
        with open(args.text_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                infos = line.split('#')
                prompt_list.append(infos[0])
                group_of_prompt.append("text_path")
                if len(infos) == 1 or (not infos[1].isdigit()):
                    est_length = True
                    length_list = []
                else:
                    length_list.append(int(infos[-1]))
    else:
        raise Exception("A text prompt, a file of text prompts, or --descriptions is required.")

    ae.to(device)
    ema_mardm.to(device)
    length_estimator.to(device)

    ae.eval()
    ema_mardm.eval()
    length_estimator.eval()

    if est_length:
        print("Since no motion length are specified, we will use estimated motion lengthes!!")
        all_token_lens = []
        text_bs = args.text_batch_size
        with torch.no_grad():
            for start in range(0, len(prompt_list), text_bs):
                batch_prompts = prompt_list[start:start + text_bs]
                text_embedding = ema_mardm.encode_text(batch_prompts)
                pred_dis = length_estimator(text_embedding)
                probs = F.softmax(pred_dis, dim=-1)
                token_lens = Categorical(probs).sample()
                all_token_lens.append(token_lens.detach().cpu())
                del text_embedding, pred_dis, probs, token_lens
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        token_lens_all = torch.cat(all_token_lens, dim = 0).to(device).long()
    else:
        token_lens_all = (torch.LongTensor(length_list) // 4).to(device).long()
    
    m_length_all = token_lens_all * 4
    captions = prompt_list

    sample = 0
    kinematic_chain = kit_kinematic_chain if args.dataset_name == 'kit' else t2m_kinematic_chain

    def prompt_done(group: str, idx: int) -> bool:
        s_path = Path(result_dir) / group / str(idx)
        if not s_path.exists():
            return False
        return (len(list(s_path.glob("*.mp4"))) > 0) or (len(list(s_path.glob("*.npy"))) > 0)

    pending_indices = [i for i in range(len(prompt_list)) if not prompt_done(group_of_prompt[i], i)]
    if not pending_indices:
        print(f"All {len(prompt_list)} prompts already have outputs under {result_dir}. Nothing to do.")
        return

    completed_count = len(prompt_list) - len(pending_indices)

    pending_indices = apply_index_filter(
        pending=pending_indices,
        n_total=len(prompt_list),
        shard_size=args.shard_size,
        shard_index=args.shard_index
    )

    if args.shard_index is not None:
        print(
            f"Index filter applied (shard_index={args.shard_index}). "
            f"Now considering {len(pending_indices)} pending prompts out of {len(prompt_list)} total."
        )

    if not pending_indices:
        if args.shard_index is not None and args.shard_size is not None:
            lo = args.shard_index * args.shard_size
            hi = min((args.shard_index + 1) * args.shard_size - 1, len(prompt_list) - 1)
            print(f"No pending prompts in selected shard {args.shard_index} (global indices {lo}-{hi}). Nothing to do.")
        else:
            print("No pending prompts. Nothing to do.")
        return

    print(f"Found {completed_count} completed; generating {len(pending_indices)} prompts:")
    print(f"First few pending indices: {pending_indices[:10]}")

    batch_id_global = 0

    for r in range(args.repeat_times):
        print("-->Repeat %d" % r)

        for batch_indices in chunked(pending_indices, args.batch_size):
            batch_id_global += 1
            print(f"  -> Batch {batch_id_global} (size={len(batch_indices)}): {batch_indices[:10]}")

            batch_captions = [prompt_list[i] for i in batch_indices]
            batch_token_lens = token_lens_all[batch_indices]
            batch_m_length = m_length_all[batch_indices]

            with torch.no_grad():
                try:
                    pred_latents = ema_mardm.generate(
                        batch_captions,
                        batch_token_lens,
                        args.time_steps,
                        args.cfg,
                        temperature = args.temperature,
                        hard_pseudo_reorder = args.hard_pseudo_reorder
                    )
                except AssertionError as e:
                    print(f"ODEINT failed for batch starting with {batch_indices[:3]}: {e}")
                    continue
                pred_motions = ae.decode(pred_latents)
                pred_motions = pred_motions.detach().cpu().numpy()
                data = pred_motions * std + mean

            for local_i, (orig_idx, caption, joint_data) in enumerate(zip(batch_indices, batch_captions, data)):
                group = group_of_prompt[orig_idx]
                ml = int(batch_m_length[local_i])
                print(f"    ----> {group}/{orig_idx}: {caption}  len={ml}")
                s_path = pjoin(result_dir, group, str(orig_idx))
                os.makedirs(s_path, exist_ok=True)
                joint_data = joint_data[:ml]
                joint = recover_from_ric(torch.from_numpy(joint_data).float(), nb_joints).numpy()
                print(
                    "    joint stats:",
                    "min", np.nanmin(joint),
                    "max", np.nanmax(joint),
                    "nan?", np.isnan(joint).any(),
                    "inf?", np.isinf(joint).any()
                )
                cap_stem = safe_stem(caption)
                mp4_name = f"{cap_stem}_sample{orig_idx}_repeat{r}_len{ml}.mp4"
                npy_name = f"{cap_stem}_sample{orig_idx}_repeat{r}_len{ml}.npy"
                save_mp4 = pjoin(s_path, mp4_name)
                save_npy = pjoin(s_path, npy_name)
                plot_3d_motion(save_mp4, kinematic_chain, joint, title=caption, fps=20)
                np.save(save_npy, joint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type = int, default = 8)
    parser.add_argument("--text_batch_size", type = int, default = 64, help = "Batch size for CLIP text encoding / length estimation (lower if OOM).")
    parser.add_argument('--name', type=str, default='MARDM')
    parser.add_argument('--ae_name', type=str, default="AE")
    parser.add_argument('--ae_model', type=str, default='AE_Model')
    parser.add_argument('--model', type=str, default='MARDM-SiT-XL')
    parser.add_argument('--dataset_name', type=str, default='t2m')
    parser.add_argument('--dataset_dir', type=str, default='./datasets')

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument("--time_steps", default=18, type=int)
    parser.add_argument("--cfg", default=4.5, type=float)
    parser.add_argument("--temperature", default=1, type=float)

    parser.add_argument('--text_prompt', default='', type=str)
    parser.add_argument('--text_path', type=str, default="")
    parser.add_argument("--motion_length", default=0, type=int)
    parser.add_argument("--descriptions", action="store_true", help="If set, read prompts from descriptions_dir/*.txt; one prompt per line.")
    parser.add_argument("--descriptions_dir", type=str, default="descriptions", help="Directory containing *.txt files of descriptions (one per line).")
    parser.add_argument("--repeat_times", default=1, type=int)
    parser.add_argument('--hard_pseudo_reorder', action="store_true")

    parser.add_argument(
        "--shard_index",
        type=int,
        default=None,
        help="Single batch index to include (0-based). Batch k = [k*batch_size .. (k+1)*batch_size-1]."
    )
    parser.add_argument("--shard_size", type = int, default = None, help = "How many prompt indices each Slurm array task owns (e.g., 437). If None, no sharding is applied.")
    arg = parser.parse_args()
    main(arg)
