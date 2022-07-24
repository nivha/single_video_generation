import os
import argparse
import ast
import numpy as np

import utils
from utils import scale_utils
from utils.main_utils import now, save_vid, read_original_video, get_raft_stuff, tokenize, save_vid, torchvid2mp4
from pnn.models import PMPNN3D
from pnn import patch_match

import threadpoolctl
thread_limit = threadpoolctl.threadpool_limits(limits=8)


fix_extra = lambda x: x[:, -3:, ...] if x.shape[1] != 2 else x.norm(dim=1, keepdim=True)


def analogies_loop(o, a_pyr, b_pyr, extra_a, extra_b, downscales, upscale_factors, a_out_shapes, b_out_shapes):
    import torch

    # patchmatch params
    o.chunk_size = 40
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    o.device_fold = 'cuda:1' if num_gpus >= 2 else 'cuda:0'
    o.reuse = True
    o.reuse_k = False
    o.max_unfolded_size_gb = 20


    max_flow_level = int((o.n_stages-1) * o.n_max_flow)
    start = now()

    cur_index, cur_k_index, cur_k_size, cur_q_size = None, None, None, None
    for level in range(o.n_stages):
        time_start_level = now()

        # set q/k/v
        v_vid = b_pyr[level]
        # first level, where all the magic happens
        if level == 0:
            extra_a_l = utils.resize_right.resize(extra_a, scale_factors=[1, 1] + downscales[level], out_shape=a_out_shapes[level])
            extra_b_l = utils.resize_right.resize(extra_b, scale_factors=[1, 1] + downscales[level], out_shape=b_out_shapes[level])

            q_vid = extra_a_l
            k_vid = extra_b_l

        # in the rest of the levels
        else:
            extra_a_l = utils.resize_right.resize(extra_a, scale_factors=[1, 1] + downscales[level], out_shape=a_out_shapes[level])
            extra_b_l = utils.resize_right.resize(extra_b, scale_factors=[1, 1] + downscales[level], out_shape=b_out_shapes[level])

            q_vid = utils.resize_right.resize(q_vid, scale_factors=[1, 1] + upscale_factors[level], out_shape=a_out_shapes[level])
            k_vid = utils.resize_right.resize(b_pyr[level - 1], scale_factors=[1, 1] + upscale_factors[level], out_shape=b_out_shapes[level])

        # make qkv-thing
        J = o.Js[level]
        ks = o.Ks[level]
        alpha = o.alpha
        bidi = alpha is not None
        for j in range(J):

            if j == 1:
                k_vid = v_vid

            if max_flow_level is not None and level < max_flow_level and (level > 0 or (level == 0 and j >= 1)):
                q_vid = torch.cat([q_vid, extra_a_l], dim=1)
                if j == 0 or j == 1:
                    k_vid = torch.cat([k_vid, extra_b_l], dim=1)

            save_vid(fix_extra(q_vid.clone()), f'{o.results_dir}/{level}/q')
            save_vid(fix_extra(k_vid.clone()), f'{o.results_dir}/{level}/k')
            save_vid(fix_extra(v_vid.clone()), f'{o.results_dir}/{level}/v')

            print(level, j, '*' * 100)
            print(now(),
                  f'level: {level}/{o.n_stages}:{j}/{J} ; q-shape: {q_vid.shape}; k-shape: {k_vid.shape}; v-shape: {v_vid.shape}')

            device_fold = o.device_fold
            unfolded_size_gb = q_vid.numel() * np.prod(ks) * (torch.finfo(q_vid.dtype).bits / 8) / 1e9
            memory_efficient = o.max_unfolded_size_gb is not None and unfolded_size_gb >= o.max_unfolded_size_gb
            # print(f'device_fold={device_fold}', f'unfolded_size={unfolded_size_gb:.2f}GB',
            #       f'memory_efficient={memory_efficient}', sep='  ')

            # patch-match pnn3d
            pnn = PMPNN3D(
                kernel_size=ks,
                reduce=o.reduce,
                device=o.device,
                steps=(8, 4, 1) * 5,
                radii=(1, 1, 1) * 5,
                dist_fn=patch_match.l2_dist,
                chunk_size=o.chunk_size,
                device_fold=device_fold,
                scale_origin=False,
                use_padding=False,
                with_noisy=True,
                with_noisy_neighbors=True,
                debug=False,
            )
            q_vid, D, cur_index, cur_k_index, cur_k_size, cur_q_size, *debug_outs = pnn.reconstruct_qkv(
                q_vid, k_vid, v_vid, bidi=bidi, alpha=alpha,
                old_index=cur_index, old_k_size=cur_k_size,
                old_k_index=cur_k_index, old_q_size=cur_q_size,
                memory_efficient=memory_efficient,
            )
            if not o.reuse:
                cur_index, cur_k_size = None, None
            if not o.reuse_k:
                cur_k_index = None

        assert q_vid.shape[1] == 3
        q_vid = q_vid[:, -3:, ...]  # TODO
        save_vid(q_vid, f'{o.results_dir}/{level}/r')
        print(now(), f'finished level {level}/{o.n_stages} took: {now() - time_start_level}')

    total_time = now() - start
    print(now(), f'DONE! took: {total_time}')

    # Save results to mp4
    torchvid2mp4(q_vid, os.path.join(o.results_dir, 'r.mp4'), fps=10)
    torchvid2mp4(a_pyr[-1], os.path.join(o.results_dir, 'a.mp4'), fps=10)
    torchvid2mp4(b_pyr[-1], os.path.join(o.results_dir, 'b.mp4'), fps=10)
    print('Saved results to:', o.results_dir)


def process_flow(flow_vid, out_shape, n_bins):
    extra = flow_vid.norm(dim=1, keepdim=True)
    extra = utils.resize_right.resize(extra, out_shape=out_shape)
    extra = tokenize(extra, n_bins=n_bins, strategy='kmeans')
    extra = (extra / extra.max()).pow(0.25)

    return extra


def run_everything(o):

    os.environ['CUDA_VISIBLE_DEVICES'] = o.gpu
    import torch
    o.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    assert len(o.kernel_size) == 3
    print('RESULTS_DIR:', o.results_dir)
    os.makedirs(o.results_dir, exist_ok=True)

    # read content video (a)
    a_vid = read_original_video(o.a_frames_dir, o.a_start_frame, o.a_end_frame, o.a_max_size, o.device)
    _, C, T, H, W = a_vid.shape
    downscales, upscale_factors, a_out_shapes = scale_utils.get_scales_out_shapes(T, H, W, o.downfactor, o.min_size)
    a_pyr = scale_utils.create_spatio_temporal_pyramid(a_vid, downscales, a_out_shapes, verbose=True, device=o.device)
    print('*' * 25)

    # read style video (b)
    b_vid = read_original_video(o.b_frames_dir, o.b_start_frame, o.b_end_frame, o.b_max_size, o.device)
    b_pyr = scale_utils.create_spatio_temporal_pyramid(b_vid, downscales, out_shapes=None, verbose=True, device=o.device)
    b_out_shapes = [tuple(l.shape[-3:]) for l in b_pyr]
    print('*' * 25)

    # define some more stuff
    o.n_stages = len(a_out_shapes)
    o.Js = [o.J] * o.n_stages
    o.Ks = [o.kernel_size] * o.n_stages
    o.bidialphas = [o.alpha] * o.n_stages

    for i, outs in enumerate(range(o.n_stages)):
        print(f'LEVEL {i:3} SIZE {str(a_out_shapes[i]):35} KERNEL {str(o.Ks[i]):15} J {o.Js[i]} alpha {o.bidialphas[i]}')

    flows_a = get_raft_stuff(a_vid)
    flows_b = get_raft_stuff(b_vid)

    extra_a = process_flow(flows_a, a_out_shapes[-1], o.a_n_bins)
    extra_b = process_flow(flows_b, b_out_shapes[-1], o.b_n_bins)

    analogies_loop(o, a_pyr, b_pyr, extra_a, extra_b, downscales, upscale_factors, a_out_shapes, b_out_shapes)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2list(v):
    return ast.literal_eval(v)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='', type=str, default='0')
    parser.add_argument('--results_dir', default='./result', help='')
    parser.add_argument('--a_frames_dir', help='')
    parser.add_argument('--b_frames_dir', help='')
    parser.add_argument('--a_n_bins', help='', type=int, default=None)
    parser.add_argument('--b_n_bins', help='', type=int, default=None)

    parser.add_argument('--a_start_frame', default=1, help='first frame to read', type=int)
    parser.add_argument('--a_end_frame', default=30, help='end frame to read', type=int)
    parser.add_argument('--a_max_size', default=256, help='', type=int)
    parser.add_argument('--b_start_frame', default=1, help='first frame to read', type=int)
    parser.add_argument('--b_end_frame', default=30, help='end frame to read', type=int)
    parser.add_argument('--b_max_size', default=256, help='', type=int)

    parser.add_argument('--downfactor', default='(0.9,0.9)', type=str2list, help='')
    parser.add_argument('--min_size', default='(3,20)', type=str2list, help='')
    parser.add_argument('--kernel_size', default='(3,5,5)', type=str2list, help='')
    parser.add_argument('--J', default=1, help='', type=int)
    parser.add_argument('--reduce', default='median', help='', choices=['mean', 'median'])
    parser.add_argument('--alpha', default=1, help='', type=float)
    parser.add_argument('--n_max_flow', default=0.5, help='', type=float)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    o = get_args()
    print(now(), o)
    run_everything(o)
