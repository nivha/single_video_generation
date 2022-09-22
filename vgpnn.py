import os
import torch
import numpy as np

import utils
from utils.main_utils import now, save_vid, read_original_video
from pnn.models import PNN3D, PMPNN3D
from utils import scale_utils
from pnn import patch_match


class VGPNN:
    def __init__(self, pyr, q0,
                 upscale_factors, v_out_shapes, r_out_shapes, n_stages,
                 Js, Ks, reduce='median'):
        self.pyr = pyr
        self.q0 = q0
        self.upscale_factors = upscale_factors
        self.v_out_shapes = v_out_shapes
        self.r_out_shapes = r_out_shapes
        self.n_stages = n_stages
        self.Js = Js
        self.Ks = Ks
        self.reduce = reduce

    def forward(self, q0, noises_dict=None, noises_amps=None, n_stages=None,
                return_input=False, save_dir=None):
        q_vid = q0.clone()  # otherwise it changes input tensor..
        device = q_vid.device

        if n_stages is None:
            n_stages = self.n_stages

        for level in range(n_stages):
            # get next k/v
            if level == 0:
                k_vid = self.pyr[level]
            else:
                q_vid = utils.resize_right.resize(q_vid, scale_factors=[1, 1] + self.upscale_factors[level], out_shape=self.r_out_shapes[level])
                k_vid = utils.resize_right.resize(self.pyr[level - 1], scale_factors=[1, 1] + self.upscale_factors[level], out_shape=self.v_out_shapes[level])
            v_vid = self.pyr[level]

            if noises_dict is not None and level in noises_dict:
                q_vid += noises_dict[level] * noises_amps.get(level, 1)

            # save results to output dir
            if save_dir:
                save_vid(q_vid, f'{save_dir}/{level}/q')
                save_vid(k_vid, f'{save_dir}/{level}/k')
                save_vid(v_vid, f'{save_dir}/{level}/v')
            # save input for returning
            if return_input:
                q_in = q_vid.data.clone()

            # make qkv-thing
            J = self.Js[level]
            ks = self.Ks[level]
            for j in range(J):
                if j == 1:
                    k_vid = self.pyr[level]

                pnn = PNN3D(
                    kernel_size=ks, reduce=self.reduce, device=device,
                )
                q_vid, D, cur_index, cur_k_index, cur_k_size, cur_q_size = pnn.reconstruct_qkv_diff(
                    q_vid, k_vid, v_vid,
                )

        if return_input:
            return q_in, q_vid
        else:
            return q_vid


class PMVGPNN:
    def __init__(self, pyr, q0,
                 upscale_factors, v_out_shapes, r_out_shapes, n_stages,
                 Js, Ks, reduce='median',
                 chunk_size=40, max_unfolded_size_gb=20, device='cuda:0'):
        self.device = device
        self.pyr = pyr
        self.q0 = q0
        self.upscale_factors = upscale_factors
        self.v_out_shapes = v_out_shapes
        self.r_out_shapes = r_out_shapes
        self.n_stages = n_stages
        self.Js = Js
        self.Ks = Ks
        self.reduce = reduce

        # patchmatch
        self.dist_fn = patch_match.l2_dist
        num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        self.device_fold = torch.device(('cuda:1' if num_gpus >= 2 else 'cuda:0') if torch.cuda.is_available() else 'cpu')
        self.scale_origin = False
        self.with_noisy = True
        self.with_noisy_neighbors = True
        self.reuse = True
        self.reuse_k = False
        self.chunk_size = chunk_size
        self.max_unfolded_size_gb = max_unfolded_size_gb
        self.debug = True

        if not self.device.type.startswith('cuda'):
            self.device_fold = 'cpu'

    def forward(self, q0, noises_dict=None, noises_amps=None, n_stages=None,
                return_input=False, save_dir=None, verbose=False):
        if verbose:
            print(f'DEVICES: device={self.device} device_fold={self.device_fold}')

        q_vid = q0.clone()  # otherwise it changes input tensor..
        device = q_vid.device

        if n_stages is None:
            n_stages = self.n_stages

        for level in range(n_stages):
            time_start_level = now()

            # get next k/v
            if level == 0:
                k_vid = self.pyr[level]
            else:
                q_vid = utils.resize_right.resize(q_vid, scale_factors=[1, 1] + self.upscale_factors[level], out_shape=self.r_out_shapes[level])
                k_vid = utils.resize_right.resize(self.pyr[level - 1], scale_factors=[1, 1] + self.upscale_factors[level], out_shape=self.v_out_shapes[level])
            v_vid = self.pyr[level]

            if noises_dict is not None and level in noises_dict:
                q_vid += noises_dict[level] * noises_amps.get(level, 1)

            # save results to output dir
            if save_dir:
                save_vid(q_vid, f'{save_dir}/{level}/q')
                save_vid(k_vid, f'{save_dir}/{level}/k')
                save_vid(v_vid, f'{save_dir}/{level}/v')
            # save input for returning
            if return_input:
                q_in = q_vid.data.clone()

            # make qkv-thing
            J = self.Js[level]
            ks = self.Ks[level]
            cur_index, cur_k_index, cur_k_size, cur_q_size = None, None, None, None
            for j in range(J):
                if verbose:
                    print(now(), os.environ['CUDA_VISIBLE_DEVICES'],
                          f'level: {level}/{n_stages}:{j}/{J} ; q-shape: {q_vid.shape}; k-shape: {k_vid.shape}; v-shape: {v_vid.shape}')

                if j == 1:
                    k_vid = self.pyr[level]

                device_fold = self.device_fold
                unfolded_size_gb = q_vid.numel() * np.prod(ks) * (torch.finfo(q_vid.dtype).bits / 8) / 1e9
                memory_efficient = self.max_unfolded_size_gb is not None and unfolded_size_gb >= self.max_unfolded_size_gb
                # print(f'device_fold={device_fold}', f'unfolded_size={unfolded_size_gb:.2f}GB', f'memory_efficient={memory_efficient}', sep='  ')

                # patch-match pnn3d
                pnn = PMPNN3D(
                    kernel_size=ks, reduce=self.reduce, use_padding=False,
                    reduce_std=None, device=device,
                    steps=(8, 4, 2, 1) * 5,
                    radii=(2, 2, 1, 1) * 5,
                    dist_fn=self.dist_fn, chunk_size=self.chunk_size,
                    device_fold=device_fold, scale_origin=self.scale_origin,
                    with_noisy=self.with_noisy, with_noisy_neighbors=self.with_noisy_neighbors,
                    debug=self.debug
                )
                q_vid, D, cur_index, cur_k_index, cur_k_size, cur_q_size, *debug_outs = pnn.reconstruct_qkv(
                    q_vid, k_vid, v_vid, bidi=False, alpha=None,
                    old_index=cur_index, old_k_size=cur_k_size,
                    old_k_index=cur_k_index, old_q_size=cur_q_size,
                    memory_efficient=memory_efficient,
                )
                if self.debug:
                    hists = debug_outs[0]
                if not self.reuse:
                    cur_index, cur_k_size = None, None
                if not self.reuse_k:
                    cur_k_index = None

            if verbose:
                print(now(), f'finished level {level}/{n_stages} time: {now() - time_start_level}')

            if save_dir:
                save_vid(q_vid, f'{save_dir}/{level}/r')

        if return_input:
            return q_in, q_vid
        else:
            return q_vid


def get_vgpnn(frames_dir, start_frame, end_frame, device,
             max_size, min_size, downfactor, J, kernel_size,
             sthw, reduce, J_start_from=1, ext='png', verbose=False, vgpnn_type=None):

    o = lambda: None
    # input parameters
    o.device = device
    o.start_frame = start_frame
    o.end_frame = end_frame
    o.frames_dir = frames_dir
    # algorithm parameters
    o.max_size = max_size
    o.min_size = min_size  # (T,H,W)
    o.downfactor = downfactor
    o.J = J
    o.J_start_from = J_start_from
    o.kernel_size = kernel_size
    o.st, o.sh, o.sw = sthw
    o.reduce = reduce

    # read video and create spatio-temporal pyramid
    orig_vid = read_original_video(o.frames_dir, o.start_frame, o.end_frame, o.max_size, o.device, verbose=verbose, ext=ext)
    _, C, T, H, W = orig_vid.shape
    downscales, upscale_factors, out_shapes = scale_utils.get_scales_out_shapes(T, H, W, o.downfactor, o.min_size)
    pyr = scale_utils.create_spatio_temporal_pyramid(orig_vid, downscales, out_shapes, verbose=verbose, device=o.device, temporal_nearest=False)

    t_, h_, w_ = scale_utils.get_out_shapes(T, H, W, o.st, o.sh, o.sw)
    resized_vid = utils.resize_right.resize(orig_vid, scale_factors=[1, 1, o.st, o.sh, o.sw], out_shape=[1, 1, t_, h_, w_])
    ret_pyr = scale_utils.create_spatio_temporal_pyramid(resized_vid, downscales, out_shapes=None, verbose=verbose, device=o.device)
    ret_out_shapes = [tuple(l.shape[-3:]) for l in ret_pyr]
    q0 = ret_pyr[0].clone()

    assert ret_pyr[0].shape[2] >= o.kernel_size[0], f'smallest pyramid level has less frames {ret_pyr[0].shape[2]} than temporal kernel-size {o.kernel_size[0]}. You may want to increase min_size of the temporal dimension'

    # upward stuff
    o.n_stages = len(out_shapes)
    o.Ks = [o.kernel_size] * o.n_stages
    o.Js = [o.J] * o.n_stages
    for i in range(o.J_start_from):
        o.Js[i] = 1
    # set J and kernel_size in finer levels
    for i, outs in enumerate(ret_out_shapes):
        n_pixels = outs[-1] * outs[-2] * outs[-3]
        if n_pixels > 50000 * 15:
            o.Js[i] = o.J
        if n_pixels > 3000000:
            o.Js[i] = 1
        if n_pixels > 500000 * 15:
            o.Ks[i] = (3, 5, 5)

    if vgpnn_type == 'vanilla' or vgpnn_type is None:
        vgpnn_module = VGPNN
    elif vgpnn_type == 'pm':
        vgpnn_module = PMVGPNN
    else:
        raise

    vgpnn = vgpnn_module(pyr, q0, upscale_factors, out_shapes, ret_out_shapes, o.n_stages, o.Js, o.Ks, reduce='median', device=device)
    return vgpnn, orig_vid
