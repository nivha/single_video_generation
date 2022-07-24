# __author__ = "Ben Feinstein (ben.feinstein@weizmann.ac.il)"

import itertools
import gc
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _ntuple, _triple

from pnn.patch_match import tensor2patchesnd as unfold3d
from pnn.patch_match import extract_patches_by_index
from pnn.fold3d import fold3d as _fold3d

NAN = float('nan')

__all__ = ['unfold3d', 'fold3d']


def fold3d(
        input,  # not patches!!!
        index,
        kernel_size,
        chunk_size=None,
        stride=1,
        use_padding=False,
        reduce='mean',
        aggressive_gc=0,
        **kwargs):  # noqa
    # input dimensions (8D): 1, c, t, h, w
    # index dimensions (): 1, ndim, *size
    # output dimensions (5D): 1, c, t, h, w
    ndim = 3
    assert input.size(0) == index.size(0) == 1
    assert index.size(1) == ndim
    if isinstance(chunk_size, int) and chunk_size == -1:
        chunk_size = None
    elif isinstance(chunk_size, Iterable) and  all(csz == -1 for csz in chunk_size):  # noqa  # yapf: disable
        chunk_size = None

    if chunk_size is None:
        return _fold3d_unchunked(input,
                                 index,
                                 kernel_size,
                                 stride=stride,
                                 use_padding=use_padding,
                                 reduce=reduce,
                                 **kwargs)

    return _fold3d_chunked(input,
                           index,
                           kernel_size,
                           chunk_size=chunk_size,
                           stride=stride,
                           use_padding=use_padding,
                           reduce=reduce,
                           aggressive_gc=aggressive_gc,
                           **kwargs)


def _compute_output_size3d(size, kernel_size, stride, use_padding):
    t, h, w = size = _triple(size)
    kt, kh, kw = kernel_size = _triple(kernel_size)
    st, sh, sw = stride = _triple(stride)
    if use_padding:
        pt, ph, pw = (kt - st, kh - sh, kw - sw)
        ot = st * (t - 1) + kt - 2 * pt
        oh = sh * (h - 1) + kh - 2 * ph
        ow = sw * (w - 1) + kw - 2 * pw
    else:
        ot = st * (t - 1) + kt
        oh = sh * (h - 1) + kh
        ow = sw * (w - 1) + kw
    return (ot, oh, ow)


def _get_chunk_slice(chunk_index, chunk_size, stride=None):
    assert len(chunk_index) == len(chunk_size)
    assert stride is None or len(chunk_index) == len(stride)
    chunk_slice = [slice(None), slice(None)]
    if stride is None:
        stride = _ntuple(len(chunk_index))(chunk_size)
    # chunk_size = _ntuple(len(chunk_index))(chunk_size)
    # stride = _ntuple(len(chunk_index))(stride)
    for cidx, csz, st in zip(chunk_index, chunk_size, stride):
        chunk_slice.append(slice(cidx * st, cidx * st + csz))
    chunk_slice = tuple(chunk_slice)
    return chunk_slice


def _get_chunk_size(chunk_size, q_size):
    ndim = len(q_size)
    chunk_size = _ntuple(ndim)(chunk_size)
    chunk_size = tuple([
        min(q_size[dim], chunk_size[dim])
        if chunk_size[dim] != -1 else q_size[dim] for dim in range(ndim)
    ])
    return chunk_size


def _get_chunk_iter(chunk_size, q_size):
    assert len(chunk_size) == len(q_size)
    return [range((qsz + csz - 1) // csz) for qsz, csz in zip(q_size, chunk_size)]  # yapf: disable  # noqa


def _fold3d_chunked(input, index, kernel_size, chunk_size, stride, use_padding,
                    reduce, aggressive_gc, **kwargs):
    ndim = 3
    kt, kh, kw = kernel_size = _triple(kernel_size)
    st, sh, sw = stride = _triple(stride)
    ct, ch, cw = chunk_size = _triple(chunk_size)
    pt, ph, pw = (kt - st, kh - sh, kw - sw)
    assert all(st == 1 for st in stride)
    if reduce == 'mean':
        weights = input.new_ones(size=(input.shape[0], 1, *input.shape[2:]))
    else:
        weights = None
    if not use_padding:
        input = F.pad(input, [0, kw, 0, kh, 0, kt], mode='constant', value=NAN)
        if weights is not None:
            weights = F.pad(weights, [0, kw, 0, kh, 0, kt], mode='constant', value=NAN)  # noqa  # yapf: disable
        index = F.pad(index, [pw, pw, ph, ph, pt, pt],
                      mode='constant',
                      value=-1).to(index)
    else:
        input = F.pad(input, [pw, pw, ph, ph, pt, pt],
                      mode='constant',
                      value=NAN)
        if weights is not None:
            weights = F.pad(weights, [pw, pw, ph, ph, pt, pt],
                            mode='constant',
                            value=0)
    patches = unfold3d(input, kernel_size, stride, use_padding=False)
    if weights is not None:
        w_patches = unfold3d(weights, kernel_size, stride, use_padding=False)
    else:
        w_patches = None
    if patches.dim() != 8:
        raise ValueError('expects a 8D tensor as input')
    it, ih, iw = index.shape[-3:]
    c = input.size(1)
    q_size = (it - 2 * pt, ih - 2 * ph, iw - 2 * pw)
    output_size = _compute_output_size3d((it, ih, iw), (kt, kh, kw), (st, sh, sw), True)  # noqa  # yapf: disable
    output = input.new_zeros(size=(1, c, *output_size))
    ct, ch, cw = chunk_size = _get_chunk_size(chunk_size, q_size)
    chunk_size_padded2 = (ct + 2 * pt, ch + 2 * ph, cw + 2 * pw)  # noqa  # yapf: disable
    chunk_size_padded1 = (ct + pt, ch + ph, cw + pw)
    # print(output_size, (it, ih, iw))
    # print(chunk_size, chunk_size_padded1, chunk_size_padded2)
    chunk_iters = _get_chunk_iter(chunk_size_padded1,
                                  (it - pt, ih - ph, iw - pw))
    _reduce = 'sum' if reduce == 'mean' else reduce
    for chunk_index in itertools.product(*chunk_iters):
        chunk_slice_in = _get_chunk_slice(chunk_index, chunk_size_padded2, chunk_size_padded1)  # noqa  # yapf: disable
        chunk_slice_out = _get_chunk_slice(chunk_index, chunk_size_padded1)
        # print(chunk_slice_out[2:], chunk_slice_in[2:])
        index_chunk = index[chunk_slice_in]
        patches_chunk = extract_patches_by_index(patches, index_chunk)
        # TODO: MAKE NaN HERE (patches_chunk)
        # TODO: hack!
        # predicate = patches_chunk.abs().reshape(*patches_chunk.shape[:-4], -1).max(dim=-1)[0] > 10000
        predicate = patches_chunk.abs().sum(dim=(-4, -3, -2, -1)) > 50000
        # print('P[Nan] =', predicate.to(dtype=torch.float32).mean().item())
        patches_chunk[predicate] = float('nan')
        permutation = ([0] + list(range(ndim + 1, 2 * (ndim + 1))) + list(range(1, ndim + 1)))  # noqa  # yapf: disable
        patches_chunk = patches_chunk.permute(*permutation).contiguous()
        output_chunk = _nan_fold3d(patches_chunk,
                                   stride=stride,
                                   reduce=_reduce,
                                   **kwargs)
        if reduce == 'mean':
            assert w_patches is not None
            w_patches_chunk = extract_patches_by_index(w_patches, index_chunk)
            w_patches_chunk = w_patches_chunk.permute(*permutation).contiguous()  # noqa  # yapf: disable
            count_chunk = _nan_fold3d(w_patches_chunk,
                                      stride=stride,
                                      reduce=_reduce,
                                      **kwargs)
            output_chunk /= count_chunk
            del w_patches_chunk, count_chunk
        output[chunk_slice_out] = output_chunk
        del index_chunk, patches_chunk, output_chunk, predicate
        if aggressive_gc >= 3:
            gc.collect()
            if aggressive_gc >= 4 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    if aggressive_gc >= 1:
        gc.collect()
        if aggressive_gc >= 2 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return output


def _fold3d_unchunked(input,
                      index,
                      kernel_size,
                      stride,
                      use_padding,
                      reduce,
                      **kwargs):
    patches = unfold3d(input, kernel_size, stride, use_padding)
    ndim = (patches.dim() - 2) // 2
    patches = extract_patches_by_index(patches, index)
    # TODO: MAKE NaN HERE (patches)
    # TODO: hack!
    # predicate = patches.abs().reshape(*patches.shape[:-4], -1).max(dim=-1)[0] > 10000
    # predicate = patches.abs().sum(dim=(-4,-3,-2,-1)) > 50000
    # print('P[Nan] =', predicate.to(dtype=torch.float32).mean().item())
    # patches[predicate] = float('nan')
    permutation = ([0] + list(range(ndim + 1, 2 * (ndim + 1))) +
                   list(range(1, ndim + 1)))
    patches = patches.permute(*permutation).contiguous()
    return _fold3d(patches,
                   stride=stride,
                   use_padding=use_padding,
                   reduce=reduce,
                   **kwargs)


def _nan_fold3d(input, stride, reduce, **kwargs):  # noqa
    # input dimensions (8D): n, c, kt, kh, kw, t', h', w'
    # output dimensions (5D): n, c, t, h, w
    if input.dim() != 8:
        raise ValueError('expects a 8D tensor as input')
    n, c, kt, kh, kw, t, h, w = input.shape
    st, sh, sw = stride = _triple(stride)
    if reduce == 'sum':
        output = _nan_fold3d_sum(input, stride)
    elif reduce == 'median':
        return _nan_fold3d_median(input, stride)
    else:
        raise ValueError(f'unknown reduction: {reduce}')
    return output


def _nan_fold3d_sum(input, stride):
    if input.dim() != 8:
        raise ValueError('expects a 8D tensor as input')
    n, c, kt, kh, kw, t, h, w = input.shape
    st, sh, sw = stride = _triple(stride)
    dt, dh, dw = kt // st, kh // sh, kw // sw
    if kt % st != 0:
        raise ValueError('kt should be divisible by st')
    if kh % sh != 0:
        raise ValueError('kh should be divisible by sh')
    if kw % sw != 0:
        raise ValueError('kw should be divisible by sw')
    pt, ph, pw = (kt - st, kh - sh, kw - sw)
    ot = st * (t - 1) + kt - 2 * pt
    oh = sh * (h - 1) + kh - 2 * ph
    ow = sw * (w - 1) + kw - 2 * pw
    # output = input.new_full(size=(dh * dw, n, c, oh, ow), fill_value=float('nan'))  # XXX: debug  # noqa
    output = input.new_zeros(size=(dt * dh * dw, n, c, ot, oh, ow))
    for it in range(kt):
        for ih in range(kh):
            for iw in range(kw):
                iit, iih, iiw = it // st, ih // sh, iw // sw
                ii = iit * dw * dh + iih * dw + iiw
                siit, siih, siiw = (kt - it - 1) // st, (kh - ih - 1) // sh, ( kw - iw - 1) // sw  # noqa  # yapf: disable
                output[ii, :, :, it % st::st, ih % sh::sh, iw % sw::sw] = input[:, :, it, ih, iw, siit:t - iit, siih:h - iih, siiw:w - iiw]  # noqa  # yapf: disable
    output = torch.nansum(output, dim=0)
    return output


def _nan_fold3d_median(input, stride):
    if input.dim() != 8:
        raise ValueError('expects a 8D tensor as input')
    n, c, kt, kh, kw, t, h, w = input.shape
    st, sh, sw = stride = _triple(stride)
    dt, dh, dw = kt // st, kh // sh, kw // sw
    if kt % st != 0:
        raise ValueError('kt should be divisible by st')
    if kh % sh != 0:
        raise ValueError('kh should be divisible by sh')
    if kw % sw != 0:
        raise ValueError('kw should be divisible by sw')

    pt, ph, pw = (kt - st, kh - sh, kw - sw)
    ot = st * (t - 1) + kt - 2 * pt
    oh = sh * (h - 1) + kh - 2 * ph
    ow = sw * (w - 1) + kw - 2 * pw
    output = input.new_zeros(size=(dt * dh * dw, n, c, ot, oh, ow))
    for it in range(kt):
        for ih in range(kh):
            for iw in range(kw):
                iit, iih, iiw = it // st, ih // sh, iw // sw
                ii = iit * dw * dh + iih * dw + iiw
                siit, siih, siiw = (kt - it - 1) // st, (kh - ih - 1) // sh, ( kw - iw - 1) // sw  # noqa  # yapf: disable
                output[ii, :, :, it % st::st, ih % sh::sh, iw % sw::sw] = input[:, :, it, ih, iw, siit:t - iit, siih:h - iih, siiw:w - iiw]  # noqa  # yapf: disable
    output = torch.nanmedian(output, dim=0)[0]

    return output
