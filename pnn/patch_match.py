# __author__ = "Ben Feinstein (ben.feinstein@weizmann.ac.il)"

import itertools
import gc
import warnings
from collections.abc import Iterable
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _ntuple

__all__ = [
    'tensor2patchesnd',
    'get_initial_nnf',
    'get_resized_nnf',
    'get_orig',
    'l2_dist',
    'l1_dist',
    'patch_match_',
    'patch_match',
    'extract_patches_by_index',
    'compute_dist',
]


def tensor2patchesnd(input, kernel_size, stride=1, use_padding=False):
    assert input.size(0) == 1
    ndim = input.dim() - 2
    kernel_size = _ntuple(ndim)(kernel_size)
    stride = _ntuple(ndim)(stride)
    if use_padding:
        input = F.pad(input, _get_padding_unfold(kernel_size))
    patches = input.movedim(1, -1).contiguous()
    for dim, (ksz, st) in enumerate(zip(kernel_size, stride), 1):
        patches = patches.unfold(dim, ksz, st)
    return patches


def get_initial_nnf(orig, k_size):
    return get_random_nnf(orig, k_size, n=1)


def get_resized_nnf(orig, k_size, old_index, old_k_size):
    modes = {1: 'linear', 2: 'bilinear', 3: 'trilinear'}
    ndim = orig.size(1)
    q_size = orig.shape[2:]
    assert ndim in modes
    assert orig.size(0) == old_index.size(0) == 1
    assert orig.size(1) == old_index.size(1)
    assert ndim == orig.dim() - 2 == old_index.dim() - 2
    assert ndim == len(k_size) == len(old_k_size)
    index = F.interpolate(old_index,
                          size=q_size,
                          mode=modes[ndim],
                          align_corners=True)
    for dim, (oksz, ksz) in enumerate(zip(old_k_size, k_size)):
        index[:, dim].mul_((ksz - 1) / max(1, oksz - 1))
        index[:, dim].clip_(0, ksz - 1)
    if index.is_floating_point():
        index.round_()
    nnf = index - orig
    return nnf


def get_orig(q_size, k_size=None, dtype=None, device=None):
    if k_size is None:
        k_size = q_size
    else:
        assert len(q_size) == len(k_size)
        warnings.warn('passing k_size to get_orig is deprecated',
                      DeprecationWarning,
                      stacklevel=2)
    ndim = len(q_size)
    to = dict(dtype=dtype, device=device)
    origs = [
        torch.linspace(start=0, end=k_size[dim] - 1, steps=q_size[dim], **to)
        for dim in range(ndim)
    ]
    orig = torch.stack(torch.meshgrid(*origs), dim=0).unsqueeze(0)
    return orig


def l2_dist(a, b, **kwargs):
    c = a - b
    c = c.pow_(2)
    c = c.sum(**kwargs)
    return c


def l1_dist(a, b, **kwargs):
    c = a - b
    c = c.abs_()
    c = c.sum(**kwargs)
    return c


@torch.no_grad()
def patch_match_(nnf,
                 q_patches,
                 k_patches,
                 orig,
                 steps=1,
                 radii=1,
                 num_noisy=1,
                 dist_fn=l2_dist,
                 k_weights=None,
                 k_bias=None,
                 chunk_size=-1,
                 with_noisy_neighbors=True,
                 index_dtype=torch.long,
                 aggressive_gc=0,
                 debug=False):
    ndim = nnf.size(1)
    pdims = [-i for i in range(1, ndim + 2)]
    k_size = k_patches.shape[1:1 + ndim]
    q_size = q_patches.shape[1:1 + ndim]
    assert q_patches.size(0) == k_patches.size(0) == 1
    assert len(k_size) == len(q_size) == ndim
    assert orig.size(0) == 1
    assert nnf.shape[1:] == orig.shape[1:]
    assert aggressive_gc == 0
    # assert isinstance(step, int) and step >= 1

    chunk_size = _get_chunk_size(chunk_size, q_size, k_size)
    extended_nnf = [nnf]
    extended_nnf.append(get_random_nnf(orig, k_size, n=1))  # XXX useless!
    if num_noisy:
        extended_nnf.append(get_noisy_nnf(nnf, radii=radii, n=num_noisy))
    neighbors_nnf = get_neighbor_nnf(nnf, steps=steps)
    extended_nnf.append(neighbors_nnf)
    if with_noisy_neighbors:
        extended_nnf.append(get_noisy_nnf(neighbors_nnf, radii=radii, n=neighbors_nnf.size(0)))  # yapf: disable  # noqa
    extended_nnf = torch.cat(extended_nnf, dim=0)  # yapf: disable  # noqa
    if debug:
        hist = torch.zeros(size=(extended_nnf.size(0),), dtype=torch.long, device=extended_nnf.device)  # yapf: disable  # noqa  # TODO: debug
    chunk_iters = [range((qsz + csz - 1) // csz) for qsz, csz in zip(q_size, chunk_size)]  # yapf: disable  # noqa
    for chunk_index in itertools.product(*chunk_iters):
        chunk_slice = _get_chunk_slice(chunk_index, chunk_size)
        extended_nnf_chunk = extended_nnf[chunk_slice]
        orig_chunk = orig[chunk_slice]
        index_chunk = nnf_to_index(extended_nnf_chunk, orig_chunk, k_size,
                                   dtype=index_dtype)  # yapf: disable  # noqa
        q_patches_chunk = q_patches[chunk_slice[1:]]
        o_patches_chunk = extract_patches_by_index(k_patches, index_chunk)
        dists_chunk = dist_fn(q_patches_chunk, o_patches_chunk, dim=pdims)
        if k_weights is not None:
            k_weights_chunk = extract_patches_by_index(k_weights, index_chunk)
            dists_chunk *= k_weights_chunk
            del k_weights_chunk
        if k_bias is not None:
            k_bias_chunk = extract_patches_by_index(k_bias, index_chunk)
            dists_chunk += k_bias_chunk
            del k_bias_chunk
        # del q_patches_chunk, o_patches_chunk, orig_chunk, index_chunk  # yapf: disable  # noqa
        del q_patches_chunk, o_patches_chunk, extended_nnf_chunk  # , orig_chunk, index_chunk  # yapf: disable  # noqa
        best_chunk = dists_chunk.argmin(dim=0, keepdims=True)
        if debug:
            hist += torch.bincount(best_chunk.flatten(), minlength=hist.size(0))
        best_chunk = best_chunk.unsqueeze(1).expand(-1, ndim, *([-1] * ndim))
        # best_nnf_chunk = torch.gather(extended_nnf_chunk, 0, best_chunk)  # TODO: remove  # yapf: disable  # noqa
        best_nnf_chunk = torch.gather(index_chunk.type_as(nnf), 0, best_chunk).sub_(orig_chunk)  # yapf: disable  # noqa
        nnf[chunk_slice].copy_(best_nnf_chunk)
        # del dists_chunk, best_chunk, best_nnf_chunk, extended_nnf_chunk  # yapf: disable  # noqa
        del dists_chunk, best_chunk, best_nnf_chunk, orig_chunk, index_chunk  # , extended_nnf_chunk  # yapf: disable  # noqa
        # if aggressive_gc >= 3:
        #     gc.collect()
        #     if aggressive_gc >= 4 and torch.cuda.is_available():
        #         torch.cuda.empty_cache()

    del neighbors_nnf, extended_nnf
    # if aggressive_gc >= 1:
    #     gc.collect()
    #     if aggressive_gc >= 2 and torch.cuda.is_available():
    #         torch.cuda.empty_cache()

    if debug:
        return nnf, hist
    return nnf


def patch_match(nnf, *args, **kwargs):
    nnf = nnf.clone()
    return patch_match_(nnf, *args, **kwargs)


def extract_patches_by_index(patches, index):
    return patches[tuple([0, *index.unbind(dim=1)])]


def get_random_nnf(orig, k_size, n=1):
    assert orig.size(0) == 1 or orig.size(0) == n
    assert orig.size(1) == len(k_size)
    rand_kwargs = dict(size=(n, 1, *orig.shape[2:]), device=orig.device, dtype=orig.dtype)  # yapf: disable  # noqa
    rand_nnf = torch.cat([torch.randint(low=0, high=ksz, **rand_kwargs) for ksz in k_size], dim=1)  # yapf: disable  # noqa
    rand_nnf -= orig
    return rand_nnf


def get_noisy_nnf(nnf, radii=1, n=1):
    assert nnf.size(0) == 1 or nnf.size(0) == n
    ndim = nnf.size(1)
    radii = _ntuple(ndim)(radii)
    rand_kwargs = dict(size=(n, 1, *nnf.shape[2:]), device=nnf.device, dtype=nnf.dtype)  # yapf: disable  # noqa
    noisy_nnf = torch.cat([torch.randint(low=-nsz, high=nsz + 1, **rand_kwargs) for nsz in radii], dim=1)  # yapf: disable  # noqa
    noisy_nnf += nnf
    return noisy_nnf


def get_neighbor_nnf(nnf, steps=1):
    ndim = nnf.size(1)
    steps = _ntuple(ndim)(steps)
    neighbors_nnf = nnf.new_zeros(2 * ndim, *nnf.shape[1:])
    for (dim, step), i in itertools.product(enumerate(steps), [0, 1]):
        if step >= nnf.size(dim + 2):
            continue
        slc_in = [0, slice(None), *(slice(None) for j in range(ndim))]
        slc_out = [
            2 * dim + i,
            slice(None), *[slice(None) for j in range(ndim)]
        ]  # noqa
        slc_in[dim + 2] = slice((1 - i) * step, nnf.size(dim + 2) - i * step)
        slc_out[dim + 2] = slice(i * step, nnf.size(dim + 2) - (1 - i) * step)
        neighbors_nnf[tuple(slc_out)] = nnf[tuple(slc_in)]
    return neighbors_nnf


def nnf_to_index(nnf, orig, size, dtype=torch.long):
    ndim = nnf.size(1)
    assert len(size) == ndim
    assert orig.size(0) == 1
    assert nnf.shape[1:] == orig.shape[1:]
    index = nnf.clone()
    orig = orig.to(dtype=index.dtype, device=index.device)
    for dim in range(ndim):
        index[:, dim] += orig[:, dim]
        index[:, dim].clip_(0, size[dim] - 1)
    if index.is_floating_point():
        index = index.round_()
    return index.to(dtype=dtype)


def compute_dist(q_patches, k_patches, index, chunk_size, dist_fn=l2_dist,
                 aggressive_gc=0):
    assert aggressive_gc == 0
    assert q_patches.dim() == k_patches.dim()
    ndim = (q_patches.dim() - 2) // 2
    pdims = [-i for i in range(1, ndim + 2)]
    assert q_patches.shape[ndim + 1:] == k_patches.shape[ndim + 1:]
    q_size = q_patches.shape[1:ndim + 1]
    k_size = k_patches.shape[1:ndim + 1]  # noqa
    chunk_size = _ntuple(ndim)(chunk_size)
    chunk_iters = [range((qsz + csz - 1) // csz) for qsz, csz in zip(q_size, chunk_size)]  # yapf: disable  # noqa
    dist = q_patches.new_zeros(size=q_patches.shape[:ndim + 1])
    for chunk_index in itertools.product(*chunk_iters):
        chunk_slice = _get_chunk_slice(chunk_index, chunk_size)
        index_chunk = index[chunk_slice]
        q_patches_chunk = q_patches[chunk_slice[1:]]
        o_patches_chunk = extract_patches_by_index(k_patches, index_chunk)
        dist[chunk_slice[1:]] = dist_fn(q_patches_chunk, o_patches_chunk, dim=pdims)  # yapf: disable  # noqa
        del chunk_slice, index_chunk, q_patches_chunk, o_patches_chunk
        # if aggressive_gc >= 3:
        #     gc.collect()
        #     if aggressive_gc >= 4 and torch.cuda.is_available():
        #         torch.cuda.empty_cache()

    # if aggressive_gc >= 1:
    #     gc.collect()
    #     if aggressive_gc >= 2 and torch.cuda.is_available():
    #         torch.cuda.empty_cache()

    return dist


def _get_padding_unfold(kernel_size):
    padding = []
    for ksz in kernel_size[::-1]:
        padding.extend([ksz - 1, ksz - 1])
    return tuple(padding)


def _get_chunk_slice(chunk_index, chunk_size):
    assert len(chunk_index) == len(chunk_size)
    chunk_slice = [slice(None), slice(None)]
    for cidx, csz in zip(chunk_index, chunk_size):
        chunk_slice.append(slice(cidx * csz, (cidx + 1) * csz))
    chunk_slice = tuple(chunk_slice)
    return chunk_slice


def _get_chunk_size(chunk_size, q_size, k_size):
    ndim = len(k_size)
    chunk_size = _ntuple(ndim)(chunk_size)
    chunk_size = tuple([
        min(q_size[dim], chunk_size[dim])
        if chunk_size[dim] != -1 else k_size[dim] for dim in range(ndim)
    ])
    return chunk_size
