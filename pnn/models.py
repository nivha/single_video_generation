# __author__ = "Ben Feinstein (ben.feinstein@weizmann.ac.il)"

from typing import Iterable
import torch
from tqdm.auto import tqdm

from pnn import fold3d, fold_utils, nnlookup, patch_match, patch_match_fold3d

__all__ = ['PNN3D', 'PMPNN3D']
GB = 1024**3
BYTES_PER_VALUE = 4


def vid2patches(x, kernel_size, stride=1, use_padding=False):
    y = fold3d.unfold3d(x, kernel_size=kernel_size, stride=stride, use_padding=use_padding)
    y2d, y_size, y_ndim = fold_utils.view_as_2d(y)
    return y2d, y_size, y_ndim


def patches2vid(z2d, y_size, y_ndim, stride, use_padding, reduce, reduce_std, dists=None):
    z_ = fold_utils.view_2d_as(z2d, y_size, y_ndim)
    z = fold3d.fold3d(z_, stride=stride, use_padding=use_padding, reduce=reduce,
                      std=reduce_std, dists=dists)
    return z


class PNN3D:
    def __init__(self, kernel_size=(3, 7, 7), reduce='median', stride=1, use_padding=False, device='cuda',
                 full_batch_size=2 ** 31, faiss_batch_size=128, lookup='full', reduce_std=1.7,
                 differentiable=False,
                 ):
        self.kernel_size = kernel_size
        self.reduce = reduce
        self.reduce_std = reduce_std
        self.stride = stride
        self.use_padding = use_padding
        self.device = torch.device(device)
        self.faiss_batch_size = None  # faiss_batch_size
        self.full_batch_size = full_batch_size
        self.qk_lookup = self.qk_lookup_full if lookup == 'full' else self.qk_lookup_faiss

    def qk_lookup_faiss(self, q_vid, k_vid, w=None):
        assert w is None
        q, q_size, q_dim = vid2patches(q_vid, self.kernel_size, self.stride, self.use_padding)

        q = q.contiguous()
        # input('q')
        k, k_size, k_dim = vid2patches(k_vid, self.kernel_size, self.stride, self.use_padding)
        # k2, k_size, k_dim = self.vid2patches(torch.flip(k_vid, (2,)))
        # k = torch.cat((k, k2), 0)
        k = k.contiguous()
        # input('k')
        # print(now(), 'L2LOOKUP: creating index')
        index = faiss_search.create_index(k, cuda=self.device.type=='cuda')
        # input('index')
        # print(now(), 'L2LOOKUP: searching index')
        D, I = faiss_search.index_search(index, q, batch_size=self.faiss_batch_size)
        return D, I, q_size, q_dim

    @torch.no_grad()
    def qk_lookup_full(self, q_vid, k_vid, w=None):
        q, q_size, q_dim = vid2patches(q_vid, self.kernel_size, self.stride, self.use_padding)
        k, k_size, k_dim = vid2patches(k_vid, self.kernel_size, self.stride, self.use_padding)
        n_q = q.size(0)
        n_k = k.size(0)
        assert q.size(1) == k.size(1)
        # d = q.size(1)
        # print(d)
        # bytes_per_query = d * n_k
        # batch_size = min(self.max_mem // bytes_per_query, n_q)
        qbatch = (self.full_batch_size + n_k - 1) // n_k
        assert qbatch >= 1
        idxs = []
        for i in range((n_q + qbatch - 1) // qbatch):
            qi = q[i * qbatch:(i + 1) * qbatch]
            idxs.append(nnlookup.nn_lookup2d(qi, k, weight=w))
        I = torch.cat(idxs, dim=0)
        D = (q - k[I]).pow(2).sum(dim=1)
        return D.unsqueeze(1), I.unsqueeze(1), q_size, q_dim

    def compute_weights(self, q_vid, k_vid, alpha=1.):
        Dk, Ik, k_size, k_dim = self.qk_lookup_faiss(k_vid, q_vid)
        w = 1 / (alpha + Dk)#.pow(2))
        return w.transpose(0, 1), Dk, Ik, k_size, k_dim

    def kv_replace(self, v_vid, I):
        v_patches, v_size, v_dim = vid2patches(v_vid, self.kernel_size, self.stride, self.use_padding)
        # v_patches2, v_size, v_dim = self.vid2patches(torch.flip(v_vid, (2,)))
        # v_patches = torch.cat((v_patches, v_patches2), 0)
        v_patches = v_patches.contiguous()
        r_patches = v_patches[I.squeeze()]
        return r_patches

    def reconstruct_qkv(self, q_vid, k_vid, v_vid=None, bidi=False, alpha=1.):
        assert (not bidi) or self.qk_lookup == self.qk_lookup_full, 'bidi must be used with lookup=full'
        w = Dk = Ik = k_size = k_dim = None
        if bidi:
            w, Dk, Ik, k_size, k_dim = self.compute_weights(q_vid, k_vid, alpha)
        D, I, q_size, q_dim = self.qk_lookup(q_vid, k_vid, w=w)

        if v_vid is None:
            v_vid = k_vid
        r_patches = self.kv_replace(v_vid, I)

        r_vid = patches2vid(r_patches, q_size, q_dim, self.stride, self.use_padding, self.reduce, self.reduce_std, dists=D)
        # diff = -1# (r_vid - v_vid).abs().sum().item()
        return r_vid, D, I, Ik, k_size, q_size

    def reconstruct_qkv_diff(self, q_vid, k_vid, v_vid=None, bidi=False, alpha=1.):
        assert (not bidi) or self.qk_lookup == self.qk_lookup_full, 'bidi must be used with lookup=full'
        w = Dk = Ik = k_size = k_dim = None
        if bidi:
            w, Dk, Ik, k_size, k_dim = self.compute_weights(q_vid, k_vid, alpha)
        D, I, q_size, q_dim = self.qk_lookup(q_vid, k_vid, w=w)

        if v_vid is None:
            v_vid = k_vid
        # replacement of self.kv_replace(v_vid, I)
        q_patches, q_size, q_dim = vid2patches(q_vid, self.kernel_size, self.stride, self.use_padding)
        v_patches, v_size, v_dim = vid2patches(v_vid, self.kernel_size, self.stride, self.use_padding)
        v_patches = v_patches.contiguous()
        # r_patches = v_patches[I.squeeze()]
        # quantized_latents = latents + (quantized_latents - latents).detach()
        q_patches = q_patches + (v_patches[I.squeeze()] - q_patches).detach()

        r_vid = patches2vid(q_patches, q_size, q_dim, self.stride, self.use_padding, self.reduce, self.reduce_std, dists=D)
        return r_vid, D, I, Ik, k_size, q_size



class PMPNN3D:
    def __init__(self,
                 kernel_size=(3, 7, 7),
                 reduce='median',
                 stride=1,
                 use_padding=False,
                 chunk_size=50,
                 reduce_std=None,
                 steps=(8, 4, 1),
                 radii=1,
                 device='cuda',
                 device_fold='cuda',
                 scale_origin=False,
                 dist_fn=patch_match.l2_dist,
                 with_noisy=True,
                 with_noisy_neighbors=True,
                 aggressive_gc=0,
                 verbose=False,
                 debug=False):
        if isinstance(steps, Iterable) or isinstance(radii, Iterable):
            if not isinstance(steps, Iterable):
                steps = [steps] * len(radii)
            if not isinstance(radii, Iterable):
                radii = [radii] * len(steps)
            assert isinstance(steps, Iterable) and isinstance(radii, Iterable)
            assert len(steps) == len(radii)
        self.kernel_size = kernel_size
        self.reduce = reduce
        self.reduce_std = reduce_std
        self.stride = stride
        self.use_padding = use_padding
        self.chunk_size = chunk_size
        self.steps = steps
        self.radii = radii
        self.device = device
        self.device_fold = device_fold
        self.scale_origin = scale_origin
        self.dist_fn = dist_fn
        self.with_noisy = with_noisy
        self.with_noisy_neighbors = with_noisy_neighbors
        self.aggressive_gc = aggressive_gc
        self.verbose = verbose
        self.debug = debug

    def vid2patches(self, vid):
        return patch_match_fold3d.unfold3d(vid,
                                           kernel_size=self.kernel_size,
                                           stride=self.stride,
                                           use_padding=self.use_padding)

    def create_out_vid(self, vid, index, chunk_size=None):
        vid = vid.to(device=self.device_fold)
        index = index.to(device=self.device_fold)
        out_vid = patch_match_fold3d.fold3d(input=vid,
                                            index=index,
                                            kernel_size=self.kernel_size,
                                            chunk_size=chunk_size,
                                            stride=self.stride,
                                            use_padding=self.use_padding,
                                            reduce=self.reduce,
                                            std=self.reduce_std,
                                            aggressive_gc=self.aggressive_gc)
        out_vid = out_vid.to(device=self.device)  # TODO ?
        return out_vid

    @torch.no_grad()
    def qk_lookup(self,
                  q_patches,
                  k_patches,
                  k_weights=None,
                  k_bias=None,
                  old_index=None,
                  old_k_size=None,
                  compute_dist=False):
        # assert k_weights is None
        ndim = (q_patches.dim() - 2) // 2
        q_size = q_patches.shape[1:ndim + 1]
        k_size = k_patches.shape[1:ndim + 1]
        k_size_for_origin = k_size if self.scale_origin else None
        orig = patch_match.get_orig(q_size,
                                    k_size=k_size_for_origin,
                                    device=self.device)
        if old_index is not None:
            old_index = old_index.to(orig)
            nnf = patch_match.get_resized_nnf(orig=orig,
                                              k_size=k_size,
                                              old_index=old_index,
                                              old_k_size=old_k_size)
        else:
            nnf = patch_match.get_initial_nnf(orig, k_size)

        hist = [] if self.debug else None
        for steps, radii in tqdm(zip(self.steps, self.radii), disable=not self.verbose, total=len(self.steps)):
        # for steps, radii in zip(self.steps, self.radii):
            out = patch_match.patch_match_(
                nnf,
                q_patches,
                k_patches,
                orig,
                steps=steps,
                radii=radii,
                num_noisy=1 if self.with_noisy else 0,
                k_weights=k_weights,
                k_bias=k_bias,
                chunk_size=self.chunk_size,
                dist_fn=self.dist_fn,
                with_noisy_neighbors=self.with_noisy_neighbors,
                aggressive_gc=self.aggressive_gc,
                debug=self.debug)

            if self.debug:
                nnf = out[0]
                hist.extend(out[1])
            else:
                nnf = out
            del out

        index = patch_match.nnf_to_index(nnf, orig, k_size)

        if compute_dist:
            dist = patch_match.compute_dist(
                q_patches,
                k_patches,
                index=index,
                chunk_size=self.chunk_size,
                aggressive_gc=self.aggressive_gc)
        else:
            dist = -1

        return index, dist, nnf, orig, hist

    def compute_weights(self,
                        q_patches,
                        k_patches,
                        alpha,
                        old_k_index=None,
                        old_q_size=None):
        # reverse search
        k_index, k_dist, _, _, _ = self.qk_lookup(q_patches=k_patches,
                                                  k_patches=q_patches,
                                                  old_index=old_k_index,
                                                  old_k_size=old_q_size,
                                                  compute_dist=True)
        k_weights = 1. / (alpha + k_dist)
        return k_weights, k_index

    @torch.no_grad()
    def reconstruct_qkv(self,
                        q_vid,
                        k_vid,
                        v_vid=None,
                        bidi=False,
                        alpha=None,
                        k_weights=None,
                        k_bias=None,
                        old_index=None,
                        old_k_size=None,
                        old_k_index=None,
                        old_q_size=None,
                        memory_efficient=False,
                        compute_dist=False):
        assert v_vid is None or k_vid.shape[2:] == v_vid.shape[2:], f"keys and values must have the same size! (k:{k_vid.shape[2:]}, v:{v_vid.shape[2:]})"

        # assert (not bidi) or self.qk_lookup == self.qk_lookup_full, 'bidi must be used with lookup=full'  # noqa
        # assert bidi is False
        q_vid = q_vid.to(device=self.device)
        k_vid = k_vid.to(device=self.device)
        q_patches = self.vid2patches(q_vid)
        k_patches = self.vid2patches(k_vid)

        if bidi:
            k_weights, k_index = self.compute_weights(q_patches,
                                                      k_patches,
                                                      alpha,
                                                      old_k_index=old_k_index,
                                                      old_q_size=old_q_size)
        else:
            k_index = None
        index, dist, _, _, hist = self.qk_lookup(q_patches,
                                                 k_patches,
                                                 k_weights=k_weights,
                                                 k_bias=k_bias,
                                                 old_index=old_index,
                                                 old_k_size=old_k_size,
                                                 compute_dist=compute_dist)
        if v_vid is None:
            v_vid = k_vid

        if memory_efficient:
            r_vid = self.create_out_vid(v_vid, index, chunk_size=self.chunk_size)  # noqa  # yapf: disable
        else:
            r_vid = self.create_out_vid(v_vid, index)

        ndim = (q_patches.dim() - 2) // 2
        q_size = q_patches.shape[1:ndim + 1]
        k_size = k_patches.shape[1:ndim + 1]

        if self.debug:
            return r_vid, dist, index, k_index, k_size, q_size, hist
        else:
            return r_vid, dist, index, k_index, k_size, q_size
