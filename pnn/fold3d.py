# __author__ = "Ben Feinstein (ben.feinstein@weizmann.ac.il)"

import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn.modules.utils import _triple

__all__ = ['unfold3d', 'fold3d']


def unfold3d(input, kernel_size, stride=1, use_padding=True):
    # input dimensions (5D): n, c, t, h, w
    # kernel: kt, kh, kw
    # output dimensions (8D): n, c, kt, kh, kw, t', h', w'
    # output can be viewed as: n, c * kt * kh * kw, t' * h' * w'
    if input.dim() != 5:
        raise ValueError('expects a 5D tensor as input')
    n, c, t, h, w = input.size()
    kt, kh, kw = kernel_size = _triple(kernel_size)
    st, sh, sw = stride = _triple(stride)
    if use_padding:
        pt, ph, pw = padding = (kt - st, kh - sh, kw - sw)
    else:
        pt, ph, pw = padding = (0, 0, 0)
    ot = (t + 2 * pt - kt) // st + 1
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1

    output = F.pad(input, (pw, pw, ph, ph, pt, pt))
    output = output.unfold(2, kt, st).unfold(3, kh, sh).unfold(4, kw, sw)
    output = output.permute(0, 1, 5, 6, 7, 2, 3, 4)
    assert output.shape == (n, c, kt, kh, kw, ot, oh, ow)
    # output = output.view(n, c, kt, kh, kw, ot, oh, ow)
    return output


def fold3d(input, stride=1, use_padding=True, reduce='mean', std=1.7, dists=None):  # noqa
    # input dimensions (8D): n, c, kt, kh, kw, t', h', w'
    # output dimensions (5D): n, c, t, h, w
    if input.dim() != 8:
        raise ValueError('expects a 8D tensor as input')
    n, c, kt, kh, kw, t, h, w = input.shape
    st, sh, sw = stride = _triple(stride)
    # TODO: MAKE NaN HERE (input)
    if reduce == 'sum':
        output = _fold3d_sum(input, stride, use_padding)
    elif reduce == 'median':
        return _fold3d_median(input, stride, use_padding)
    elif reduce == 'mean':
        weights = _get_weights_fold3d_mean(input, kt, kh, kw)
        output = _fold3d_sum(input, stride, use_padding)
        if use_padding:
            norm = weights[:, :, ::st, ::sh, ::sw, :, :, :].sum()
        else:
            weights = weights.expand(1, 1, kt, kh, kw, t, h, w)
            norm = _fold3d_sum(weights, stride, use_padding)
            norm[norm == 0] = 1
        output = output / norm
    elif reduce == 'weighted_mean':
        weights = _get_weights_fold3d_weighted_mean(input, kt, kh, kw, st, sh, sw, std)
        output = _fold3d_sum(input * weights, stride, use_padding)
        if use_padding and (st, sh, sw) == (1, 1, 1):
            norm = weights.sum()
        else:
            weights = weights.expand(1, 1, kt, kh, kw, t, h, w)
            norm = _fold3d_sum(weights, stride, use_padding)
            norm[norm == 0] = 1
        output = output / norm
    elif reduce == 'wexler_mean':
        assert dists is not None
        weights = _get_weights_fold3d_wexler_mean(input, dists, std)
        output = _fold3d_sum(input * weights, stride, use_padding)
        weights = weights.expand(n, 1, kt, kh, kw, t, h, w)
        norm = _fold3d_sum(weights, stride, use_padding)
        norm[norm == 0] = 1
        output = output / norm
    else:
        raise ValueError(f'unknown reduction: {reduce}')
    return output


def _fold3d_sum(input, stride, use_padding):
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
    if use_padding:
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
                    siit, siih, siiw = (kt - it - 1) // st, (kh - ih - 1) // sh, (kw - iw - 1) // sw
                    output[ii, :, :, it % st::st, ih % sh::sh, iw % sw::sw] = input[:, :, it, ih, iw, siit:t - iit, siih:h - iih, siiw:w - iiw]  # noqa
        output = torch.sum(output, dim=0)
    else:
        if st != 1 or sh != 1 or sw != 1:
            raise NotImplementedError('fold3d_sum with use_padding==False and stride!=1 is not implemented')  # noqa
        ot = st * (t - 1) + kt
        oh = sh * (h - 1) + kh
        ow = sw * (w - 1) + kw
        output = input.new_zeros(size=(dt * dh * dw, n, c, ot, oh, ow))  # noqa
        # output = input.new_zeros(size=(dh * dw, n, c, oh, ow))  # XXX: debug
        for it in range(kt):
            for ih in range(kh):
                for iw in range(kw):
                    iit, iih, iiw = it // st, ih // sh, iw // sw
                    ii = iit * dw * dh + iih * dw + iiw
                    output[ii, :, :, it:t + it:st, ih:h + ih:sh, iw:w + iw:sw] = input[:, :, it, ih, iw, :, :, :]  # noqa
        output = torch.sum(output, dim=0)
    return output


def _fold3d_median(input, stride, use_padding):
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

    if use_padding:
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
                    siit, siih, siiw = (kt - it - 1) // st, (kh - ih - 1) // sh, (kw - iw - 1) // sw
                    output[ii, :, :, it % st::st, ih % sh::sh, iw % sw::sw] = input[:, :, it, ih, iw, siit:t - iit, siih:h - iih, siiw:w - iiw]  # noqa
        output = torch.nanmedian(output, dim=0)[0]  # TODO: median -> nanmedian

    else:
        if not hasattr(torch, 'nanmedian'):
            raise RuntimeError('fold3d_median with use_padding==False depends on torch.nanmedian()')  # noqa
            # pass  # XXX: debug
        if st != 1 or sh != 1 or sw != 1:
            raise NotImplementedError('fold3d_median with use_padding==False and stride!=1 is not implemented')  # noqa
        ot = st * (t - 1) + kt
        oh = sh * (h - 1) + kh
        ow = sw * (w - 1) + kw
        output = input.new_full(size=(dt * dh * dw, n, c, ot, oh, ow), fill_value=float('nan'))  # noqa
        # output = input.new_zeros(size=(dh * dw, n, c, oh, ow))  # XXX: debug
        for it in range(kt):
            for ih in range(kh):
                for iw in range(kw):
                    iit, iih, iiw = it // st, ih // sh, iw // sw
                    ii = iit * dw * dh + iih * dw + iiw
                    output[ii, :, :, it:t + it:st, ih:h + ih:sh, iw:w + iw:sw] = input[:, :, it, ih, iw, :, :, :]  # noqa
        output = torch.nanmedian(output, dim=0)[0]

    return output


def _get_weights_fold3d_mean(input, kt, kh, kw):
    weights = input.new_ones(size=(kt, kh, kw))
    return weights.view(1, 1, kt, kh, kw, 1, 1, 1)


def _get_weights_fold3d_weighted_mean(input, kt, kh, kw, st, sh, sw, std):
    to = {'device': input.device, 'dtype': input.dtype}
    gt = st * torch.linspace(-1, 1, kt, **to)
    gh = sh * torch.linspace(-1, 1, kh, **to)
    gw = sw * torch.linspace(-1, 1, kw, **to)
    nt = torch.exp(-0.5 * (gt / std).pow(2))
    nh = torch.exp(-0.5 * (gh / std).pow(2))
    nw = torch.exp(-0.5 * (gw / std).pow(2))
    weights = torch.einsum('t,h,w->thw', nt, nh, nw)
    return weights.view(1, 1, kt, kh, kw, 1, 1, 1)


def _get_weights_fold3d_wexler_mean(input, dists, std):
    n = input.shape[0]
    t, h, w = input.shape[-3:]
    dists = dists.view(n, t, h, w)
    weights = torch.exp(-dists / (2 * std ** 2))
    return weights.view(n, 1, 1, 1, 1, t, h, w)
