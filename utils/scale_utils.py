import numpy as np
import torch
from functools import partial

from utils.resize_right import resize
import utils.interp_methods as interp_methods
from utils.image import img_read


def round2even(x):
    if x % 2 != 0:
        return x + 1
    else:
        return x


def get_out_shapes(T, H, W, st, sh, sw):
    T_ = round2even(np.ceil(T * st).astype(int))
    H_ = round2even(np.ceil(H * sh).astype(int))
    W_ = round2even(np.ceil(W * sw).astype(int))
    return (T_, H_, W_)


def get_frame_resizer(first_frame_path, max_size=None, target_shape=None):
    first_frame = img_read(first_frame_path, device='cpu')
    _, C, H, W = first_frame.shape

    if max_size is not None:
        scale1 = min(max_size / H, 1)
        H_ = round2even(np.ceil(H * scale1).astype(int))
        W_ = round2even(np.ceil(W * scale1).astype(int))
        return partial(resize, scale_factors=scale1, out_shape=[H_, W_])
    elif target_shape is not None:
        scale1 = min(max(target_shape) / H, 1)
        return partial(resize, scale_factors=scale1, out_shape=target_shape)


def get_out_shape(scale, T, H, W):
    out_shapes_float = scale * torch.tensor([T, H, W])
    int_idxs = (out_shapes_float - out_shapes_float.round()).abs() < 1e-5
    out_shapes = out_shapes_float.ceil()
    out_shapes[int_idxs] = out_shapes_float[int_idxs].round()
    out_shapes = out_shapes.long().tolist()
    return out_shapes


def get_scales_out_shapes(T, H, W, downfactor, min_size):
    assert T >= min_size[0], f"min_size ({min_size[0]},{min_size[1]}) larger than original size ({T},{H},{W}) (it must be smaller)"

    scales = []
    out_shapes = []
    out_shape = [T, H, W]
    i = 0
    while True:
        scale = torch.tensor([downfactor[0], downfactor[1], downfactor[1]]).pow(i)

        if out_shape[0] <= min_size[0] and min(out_shape[1], out_shape[2]) <= min_size[1]:
            break

        out_shape = get_out_shape(scale, T, H, W)
        # fix out_shapes if needed
        if out_shape[0] < min_size[0]:
            scale[0] = (min_size[0] / out_shapes[-1][0]) * scales[-1][0]
        if min(out_shape[1], out_shape[2]) <= min_size[1]:
            min_idx = np.argmin([out_shape[1], out_shape[2]])
            ss = (min_size[1] / out_shapes[-1][1+min_idx]) * scales[-1][1]
            scale[1] = ss
            scale[2] = ss

        out_shape = get_out_shape(scale, T, H, W)

        scales.append(scale)
        out_shapes.append(out_shape)

        i += 1
        if i >= 50:
            raise Exception('something is wrong in compute scales')

    out_shapes = list(reversed(out_shapes))
    scales = torch.stack(scales).flipud()
    upscale_factors = torch.cat([torch.tensor([[1,1,1]]), scales[1:] / scales[:-1]])
    return scales.tolist(), upscale_factors.tolist(), out_shapes


def create_spatio_temporal_pyramid(video, downscales, out_shapes, device='cuda',
                                   interp_method=interp_methods.cubic,
                                   temporal_nearest=False, verbose=True):
    pyr = []
    for i in range(len(downscales)):

        if temporal_nearest:
            ooo = [out_shapes[i][0], out_shapes[i][1], out_shapes[i][2]] if out_shapes is not None else None
            # scale spatially
            sss = [1, downscales[i][1], downscales[i][2]]
            space_scaled = resize(video, scale_factors=sss, out_shape=ooo).to(device)
            # scale temporally
            sss = [downscales[i][0], 1, 1]
            scaled = resize(space_scaled, scale_factors=sss, out_shape=ooo, interp_method=interp_methods.box, antialiasing=False).to(device)
        else:
            sss = [downscales[i][0], downscales[i][1], downscales[i][2]]
            ooo = [out_shapes[i][0], out_shapes[i][1], out_shapes[i][2]] if out_shapes is not None else None
            scaled = resize(video, scale_factors=sss, out_shape=ooo, interp_method=interp_method).to(device)

        if verbose: print(scaled.shape)
        pyr.append(scaled)

    return pyr
