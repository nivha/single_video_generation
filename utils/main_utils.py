import os
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
import numpy as np
from matplotlib import animation
import torchvision.io


import utils
from utils import scale_utils
from utils.image import img_read
import datetime


def now():
    return datetime.datetime.now()  # .strftime('%H:%d:%S')


def save_vid(box, frames_dir):
    N, C, T, H, W = box.shape
    for t in range(T):
        f = box[:, :, t, :, :]
        os.makedirs(f'{frames_dir}', exist_ok=True)
        plt.imsave(f'{frames_dir}/{t}.png', utils.image.tensor2npimg(f, vmin=-1, vmax=1))


def read_frames(frames_dir, start_frame, end_frame, frame_resizer=None, device='cuda', verbose=True, ext='png'):
    frames = []
    for fi in tqdm(range(start_frame, end_frame + 1), disable=not verbose):
        frame_path = os.path.join(frames_dir, f'{fi}.{ext}')

        x = img_read(frame_path, device=device)
        if frame_resizer is not None:
            x = frame_resizer(x)

        frames.append(x[:, :3, :, :])

    return frames


def read_original_video(frames_dir, start_frame, end_frame, max_size=None, device='cuda', verbose=True, ext='png'):
    # read frames to video
    first_frame_path = os.path.join(frames_dir, f'{start_frame}.{ext}')
    resizer = None
    if max_size is not None:
        resizer = scale_utils.get_frame_resizer(first_frame_path, max_size=max_size, target_shape=None) if max_size is not None else None
    frames = read_frames(frames_dir, start_frame, end_frame, resizer, device=device, verbose=verbose, ext=ext)
    frames = [utils.image.img_255_to_m11(f) for f in frames]
    orig_vid = torch.cat(frames, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
    return orig_vid


def torchvid2mp4(vid, path, fps=10):
    torchvision.io.write_video(path, utils.image.tensor2npimg(vid, to_numpy=False).permute(1, 2, 3, 0), fps=fps)


###############################################################################
#                   HTML Video (for notebooks)                                #
###############################################################################
def html_vid(vid, interval=100):
    """
        Use in jupyter:
        anim = html_vid(q_vid)
        HTML(anim.to_html5_video())
    """
    video = vid.detach().cpu().numpy()[0]
    video = np.transpose(video, (1, 2, 3, 0))
    video = (video + 1) / 2
    video = np.clip(video, 0, 1)
    fig = plt.figure()
    fig.tight_layout()
    im = plt.imshow(video[0, :, :, :])
    plt.axis('off')
    plt.close()  # this is required to not display the generated image

    def init():
        im.set_data(video[0, :, :, :])

    def animate(i):
        im.set_data(video[i, :, :, :])
        return im

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=video.shape[0], interval=interval)
    return anim


###############################################################################
#                   Analogies related (dynamic structure)                     #
###############################################################################
@torch.no_grad()
def get_raft_stuff(vid):
    import sys
    sys.path.append('./raft/core')
    from raft import RAFT
    # from utils import flow_viz
    from utils.utils import InputPadder
    from tqdm.auto import tqdm

    class A:
        def __contains__(self, key):
            return key in self.__dict__

    args = A()
    args.model = "./raft/models/raft-sintel.pth"
    if not os.path.exists(args.model):
        raise ValueError('Please download raft-sintel.pth model from https://github.com/princeton-vl/RAFT (and place it in ./raft/models/raft-sintel.pth)')
    args.small = False
    args.mixed_precision = False
    args.alternate_corr = False # for memory efficient (we don't need it)
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(vid.device)
    model.eval()

    flows = []
    for t in tqdm(range(vid.shape[2]-1)):
        image1 = vid[:,:,t,:,:].add(1).mul(255/2)
        image2 = vid[:,:,t+1,:,:].add(1).mul(255/2)
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
        fl = padder.unpad(flow_up).unsqueeze(2).data
        flows.append(fl)
    flows = torch.cat(flows, dim=2)
    return flows


def tokenize(vid, n_bins=20, strategy='kmeans', dither=False):
    from sklearn.preprocessing import KBinsDiscretizer
    n, c, t, h, w = vid.shape
    assert c == 1
    v = vid.reshape(n*c*t*h*w,1).cpu().numpy()
    enc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    enc.fit(v)
    if dither:
        vc = enc.transform(v + np.random.randn(*v.shape) * v.std() * 0.1)
    else:
        vc = enc.fit_transform(v)
    y = torch.from_numpy(vc).cuda().reshape(n, c, t, h, w)
    return y

