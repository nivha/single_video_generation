import os
import argparse
import ast
import threadpoolctl
thread_limit = threadpoolctl.threadpool_limits(limits=8)


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
    parser = argparse.ArgumentParser(description='')

    # general parameters
    parser.add_argument('--gpu', default='0', help='which gpu to use')
    parser.add_argument('--results_dir', default='./results/generation', help='where to save the results')

    # VGPNN params
    parser.add_argument('--frames_dir', help='path to directory with frames (see README for video format)')
    parser.add_argument('--start_frame', default=1, type=int, help='use this #d.png as first frame in the video')
    parser.add_argument('--end_frame', default=15, type=int, help='use this #d.png as last frame in the video')
    parser.add_argument('--max_size', default=144, type=int, help='maximal spatial size (same as in SinGAN)')
    parser.add_argument('--min_size', default='(3,15)', type=str2list, help='shape of the coarsest pyramid level (T,HW)')
    parser.add_argument('--downfactor', default='(0.85,0.85)', type=str2list, help='downscale factor of the pyramid (T,HW)')
    parser.add_argument('--J', default=5, type=int, help='number of EM-iterations within level')
    parser.add_argument('--J_start_from', default=1, type=int, help='all levels before this will us J=1')
    parser.add_argument('--kernel_size', default='(3,7,7)', type=str2list, help='')
    parser.add_argument('--sthw', default='(0.5,1,1)', type=str2list, help='scaling factor of the output w.r.t to the input video')
    parser.add_argument('--reduce', default='median', help='patchmatch fold (vote) function')
    parser.add_argument('--vgpnn_type', default='pm', choices=['pm', 'vanilla'], help='vanilla means exhaustive search, pm=patchmatch')
    parser.add_argument('--use_noise', default='true', type=str2bool, help='add noise at coarsest level')
    parser.add_argument('--noise_amp', default=5, type=float, help='amplitude of noise to add to coarsest level')
    parser.add_argument('--verbose', default='true', type=str2bool, help='')
    parser.add_argument('--save_intermediate', default='true', type=str2bool, help='save intermediate results')

    args = parser.parse_args()

    args.save_intermediate_path = ''
    if args.save_intermediate:
        args.save_intermediate_path = args.results_dir

    return args


if __name__=="__main__":
    args = get_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    import torchvision.io
    from utils import image
    from utils.main_utils import now, save_vid
    from html_results.html_utils import create_results_html
    import vgpnn
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    VGPNN, orig_vid = vgpnn.get_vgpnn(
        frames_dir=args.frames_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        device=args.device,
        max_size=args.max_size,
        min_size=args.min_size,
        downfactor=args.downfactor,
        J=args.J,
        J_start_from=args.J_start_from,
        kernel_size=args.kernel_size,
        sthw=args.sthw,
        reduce=args.reduce,
        vgpnn_type=args.vgpnn_type,
    )

    print('Pyramid:')
    for i, outs in enumerate(VGPNN.r_out_shapes):
        assert outs[0]>=VGPNN.Ks[i][0], 'kernel size must be bigger than size of video'
        print(f'LEVEL {i:3} SIZE {str(outs):17} KERNEL {str(VGPNN.Ks[i]):15} J {VGPNN.Js[i]}')

    if args.use_noise:
        z = torch.randn_like(VGPNN.q0).to(args.device)
    else:
        z = torch.zeros_like(VGPNN.q0).to(args.device)

    time_start = now()
    vgpnn_out = VGPNN.forward(
        VGPNN.q0, noises_dict={0: z}, noises_amps={0: args.noise_amp},
        n_stages=None, return_input=False,
        save_dir=args.save_intermediate_path,
        verbose=args.verbose,
    )
    print('TOTAL TIME:', now() - time_start)

    # Save results (frames and to mp4)
    save_vid(vgpnn_out, f'{args.results_dir}/frames')
    vid_ = image.tensor2npimg(vgpnn_out, to_numpy=False).permute(1, 2, 3, 0)
    torchvision.io.write_video(f'{args.results_dir}/output.mp4', vid_, fps=10)
    print('saved results to:', args.results_dir)

    if args.save_intermediate:
        # show all intermediate results in HTML format
        base_dir = args.results_dir
        html_filename = 'result.html'
        vid_folders = [
            (f'scale_{s}', [f'{s}/v', f'{s}/q', f'{s}/r'], 'green', 'flex', 30, True) for s in reversed(range(VGPNN.n_stages))
        ]
        create_results_html(vid_folders, base_dir, html_filename, frame_rate=10)

