# VGPNN: Diverse Generation from a Single Video Made Possible

[Project](https://nivha.github.io/vgpnn) | [Arxiv](https://arxiv.org/abs/2109.08591) 
#### Pytorch implementation of the paper: "Diverse Generation from a Single Video Made Possible"
(To appear in ECCV 2022)

## Code

### Data

Please download the videos from this [Dropbox Videos Folder](https://www.dropbox.com/sh/gbt6rnm3b7afdk5/AAAXqWZGt0LUEjnYy-5-khyXa?dl=0) into ```./data``` folder.

Note that a video is represented as a directory with PNG files in the format \<frame number\>.png

For example:
```
some/path/my_video/
   1.png
   2.png
   3.png
   ...
```

### Video generation

To generate a new sample from a single video

```
python run_generation.py --gpu 0 --frames_dir <path to frames dir> --start_frame <number of first frame> --end_frame <number of last frame>
```

Examples:

```
python run_generation.py --frames_dir=data/air_balloons --start_frame=66 --end_frame=80
python run_generation.py --frames_dir=data/air_balloons --start_frame=66 --end_frame=165 --max_size=360 --sthw='(0.5,1,1)'
```


###  Video analogies

Please download raft-sintel.pth model from [RAFT](https://github.com/princeton-vl/RAFT) (or directly from [here](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT)) and place it in ```./raft/models/raft-sintel.pth```

To compute a new video with the spatio-temporal layout of video A and the appearance of video B:

```
python run_analogies.py --a_frames_dir <A frames dir> --b_frames_dir <B frames dir> --a_n_bins <A: number of dynamic bins> --b_n_bins <B: number of dynamic bins> --results_dir <results dir>
```

For example:

```
python run_analogies.py --a_frames_dir data/waterfall_Qo3OM5sPUPM --b_frames_dir data/lava_m_e7jUfvt-I --a_n_bins 4 --b_n_bins 8 --results_dir results/wfll2lava
```


### Video Retargeting

Retargeting is similar to generation but with a different aspect ratio for the output, and without adding any noise.

```
python run_generation.py --gpu 0 --frames_dir <path to frames dir> --start_frame <number of first frame> --end_frame <number of last frame> --use_noise False --sthw '(ST,SH,SW)'
```

Where ```(ST,SH,SW)``` are the required scales for the temporal, height and width dimensions, respectively. E.g., ```(1,1,1)``` will not change the result, where ```(1,1,0.5)``` will generate a retargeted result with the same height and number of frames, but with half the width of the input.


For example:

```
python run_generation.py --gpu 0 --frames_dir data/airballoons_QGAMTlI6XxY --start_frame 66 --end_frame 80 --max_size 360 --use_noise False --min_size '(3,40)' --kernel_size '(3,7,7)' --downfactor '(0.87,0.82)' --sthw '(1,1,0.6)'
```


## Citation
If you find our project useful for your work please cite:

```
@article{haim2021vgpnn,
  author    = {Haim, Niv and Feinstein, Ben and Granot, Niv and Shocher, Assaf and Bagon, Shai and Dekel, Tali and Irani, Michal},
  title     = {Diverse Generation from a Single Video Made Possible},
  journal   = {arXiv preprint arXiv:2109.08591},
  year      = {2021},
}
```
