# One-Liner importing all important stuff for notebooks.
# Just put this at the top line of the notebook:
# from common_utils.import_for_notebooks import *

import os

from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import PIL
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
import torch.nn.functional as F

import threadpoolctl
thread_limit = threadpoolctl.threadpool_limits(limits=8)
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
np.seterr(all='raise')

import utils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('IMPORTANT IMPORTED!')
print(f'device={device}')
