# imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _thread
import json
import logging
import os
import random

import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import time
import pickle
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import collections
import itertools
import time
from tqdm import tqdm
import os
from utils import *
from dataloader import *
from main import *

args = Args(
    cuda=True,  
    do_train=True,  
    do_test=True,  
    data_path="/kaggle/input/kg-data/FB15k-237-betae",  
    negative_sample_size=128,  
    batch_size=512,  
    hidden_dim=800,  
    gamma=60.0,  
    learning_rate=0.0001,  
    max_steps=1000,  
    cpu_num=3,  
    test_batch_size=4,  
    geo='laplace',  # Changed from 'gamma' to 'laplace' for LapE  
    drop=0.1,  
    valid_steps=500,  
    gamma_mode="(1600,4)",  # Using GammaE settings as reference  
    seed=42  
)

main(args)