from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
import torchvision.transforms.functional as tx
import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else "cpu"
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import binary_cross_entropy
import configparser
def run_once(func):
    ''' a declare wrapper function to call only once, use @run_once declare keyword '''
    def wrapper(*args, **kwargs):
        if 'result' not in wrapper.__dict__:
            wrapper.result = func(*args, **kwargs)
        return wrapper.result
    return wrapper

@run_once
def read_config():
    conf = configparser.ConfigParser()
    candidates = ['config_default.ini', 'config.ini']
    conf.read(candidates)
    return conf

config = read_config() # keep the line as top as possible