import torch, time, os, shutil
from tqdm import tqdm
import pywt, os, copy
import torch
from scipy import signal
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dateset_hf import DatasetECG
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import *
from models.endecoder import *
from torch.utils.tensorboard import SummaryWriter
import collections
from apex import amp

if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    #device='cuda'
    model=resnet18(num_classes=3,use_freq=True)
    #model.to(device)
    #model.load_state_dict(torch.load('./ckpt/resnet/epoch_0.pth'))
    for name,layer in model.named_children():
        print(name,'#',layer)