import warnings
warnings.filterwarnings('ignore')

import os
import cv2
import sys
import time
import h5py
import glob
import yaml
import scipy
import random
import shutil
import sklearn
import IPython
import argparse
import itertools
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as scipy_R

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms

from .CD.chamferdist.chamfer import knn_points as knn_gpu
from .EMD.emd import earth_mover_distance_unwrapped
from .KNN_CPU import nearest_neighbors as knn_cpu
from .GS_CPU.cpp_subsampling import grid_subsampling as cpp_grid_subsample



def knn_on_gpu(source_pts, query_pts, k):
    assert source_pts.device.type == 'cuda'
    assert query_pts.device.type == 'cuda'
    assert source_pts.size(0) == query_pts.size(0)
    assert source_pts.size(2) == query_pts.size(2)
    knn_idx = knn_gpu(p1=query_pts, p2=source_pts, K=k, return_nn=False, return_sorted=True)[1]
    return knn_idx


def knn_on_cpu(source_pts, query_pts, k):
    assert source_pts.device.type == 'cpu'
    assert query_pts.device.type == 'cpu'
    assert source_pts.size(0) == query_pts.size(0)
    assert source_pts.size(2) == query_pts.size(2)
    knn_idx = knn_cpu.knn_batch(source_pts, query_pts, k, omp=True)
    return knn_idx


def knn_search(source_pts, query_pts, k):
    assert source_pts.device.type == query_pts.device.type
    device_type = source_pts.device.type
    assert device_type in ['cpu', 'cuda']
    if device_type == 'cuda':
        knn_idx = knn_on_gpu(source_pts, query_pts, k)
    if device_type == 'cpu':
        knn_idx = knn_on_cpu(source_pts, query_pts, k)
    return knn_idx


def chamfer_distance_cuda(pts_s, pts_t, cpt_mode='max', return_detail=False):
    Bs, Ns, Cs, device_s = pts_s.size(0), pts_s.size(1), pts_s.size(2), pts_s.device
    Bt, Nt, Ct, device_t = pts_t.size(0), pts_t.size(1), pts_t.size(2), pts_t.device
    assert Bs == Bt
    assert Cs == Ct
    assert device_s == device_t
    assert device_s.type == 'cuda' and device_t.type == 'cuda'
    assert cpt_mode in ['max', 'avg']
    lengths_s = torch.ones(Bs, dtype=torch.long, device=device_s) * Ns
    lengths_t = torch.ones(Bt, dtype=torch.long, device=device_t) * Nt
    source_nn = knn_gpu(pts_s, pts_t, lengths_s, lengths_t, 1)
    target_nn = knn_gpu(pts_t, pts_s, lengths_t, lengths_s, 1)
    source_dist, source_idx = source_nn.dists.squeeze(-1), source_nn.idx.squeeze(-1)
    target_dist, target_idx = target_nn.dists.squeeze(-1), target_nn.idx.squeeze(-1)
    batch_dist = torch.cat((source_dist.mean(dim=-1, keepdim=True), target_dist.mean(dim=-1, keepdim=True)), dim=-1)
    if cpt_mode == 'max':
        cd = batch_dist.max(dim=-1)[0].mean()
    if cpt_mode == 'avg':
        cd = batch_dist.mean(dim=-1).mean()
    if not return_detail:
        return cd
    else:
        return cd, source_dist, source_idx, target_dist, target_idx


def earth_mover_distance_cuda(pts_1, pts_2):
    assert pts_1.size(0) == pts_2.size(0)
    assert pts_1.size(2) == pts_2.size(2)
    assert pts_1.device == pts_2.device
    B, N1, C, device = pts_1.size(0), pts_1.size(1), pts_1.size(2), pts_1.device
    B, N2, C, device = pts_2.size(0), pts_2.size(1), pts_2.size(2), pts_2.device
    assert device.type == 'cuda'
    assert C in [1, 2, 3]
    if C < 3:
        pts_1 = torch.cat((pts_1, torch.zeros(B, N1, 3-C).to(device)), dim=-1)
        pts_2 = torch.cat((pts_2, torch.zeros(B, N2, 3-C).to(device)), dim=-1)
    dist_1 = earth_mover_distance_unwrapped(pts_1, pts_2, transpose=False) / N1
    dist_2 = earth_mover_distance_unwrapped(pts_2, pts_1, transpose=False) / N2
    emd = ((dist_1 + dist_2) / 2).mean()
    return emd


