''' Dataset for The aligned, reduced, partitioned S3DIS dataset 
    Provides functionality for train/test on partitioned sets as well 
    as testing on entire spaces via get_random_partitioned_space()
    '''

import os
import sys
from glob import glob
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import copy


class S3DIS(Dataset):
    def __init__(self, root, area_nums, split='train', npoints=4096, r_prob=0.25, prefix=None):
        self.root = root
        self.area_nums = area_nums # i.e. '1-4' # areas 1-4
        self.split = split.lower() # use 'test' in order to bypass augmentations
        self.npoints = npoints     # use  None to sample all the points
        self.r_prob = r_prob       # probability of rotation

        # glob all hdf paths
        areas = glob(os.path.join(root, f'Area_[{area_nums}]*'))

        # check that datapaths are valid, if not raise error
        if len(areas) == 0:
            raise FileNotFoundError("NO VALID FILEPATHS FOUND!")

        for p in areas:
            if not os.path.exists(p):
                raise FileNotFoundError(f"PATH NOT VALID: {p} \n")

        # get all datapaths
        self.data_paths = []
        for area in areas:
            if prefix:
                self.data_paths += glob(os.path.join(area, f'{prefix}*.hdf5'), recursive=True)
            else:
                self.data_paths += glob(os.path.join(area, '*.hdf5'), recursive=True)
        
        # get unique space identifiers (area_##\\spacename_##_)
        self.space_ids = []
        for fp in self.data_paths:
            area, space = fp.split(os.sep)[-2:]
            space_id = os.sep.join([area, '_'.join(space.split('_')[:2])]) + '_'
            self.space_ids.append(space_id)

        self.space_ids = list(set(self.space_ids))
        
        # preload data to save on reading hdf5 files later
        self.space_data = []
        for path in self.data_paths:
            self.space_data.append(pd.read_hdf(path, key='space_slice').to_numpy())


    def __getitem__(self, idx):
        # read data from hdf5
        space_data = copy.deepcopy(self.space_data[idx])
        #space_data = pd.read_hdf(self.data_paths[idx], key='space_slice').to_numpy()
        points = space_data[:, :3] # xyz points
        targets = space_data[:, 3]    # integer categories

        # down sample point cloud
        if self.npoints:
            points, targets = self.downsample(points, targets)

        # add Gaussian noise to point set if not testing
        if self.split != 'test':
            # add N(0, 1/100) noise
            points += np.random.normal(0., 0.01, points.shape)

            # add random rotation to the point cloud with probability
            if np.random.uniform(0, 1) > 1 - self.r_prob:
                points = self.random_rotate(points)


        # Normalize Point Cloud to (0, 1)
        points = self.normalize_points(points)

        # convert to torch
        points = torch.from_numpy(points).type(torch.float32)
        targets = torch.from_numpy(targets).type(torch.LongTensor)

        return points, targets
        

    def get_random_partitioned_space(self):
        ''' Obtains a Random space. In this case the batchsize would be
            the number of partitons that the space was separated into.
            This is a special function for testing.
            '''

        # get random space id
        idx = random.randint(0, len(self.space_ids) - 1)
        space_id = self.space_ids[idx]

        # get all filepaths for randomly selected space
        space_paths = []
        space_path_idx = []
        for fpath in self.data_paths:
            if space_id in fpath:
                space_paths.append(fpath)
                space_path_idx.append(self.data_paths.index(fpath))
        
        # assume npoints is very large if not passed
        if not self.npoints:
            self.npoints = 20000

        points = np.zeros((len(space_paths), self.npoints, 3))
        targets = np.zeros((len(space_paths), self.npoints))

        # obtain data
        for i, space_path_i in enumerate(space_path_idx):
            space_data = copy.deepcopy(self.space_data[space_path_i])
            #space_data = pd.read_hdf(space_path, key='space_slice').to_numpy()
            _points = space_data[:, :3] # xyz points
            _targets = space_data[:, 3] # integer categories

            # downsample point cloud
            _points, _targets = self.downsample(_points, _targets)

            # add points and targets to batch arrays
            points[i] = _points
            targets[i] = _targets

        # convert to torch
        points = torch.from_numpy(points).type(torch.float32)
        targets = torch.from_numpy(targets).type(torch.LongTensor)

        return points, targets
        

    def downsample(self, points, targets):
        if len(points) > self.npoints:
            choice = np.random.choice(len(points), self.npoints, replace=False)
        else:
            # case when there are less points than the desired number
            choice = np.random.choice(len(points), self.npoints, replace=True)
        points = points[choice, :] 
        targets = targets[choice]

        return points, targets

    
    @staticmethod
    def random_rotate(points):
        ''' randomly rotates point cloud about vertical axis.
            Code is commented out to rotate about all axes
            '''
        # construct a randomly parameterized 3x3 rotation matrix
        phi = np.random.uniform(-np.pi, np.pi)
        theta = np.random.uniform(-np.pi, np.pi)
        psi = np.random.uniform(-np.pi, np.pi)

        rot_x = np.array([
            [1,              0,                 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi) ]])

        rot_y = np.array([
            [np.cos(theta),  0, np.sin(theta)],
            [0,                 1,                0],
            [-np.sin(theta), 0, np.cos(theta)]])

        rot_z = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi),  0],
            [0,              0,                 1]])

        # rot = np.matmul(rot_x, np.matmul(rot_y, rot_z))
        
        return np.matmul(points, rot_z)


    @staticmethod
    def normalize_points(points):
        ''' Perform min/max normalization on points
            Same as:
            (x - min(x))/(max(x) - min(x))
            '''
        points = points - points.min(axis=0)
        points /= points.max(axis=0)

        return points


    def __len__(self):
        return len(self.data_paths)
