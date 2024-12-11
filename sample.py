from pathlib import Path
import numpy as np
import copy
import open3d as o3d
#import open3d.visualization.rendering as rendering
#import simple_3dviz as s3d
from matplotlib import pyplot as plt
import os

from torch.utils.data import Dataset, DataLoader

class S3DISDataset(Dataset):
  class2label = {
    'ceiling'  : 0,
    'floor'    : 1,
    'wall'     : 2,
    'beam'     : 3,
    'column'   : 4,
    'window'   : 5,
    'door'     : 6,
    'table'    : 7,
    'chair'    : 8,
    'sofa'     : 9,
    'bookcase' : 10,
    'board'    : 11,
    'stairs'   : 12,
    'clutter'  : 13
  }

  label2class = [
    'ceiling',
    'floor',
    'wall',
    'beam',
    'column',
    'window',
    'door',
    'table',
    'chair',
    'sofa',
    'bookcase',
    'board',
    'stairs',
    'clutter'
  ]

  COLOR_MAP_ARRAY = np.array([
    [47./255., 79./255., 79./255.],    # ceiling - darkslategray
    [139./255., 69./255., 19./255.],   # floor - saddlebrown
    [34./255., 139./255., 34./255.],   # wall - forestgreen
    [75./255., 0./255., 130./255.],    # beam - indigo
    [255./255., 0./255., 0./255.],     # column - red 
    [255./255., 255./255., 0./255.],   # window - yellow
    [0./255., 255./255., 0./255.],     # door - lime
    [0./255., 255./255., 255./255.],   # table - aqua
    [0./255., 0./255., 255./255.],     # chair - blue
    [255./255., 0./255., 255./255.],   # sofa - fuchsia
    [238./255., 232./255., 170./255.], # bookcase - palegoldenrod
    [100./255., 149./255., 237./255.], # board - cornflower
    [255./255., 105./255., 180./255.], # stairs - hotpink
    [0./255., 0./255., 0./255.]        # clutter - black
  ])

  def __init__(self):
    self.xyz = np.array([])
    self.rgb = np.array([])
    self.label = np.array([])
    self.roomIdx = np.array([])
    self.itemIdx = np.array([])
    self.roomOff = np.array([])
    self.itemOff = np.array([])

  def load_data(self, rootPath, areas):
    roomIdx = 0
    itemIdx = 0
    xyz_list = []
    rgb_list = []
    label_list = []
    roomIdx_list = []
    roomOff_list = [0]
    itemIdx_list = []
    itemOff_list = [0]

    classNames = S3DISDataset.class2label.keys()

    offset = 0;
    for area in areas:
      rp = Path(rootPath).joinpath(area)
      rp_children = os.listdir(rp)
      rp_children.sort()
      for child in rp_children:
        child_path = rp.joinpath(child)
        if os.path.isdir(child_path):
          paths = list(child_path.glob('**/Annotations/*.txt'))
          print('Room Idx:', roomIdx)
          for path in paths:
            print('Loading', path)
            className = path.stem.split('_')[0]
            if className not in classNames:
              className = 'clutter';
            raw_data = np.loadtxt(path, dtype=np.float32)
            xyz_list.append(copy.deepcopy(raw_data[:,0:3]))
            rgb_list.append(copy.deepcopy(raw_data[:,3:6]))
            label_list.append(np.ones((raw_data.shape[0],1), dtype=int)*S3DISDataset.class2label[className])
            itemIdx_list.append(np.ones((raw_data.shape[0],1), dtype=int)*itemIdx)
            roomIdx_list.append(np.ones((raw_data.shape[0],1), dtype=int)*roomIdx)
            itemIdx += 1
            offset += raw_data.shape[0]
            itemOff_list.append(offset)
          roomOff_list.append(offset)
          roomIdx += 1

    self.xyz = np.concatenate(xyz_list, 0)
    self.rgb = np.concatenate(rgb_list, 0)
    self.label = np.concatenate(label_list, 0)
    self.roomIdx = np.concatenate(roomIdx_list, 0)
    self.itemIdx = np.concatenate(itemIdx_list, 0)
    self.roomOff = np.array(roomOff_list, dtype=int)
    self.itemOff = np.array(itemOff_list, dtype=int)

  def save_data_to_npy(self, rootPath, prefix):
    np.save(Path(rootPath).joinpath(prefix + '_xyz.npy'), self.xyz)
    np.save(Path(rootPath).joinpath(prefix + '_rgb.npy'), self.rgb)
    np.save(Path(rootPath).joinpath(prefix + '_label.npy'), self.label)
    np.save(Path(rootPath).joinpath(prefix + '_roomIdx.npy'), self.roomIdx)
    np.save(Path(rootPath).joinpath(prefix + '_itemIdx.npy'), self.itemIdx)
    np.save(Path(rootPath).joinpath(prefix + '_roomOff.npy'), self.roomOff)
    np.save(Path(rootPath).joinpath(prefix + '_itemOff.npy'), self.itemOff)

  def load_data_from_npy(self, rootPath, prefix):
    self.xyz = np.load(Path(rootPath).joinpath(prefix + '_xyz.npy'))
    self.rgb = np.load(Path(rootPath).joinpath(prefix + '_rgb.npy'))
    self.label = np.load(Path(rootPath).joinpath(prefix + '_label.npy'))
    self.roomIdx = np.load(Path(rootPath).joinpath(prefix + '_roomIdx.npy'))
    self.itemIdx = np.load(Path(rootPath).joinpath(prefix + '_itemIdx.npy'))
    self.roomOff = np.load(Path(rootPath).joinpath(prefix + '_roomOff.npy'))
    self.itemOff = np.load(Path(rootPath).joinpath(prefix + '_itemOff.npy'))

  def __getitem__(self, idx):
    sidx = self.roomOff[idx]
    eidx = self.roomOff[idx+1]
    return self.xyz[sidx:eidx,:], self.label[sidx:eidx]

  def __len__(self):
    return len(self.roomOff)-1

  def numRooms(self):
    return len(self.roomOff)-1

  def room_points(self, idx):
    return self.xyz[self.roomOff[idx]:self.roomOff[idx+1],:]

  def room_colors(self, idx):
    return self.rgb[self.roomOff[idx]:self.roomOff[idx+1],:]
  
  def room_colormap_colors(self, idx):
    num = self.roomOff[idx+1] - self.roomOff[idx]
    cmap = np.ndarray((num,3))
    for i in range(self.roomOff[idx], self.roomOff[idx+1]):
      cmap[i-self.roomOff[idx],:] = S3DISDataset.COLOR_MAP_ARRAY[self.label[i]]
    return cmap

  def room_labels(self, idx):
    return selz.label[self.roomOff[idx]:self.roomOff[idx+1],:]

if __name__ == '__main__':
  rootPath = '/scratch/hercules/users/adz8/point-cloud-datasets/S3DIS/Stanford3dDataset_v1.2_Aligned_Version'

  #trainset = S3DISDataset()
  #trainset.load_data(rootPath, ['Area_1'])
  #print(len(trainset))
  #trainset.save_data_to_npy(rootPath, 'train_aligned')

  trainset = S3DISDataset()
  trainset = S3DISDataset()
  trainset.load_data_from_npy(rootPath, 'train_aligned')
  print(len(trainset))

  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(trainset.room_points(20))
  pcd.colors = o3d.utility.Vector3dVector(trainset.room_colormap_colors(20))
  o3d.visualization.draw_geometries([pcd])

