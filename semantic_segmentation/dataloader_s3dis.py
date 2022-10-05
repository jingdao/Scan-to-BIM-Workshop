import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import open3d as o3d
import json
import h5py

class_names = ['clutter', 'board', 'bookcase', 'beam', 'chair', 'column', 'door', 'sofa', 'table', 'window', 'ceiling', 'floor', 'wall']
class_colors = [
    [200,200,200], #clutter
    [0,100,100], #board
    [255,0,0], #bookcase
    [255,200,200], #beam
    [0,0,100], #chair
    [0,255,255], #column
    [0,100,0], #door
    [255,0,255], #sofa
    [50,50,50], #table
    [0,255,0], #window
    [255,255,0], #ceiling
    [0,0,255], #floor
    [255,165,0], #wall
]

def loadFromH5(filename, load_labels=True):
	f = h5py.File(filename,'r')
	all_points = f['points'][:]
	count_room = f['count_room'][:]
	tmp_points = []
	idp = 0
	for i in range(len(count_room)):
		tmp_points.append(all_points[idp:idp+count_room[i], :])
		idp += count_room[i]
	f.close()
	room = []
	labels = []
	class_labels = []
	if load_labels:
		for i in range(len(tmp_points)):
			room.append(tmp_points[i][:,:-2])
			labels.append(tmp_points[i][:,-2].astype(int))
			class_labels.append(tmp_points[i][:,-1].astype(int))
		return room, labels, class_labels
	else:
		return tmp_points

class SemSegDataset(data.Dataset):
    def __init__(self, root, area=1, N=2048, grid_resolution=5.0):
        self.N = N
        self.grid_resolution = grid_resolution
        self.root = root
        self.area = area
        self.normalized_points = []
        self.predicted_labels = []
        self.points, _, self.labels = loadFromH5('%s/s3dis_area%d.h5' % (root,area))
        print('Created dataset from area=%s with %d rooms' % (area, len(self.labels))) 

        # normalize XYZ coordinates, concatenate with room coordinates
        for i in range(len(self.points)):
            P = np.zeros((len(self.points[i]), 9))
            P[:, :3] = self.points[i][:, :3].copy()
            centroid_xy = (P[:, :2].max() + P[:, :2].min()) / 2
            centroid_z = P[:,2].min()
            P[:,:2] -= centroid_xy
            P[:,2] -= centroid_z
            P[:, 3:6] = self.points[i][:, 3:6].copy()
            room_coordinates = (P[:,:3] - P[:,:3].min(axis=0)) / (P[:,:3].max(axis=0) - P[:,:3].min(axis=0))
            P[:, 6:9] = room_coordinates
            self.normalized_points.append(P)

    def __getitem__(self, index):
        pc = self.normalized_points[index].astype(np.float32)
        resample_idx = np.random.choice(len(pc), self.N, replace=len(pc)<self.N)
        pc = torch.from_numpy(pc[resample_idx].T)
        cls = torch.from_numpy(self.labels[index][resample_idx])
        return pc, cls

    def __len__(self):
        return len(self.labels)

    def colorize_points(self, room_id, output_file, use_predictions=False):
        visualized_point_cloud = np.zeros((len(self.points[room_id]), 6))
        visualized_point_cloud[:, :3] = self.points[room_id][:, :3]
        for c in range(len(class_names)):
            if use_predictions and len(self.predicted_labels) > 0:
                visualized_point_cloud[self.predicted_labels[room_id]==c, 3:6] = class_colors[c]
            else:
                visualized_point_cloud[self.labels[room_id]==c, 3:6] = class_colors[c]
        np.savetxt(output_file, visualized_point_cloud)
        print('Saved %d points to %s' % (len(visualized_point_cloud), output_file))

if __name__=='__main__':
    train_dataset = SemSegDataset(root='data', area=1, N=2048)
    validation_dataset = SemSegDataset(root='data', area=2, N=2048)
    test_dataset = SemSegDataset(root='data', area=3, N=2048)

    pc, cls = train_dataset[0]
    assert pc.shape == torch.Size([9, 2048])
    assert cls.shape == torch.Size([2048])
#    for i in range(len(train_dataset)):
#        train_dataset.colorize_points(i, '%d_viz.txt' % i)
