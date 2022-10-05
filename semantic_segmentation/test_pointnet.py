#!/usr/bin/python
import numpy as np
import torch
from pointnet import PointNet
from dataloader_s3dis import SemSegDataset, class_names, class_colors
import os
import sys

num_resampled_points = 1024
num_class = len(class_names)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

model = PointNet(num_class = num_class).to(device)
model = PointNet(num_class = num_class).to(device)
model_path = 'pointnet.pth'
if os.path.exists(model_path):
    print('Loading PointNet model from', model_path)
    model.load_state_dict(torch.load(model_path))
else:
    print('Failed to load model from', model_path)
    sys.exit(1)
model.eval()

test_dataset = SemSegDataset(root='data', area=3, N=num_resampled_points)

for i, points in enumerate(test_dataset.normalized_points):
    shuffle_idx = np.arange(len(points))
    np.random.shuffle(shuffle_idx)
    num_batches = int(np.ceil(1.0 * len(points) / num_resampled_points))
    input_points = np.zeros((1, num_resampled_points, 9), dtype=np.float32)
    predicted_labels = np.zeros(len(points), dtype=int)
    print('Processing room %d with %d batches' % (i, num_batches))
    for batch_id in range(num_batches):
        start_idx = batch_id * num_resampled_points
        end_idx = (batch_id + 1) * num_resampled_points
        valid_idx = min(len(points), end_idx)
        if end_idx <= len(points):
            input_points[0, :valid_idx-start_idx] = points[shuffle_idx[start_idx:valid_idx],:]
        else:
            input_points[0, :valid_idx-start_idx] = points[shuffle_idx[start_idx:valid_idx],:]
            input_points[0, valid_idx-end_idx:] = points[np.random.choice(range(len(points)), end_idx-valid_idx, replace=True),:]

        with torch.no_grad():
            pred = model(torch.from_numpy(input_points.transpose(0,2,1)).to(device))
            pred = pred[0].data.max(1)[1]
            predicted_labels[shuffle_idx[start_idx:valid_idx]] = pred[:valid_idx-start_idx].cpu().numpy()
    test_dataset.predicted_labels.append(predicted_labels)

for i in range(len(test_dataset)):
    test_dataset.colorize_points(i, '%d_viz.txt' % i, use_predictions=True)
