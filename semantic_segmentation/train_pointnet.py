#!/usr/bin/python
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from pointnet import PointNet
from dataloader import SemSegDataset, class_names, class_colors
import matplotlib.pyplot as plt

# training parameters
learning_rate = 1e-4
batch_size = 10
max_epochs = 100
num_resampled_points = 256
num_class = len(class_names)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

model = PointNet(num_class = num_class).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print('PointNet model:')
print(model)

train_dataset = SemSegDataset(root='data', split='train', N=num_resampled_points)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
validation_dataset = SemSegDataset(root='data', split='validation', N=num_resampled_points)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
num_train_batches = int(np.ceil(len(train_dataset) / batch_size))
num_validation_batches = int(np.ceil(len(validation_dataset) / batch_size))

for epoch in range(max_epochs):
    train_loss, train_correct, train_accuracy = 0, 0, 0
    for i, data in enumerate(train_dataloader):
        points, target = data
        # put the model in training mode
        model = model.train()
        # move this batch of data to the GPU if device is cuda
        points, target = points.to(device), target.to(device)
        # run a forward pass through the neural network and predict the outputs
        pred = model(points)
        pred_1d = pred.view(-1, num_class)
        target_1d = target.view(-1, 1)[:, 0]
        # compare the prediction vs the target labels and determine the negative log-likelihood loss
        loss = F.nll_loss(pred_1d, target_1d)
        # perform backpropagation to update the weights of the network based on the computed loss function
        loss.backward()
        optimizer.step()
        pred_choice = pred_1d.data.max(1)[1]
        train_loss += loss.item()
        train_correct += pred_choice.eq(target_1d).sum().item()
#        print(pred.shape, target.shape, pred_choice.shape, train_correct)
    train_loss /= num_train_batches
    train_accuracy = train_correct / len(train_dataset) / num_resampled_points
    print('[Epoch %d] train loss: %.3f accuracy: %.3f' % (epoch, train_loss, train_accuracy))

    if epoch % 10 == 9: # run validation every 10 epochs
        validation_loss, validation_correct, validation_accuracy = 0, 0, 0
        tp_per_class = [0] * num_class
        fp_per_class = [0] * num_class
        fn_per_class = [0] * num_class
        for j, data in enumerate(validation_dataloader):
            points, target = data
            points, target = points.to(device), target.to(device)
            # put the model in evaluation mode
            model = model.eval()
            with torch.no_grad():
                pred = model(points)
                pred_1d = pred.view(-1, num_class)
                target_1d = target.view(-1, 1)[:, 0]
                loss = F.nll_loss(pred_1d, target_1d)
                pred_choice = pred_1d.data.max(1)[1]
                validation_loss += loss.item()
                validation_correct += pred_choice.eq(target_1d).sum().item()
                for i in range(num_class):
                    tp_per_class[i] += ((pred_choice==i) & (target_1d==i)).sum().item()
                    fp_per_class[i] += ((pred_choice==i) & (target_1d!=i)).sum().item()
                    fn_per_class[i] += ((pred_choice!=i) & (target_1d==i)).sum().item()
        validation_loss /= num_validation_batches
        validation_accuracy = validation_correct / len(validation_dataset) / num_resampled_points
        print('[Epoch %d] validation  loss: %.3f accuracy: %.3f' % (epoch, validation_loss, validation_accuracy))
        for i in range(num_class):
            precision = 1.0 * tp_per_class[i] / (tp_per_class[i] + fp_per_class[i] + 1e-6)
            recall = 1.0 * tp_per_class[i] / (tp_per_class[i] + fn_per_class[i] + 1e-6)
            print('%10s: recall %.3f precision %.3f' % (class_names[i], precision, recall))

torch.save(model.state_dict(), 'pointnet.pth')
