import torch
import torch.nn.functional as F
import torch.nn as nn

class PointNet(nn.Module):
    def __init__(self, num_class=10):
        super(PointNet, self).__init__()
        self.num_class = num_class
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1088, 256, 1)
        self.conv5 = torch.nn.Conv1d(256, num_class, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(256)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = self.bn3(self.conv3(x2))
        x4 = torch.max(x3, 2, keepdim=True)[0]
        x4 = x4.view(-1, 1024, 1).repeat(1, 1, x.size()[2])
        x5 = torch.cat([x4, x1], 1)
        x6 = F.relu(self.bn4(self.conv4(x5)))
        x7 = self.conv5(x6)
        x7 = x7.transpose(2,1).contiguous()
        x7 = F.log_softmax(x7.view(-1,self.num_class), dim=-1)
        x7 = x7.view(x.size()[0], x.size()[2], self.num_class)
        return x7

if __name__ == '__main__':
    model = PointNet(10)
    xyz = torch.rand(10, 3, 2048)
    print(model)
    print(model(xyz).shape)
