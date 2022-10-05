import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import open3d as o3d
import json

class_names = ['other', 'column', 'door', 'wall']
class_colors = [
    [100, 100, 100],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
]

class SemSegDataset(data.Dataset):
    def __init__(self, root, split='train', N=1024, grid_resolution=5.0):
        self.N = N
        self.grid_resolution = grid_resolution
        self.root = root
        self.split = split
        self.class_labels = None
        self.visualized_point_cloud = None
        self.grid_points = []
        self.labels = []
        
        pcd_object = o3d.io.read_point_cloud(os.path.join(root, self.split + ".ply"))
        # Convert to a NumPy array for further processing
        self.initial_pcd = np.array(pcd_object.points)
        print(split, "point cloud shape is: ", self.initial_pcd.shape)
        print(split, "point cloud dimensions are: [%.2f,%.2f] [%.2f,%.2f] [%.2f,%.2f]" % 
            (self.initial_pcd[:,0].min(), self.initial_pcd[:,0].max(),
            self.initial_pcd[:,1].min(), self.initial_pcd[:,1].max(),
            self.initial_pcd[:,2].min(), self.initial_pcd[:,2].max()))

        if split!='test':
            self.class_labels = np.zeros(len(self.initial_pcd), dtype=int)
            for c in range(1, len(class_names)):
                structures = json.load(open(os.path.join(root, self.split + '_' + class_names[c] + "s.json")))
                structureslist = []
                print("Processing", len(structures), class_names[c], 'objects')
                for structure in structures:
                    x1, y1, z1 = structure.get("start_pt", structure.get("loc", [.0, .0, .0]))
                    x2, y2, z2 = structure.get("end_pt", structure.get("loc", [.0, .0, .0]))
                    width = structure.get("width", .0)
                    depth = structure.get("depth", .0)
                    height = structure.get("height", 0)
                    rotation = structure.get("rotation", 0)
                    structureslist.append([
                        x1, y1, z1,
                        x2, y2, z2,
                        width, depth, height,
                        class_names[c], rotation
                    ])
                for structure in structureslist:
                    x1, y1, z1 = structure[:3]
                    x2, y2, z2 = structure[3:6]

                    width, depth, height = structure[6:9]
                    w2, d2, h2 = (width / 2), (depth / 2), (height / 2)

                    d = np.array([(x2 - x1), (y2 - y1)], dtype=np.float32)
                    if d.sum() == 0:
                        sides = np.array([
                            [-w2, +d2],
                            [-w2, -d2],
                            [+w2, -d2],
                            [+w2, +d2]
                        ], dtype=np.float32)
                        sides = np.tile(sides, (2, 1))

                        rotation: float = structure[10] # Structure specicic rotation (degrees)
                        if rotation != 0:
                            rotation = np.deg2rad(rotation)
                            R = np.array([
                                [np.cos(rotation), -np.sin(rotation)],
                                [np.sin(rotation), np.cos(rotation)]
                            ])
                            for i in range(sides.shape[0]):
                                sides[i] = np.dot(R, sides[i])
                    else:
                        d /= np.linalg.norm(d + np.finfo(np.float32).eps)
                        dx = -d[1] * w2
                        dy = d[0] * (w2 if depth == .0 else d2)

                        sides = np.array([
                            [-dx, +dy],
                            [-dx, +dy],
                            [+dx, -dy],
                            [+dx, -dy]
                        ], dtype=np.float32)
                        sides = np.tile(sides, (2, 1))
                    x = np.asarray([x1, x2, x2, x1, x1   , x2   , x2   , x1   ])
                    y = np.asarray([y1, y2, y2, y1, y1   , y2   , y2   , y1   ])
                    z = np.asarray([z1, z1, z1, z1, z1+h2, z1+h2, z1+h2, z1+h2])

                    corners = np.vstack([x, y, z]).transpose()
                    corners[:, :2] += sides
                    v1 = corners[0, :2] - corners[1, :2]
                    v2 = corners[2, :2] - corners[1, :2]
                    c1 = np.linalg.norm(v1)
                    v1 /= c1
                    c2 = np.linalg.norm(v2)
                    v2 /= c2
                    dp1 = (self.initial_pcd[:, :2] - corners[1, :2]).dot(v1)
                    dp2 = (self.initial_pcd[:, :2] - corners[1, :2]).dot(v2)
                    mask = (dp1 >= 0) & (dp1 <= c1) & (dp2 >= 0) & (dp2 <= c2) & (self.initial_pcd[:, 2] >= z1) & (self.initial_pcd[:, 2] <= z1 + h2)
                    self.class_labels[mask] = c
#                break
            print('Counts:', np.unique(self.class_labels, return_counts=True))

        grid = np.round(self.initial_pcd[:,:2]/self.grid_resolution).astype(int)
        grid_set = set([tuple(g) for g in grid])
        num_points_per_cell = []
        for g in grid_set:
            grid_mask = np.all(grid==g, axis=1)
            grid_points = self.initial_pcd[grid_mask, :3].copy()
            centroid_xy = np.array(g)*grid_resolution
            centroid_z = grid_points[:,2].min()
            grid_points[:,:2] -= centroid_xy
            grid_points[:,2] -= centroid_z
            num_points_per_cell.append(len(grid_points))
            if split=='test':
                self.grid_points.append(grid_points)
            else:
                masked_labels = self.class_labels[grid_mask]
                if (masked_labels>0).sum() > 0.1 * len(grid_points):
                    self.labels.append(self.class_labels[grid_mask])
                    self.grid_points.append(grid_points)
        print('Extracted %d/%d grid cells from %d points: %d +- %d points per cell' % (len(self.grid_points), len(grid_set), len(self.initial_pcd), np.mean(num_points_per_cell), np.std(num_points_per_cell)))

    def __getitem__(self, index):
        pc = self.grid_points[index].astype(np.float32)
        resample_idx = np.random.choice(len(pc), self.N, replace=len(pc)<self.N)
        pc = torch.from_numpy(pc[resample_idx].T)
        cls = torch.from_numpy(self.labels[index][resample_idx])
        return pc, cls

    def __len__(self):
        return len(self.labels)

    def colorize_points(self, class_labels, output_file=None):
        self.visualized_point_cloud = np.zeros((len(self.initial_pcd), 6))
        self.visualized_point_cloud[:, :3] = self.initial_pcd
        for c in range(len(class_names)):
            self.visualized_point_cloud[class_labels==c, 3:6] = class_colors[c]
        if output_file is not None:
            np.savetxt(output_file, self.visualized_point_cloud)

if __name__=='__main__':
    train_dataset = SemSegDataset(root='data', split='train')
    validation_dataset = SemSegDataset(root='data', split='validation')
    test_dataset = SemSegDataset(root='data', split='test')

    train_dataset.colorize_points(train_dataset.class_labels, 'train_viz.txt')
    validation_dataset.colorize_points(validation_dataset.class_labels, 'validation_viz.txt')
