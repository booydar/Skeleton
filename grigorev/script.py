import os
import sys
import pickle
import time
import numpy as np
from tqdm.auto  import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler
from torch_scatter import scatter_mean

from torch_geometric.data import DataLoader
from torch_geometric.nn import (NNConv, GMMConv, GraphConv, Set2Set)
from torch_geometric.data import (InMemoryDataset, Data)
import torch_geometric.transforms as T
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (SplineConv, graclus, max_pool, max_pool_x, global_mean_pool)

from torch_geometric.datasets import MNISTSuperpixels

class MNISTSkeleton(InMemoryDataset):
    r"""The skeleton on MNIST dataset
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """
    def __init__(self, root, dataset="train", transform=None, pre_transform=None, pre_filter=None):
        super(MNISTSkeleton, self).__init__(root, transform, pre_transform, pre_filter)
        if dataset == "train":
            path = self.processed_paths[0]
        elif dataset == "test":
            path = self.processed_paths[1]
        elif dataset == "val":
            path = self.processed_paths[2]
        self.data, self.slices = torch.load(path)
        
    @property
    def raw_file_names(self):
        return ['train', 'test', 'val']

    @property
    def processed_file_names(self):
        return ['train_prc', 'test_prc', 'val_prc']
    
    def download(self):
        pass
    
    def get_edges_ged_rad(self, skel_features):
        edges_x = []
        edges_y = []
        degs = set()
        rads = set()
        for i, d in enumerate(skel_features):
            if i % 4 == 0:
                edges_x.append(d)
            if i % 4 == 1:
                edges_y.append(d)
            if i % 4 == 2:
                degs.add((edges_x[-1], edges_y[-1], d))
            if i % 4 == 3:
                rads.add((edges_x[-1], edges_y[-1], d))
    
        return edges_x, edges_y, list(degs), list(rads)
    
    def get_dataset(self, skel_features):
        pos = []
        features = []
        node_slice = [0]
        edge_index = []
        edge_slice = [0]
        node_count = 0
        edge_global_count = 0
                
        for x in skel_features:
            xx, yy, deg, rad = self.get_edges_ged_rad(x)
            skeleton = {}
            n = 0
            for i in range(len(deg)):
                x, y, d = deg[i]
                if i >= len(rad):
                    r = 0
                else:
                    _, _, r = rad[i]
                skeleton[(x, y)] = n
                n += 1
                pos.append([x, y])
                features.append([d, r])
                
            node_count += len(deg)
            node_slice.append(node_count)
            for i in range(0, len(xx), 2):
                if (xx[i], yy[i]) in skeleton and (xx[i+1], yy[i+1]) in skeleton:
                    edge_index.append([skeleton[(xx[i], yy[i])], skeleton[(xx[i+1], yy[i+1])]]) 
                    edge_global_count += 1
            edge_slice.append(edge_global_count)
            
        return torch.tensor(pos), torch.tensor(features), torch.tensor(node_slice), torch.tensor(edge_index).t().contiguous(), torch.tensor(edge_slice)
        
    def process(self):
        for raw_path, path in zip(self.raw_paths, self.processed_paths):
            with open(raw_path, "rb") as fin:
                data = pickle.load(fin)
                Y, X_skel_features = data["labels"], data["skel_features"]
            pos, features, node_slice, edge_index, edge_slice = self.get_dataset(X_skel_features)            
            node_slice, Y = node_slice.to(torch.long), torch.as_tensor(Y).to(torch.long)
            graph_slice = torch.arange(Y.size(0,)+1)
            self.data = Data(x=features, y=Y, pos=pos, edge_index=edge_index)
            self.slices = {
                'x': node_slice,
                'y': graph_slice,
                'pos': node_slice,
                'edge_index': edge_slice
            }

            if self.pre_filter is not None:
                data_list = [self.get(idx) for idx in range(len(self))]
                data_list = [d for d in data_list if self.pre_filter(d)]
                self.data, self.slices = self.collate(data_list)

            if self.pre_transform is not None:
                data_list = [self.get(idx) for idx in range(len(self))]
                data_list = [self.pre_transform(data) for data in data_list]
                self.data, self.slices = self.collate(data_list)

            torch.save((self.data, self.slices), path)


class MoNet(nn.Module):
    def __init__(self, num_features):
        super(MoNet, self).__init__()
        self.conv1 = GMMConv(in_channels=num_features, out_channels=32, dim=2)
        self.batchnorm_1 = torch.nn.BatchNorm1d(32)
        self.conv2 = GMMConv(in_channels=32, out_channels=64, dim=2)
        self.batchnorm_2 = torch.nn.BatchNorm1d(64)
        self.conv3 = GMMConv(in_channels=64, out_channels=64, dim=2)
        self.batchnorm_3 = torch.nn.BatchNorm1d(64)
        self.fc1 = torch.nn.Linear(64, 80)
        self.fc2 = torch.nn.Linear(80, 10)
        
    def forward(self, data):
        data.x = F.elu(self.batchnorm_1(self.conv1(data.x, data.edge_index, data.edge_attr)))
       
        data.x = F.elu(self.batchnorm_2(self.conv2(data.x, data.edge_index, data.edge_attr)))

        data.x = F.elu(self.batchnorm_3(self.conv3(data.x, data.edge_index, data.edge_attr)))

        x = global_mean_pool(data.x, data.batch)

        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.elu(self.fc2(x))
        return F.log_softmax(x, dim=1)
      

class MPNNNet(nn.Module):
    def __init__(self, num_features, aggr='mean', processing_steps=8, message_passing_steps=4, dim=64):
        super(MPNNNet, self).__init__()
        self.message_passing_steps = message_passing_steps
        
        self.lin0 = torch.nn.Linear(num_features, dim) # Change 2 to 1 for superpixels
        nn = torch.nn.Sequential(torch.nn.Linear(2, 128), torch.nn.ReLU(), torch.nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr=aggr, root_weight=False)
        self.gru = torch.nn.GRU(dim, dim)
        
        self.set2set = Set2Set(dim, processing_steps=processing_steps)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 10)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(self.message_passing_steps):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        return F.log_softmax(self.lin2(out), dim=1)

class SplineCNN(nn.Module):
    def __init__(self, num_features, kernel=5, dim=2, num_classes=10):
        super(SplineCNN, self).__init__()
        self.conv1 = SplineConv(num_features, 32, dim, kernel)
        self.conv2 = SplineConv(32, 64, dim, kernel)
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
    
        x = global_mean_pool(data.x, data.batch)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)
     
    
class GNNNet(torch.nn.Module):
    def __init__(self, num_features, num_classes=10):
        super(GNNNet, self).__init__()
        self.conv1 = GraphConv(num_features, 32)
        self.conv2 = GraphConv(32, 64)
        self.conv3 = GraphConv(64, 64)
        self.fc1 = torch.nn.Linear(64, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, num_classes)
 
    def reset_parameters(self):
        for (name, module) in self._modules.items():
            module.reset_parameters()
 
    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index))
        data.x = F.elu(self.conv2(data.x, data.edge_index))
        data.x = F.elu(self.conv3(data.x, data.edge_index))
        x_1 = scatter_mean(data.x, data.batch, dim=0)
        x = x_1
 
        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def get_train_val_loader(path, train_batch_size=64, val_batch_size=64, val_split=1/12):
    train_dataset = MNISTSuperpixels(path, "train", transform=T.Cartesian())
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(val_split * dataset_size)
    np.random.seed(43)
    np.random.shuffle(indices)
    train_indices = indices[split:]
    val_indices = indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    validation_loader = DataLoader(train_dataset, batch_size=val_batch_size,
                                                sampler=val_sampler, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, 
                                                sampler=train_sampler, shuffle=False)
    return train_loader, validation_loader


def train(epoch, model, train_loader, device, optimizer):
    model.train()

    for data in tqdm(train_loader, leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        responce = model(data)
        F.nll_loss(responce, data.y).backward()
        optimizer.step()


def validate(model, test_loader, device):
    model.eval()
    correct = 0

    for data in test_loader:
        data = data.to(device)
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(test_loader.sampler)


def process_model(network, out_file_name, train_loader, validation_loader, test_loader,
                  init_lr=0.01, num_epochs=150):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = network(train_loader.dataset.num_features).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.8, patience=5, min_lr=0.00001, verbose=True)

    with open(out_file_name, 'w') as file:
      start_time = time.time()
      for epoch in tqdm(range(num_epochs)):
        train(epoch, model, train_loader, device, optimizer)
        test_acc = validate(model, validation_loader, device)
        scheduler.step(test_acc)
        print('Epoch: {:02d}, Time: {:.4f}, Validation Accuracy: {:.4f}'\
              .format(epoch, time.time() - start_time, test_acc), file=file)

      start_time = time.time()
      test_acc = validate(model, test_loader, device)
      print('Test, Time: {:.4f}, Accuracy: {:.4f}'\
            .format(time.time() - start_time, test_acc), file=file)
 
    return model


if __name__ == '__main__':

    # SuperPixels dataset
    sp_path = os.path.join(os.path.dirname(os.path.realpath("/")), "MNISTSuperpixel")
    sp_test_dataset = MNISTSuperpixels(sp_path, train=False, transform=T.Cartesian())
    sp_train_loader_25, sp_val_loader_25 = get_train_val_loader(sp_path, train_batch_size=25, val_batch_size=25)
    sp_train_loader_64, sp_val_loader_64 = get_train_val_loader(sp_path, train_batch_size=64, val_batch_size=64)
    sp_test_loader = DataLoader(sp_test_dataset, batch_size=1, shuffle=False)

    # Skeletons dataset
    sk_path = "dataset"
    sk_train_dataset = MNISTSkeleton(sk_path, "train", transform=T.Polar())
    sk_test_dataset = MNISTSkeleton(sk_path, "test", transform=T.Polar())
    sk_val_dataset = MNISTSkeleton(sk_path, "val", transform=T.Polar())

    sk_train_loader = DataLoader(sk_train_dataset, batch_size=64, shuffle=True)
    sk_val_loader = DataLoader(sk_val_dataset, batch_size=64, shuffle=False)
    sk_test_loader = DataLoader(sk_test_dataset, batch_size=64, shuffle=False)

    # Train
    num_epochs = 150
    print('MoNet Superpixel starts:')
    process_model(MoNet, 'MoNet_SuperPixels.txt', sp_train_loader_25, sp_val_loader_25, sp_test_loader, init_lr=0.01, num_epochs=num_epochs)
    print('MoNet Skeletons starts:')
    process_model(MoNet, 'MoNet_Skeletons.txt', sk_train_loader, sk_val_loader, sk_test_loader, init_lr=0.01, num_epochs=num_epochs)

    print('MPNNNet Superpixel starts:')
    process_model(MPNNNet, 'MPNNNet_SuperPixels.txt', sp_train_loader_64, sp_val_loader_64, sp_test_loader, init_lr=0.001, num_epochs=num_epochs)
    print('MPNNNet Skeletons starts:')
    process_model(MPNNNet, 'MPNNNet_Skeletons.txt', sk_train_loader, sk_val_loader, sk_test_loader, init_lr=0.001, num_epochs=num_epochs)

    print('GNNNet Superpixel starts:')
    process_model(GNNNet, 'GNNNet_SuperPixels.txt', sp_train_loader_64, sp_val_loader_64, sp_test_loader, init_lr=0.001, num_epochs=num_epochs)
    print('GNNNet Skeletons starts:')
    process_model(GNNNet, 'GNNNet_Skeletons.txt', sk_train_loader, sk_val_loader, sk_test_loader, init_lr=0.001, num_epochs=num_epochs)

    print('SplineCNN Superpixel starts:')
    process_model(SplineCNN, 'SplineCNN_SuperPixels.txt', sp_train_loader_64, sp_val_loader_64, sp_test_loader, init_lr=0.01, num_epochs=num_epochs)
    print('SplineCNN Skeletons starts:')
    process_model(SplineCNN, 'SplineCNN_Skeletons.txt', sk_train_loader, sk_val_loader, sk_test_loader, init_lr=0.01, num_epochs=num_epochs)