import torch
import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
curPath = os.path.abspath(os.path.dirname(__file__))

def getData_cifar(setting, config):
    """
    Load CIFAR10 dataset

    :param setting: 'iid' or 'noniid'
    :param config:  configuration of the method
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_noniid = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    num_sample = 50000
    num_class = 10
    num_perc = int(num_sample / num_class)

    test_batchSize = config['test_batchSize']

    if setting == 'iid':
        train_dataset = datasets.CIFAR10(root=curPath + '/dataset', train=True, download=True,
                                       transform=transform)
    elif setting == 'noniid':
        train_dataset = datasets.CIFAR10(root=curPath + '/dataset', train=True, download=True,
                                       transform=transform)
        labels = np.array(train_dataset.targets)
        idxs = np.arange(num_sample)
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        class_dataset = [
            Subset(train_dataset, list(idxs[i*num_perc: (i+1)*num_perc]))
            for i in range(10)]

    test_dataset = datasets.CIFAR10(curPath + '/dataset', train=False, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=test_batchSize,
                              shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batchSize,
                             shuffle=False, num_workers=0, pin_memory=False)

    # evenly split the data into workers
    num_data_node = int(len(train_dataset) / config['nodeSize'])
    print(num_data_node)
    if setting == 'iid':
        train_dataset_subset = [
            Subset(train_dataset, list(range(i * num_data_node, (i + 1) * num_data_node)))
            for i in range(config['nodeSize'])]
    elif setting == 'noniid':
        n = config['nodeSize']
        train_dataset_subset1 = []
        train_dataset_subset2 = []
        train_dataset_subset = []
        for i in range(n):
            train_dataset_subset1.append(Subset(class_dataset[int(i/(n/10))],
                                                list(range(int((i % (n/10)) * (25000/n)),
                                                           int(((i % (n/10)) + 1) * (25000/n))))))
            for j in range(10):
                if j == 0:
                    train_dataset_subset2.append(Subset(class_dataset[j],
                                                        list(range(int(2500+i*(2500/n)),
                                                                   int(2500+(i+1)*(2500/n))))))
                    continue
                subset2 = Subset(class_dataset[j], list(range(int(2500+i*(2500/n)), int(2500+(i+1)*(2500/n)))))
                train_dataset_subset2[i] = ConcatDataset([train_dataset_subset2[i], subset2])
            train_dataset_subset.append(ConcatDataset([train_dataset_subset1[i], train_dataset_subset2[i]]))

    return train_dataset_subset, train_loader, test_loader

