# -*- coding: utf-8 -*-

import random
from abc import ABC
from argparse import ArgumentParser
from collections import OrderedDict
from os import path, makedirs

import numpy as np
import torch
from torch.nn import Conv2d, MaxPool2d, ReLU, Linear, Module, Flatten, Sequential, BatchNorm2d, Dropout
from torch.nn.functional import pairwise_distance
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize


# siamese dataset
class SiameseDataset(object):
    def __init__(self, pair_indexes_with_target, images):
        super(Dataset).__init__()
        self.pair_indexes = []
        self.targets = []
        for pair_index, target in pair_indexes_with_target:
            self.pair_indexes.append(pair_index)
            self.targets.append(target)
        self.images = images
        self.pair_indexes = torch.from_numpy(np.array(self.pair_indexes))
        self.targets = torch.from_numpy(np.array(self.targets, dtype=np.int32))

    def __getitem__(self, index):
        first_index, second_index = self.pair_indexes[index]
        first, second = self.images[first_index], self.images[second_index]
        inputs = first, second
        return inputs, self.targets[index]

    def __len__(self):
        return torch.numel(self.targets)


# train siamese epoch
def train_siamese_epoch(model, criterion, device, loader, optimizer,
                        make_prediction=False, show_batch_info=True, show_epoch_info=True):
    number_of_batches = len(loader)
    model.train()
    epoch_info = dict()
    epoch_info['samples'] = len(loader.dataset)
    batch_info_list = []
    processed_samples = 0
    for batch_index, batch in enumerate(iterable=loader):
        batch_inputs, batch_target = batch
        assert len(batch_inputs) > 0
        batch_info = train_siamese_batch(batch_inputs=batch_inputs, batch_target=batch_target,
                                         model=model, criterion=criterion, device=device,
                                         optimizer=optimizer, make_prediction=make_prediction)
        batch_info_list.append(batch_info)
        processed_samples += batch_info['size']
        if show_batch_info:
            print('Batch: [{batch_no}/{number_of_batches}], '
                  'Processed=[{processed_samples}/{samples}], '
                  'Loss={batch_loss:.8f}'
                  .format(batch_no=(batch_index + 1), number_of_batches=number_of_batches,
                          processed_samples=processed_samples, samples=epoch_info['samples'],
                          batch_loss=batch_info['loss']))
    epoch_info['batches'] = batch_info_list
    epoch_info['loss'] = sum(batch_info['loss'] for batch_info in batch_info_list) / epoch_info['samples']
    if make_prediction:
        epoch_info['correct'] = sum(batch_info['correct'] for batch_info in batch_info_list)
        epoch_info['accuracy'] = 100.0 * (epoch_info['correct'] / epoch_info['samples'])
    if show_epoch_info:
        print('Epoch: Samples={epoch_samples}, Loss={epoch_loss:.8f}'
              .format(epoch_loss=epoch_info['loss'], epoch_samples=epoch_info['samples']))
    return epoch_info


# train siamese batch
def train_siamese_batch(batch_inputs, batch_target, model, criterion, device, optimizer, make_prediction=False):
    assert len(batch_inputs) > 0
    batch_info = dict()
    batch_info['size'] = len(batch_inputs[0])
    batch_target = batch_target.to(device=device)
    for i in range(len(batch_inputs)):
        batch_inputs[i] = batch_inputs[i].to(device=device)
    optimizer.zero_grad()
    for i in range(len(batch_inputs)):
        if batch_inputs[i].dim() == 3:
            # [batch, height, width] -> [batch, channel, height, width]
            batch_inputs[i] = batch_inputs[i].unsqueeze(dim=1)
    batch_outputs = model(batch_inputs)
    assert len(batch_outputs) > 0
    assert len(batch_outputs) == len(batch_inputs)
    batch_loss = criterion(outputs=batch_outputs, target=batch_target)
    batch_loss.backward()
    batch_info['loss'] = batch_loss.item()
    if make_prediction:
        output_distance = pairwise_distance(x1=batch_outputs[0], x2=batch_outputs[1])
        batch_prediction = (output_distance < 1).long().squeeze()
        batch_info['correct'] = batch_prediction.eq(batch_target.view_as(other=batch_prediction)).sum().item()
        batch_info['accuracy'] = 100. * batch_info['correct'] / batch_info['size']
    optimizer.step()
    return batch_info


# test siamese epoch
def test_siamese_epoch(model, criterion, device, loader):
    batches = len(loader)
    samples = len(loader.dataset)
    model.eval()
    batch_losses = []
    batch_corrects = []
    with torch.no_grad():
        processed_samples = 0
        for batch_index, batch in enumerate(iterable=loader):
            batch_no = batch_index + 1
            batch_inputs, batch_target = batch
            assert len(batch_inputs) > 0
            batch_size = len(batch_inputs[0])
            processed_samples += batch_size
            batch_target = batch_target.to(device=device)
            for i in range(len(batch_inputs)):
                batch_inputs[i] = batch_inputs[i].to(device=device)
            for i in range(len(batch_inputs)):
                if batch_inputs[i].dim() == 3:
                    # [batch, height, width] -> [batch, channel, height, width]
                    batch_inputs[i] = batch_inputs[i].unsqueeze(dim=1)
            batch_outputs = model(batch_inputs)
            assert len(batch_outputs) > 0
            assert len(batch_outputs) == len(batch_inputs)
            batch_loss = criterion(outputs=batch_outputs, target=batch_target)
            batch_losses.append(batch_loss)
            output_distance = pairwise_distance(x1=batch_outputs[0], x2=batch_outputs[1])
            batch_prediction = (output_distance < 1).long().squeeze()
            batch_correct = batch_prediction.eq(batch_target.view_as(other=batch_prediction)).sum().item()
            batch_corrects.append(batch_correct)
            batch_accuracy = 100. * batch_correct / batch_size
            print('Batch: [{batch_no}/{batches}], '
                  'Processed=[{processed_samples}/{samples}], '
                  'Loss={batch_loss:.8f}, '
                  'Correct=[{batch_correct}/{batch_size}], '
                  'Accuracy={batch_accuracy:.2f}%'.format(batch_no=batch_no, batches=batches,
                                                          processed_samples=processed_samples, samples=samples,
                                                          batch_size=batch_size, batch_correct=batch_correct,
                                                          batch_accuracy=batch_accuracy, batch_loss=batch_loss))

        epoch_loss = sum(batch_losses) / samples
        epoch_correct = sum(batch_corrects)
        epoch_accuracy = 100.0 * (sum(batch_corrects) / samples)
        print('Epoch:',
              'Loss={epoch_loss:.8f}, '
              'Correct=[{epoch_correct}/{samples}], '
              'Accuracy={epoch_accuracy:.2f}%'.format(epoch_loss=epoch_loss, epoch_correct=epoch_correct,
                                                      epoch_accuracy=epoch_accuracy, samples=samples))


# siamese cnn model
class SiameseCNN(Module, ABC):
    def __init__(self):
        super(SiameseCNN, self).__init__()
        self.feature_extraction = Sequential(
            OrderedDict([
                ('conv_1', Conv2d(in_channels=1, out_channels=16, stride=(1, 1), kernel_size=(3, 3), padding=0)),
                # 16x26x26
                ('bn_1', BatchNorm2d(num_features=16)),
                ('relu_1', ReLU(inplace=False)),
                ('conv_2', Conv2d(in_channels=16, out_channels=32, stride=(1, 1), kernel_size=(3, 3), padding=0)),
                # 32x24x24
                ('bn_2', BatchNorm2d(num_features=32)),
                ('relu_2', ReLU(inplace=False)),
                ('pool_1', MaxPool2d(stride=2, kernel_size=2, padding=0)),  # 32x12x12
                ('drop_1', Dropout(p=0.1)),
                ('conv_3', Conv2d(in_channels=32, out_channels=64, stride=(1, 1), kernel_size=(3, 3), padding=0)),
                # 64x10x10
                ('bn_3', BatchNorm2d(num_features=64)),
                ('relu_3', ReLU(inplace=False)),
                ('conv_4', Conv2d(in_channels=64, out_channels=128, stride=(1, 1), kernel_size=(3, 3), padding=0)),
                # 128x8x8
                ('bn_4', BatchNorm2d(num_features=128)),
                ('relu_4', ReLU(inplace=False)),
                ('pool_2', MaxPool2d(stride=2, kernel_size=2, padding=0)),  # 128x4x4
                ('drop_2', Dropout(p=0.1)),
            ])
        )
        self.classifier = Sequential(
            OrderedDict([
                ('flatten', Flatten(start_dim=1)),  # 2048
                ('fc_1', Linear(in_features=2048, out_features=1024)),  # 1024
                ('relu_5', ReLU(inplace=False)),
                ('fc_2', Linear(in_features=1024, out_features=2)),  # 2
            ])
        )

    def forward_single(self, x):
        features = self.feature_extraction(x.float())
        classes = self.classifier(features)
        return classes

    def forward(self, inputs: list):
        assert len(inputs) > 0
        return [self.forward_single(inputs[i]) for i in range(len(inputs))]


class ContrastiveLoss(torch.nn.Module, ABC):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, outputs, target):
        assert len(outputs) == 2
        assert outputs[0].size() == outputs[1].size()
        assert outputs[0].dim() == 2
        assert outputs[1].dim() == 2
        euclidean_distance = torch.sqrt(torch.sum(torch.pow((outputs[0] - outputs[1]), exponent=2), dim=1))
        assert outputs[0].size()[0] == target.shape[0]
        assert outputs[0].size()[0] > 0
        assert target.dim() == 1
        losses = (target * torch.pow(euclidean_distance, 2) + (1 - target) * torch.pow(
            torch.clamp((self.margin - euclidean_distance), min=0.0), 2)) / 2.0
        loss = torch.sum(losses) / torch.numel(losses)
        return loss


def generate_class_with_indexes(targets):
    class_with_indexes = dict()
    for index in range(len(targets)):
        target = int(targets[index])
        if target not in class_with_indexes.keys():
            class_with_indexes[target] = []
        class_with_indexes[target].append(index)
    return class_with_indexes


def generate_pair_indexes_with_target(class_with_indexes):
    pair_indexes_with_target = []
    classes = list(class_with_indexes.keys())
    for class_ in classes:
        indexes = class_with_indexes.get(class_)
        for index in indexes:
            random_similar_index = index
            while random_similar_index == index:
                random_similar_index = random.choice(indexes)
            target = 1
            pair_index_with_target = (index, random_similar_index), target
            pair_indexes_with_target.append(pair_index_with_target)
            for other_class in classes:
                if other_class > class_:
                    not_similar_indexes = class_with_indexes.get(other_class)
                    random_not_similar_index = index
                    while random_not_similar_index == index:
                        random_not_similar_index = random.choice(not_similar_indexes)
                    target = 0
                    pair_index_with_target = (index, random_not_similar_index), target
                    pair_indexes_with_target.append(pair_index_with_target)
    return pair_indexes_with_target


# main
def main():
    # paths
    paths = dict()
    paths['project'] = '.'
    paths['dataset'] = path.join(paths['project'], 'dataset')
    paths['weight'] = path.join(paths['project'], 'weight')
    paths['train'] = path.join(paths['project'], 'train')

    # make directories if not exist
    for key in paths.keys():
        value = paths[key]
        if not path.exists(value):
            makedirs(value)

    # settings
    argument_parser = ArgumentParser()

    # argument parser
    argument_parser.add_argument('--train-batch-size', type=int, default=512, help='batch size for training')
    argument_parser.add_argument('--test-batch-size', type=int, default=512, help='batch size for testing')
    argument_parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    argument_parser.add_argument('--learning-rate', type=float, default=1e-3, help='learning rate')
    argument_parser.add_argument('--cuda', type=bool, default=True, help='enable CUDA training')
    argument_parser.add_argument('--seed', type=int, default=1, help='random seed')

    # arguments
    arguments = argument_parser.parse_args(args=[])

    # cuda
    use_cuda = arguments.cuda and torch.cuda.is_available()

    # device
    device = torch.device('cuda' if use_cuda else 'cpu')

    # seed
    torch.manual_seed(seed=arguments.seed)
    np.random.seed(arguments.seed)

    # dataset mean
    mean = 0.1307

    # dataset standard deviation
    std = 0.3081

    # dataset
    train_dataset = MNIST(root=paths['dataset'], train=True, download=True,
                          transform=Compose([ToTensor(), Normalize(mean=mean, std=std)]))
    train_images = train_dataset.data.numpy().astype(np.float32)
    train_targets = train_dataset.targets.numpy().astype(np.float32)
    train_class_with_indexes = generate_class_with_indexes(targets=train_targets)
    train_pair_indexes_with_target = generate_pair_indexes_with_target(class_with_indexes=train_class_with_indexes)
    train_siamese_dataset = SiameseDataset(pair_indexes_with_target=train_pair_indexes_with_target, images=train_images)

    test_dataset = MNIST(root=paths['dataset'], train=False,
                         transform=Compose([ToTensor(), Normalize(mean=mean, std=std)]))
    test_images = test_dataset.data.numpy()
    test_targets = test_dataset.targets.numpy()
    test_class_with_indexes = generate_class_with_indexes(targets=test_targets)
    test_pair_indexes_with_target = generate_pair_indexes_with_target(class_with_indexes=test_class_with_indexes)
    test_siamese_dataset = SiameseDataset(pair_indexes_with_target=test_pair_indexes_with_target, images=test_images)

    # loader
    train_loader = DataLoader(dataset=train_siamese_dataset, batch_size=arguments.train_batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_siamese_dataset, batch_size=arguments.test_batch_size, shuffle=False)

    # model
    model = SiameseCNN().to(device=device)

    # optimizer
    optimizer = Adam(params=model.parameters(), lr=arguments.learning_rate)

    # loss function
    criterion = ContrastiveLoss()

    # info
    print('{} Siamese Network on MNIST {}'.format('=' * 40, '=' * 40))
    print('Feature extraction: {feature_extraction_layers}'.format(feature_extraction_layers=model.feature_extraction))
    print('Classifier: {classifier_layers}'.format(classifier_layers=model.classifier))

    # break line
    print('{}'.format('*' * 100))

    # epochs
    for epoch_number in range(1, arguments.epochs + 1):
        # train info
        print('Train: Epoch: [{epoch_number}/{epochs}]'.format(epoch_number=epoch_number, epochs=arguments.epochs))

        # train epoch
        train_info = train_siamese_epoch(model=model, criterion=criterion, device=device,
                                         loader=train_loader, optimizer=optimizer)

        # break line
        print('{}'.format('*' * 100))

        # save weight
        weight_file = 'mnist_siamese_cnn_weight_epoch_{epoch_number}.pkl'.format(epoch_number=epoch_number)
        weight_file_path = path.join(paths['weight'], weight_file)
        torch.save(obj=model.state_dict(), f=weight_file_path)
        print('"{weight_file}" file is saved as "{weight_file_path}".'
              .format(weight_file=weight_file, weight_file_path=weight_file_path))

        # break line
        print('{}'.format('*' * 100))

        # save train info
        train_info_file = 'mnist_siamese_cnn_train_epoch_{epoch_number}.npy'.format(epoch_number=epoch_number)
        train_info_file_path = path.join(paths['train'], train_info_file)
        np.save(file=train_info_file_path, arr=np.array(train_info))
        print('"{train_info_file}" file is saved as "{train_info_file_path}".'
              .format(train_info_file=train_info_file, train_info_file_path=train_info_file_path))

        # break line
        print('{}'.format('*' * 100))

        # test info
        print('Test: Epoch: [{epoch_number}/{epochs}]'.format(epoch_number=epoch_number, epochs=arguments.epochs))

        # test epoch
        test_siamese_epoch(model=model, criterion=criterion, device=device, loader=test_loader)

        # break line
        print('{}'.format('*' * 100))


if __name__ == '__main__':
    main()
