import argparse
import logging
import os

import numpy as np
import torch
from torchvision.models import resnet34, googlenet, mobilenet_v3_small
import torch.nn as nn

from loaddata import loadData,load_cifar,load_cifar100
from utils import load_state_dict
def accuracy(model,data,label,device):
    model.to(device)
    data = torch.from_numpy(data)
    data = data.to(device)
    label = torch.from_numpy(label).to(device)
    label = torch.max(label,1)[1]
    output = model(data)
    pred = torch.max(output,1)[1]
    correct = pred == label
    print(pred.size(0))
    print(correct.sum())
    acc = correct.sum().item()/pred.size(0)
    print(' Accuracy is {}'.format(acc))



if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='imagenet', choices=['cifar', 'imagenet','imagenet10','cifar100','tinyimage'])
    parser.add_argument('--model',type=str,default='resnet34',choices=['resnet34','googlenet','vgg','mobilenet'])
    parser.add_argument('--attack',default='deepfool')
    args = parser.parse_args()
    if args.dataset == 'imagenet':
        output_feature = 100
        # logging.info('dataset is imagenet.')
        # train_load, val_load = loadData('dataset/', batch_size=32)
        # save_path = 'models/imagenet/normal/' + args.model + '_best.pth'
        # save_path_adamin = 'models/imagenet/' + args.model + '_best.pth'
        # save_path_end = 'models/imagenet/normal/{}_end.pth'.format(args.model)
        # save_path_adv = 'models/imagenet/'
    if args.dataset == 'cifar100':
        # logging.info('dataset is cifar100!')
        output_feature=100
        # # train_load, val_load = load_cifar100('dataset/',batch_size=64)
        # save_path = 'models/cifar100/normal/' + args.model + '_best.pth'
        # save_path_adamin = 'models/cifar100/' + args.model + '_best.pth'
        # save_path_end = 'models/imagenet/normal/{}_end.pth'.format(args.model)
    if args.dataset == 'cifar':
        # logging.info('dataset is cifar10!')
        output_feature = 10
        # train_load, val_load = load_cifar('dataset/', batch_size=128)
        # model = resnet34(pretrained=True)

        # save_path = 'models/cifar/normal/' + args.model + '_best.pth'
        # save_path_adamin = 'models/cifar/' + args.model + '_best.pth'
        # save_path_end = 'models/imagenet/normal/{}_end.pth'.format(args.model)
    if args.model == 'resnet34':
        model = resnet34(pretrained=True)
        in_feature = model.fc.in_features
        model.fc = nn.Linear(in_feature, output_feature)
        # logging.info('model is resnet34!')
    if args.model == 'googlenet':
        model = googlenet(pretrained=True)
        in_feature = model.fc.in_features
        model.fc = nn.Linear(in_feature, output_feature)
        # logging.info('model is googlenet!')
    if args.model == 'mobilenet':
        model = mobilenet_v3_small(pretrained=True)
        in_feature = 1024
        model.classifier.__dict__['_modules']['3'] = nn.Linear(in_feature, out_features=output_feature)
        # logging.info('model is mobilenet!')
    save_path = 'models/{}/normal/{}_end_1.pth'.format(args.dataset,args.model)
    save_path_adamin = 'models/{}/{}_best.pth'.format(args.dataset,args.model)
    save_path_end = 'models/{}/normal/{}_end.pth'.format(args.dataset,args.model)
    save_path_adv = 'models/{}/{}_adv_best.pth'.format(args.dataset,args.model)
    if os.path.exists(save_path):
        load_state_dict(model, torch.load(save_path))
        print('load weight success!')
    # model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)
    dataset = args.dataset.upper()
    attack = args.attack.upper()
    adv_data_path = './AdversarialExampleDatasets/{}/{}/{}_AdvExamples.npy'.format(attack,dataset,attack)
    adv_label_path = './AdversarialExampleDatasets/{}/{}/{}_TrueLabels.npy'.format(attack, dataset, attack)
    adv_data = np.load(adv_data_path)
    adv_label = np.load(adv_label_path)
    print('normal best training accuracy:')
    accuracy(model,adv_data,adv_label,device)
    load_state_dict(model,torch.load(save_path_adamin))
    print('adamin training accuracy:')
    accuracy(model,adv_data,adv_label,device)

    load_state_dict(model,torch.load(save_path_end))
    print('normal end training accuracy:')
    accuracy(model, adv_data, adv_label, device)
    load_state_dict(model, torch.load(save_path_adv))
    print('adv training accuracy:')
    accuracy(model, adv_data, adv_label, device)

