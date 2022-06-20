import torch

from convert2style import convert2stylized
from PIL import Image
from loaddata import loadData,load_cifar,load_cifar100,load_tinyimage
from pathlib import Path
from torchvision.utils import save_image
from torchvision.models import resnet34,googlenet,mobilenet_v3_small,vgg16
import torch.nn as nn
from utils import accuracy,AverageMeter,load_state_dict,state_dict
import argparse
import os
import logging
from torch.utils.tensorboard import SummaryWriter

write=SummaryWriter('log_dir')

def get_stylized_imges(images,rand):
    list_stylized_images=[]
    for j in range(len(images)):
        output=convert2stylized(images[j].unsqueeze(0),rand=rand)
        list_stylized_images.append(output)
    ret_stylized_images=torch.cat(list_stylized_images)
    return ret_stylized_images

def plot_image(image):
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.show()
if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='imagenet', choices=['cifar', 'imagenet','imagenet10','cifar100','tinyimage'])
    args = parser.parse_args()
    if args.dataset == 'imagenet':
        output_feature = 100
        logging.info('dataset is imagenet.')
        train_load, val_load = loadData('dataset/', batch_size=1)
    if args.dataset == 'cifar100':
        logging.info('dataset is cifar100!')
        output_feature=100
        train_load, val_load = load_cifar100('dataset/',batch_size=1)

    if args.dataset == 'cifar':
        logging.info('dataset is cifar10!')
        output_feature = 10
        train_load, val_load = load_cifar('dataset/', batch_size=1)
    image, _ = val_load.dataset[50]
    print(image.shape)
    plot_image(image.numpy().transpose(1,2,0))
    content = image.unsqueeze(0)
    print(content.shape)
    image = get_stylized_imges(image.unsqueeze(0).to(device),rand=False)
    image = image[0].cpu().numpy().transpose(1,2,0)
    print(image[0].shape)
    plot_image(image)



