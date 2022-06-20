import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import torch
import argparse

def loadData(path='./imagenet',batch_size=10):

    #set the path of dataset
    train_path = os.path.join(path + 'train')
    val_path = os.path.join(path + 'val')

    #set transform method, the default img size of imagenet is 224
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize(330),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalizer
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # normalizer
    ])

    #import train\val dataset
    train_data = datasets.ImageFolder(root=train_path, transform=train_transform,)
    valid_data = datasets.ImageFolder(root=val_path, transform=valid_transform,)

    #load dataset into Dataloader
    train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)

    print(len(train_data)) #1151273
    print(len(valid_data)) #50000
    return train_queue,valid_queue

def load_cifar(root='dataset/',batch_size=128):
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as transform
    from torch.utils.data import DataLoader
    transform_train=transform.Compose([
        transforms.Resize(330),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor()
    ])
    transform_test=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])


    train_data=CIFAR10(root=root,train=True,transform=transform_train,download=True)
    test_data = CIFAR10(root=root,train=False,transform=transform_test,download=True)
    train_load = DataLoader(train_data,batch_size=batch_size,shuffle=True)
    test_load = DataLoader(test_data,batch_size=batch_size,shuffle=True)
    return train_load,test_load

def load_cifar100(root='dataset/',batch_size=128):
    from torchvision.datasets import CIFAR100
    import torchvision.transforms as transform
    from torch.utils.data import DataLoader
    transform_train=transform.Compose([
        transforms.Resize(330),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor()
    ])
    transform_test=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])


    train_data=CIFAR100(root=root,train=True,transform=transform_train,download=True)
    test_data = CIFAR100(root=root,train=False,transform=transform_test,download=True)
    train_load = DataLoader(train_data,batch_size=batch_size,shuffle=True)
    test_load = DataLoader(test_data,batch_size=batch_size,shuffle=True)
    return train_load,test_load

def load_tinyimage(root='tiny-imagenet-200',batchsize=128):
    from valpre_tiny import TinyImageNet
    import torchvision.transforms as transform
    from torch.utils.data import DataLoader
    transform_train = transform.Compose([
        transforms.Resize(330),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_data = TinyImageNet(root=root,train=True,transform=transform_train)
    test_data = TinyImageNet(root=root,train=False,transform=transform_test)
    # print(train_data.images)
    train_load = DataLoader(train_data,batch_size=batchsize)
    test_load = DataLoader(test_data,batch_size=batchsize)
    return train_load, test_load
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch_size",type=int,default=50)
    parser.add_argument('--num_workers',type=int,default=0)
    args=parser.parse_args()
    loadData()