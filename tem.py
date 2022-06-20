from TinyImagenet import TinyImageNet
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

writer=SummaryWriter('log_dir')
data_dir = './tiny-imagenet-200/'
dataset_train = TinyImageNet(data_dir, train=True)
dataset_val = TinyImageNet(data_dir, train=False,transform=transforms.Compose([transforms.ToTensor()]))
cifar_test= CIFAR10('dataset',train=False)
print(dataset_val[0])
print(cifar_test[0])
train_load=DataLoader(dataset_train,batch_size=128)
val_load = DataLoader(dataset_val,batch_size=128)
cifar_load = DataLoader(cifar_test,batch_size=128)
for i, data in enumerate(val_load):
    images, target = data
    if i==0:
        writer.add_images('tinyimage',images)
