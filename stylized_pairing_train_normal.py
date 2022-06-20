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
def train_one_epoch(epoch,args):
    model.train()
    for i,data in enumerate(train_load):
        images,labels=data
        len=images.size()[0]
        images,labels=images.to(device),labels.to(device)
        stylized_images=get_stylized_imges(images,rand=args.rand)
        # print(images.shape)
        # print(stylized_images.shape)
        data=torch.cat((images,stylized_images),0)
        label=torch.cat((labels,labels))
        optimizer.zero_grad()
        output=model(data)
        loss1=loss_func(output,label)
        # loss2=loss_func2(output[:len],output[len:])
        loss=loss1
        train_loss.update(loss.cpu().data,len)
        loss.backward()
        optimizer.step()
        if i%200==0:
            write.add_images(args.dataset+'_'+args.model+'_adamin_images',data)
            # print('epoch: {} loss: {:2f}'.format(epoch,train_loss.avg))
            logging.info('epoch: {} loss: {:2f}'.format(epoch,train_loss.avg))
    # save_path='models/cifar/resnet_'+str(epoch)+'.pth'
    # torch.save(model.state_dict(),save_path)



def validate():
    model.eval()
    test_loss.reset()
    test_acc.reset()
    for i ,data in  enumerate(val_load):
        images,labels = data
        len=images.size()[0]
        images, labels = images.to(device), labels.to(device)
        # stylized_images = get_stylized_imges(images)
        # data=torch.cat((images,stylized_images),0)
        # label=torch.cat((labels,labels))
        output=model(images)
        loss=loss_func(output,labels)
        # loss2=loss_func2(output[:len],output[len:])
        # loss=loss1+loss2
        test_loss.update(loss.cpu().data,len)
        acc=accuracy(output,labels)
        test_acc.update(acc,len)
        if i%200==0:
            # print('loss:{:.2f} acc: {:.2f}'.format(test_loss.avg,test_acc.avg))
            logging.info('loss:{:.2f} acc: {:.2f}'.format(test_loss.avg,test_acc.avg))
    logging.info('overall loss:{:.2f} acc: {:.2f}'.format(test_loss.avg,test_acc.avg))
    return test_acc.avg



# for i,data in enumerate(train_load):
#     img,label=data
#     img,label=img.to(device),label.to(device)
#     for j in range(len(img)):
#         output=convert2stylized(img[j].unsqueeze(0))
#         output = output.cpu()
#         output_name = output_dir / '{:d}{:d}_interpolation{:s}'.format(
#             i,j, '.jpg')
#         save_image(output, str(output_name))
#         output_name = output_dir / 'orig_{:d}{:d}_interpolation{:s}'.format(
#             i, j, '.jpg')
#         save_image(img[j].cpu(),str(output_name))
#     print(label.data.cpu())_train

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', default='cifar', choices=['cifar', 'imagenet','imagenet10','cifar100','tinyimage'])
    parser.add_argument('--model',type=str,default='resnet34',choices=['resnet34','googlenet','vgg','mobilenet'])
    parser.add_argument('--rand',default=False)
    args = parser.parse_args()
    log_dir = 'log/'+args.dataset + '_' + args.model + '_adamin_nomal.txt'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(log_dir, mode='a'), logging.StreamHandler()])
    if args.dataset == 'imagenet':
        output_feature = 100
        logging.info('dataset is imagenet.')
        train_load, val_load = loadData('dataset/', batch_size=32)

        save_path = 'models/imagenet/' + args.model + '_normal_best.pth'
        save_path_end = 'models/imagenet/' + args.model + '_normal_end.pth'
    if args.dataset == 'imagenet10':
        output_feature = 10
        logging.info('dataset is imagenet10.')
        train_load, val_load = loadData('dataset1/', batch_size=32)
        save_path = 'models/imagenet10/' + args.model + '_normal_best.pth'
        save_path_normal_end = 'models/imagenet10/' + args.model + '_normal_end.pth'
    if args.dataset == 'cifar100':
        logging.info('dataset is cifar100!')
        output_feature=100
        train_load, val_load = load_cifar100('dataset/',batch_size=64)
        save_path = 'models/cifar100/' + args.model + '_normal_best.pth'
        save_path_end = 'models/cifar100/' + args.model + '_normal_end.pth'
    if args.dataset == 'cifar':
        logging.info('dataset is cifar10!')
        output_feature = 10
        train_load, val_load = load_cifar('dataset/', batch_size=128)
        # model = resnet34(pretrained=True)

        save_path = 'models/cifar/' + args.model + '_normal_best.pth'
        save_path_end = 'models/cifar/' + args.model + '_normal_end.pth'
    elif args.dataset == 'tinyimage':
        logging.info('dataset is tinyimagenet!')
        output_feature=200
        train_load, val_load = load_tinyimage('tiny-imagenet-200',batchsize=256)
        save_path = 'models/tinyimage/' + args.model + '_normal_best.pth'
        save_path_end = 'models/tinyimage/' + args.model + '_normal_end.pth'
    if args.model == 'resnet34':
        model = resnet34(pretrained=True)
        in_feature = model.fc.in_features
        model.fc = nn.Linear(in_feature, output_feature)
        logging.info('model is resnet34!')
    if args.model == 'googlenet':
        model = googlenet(pretrained=True)
        in_feature = model.fc.in_features
        model.fc = nn.Linear(in_feature, output_feature)
        logging.info('model is googlenet!')
    if args.model == 'mobilenet':
        model = mobilenet_v3_small(pretrained=True)
        in_feature = 1024
        model.classifier.__dict__['_modules']['3'] = nn.Linear(in_feature, out_features=output_feature)
        logging.info('model is mobilenet!')
    if args.model == 'vgg':
        model = vgg16(pretrained=True)
        in_feature = 4096
        model.classifier.__dict__['_modules']['6'] = nn.Linear(in_feature, out_features=output_feature)
        logging.info('model is vgg16!')

    if os.path.exists(save_path):
        load_state_dict(model, torch.load(save_path))
        logging.info('load weight success!')
    # model = nn.DataParallel(model, device_ids=[0, 1])
    if args.rand:
        save_path = 'models/cifar100/' + args.model + '_normal_best_rand.pth'
        save_path_end = 'models/cifar100/' + args.model + '_end_rand.pth'
    model.to(device)
    # output_dir=Path('output')
    # test_load=loadData('dataset/')

    # model=resnet34(pretrained=True)
    # print(model)

    loss_func = nn.CrossEntropyLoss()
    loss_func2 = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    train_loss = AverageMeter()

    test_loss = AverageMeter()
    test_acc = AverageMeter()


    best_acc=0
    for i in range(20):
        logging.info('starting adamin training!')
        train_one_epoch(i,args)
        acc=validate()
        if best_acc<acc:
            best_acc=acc

            torch.save(state_dict(model), save_path)
            logging.info('saving best model success!')
    torch.save(state_dict(model),save_path_end)
write.close()