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
import torch.nn.functional as F
write=SummaryWriter('log_dir')




def adv_generation(model, var_natural_images):
    model.eval()
    clone_var_nature_images = var_natural_images.clone()
    clone_var_nature_images.requires_grad = True
    logits = model(clone_var_nature_images)
    llc_label = torch.min(logits, dim=1)[1]
    loss_llc = F.cross_entropy(logits, llc_label)
    gradients_llc = torch.autograd.grad(loss_llc, clone_var_nature_images)[0]

    clone_var_nature_images.requires_grad = False

    gradients_sign = torch.sign(gradients_llc)
    eps = 0.1

    with torch.no_grad():
        adv_images = var_natural_images - eps*gradients_sign
    ret_adv_images = torch.clamp(adv_images,min=0.0,max=1.0)

    return ret_adv_images



def train_one_epoch(epoch,args):
    # model.train()
    for i, data in enumerate(train_load):
        images, labels = data
        len = images.size()[0]
        images, labels = images.to(device), labels.to(device)
        model.eval()
        adv_images = adv_generation(model, var_natural_images=images)

        model.train()
        optimizer.zero_grad()

        logits_nat = model(images)
        loss_nat = loss_func(logits_nat,labels)

        logits_adv = model(adv_images)
        loss_adv = loss_func(logits_adv,labels)
        
        loss = (loss_nat + loss_adv)/2.0
        
        train_loss.update(loss.cpu().data, len)
        loss.backward()
        optimizer.step()
        if i%200==0:
            write.add_images(args.dataset+'_'+args.model+'_adv_images',adv_images)
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
    log_dir = 'log/'+args.dataset + '_' + args.model + '_adv.txt'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(log_dir, mode='a'), logging.StreamHandler()])
    if args.dataset == 'imagenet':
        output_feature = 100
        logging.info('dataset is imagenet.')
        train_load, val_load = loadData('dataset/', batch_size=32)

        save_path = 'models/imagenet/' + args.model + '_adv_best.pth'
        save_path_end = 'models/imagenet/' + args.model + '_adv_end.pth'
    if args.dataset == 'imagenet10':
        output_feature = 10
        logging.info('dataset is imagenet10.')
        train_load, val_load = loadData('dataset1/', batch_size=32)
        save_path = 'models/imagenet10/' + args.model + '_adv_best.pth'
        save_path_end = 'models/imagenet10/' + args.model + '_adv_end.pth'
    if args.dataset == 'cifar100':
        logging.info('dataset is cifar100!')
        output_feature=100
        train_load, val_load = load_cifar100('dataset/',batch_size=64)
        save_path = 'models/cifar100/' + args.model + '_adv_best.pth'
        save_path_end = 'models/cifar100/' + args.model + '_adv_end.pth'
    if args.dataset == 'cifar':
        logging.info('dataset is cifar10!')
        output_feature = 10
        train_load, val_load = load_cifar('dataset/', batch_size=128)
        # model = resnet34(pretrained=True)

        save_path = 'models/cifar/' + args.model + '_adv_best.pth'
        save_path_end = 'models/cifar/' + args.model + '_adv_end.pth'
    elif args.dataset == 'tinyimage':
        logging.info('dataset is tinyimagenet!')
        output_feature=200
        train_load, val_load = load_tinyimage('tiny-imagenet-200',batchsize=256)
        save_path = 'models/tinyimage/' + args.model + '_adv_best.pth'
        save_path_end = 'models/tinyimage/' + args.model + '_adv_end.pth'
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