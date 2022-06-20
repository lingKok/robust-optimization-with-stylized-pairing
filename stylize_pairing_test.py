import torch
from torchvision.models import resnet34,googlenet,vgg16,mobilenet_v3_small
from utils import tensor2variable
import torch.nn as nn
from loaddata import loadData,load_cifar,load_cifar100,load_tinyimage
import os
import argparse
# from utils import AverageMeter,accuracy
import logging
from utils import load_state_dict,state_dict

# model=resnet34(pretrained=True)
# path='models/cifar/resnet_0.pth'
# if os.path.exists(path):
#     model.load_state_dict(torch.load(path))

def find_adversary(
        start_image,
        label_now,
        steps=100,
        LR=1e-4,#cifar 1e-4/1e-5
        gradient_signs_only=True,
        use_ratio=False,
        verbose=1,
):
    image_now = start_image
    image_start = image_now

    scores_list = []
    images_list = []
    L2_perturbation_list = []
    Linfty_perturbation_list = []

    for it in range(steps):

        score_now , grad= get_score_and_grad(
            image_now,
            label_now,
            model,
        )

        # early stopping
        # if (target_score is not None) and (score_now < target_score):
        #     break

        images_list.append(image_now)

        scores_list.append(score_now)
        L2_perturbation_list.append(torch.linalg.norm(image_now - image_start).cpu())
        Linfty_perturbation_list.append(torch.max(torch.abs(image_now - image_start)).cpu())

        if it == 0:
            score_start = score_now

        # if it % verbose == 0:
        #     print("image=" + str(i) + " step=" + str(it) + " OOD score=" + str(score_now))

        # if ((first_gradient_only == False) or (it == 0)):
        #     image_grad_now = get_input_grad_jit(
        #         image_now,
        #         params,
        #         use_ratio=use_ratio,
        #     )
        image_grad_now = grad
        if gradient_signs_only:
            image_grad_now = ((image_grad_now > 0.0) * 2.0 - 1.0)

        image_now = image_now + LR * image_grad_now
        image_now = torch.clip(image_now,0,1)

    return image_start, scores_list, images_list, L2_perturbation_list, Linfty_perturbation_list
def get_score_and_grad(image,label,model):
    """
    input: (1,3,224,224)
    output:0 or 1
    """
    model.eval()
    image=tensor2variable(image,device,requires_grad=True)
    output=model(image)
    loss_fn=nn.CrossEntropyLoss()
    loss=loss_fn(output,label)
    loss.backward()
    pred=torch.max(output,1)[1]
    if pred.eq(label.view_as(pred)):
        pred_res=1
    else:
        pred_res=0
    return pred_res,image.grad
def predict(image,label):
    """
    input: (1,3,224,224)
    output:0 or 1
    """
    model.eval()
    output=model(image)
    print(output.shape)
    pred=torch.max(output,1)[1]
    print(pred)
    print(label)
    if pred.eq(label.view_as(pred)):
        pred_res=1
    else:
        pred_res=0
    return pred_res
# def get_input_grad(image,label,model):
#     image=tensor2variable(image,device,requires_grad=True)
#     output = model(image)
#     loss = nn.CrossEntropyLoss(output,label)
#     loss.backward()
#     return image.grad

def get_acc_pertubation(scores_list,l2_perturbation_list,linfty_perturbation_list):
    import matplotlib.pyplot as plt
    from scipy import interpolate
    import numpy as np

    L2_to_score_interpolated_fns = []
    linfty_to_score_interpolated_fns=[]
    for i in range(len(scores_list)):
        x_now = l2_perturbation_list[i]
        y_now = scores_list[i]

        f_now = interpolate.interp1d(x_now, y_now)
        L2_to_score_interpolated_fns.append(f_now)
    L2_values_to_scan = np.linspace(
        np.min(l2_perturbation_list),
        np.max(l2_perturbation_list),
        33
    )
    for i in range(len(scores_list)):
        x_now = linfty_perturbation_list[i]
        y_now = scores_list[i]

        f_now = interpolate.interp1d(x_now, y_now)
        linfty_to_score_interpolated_fns.append(f_now)
    linfty_values_to_scan = np.linspace(
        np.min(linfty_perturbation_list),
        np.max(linfty_perturbation_list),
        100
    )
    collected_acc_l2 = []
    l2_scan=[]
    collected_acc_linfty=[]
    linfty_scan=[]


    for i, L2_now in enumerate(L2_values_to_scan):

        bools_morethanmin = L2_now >= np.min(l2_perturbation_list, axis=1)
        bools_lessthanmax = L2_now <= np.max(l2_perturbation_list, axis=1)
        bools = np.logical_and(bools_morethanmin, bools_lessthanmax)

        if np.sum(bools) > len(bools)/2:  # not missing any images

            scores_selected = [int(f_now(L2_now)) for (i, f_now) in enumerate(L2_to_score_interpolated_fns) if bools[i]]
            # print(type(scores_selected))
            # print(scores_selected)
            scores_selected = torch.tensor(scores_selected)
            acc=scores_selected.sum()/scores_selected.size()[0]
            collected_acc_l2.append(acc)
            l2_scan.append((L2_now))

        # scores_selected = [int(f_now(L2_now)) for (i, f_now) in enumerate(L2_to_score_interpolated_fns) ]
        # # print(type(scores_selected))
        # # print(scores_selected)
        # scores_selected = torch.tensor(scores_selected)
        # acc=scores_selected.sum()/scores_selected.size()[0]
        # collected_acc_l2.append(acc)
    for i ,linfty_now in enumerate(linfty_values_to_scan):
        bools_morethanmin = linfty_now >= np.min(linfty_perturbation_list, axis=1)
        bools_lessthanmax = linfty_now <= np.max(linfty_perturbation_list, axis=1)
        bools = np.logical_and(bools_morethanmin, bools_lessthanmax)

        if np.sum(bools) > len(bools)/2:  # not missing any images
            scores_selected=[int( f_now(linfty_now) )for (i,f_now)in enumerate(linfty_to_score_interpolated_fns) if bools[i]]


            scores_selected = torch.tensor(scores_selected)
            acc= scores_selected.sum()/scores_selected.size()[0]
            collected_acc_linfty.append(acc)
            linfty_scan.append(linfty_now)
    return collected_acc_l2,collected_acc_linfty,l2_scan,linfty_scan

# def acc(image,label,model):

    
if __name__ =='__main__':
    import numpy as np


    parse = argparse.ArgumentParser()
    parse.add_argument('dataset', choices=['imagenet', 'cifar','imagenet10','cifar100'])
    parse.add_argument('--save_path', type=str, default='output/acc2perturbation.pt')
    # parse.add_argument('--weight_path', type=str)
    parse.add_argument('--batch_size', type=int, default=10)
    parse.add_argument('--step', type=int, default=30)
    parse.add_argument('--model', type=str, default='resnet34', choices=['resnet34', 'googlenet','mobilenet','vgg'])
    parse.add_argument('--type',type=str,default='normal',choices=['normal','adamin','adv','rand','adamin_normal'])
    parse.add_argument('--rand',default=False)
    args = parse.parse_args()
    log_dir='log/'+args.dataset+'_'+args.model+'_'+args.type+'_test.txt'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s  : %(message)s',
                        handlers=[logging.FileHandler(log_dir, mode='w'), logging.StreamHandler()])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    if args.type == 'normal':
        weight_path = 'models/' + args.dataset + '/normal/' + args.model + '_best.pth'
        save_path = 'output/{}_{}_acc2pert_normal.pth'.format(args.dataset,args.model)
    if args.type == 'adamin':
        weight_path = 'models/' + args.dataset + '/' + args.model + '_best.pth'
        save_path = 'output/{}_{}_acc2pert_adamin.pth'.format(args.dataset, args.model)
    if args.type == 'rand':
        weight_path = 'models/' + args.dataset + '/' + args.model + '_best_rand.pth'
        save_path = 'output/{}_{}_acc2pert_rand.pth'.format(args.dataset, args.model)
    if args.type == 'adv':
        weight_path = 'models/' + args.dataset + '/' + args.model + '_adv_best.pth'
        save_path = 'output/{}_{}_acc2pert_adv.pth'.format(args.dataset, args.model)
    if args.type == 'adamin_normal':
        weight_path = 'models/' + args.dataset + '/' + args.model + '_normal_best.pth'
        save_path = 'output/{}_{}_acc2pert_adamin_normal.pth'.format(args.dataset, args.model)
    if args.dataset == 'imagenet':
        output_feature = 100
        logging.info('dataset is imagenet.')
        train_load, val_load = loadData('dataset/', batch_size=256)


    if args.dataset == 'imagenet10':
        output_feature = 10
        logging.info('dataset is imagenet.')
        train_load, val_load = loadData('dataset1/', batch_size=256)

    if args.dataset == 'cifar':
        logging.info('dataset is cifar10!')
        output_feature = 10
        train_load, val_load = load_cifar('dataset/', batch_size=256)
        # model = resnet34(pretrained=True)

    if args.dataset == 'cifar100':
        logging.info('dataset is cifar100!')
        output_feature = 100
        train_load, val_load = load_cifar100('dataset/', batch_size=256)
    elif args.dataset == 'tinyimage':
        logging.info('dataset is tinyimagenet!')
        output_feature=200
        train_load, val_load = load_tinyimage('tiny-immagenet-200',batchsize=256)


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

    if os.path.exists(weight_path):

        load_state_dict(model, torch.load(weight_path))
        logging.info('load weight success!')
    # model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)
    # model = resnet34()
    # in_feature = model.fc.in_features
    # model.fc = nn.Linear(in_feature, 10)
    # weight_path = 'models/cifar/normal/resnet34_best_2.pth'
    # load_state_dict(model, torch.load(weight_path), strict=False)
    # model.to(device)
    # train_load, test_load = load_cifar('dataset/', batch_size=256)
    iter=10
    length=args.batch_size
    images_count=args.batch_size*iter
    print(images_count)
    score_grid=np.zeros((images_count,args.step))*float('NaN')
    l2_perturbation_size=np.zeros((images_count,args.step))*float('NaN')
    linfty_perturbation_size=np.zeros((images_count,args.step))*float('NaN')
    # linfty_perturbation_list=[]
    count=0
    sum=0
    for i ,data in enumerate(val_load):
        if count ==images_count : break
        image,label=data
        image,label = image.to(device),label.to(device)
        for j in range(len(image)):
            sum += 1
            if count==images_count : break
            print(count)
            print(sum)
            res=predict(image[j].unsqueeze(0),label[j])
            if res==1:

                image_start, scores, images, L2_perturbation, Linfty_perturbation = find_adversary(image[j].unsqueeze(0),label[j].unsqueeze(0),steps=args.step)
                score_grid[count]=scores
                l2_perturbation_size[count]=L2_perturbation
                linfty_perturbation_size[count]=Linfty_perturbation
                count += 1
    print('score : ',score_grid)
    #
    # print('perturbation:',linfty_perturbation_list)
    # print(len(linfty_perturbation_list[0]))
    # print(np.max(l2_perturbation_list))
    acc_l2 , acc_linfty, l2_value2scan,linfty_value2scan=get_acc_pertubation(score_grid,l2_perturbation_size,linfty_perturbation_size)
    torch.save([acc_l2 , acc_linfty, l2_value2scan,linfty_value2scan],save_path)

