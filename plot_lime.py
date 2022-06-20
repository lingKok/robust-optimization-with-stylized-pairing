import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json

import torch
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.models import resnet34, googlenet, mobilenet_v3_small


def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()
def get_input_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    return transf

def get_input_tensors(img):
    transf = get_input_transform()
    # unsqeeze converts single image to batch of 1
    return transf(img).unsqueeze(0)


def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])

    return transf
#
def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        # transforms.Resize((256, 256)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # normalize
    ])

    return transf
# resize and take the center part of image to what our model expects

if __name__ == '__main__':
    import argparse
    from utils import load_state_dict
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='imagenet')
    parser.add_argument('--model',default='resnet34')
    args = parser.parse_args()
    if args.dataset == 'imagenet':
        output_feature = 100
        # logging.info('dataset is imagenet.')
        # train_load, val_load = loadData('dataset/', batch_size=32)
        save_path = 'models/imagenet/normal' + args.model + '_best.pth'
        save_path_adamin = 'models/imagenet/' + args.model + '_best.pth'
    if args.dataset == 'cifar100':
        # logging.info('dataset is cifar100!')
        output_feature = 100
        # train_load, val_load = load_cifar100('dataset/',batch_size=64)
        save_path = 'models/cifar100/normal' + args.model + '_best.pth'
        save_path_adamin = 'models/cifar100/' + args.model + '_best.pth'
    if args.dataset == 'cifar':
        # logging.info('dataset is cifar10!')
        output_feature = 10
        # train_load, val_load = load_cifar('dataset/', batch_size=128)
        # model = resnet34(pretrained=True)

        save_path = 'models/cifar/normal' + args.model + '_best.pth'
        save_path_adamin = 'models/cifar/' + args.model + '_best.pth'
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

    if os.path.exists(save_path):
        load_state_dict(model, torch.load(save_path_adamin))
        print('load weight success!')
    img = get_image('./dataset/val/n01729322/ILSVRC2012_val_00002568.JPEG')
    plt.imshow(img)
    print(img.size)
    # model = models.resnet34(pretrained=True)

    # idx2label, cls2label, cls2idx = [], {}, {}
    # with open(os.path.abspath('./data/imagenet_class_index.json'), 'r') as read_file:
    #     class_idx = json.load(read_file)
    #     idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    #     cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
    #     cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))}

    # img_t = get_input_tensors(img)
    # model.eval()
    # logits = model(img_t)

    # probs = F.softmax(logits, dim=1)
    # probs5 = probs.topk(5)
    # tuple((p,c, idx2label[c]) for p, c in zip(probs5[0][0].detach().numpy(), probs5[1][0].detach().numpy()))



    pill_transf = get_pil_transform()
    preprocess_transform = get_preprocess_transform()



    # test_pred = batch_predict([pill_transf(img)])
    # test_pred.squeeze().argmax()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(pill_transf(img)),
                                             batch_predict, # classification function
                                             top_labels=5,
                                             hide_color=0,
                                             num_samples=5000) # number of images that will be sent to classification function

    # temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    # img_boundry1 = mark_boundaries(temp/255.0, mask)
    # plt.imshow(img_boundry1)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=20, hide_rest=True)
    img_boundry2 = mark_boundaries(temp/255.0, mask)
    plt.imshow(img_boundry2)
    plt.show()