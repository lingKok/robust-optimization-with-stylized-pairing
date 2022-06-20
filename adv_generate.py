import os
import shutil

import torch
from art.attacks.evasion import SaliencyMapMethod, CarliniL0Method,CarliniLInfMethod,CarliniL2Method,DeepFool,ProjectedGradientDescentPyTorch,AdversarialPatch,FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from torch import nn
from torchvision.models import resnet34, googlenet, mobilenet_v3_small, vgg16

from utils import load_state_dict,state_dict

import numpy as np

# os.makedirs('./AdversarialExampleDatasets/CWL2/IMAGENET10',exist_ok=True)
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--dataset',default='cifar', choices=['imagenet', 'cifar', 'imagenet10', 'cifar100'])
parse.add_argument('--model', type=str, default='resnet34', choices=['resnet34', 'googlenet', 'mobilenet', 'vgg'])
args = parse.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_list = ['imagenet','cifar','cifar100']#['MNIST','CIFAR10','SVHN','IMAGENET10']
loss_func = nn.CrossEntropyLoss()
batch_size = 10
attack_list = ['FGSM','PGD','DEEPFOOL','JSMA','CWL2','CWL0','CWLINF']#['DEEPFOOL','FGSM','PGD','JSMA','CWL2','CWL0','CWLINF']#,
for dataset in dataset_list:
    args.dataset = dataset
    weight_path = 'models/' + args.dataset + '/normal/' + args.model + '_best.pth'
    dataset = dataset.upper()
    for attack_name in attack_list:
        adv_examples_dir = './AdversarialExampleDatasets/'+attack_name+'/'+dataset+'/'
        clean_data_location = './CleanDatasets/'
        # raw_model_location='./RawModels/'
        # raw_model_location = '{}{}/model/{}_raw.pt'.format(
        #             raw_model_location, dataset, dataset)


        if args.dataset == 'cifar':
            # logging.info('dataset is cifar10!')
            output_feature = 10
            input_shape = (3,32,32)
            # train_load, val_load = load_cifar('dataset/', batch_size=256)
            # model = resnet34(pretrained=True)

        if args.dataset == 'cifar100':
            # logging.info('dataset is cifar100!')
            output_feature = 100
            input_shape = (3, 32, 32)
            # train_load, val_load = load_cifar100('dataset/', batch_size=256)
        elif args.dataset == 'imagenet':
            # logging.info('dataset is tinyimagenet!')
            output_feature = 100
            input_shape = (3, 224, 224)
            # train_load, val_load = load_tinyimage('tiny-immagenet-200', batchsize=256)

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
        if os.path.exists(weight_path):
            load_state_dict(model, torch.load(weight_path))
            model = PyTorchClassifier(model,loss=loss_func,input_shape=input_shape,nb_classes=output_feature)
            # logging.info('load weight success!')


        # get the clean data sets / true_labels / targets (if the attack is one of the targeted attacks)
        print(
            'Loading the prepared clean samples (nature inputs and corresponding labels) that will be attacked ...... '
        )
        nature_samples = np.load('{}{}/{}_inputs.npy'.format(
            clean_data_location, dataset, dataset))
        labels_samples = np.load('{}{}/{}_labels.npy'.format(
            clean_data_location, dataset, dataset))
        if attack_name == 'PGD':
            pgd = ProjectedGradientDescentPyTorch(model,eps=0.1)
            adv_samples = pgd.generate(nature_samples)
        if attack_name == 'FGSM':
            fgsm = FastGradientMethod(model,eps=0.1)
            adv_samples = fgsm.generate(nature_samples)
        if attack_name =='JSMA':
            # raw_model = PyTorchClassifier(raw_model,loss=None,input_shape=(3,32,32),nb_classes=10,optimizer=None)
            # # raw_model.set_params()
            jsma = SaliencyMapMethod(model)
            adv_samples=jsma.generate(nature_samples)
        if attack_name == 'CWL0':
            cwl0 = CarliniL0Method(model)
            iteration = int(np.ceil(len(nature_samples) / float(batch_size)))
            adv_samples = np.array([])
            for i in range(iteration):
                start = i * 10
                end = np.minimum((i + 1) * 10, len(nature_samples))
                adv_sample = cwl0.generate(nature_samples[start:end])
                if adv_samples.size == 0:
                    adv_samples = adv_sample
                else:
                    adv_samples = np.concatenate((adv_samples, adv_sample))


        if attack_name == 'CWLINF':
            cwlinf = CarliniLInfMethod(model)
            iteration = int(np.ceil(len(nature_samples) / float(batch_size)))
            adv_samples = np.array([])
            for i in range(iteration):
                start = i * 10
                end = np.minimum((i + 1) * 10, len(nature_samples))
                adv_sample = cwlinf.generate(nature_samples[start:end])
                if adv_samples.size == 0:
                    adv_samples = adv_sample
                else:
                    adv_samples = np.concatenate((adv_samples, adv_sample))

        if attack_name == 'CWL2':
            cwl2 = CarliniL2Method(model)
            iteration = int(np.ceil(len(nature_samples)/float(batch_size)))
            adv_samples = np.array([])
            for i in range(iteration):
                start = i*10
                end = np.minimum((i+1)*10,len(nature_samples))
                adv_sample = cwl2.generate(nature_samples[start:end])
                if adv_samples.size==0:
                    adv_samples = adv_sample
                else:
                    adv_samples = np.concatenate((adv_samples,adv_sample))
        if attack_name == 'DEEPFOOL':
            dp = DeepFool(model)
            adv_samples = dp.generate(nature_samples)
        # cwl0 = CarliniL0Method(model)
        # adv_samples = cwl0.generate(nature_samples)
        adv_labels = model.predict(adv_samples)
        # adv_labels = np.load('{}{}_AdvLabels.npy'.format(adv_examples_dir, attack_name))
        adv_labels = torch.from_numpy(adv_labels)
        adv_labels = torch.max(adv_labels, 1)[1]
        adv_labels = adv_labels.cpu().numpy()
        if not os.path.isdir(adv_examples_dir):
            os.makedirs(adv_examples_dir,exist_ok=True)
        np.save('{}{}_AdvExamples.npy'.format(adv_examples_dir, attack_name), adv_samples)
        np.save('{}{}_AdvLabels.npy'.format(adv_examples_dir, attack_name), adv_labels)
        np.save('{}{}_TrueLabels.npy'.format(adv_examples_dir, attack_name), labels_samples)


