import matplotlib.pyplot as plt
import torch
import argparse
parser=argparse.ArgumentParser('plot the l_0 and l_\infty perturbation')
parser.add_argument('--dataset',default='imagenet',choices=['cifar','cifar100','imagenet','imagenet10'])
parser.add_argument('--model',default='resnet34',choices=['resnet34','googlenet','mobilenet','vgg'])
args=parser.parse_args()
normal_path='output/'+args.dataset+'_'+args.model+'_acc2pert_adamin_rand.pth'
adamin_path='output/'+args.dataset+'_'+args.model+'_acc2pert_adamin.pth'
# adv_path='output/'+args.dataset+'_'+args.model+'_acc2pert_adv.pth'
normal_acc2perturbation=torch.load(normal_path)
adamin_acc2perturbation=torch.load(adamin_path)
# adv_acc2perturbation = torch.load(adv_path)
normal_acc_l2,normal_acc_infty,normal_l2_scan,normal_infty_scan=normal_acc2perturbation
adamin_acc_l2,adamin_acc_infty,adamin_l2_scan,adamin_infty_scan=adamin_acc2perturbation
# adv_acc_l2,adv_acc_infty,adv_l2_scan,adv_infty_scan = adv_acc2perturbation

# print(normal_l2_scan.shape)
# print(normal_acc_l2.shape)
plt.figure(figsize=(12,16))
title_content = args.dataset+'_'+args.model
print(title_content)

# plt.subplot(121)
plt.plot(normal_l2_scan,normal_acc_l2,'yo-',label='random style')
plt.plot(adamin_l2_scan,adamin_acc_l2,'ro--',label='single style')
plt.xlabel(r'l$_2$ perturbation',fontsize=30)
# plt.plot(normal_infty_scan,normal_acc_infty,'yo-',label='random style')
# plt.plot(adamin_infty_scan,adamin_acc_infty,'ro--',label='single style')
# plt.xlabel(r'l$_\infty$ perturbation',fontsize=30)
plt.ylabel('Accuracy',fontsize=30)
plt.rcParams.update({'font.size':25})
plt.tick_params(labelsize=20)
plt.legend()
plt.title(title_content,fontsize=30)
# plt.subplot(122)
# plt.plot(normal_infty_scan,normal_acc_infty,'go-',label='adv')
# plt.plot(adamin_infty_scan,adamin_acc_infty,'ro--',label='sylized pairing')
# plt.xlabel(r'l$_\infty$ perturbation',fontsize=20)
# plt.ylabel('Accuracy',fontsize=20)
# plt.rcParams.update({'font.size':15})
# plt.tick_params(labelsize=20)
# # plt.suptitle(title_content,fontsize=30)
# plt.title(title_content,fontsize=30)
# plt.legend()
plt.show()