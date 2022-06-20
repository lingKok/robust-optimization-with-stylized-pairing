# robust-optimization-with-stylized-pairing

# Description
> This repository implements Robust optimization with stylized pairing  based on PyTorch API. The code was only validated on windows platform.


# How to use

## first step: train a model with stylized pairing

>python stylized_pairing_train.py cifar --model resnet34

## second step: sample the $L_0$ and $L_\infty$-norm perturbations

>python stylized_pairing_test.py cifar --model resnet34 --type adamin

## third step: plot acc2perturbation

>python plot_acc2perturbaton.py --dataset cifar --model resnet34
