# robust-optimization-with-stylized-pairing
##first step: train a model with stylized pairing
python stylized_pairing_train.py cifar --model resnet34
##second step: sample the $L_0$ $L_\infty$ norm perturbation
python stylized_pairing_test.py cifar --model resnet34 --type adamin
## third step: plot 
python plot_acc2perturbaton.py --dataset cifar --model resnet34
