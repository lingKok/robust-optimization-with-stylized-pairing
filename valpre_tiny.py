import os
import sys
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image



class TinyImageNet(Dataset):
    def __init__(self,root, train, transform):
        labels_t = []
        image_names = []

        with open('.\\tiny-imagenet-200\\wnids.txt') as wnid:
            for line in wnid:
                labels_t.append(line.strip('\n'))
        for label in labels_t:
            txt_path = '.\\tiny-imagenet-200\\train\\' + label + '\\' + label + '_boxes.txt'
            image_name = []
            with open(txt_path) as txt:
                for line in txt:
                    image_name.append(line.strip('\n').split('\t')[0])
            image_names.append(image_name)
        labels = np.arange(200)

        val_labels_t = []
        val_labels = []
        val_names = []
        with open('.\\tiny-imagenet-200\\val\\val_annotations.txt') as txt:
            for line in txt:
                val_names.append(line.strip('\n').split('\t')[0])
                val_labels_t.append(line.strip('\n').split('\t')[1])
        for i in range(len(val_labels_t)):
            for i_t in range(len(labels_t)):
                if val_labels_t[i] == labels_t[i_t]:
                    val_labels.append(i_t//1)
        val_labels = np.array(val_labels)
        self.val_labels=val_labels

        self.type = train
        if self.type :
            i = 0
            self.images = []
            for label in labels_t:

                for image_name in image_names[i]:
                    image_path = os.path.join('.\\tiny-imagenet-200\\train', label, 'images', image_name)
                    # print(type(image_path))

                    self.images.append(image_path)
                i = i + 1
            # self.images = np.array(self.images)
            # self.images = self.images.reshape(-1, 224, 224, 3)
        else:
            self.val_images = []
            for val_image in val_names:
                val_image_path = os.path.join('.\\tiny-imagenet-200\\val\\images', val_image)
                self.val_images.append(val_image_path)
            # self.val_images = np.array(self.val_images)
            # self.val_images = self.val_images.reshape(-1, 224, 224, 3)
        self.transform = transform
        # print(type(self.images[0]))

    def __getitem__(self, index):
        # label = []
        # image = []
        if self.type :
            with open(self.images[index],'rb')as f:
                image = Image.open(self.images[index])
                image = image.convert('RGB')
            label = index // 500

        else:
            with open(self.val_images[index]) as f:
                image = Image.open(self.val_images[index])
                image = image.convert('RGB')
            label = self.val_labels[index]

        return self.transform(image),label

    def __len__(self):
        lengh = 0
        if self.type:
            lengh = len(self.images)
        else:
            lengh = len(self.val_images)
        return lengh
