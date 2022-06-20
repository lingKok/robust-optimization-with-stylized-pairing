import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import net
from function import adaptive_instance_normalization, coral

import time
def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# Either --content or --contentDir should be given.

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()
decoder_path='models/decoder.pth'
decoder.load_state_dict(torch.load(decoder_path))
vgg_path='models/vgg_normalised.pth'
vgg.load_state_dict(torch.load(vgg_path))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)

content_size=512
style_size=512
content_tf = test_transform(content_size, False)
style_tf = test_transform(style_size, False)
# style_path='input/style/woman_with_hat_matisse.jpg'
def get_stylelist(style_path):
    import os
    filenames = os.listdir(style_path)
    stylelist = []
    for file in filenames:
        filename = os.path.join(style_path,file)
        style = style_tf(Image.open(filename)).to(device).unsqueeze(0)
        stylelist.append(style)
    return stylelist

# style1_path='input/style/scene_de_rue.jpg'
# style2_path = 'input/style/asheville.jpg'
# style2_path = 'input/style/brushstrokes.jpg'
# style3_path = 'input/style/'
# style=style_tf(Image.open(style_path)).to(device).unsqueeze(0)
stylelist = get_stylelist('input/style/')
def convert2stylized(content,style=stylelist,device=device,rand=False):

    # content=content_tf(content).to(device).unsqueeze(0)
    if rand:
        print('0')
        import random
        index = random.choice(range(len(style)))
    # print(index)
    else:
        # print('1')
        index = 10
    style = style[index]
    with torch.no_grad():
        output=style_transfer(vgg,decoder,content,style,1)


    return output


# content_path = 'input/content/avril.jpg'
# content=content_tf(Image.open(content_path)).to(device).unsqueeze(0)
# output = convert2stylized(content,stylelist)
# print(output)