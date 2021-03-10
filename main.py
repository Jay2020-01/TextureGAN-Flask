import base64
import io
from io import BytesIO

import requests as req
from PIL import Image
from flask import Flask, request, render_template
from flask_cors import CORS

# import for TextureGAN
from argparser import parse_arguments
from main1 import get_transforms
from dataloader import imfol
from dataloader.imfol import ImageFolder, make_dataset
import torch
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import DataLoader
import math
from torch.autograd import Variable
from utils.visualize import vis_patch, vis_image

from models import texturegan,discriminator

from utils import transforms as custom_transforms

from train import gen_input, rand_between, gen_input_rand, gen_input_exact

import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual

import warnings

import numpy as np



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def pil_loader1(imagebytes):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with Image.open(imagebytes) as img:
        return img.convert('RGB')


def load_network(model, save_path):
    model_state = torch.load(save_path)

    if "state_dict" in model_state:
        model.load_state_dict(model_state["state_dict"])
    else:
        model.load_state_dict(model_state)

        model_state = {
            'state_dict': model.cpu().state_dict(),
            'epoch': epoch,
            'iteration': iteration,
            'model': args.model,
            'color_space': args.color_space,
            'batch_size': args.batch_size,
            'dataset': dataset,
            'image_size': args.image_size
        }

    model.cuda()


def get_input(val_loader, xcenter, ycenter, patch_size, num_patch):
    img, skg, seg, eroded_seg, txt = val_loader
    img = custom_transforms.normalize_lab(img)
    skg = custom_transforms.normalize_lab(skg)
    txt = custom_transforms.normalize_lab(txt)
    seg = custom_transforms.normalize_seg(seg)
    eroded_seg = custom_transforms.normalize_seg(eroded_seg)

    bs, w, h = seg.size()

    seg = seg.view(bs, 1, w, h)
    seg = torch.cat((seg, seg, seg), 1)

    eroded_seg = eroded_seg.view(bs, 1, w, h)
    eroded_seg = torch.cat((eroded_seg, eroded_seg, eroded_seg), 1)

    temp = torch.ones(seg.size()) * (1 - seg).float()
    temp[:, 1, :, :] = 0  # torch.ones(seg[:,1,:,:].size())*(1-seg[:,1,:,:]).float()
    temp[:, 2, :, :] = 0  # torch.ones(seg[:,2,:,:].size())*(1-seg[:,2,:,:]).float()

    txt = txt.float() * seg.float() + temp

    patchsize = args.local_texture_size
    batch_size = bs
    if xcenter < 0 or ycenter < 0:
        inp, texture_loc = gen_input_rand(txt, skg, eroded_seg[:, 0, :, :] * 100,
                                          patch_size, patch_size,
                                          num_patch)
    else:
        inp, texture_loc = gen_input_exact(txt, skg, eroded_seg[:, 0, :, :] * 100, xcenter, ycenter, patch_size, 1)

    return inp, texture_loc


def get_inputv(inp):
    input_stack = torch.FloatTensor().cuda()
    input_stack.resize_as_(inp.float()).copy_(inp)
    inputv = Variable(input_stack)
    return inputv

app = Flask(__name__)
CORS(app)  # 解决跨域问题

# select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

command = '--display_port 7770 --load 0 --load_D -1 --load_epoch 105 --gpu 2 --model texturegan --feature_weight 1e2 --pixel_weight_ab 1e3 --global_pixel_weight_l 1e3 --local_pixel_weight_l 0 --style_weight 0 --discriminator_weight 1e3 --discriminator_local_weight 1e6  --learning_rate 1e-4 --learning_rate_D 1e-4 --batch_size 36 --save_every 50 --num_epoch 100000 --save_dir /home/psangkloy3/skip_leather_re/ --load_dir /home/psangkloy3/skip_leather_re/ --data_path ../../training_handbags_pretrain/ --learning_rate_D_local  1e-4 --local_texture_size 50 --patch_size_min 20 --patch_size_max 40 --num_input_texture_patch 1 --visualize_every 5 --num_local_texture_patch 1'
args = parse_arguments(command.split())

args.batch_size = 1
args.image_size = 256
args.resize_max = 256
args.resize_min = 256
# args.data_path = '/home/psangkloy3/training_handbags_pretrain/' #change to your data path
args.data_path = '/media/jay2019/DATA/Study/8.创新杯/3.现有代码/clothes_data/clothes_data'


# args.data_path = './dataset/training_handbags_pretrain'
# args.data_path = './dataset/training_shoes_pretrain'
# args.data_path = './dataset/j_test_shoes'


def get_prediction(image_bytes, txt_bytes, pos_x, pos_y):
    try:
        # code for TextureGAN


        transform = get_transforms(args)
        # val = make_dataset(args.data_path, 'val')
        # valDset = ImageFolder('val', args.data_path, transform)
        # val_display_size = 1
        # valLoader = DataLoader(dataset=valDset, batch_size=val_display_size,shuffle=True)
        # new load data
        # img_path = "./dataset/test/img.jpg"
        img_path = image_bytes
        # skg_path = "./dataset/test/skg.jpg"
        skg_path = image_bytes
        seg_path = "./dataset/test/seg.jpg"
        eroded_seg_path = "./dataset/test/eroded_seg.jpg"
        # txt_path = "./dataset/test/txt.jpg"
        txt_path = txt_bytes

        img = pil_loader1(img_path)
        skg = pil_loader1(skg_path)
        seg = pil_loader(seg_path)
        txt = pil_loader1(txt_path)
        eroded_seg = pil_loader(eroded_seg_path)
        img, skg, seg, eroded_seg, txt = transform([img, skg, seg, eroded_seg, txt])
        img = img.unsqueeze(0)
        skg = skg.unsqueeze(0)
        txt = txt.unsqueeze(0)
        seg = seg.unsqueeze(0)
        eroded_seg = eroded_seg.unsqueeze(0)
        data = [img, skg, seg, eroded_seg, txt]

        # load model
        model_location = '../../TextureGAN_data/pretrained_models/3.2_6/G_net_texturegan_18_300.pth'
        netG = texturegan.TextureGAN(5, 3, 32)
        load_network(netG, model_location)
        netG.eval()

        # get_ipython().run_line_magic('matplotlib', 'inline')
        warnings.filterwarnings('ignore')

        # start val
        color_space = 'lab'

        img, skg, seg, eroded_seg, txt = data
        # print(txt.size())
        img = custom_transforms.normalize_lab(img)
        skg = custom_transforms.normalize_lab(skg)
        txt = custom_transforms.normalize_lab(txt)
        seg = custom_transforms.normalize_seg(seg)
        eroded_seg = custom_transforms.normalize_seg(eroded_seg)

        inp, texture_loc = get_input(data, pos_x, pos_y, 50, 1)

        seg = seg != 0

        model = netG

        inpv = get_inputv(inp.cuda())
        output = model(inpv.cuda())

        out_img = vis_image(custom_transforms.denormalize_lab(output.data.double().cpu()),
                            color_space)
        inp_img = vis_patch(custom_transforms.denormalize_lab(txt.cpu()),
                            custom_transforms.denormalize_lab(skg.cpu()),
                            texture_loc,
                            color_space)
        tar_img = vis_image(custom_transforms.denormalize_lab(img.cpu()),
                            color_space)

        # plt.figure()
        # plt.imshow(np.transpose(inp_img[0], (1, 2, 0)))
        # plt.imsave("./out_img.jpg", np.transpose(out_img[0], (1, 2, 0)))
        # # plt.axis('off')
        # # plt.figure()
        # plt.figure()
        # plt.imshow(np.transpose(out_img[0], (1, 2, 0)))
        # plt.show()
        inp_img = np.transpose(inp_img[0], (1, 2, 0))
        out_img = np.transpose(out_img[0], (1, 2, 0))
        return out_img
    except Exception as e:
        print("error!!!!!!!!!!!!")
        return inp_img


@app.route("/predict", methods=["POST"])
@torch.no_grad()
def predict():
    image_link = request.form.get("imageLink")  # 传入sketch图片url
    txt_link = request.form.get("txtLink")  # 传入texture图片url
    pos_x = request.form.get("x", type=int)  # 传入x坐标
    pos_y = request.form.get("y", type=int)  # 传入y坐标

    response = req.get(image_link)
    img_bytes = BytesIO(response.content)

    response = req.get(txt_link)
    txt_bytes = BytesIO(response.content)

    image_array = get_prediction(image_bytes=img_bytes, txt_bytes=txt_bytes, pos_x=pos_x, pos_y=pos_y)

    img = Image.fromarray(np.uint8(image_array * 255))
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    image_info = base64.b64encode(byte_data)

    return image_info


@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("up.html")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)




