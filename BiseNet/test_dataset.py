#!/usr/bin/python
# -*- encoding: utf-8 -*-

from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

import yaml
import hickle as hkl
from os.path import join


def evaluate(cfg, respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):
    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    if cfg["GPU"]["enable"]:
        net.cuda(cfg["GPU"]["name"])
    net.load_state_dict(torch.load(cp))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda(cfg["GPU"]["name"]) if cfg["GPU"]["enable"] else img
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            parsing = \
            torch.nn.functional.interpolate(torch.tensor(np.expand_dims(parsing, axis=(0, 1))).type(torch.tensor),
                                            size=(cfg["mask_size"], cfg["mask_size"]))[0][0].numpy()
            hkl.dump(parsing, join(respth, image_path[:-4]))


if __name__ == "__main__":
    f = open("config.yaml", 'r')
    config = yaml.safe_load(f)["make_datase"]
    f.close()
    evaluate(config["GPU"], dspth=config["data_path"], respth=config["out_dir_path"], cp=config["model_path"])
