import os
import torch

from MaskGan import GAN
from ResNet import ResNet
from BiseNet import MaskModel


class Implementation:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gan = GAN(cfg["gan"])
        self.attr_clf = ResNet(cfg["resnet"])
        self.mask_model = MaskModel(cfg["mask"])

        os.makedirs(cfg["env"]["out_dir"], exist_ok=True)
        self.attrs_d = {i: att for i, att in enumerate(cfg["env"]["attributes"])}
        self.d_attrs = {att: i for i, att in enumerate(cfg["env"]["attributes"])}

    def get_attrs(self, image_path):
        changes_ = []

        attr_a = self.attr_clf.inference(image_path)
        attr_b = attr_a.clone()
        self.sprintf_attrs(attr_a)
        attr_b = attr_b.type(torch.float)
        attr_b = (attr_b * 2 - 1) * 0.5
        print("Select attr to change - 'att:value, att:value'")
        changes = input().split(", ")
        for c in changes:
            att, value = c.split(":")
            changes_.append(att)
            attr_b[0][self.d_attrs[att]] = float(value)
        return attr_a, attr_b, changes_

    def generate(self, image_path):
        attr_a, attr_b, changes_ = self.get_attrs(image_path)
        mask = self.mask_model.predict(image_path)
        self.gan.generate(image_path, attr_a, attr_b, mask, path_save=self.cfg["env"]["out_dir"], changes_=changes_)

    def sprintf_attrs(self, attrs):
        for i, att in enumerate(attrs[0]):
            print(self.attrs_d[i], att.item())
