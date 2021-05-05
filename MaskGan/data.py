import torch
import random
import numpy as np
import matplotlib.image as img

from os.path import join
from torchvision import transforms
from torch.utils.data import Dataset

from MaskGan.mask_generate import get_influence_mask, get_influence_mask_v2


def load_data(cfg):
    f = open(join(cfg["base"], cfg["attrs"]), 'r')

    all_attrs = f.readline()[:-1].split(",")[1:]
    all_attrs = {k: i + 1 for i, k in enumerate(all_attrs)}

    attrs = [all_attrs.get(att) for att in cfg["attributes"]]
    images = np.loadtxt(join(cfg["base"], cfg["attrs"]), skiprows=1, usecols=[0], dtype=np.str, delimiter=',')
    attr_images = np.loadtxt(join(cfg["base"], cfg["attrs"]), skiprows=1, usecols=attrs, dtype=np.int, delimiter=',')

    return images, attr_images


class DatasetLoader(Dataset):
    def __init__(self, path_images, path_masks, image_names, attr_images, mode='train', ret_mask_path=False):
        super().__init__()
        self.labels = attr_images
        self.images = image_names
        self.path_images = path_images
        self.path_masks = path_masks
        self.ret_mask_path = ret_mask_path

        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.RandomAffine(0, translate=(0.1, 0.1)),
                                        transforms.CenterCrop(190),
                                        transforms.Resize((128, 128)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                             std=[0.5, 0.5, 0.5])])

        if mode == 'train':
            self.images = self.images[:182000]
            self.labels = self.labels[:182000]
        if mode == 'valid':
            self.images = self.images[182000:182637]
            self.labels = self.labels[182000:182637]

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_attr = self.labels[index]
        img_path = join(self.path_images, img_name)
        image = img.imread(img_path)

        if self.transform is not None:
            image = self.transform(image)

        attr_a = torch.tensor((img_attr + 1) // 2)
        idx = random.randrange(0, len(self.images))
        attr_b = torch.tensor((self.labels[idx] + 1) // 2)

        #mask = get_influence_mask(join(self.path_masks, self.images[index][:-4]), (attr_a != attr_b).numpy())
        mask = get_influence_mask_v2(join(self.path_masks, self.images[index][:-4]), attr_a.numpy())

        if self.ret_mask_path:
            return image, attr_a, attr_b, torch.tensor(int(self.images[index][:-4]))

        return image, attr_a, attr_b, torch.tensor(mask).type(torch.float)
