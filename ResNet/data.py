import torch
import numpy as np
import matplotlib.image as img

from os.path import join
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from MaskGan import load_data


transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.RandomAffine(0, translate=(0.1, 0.1)),
                                transforms.CenterCrop(190),
                                transforms.Resize((128, 128)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                     std=[0.5, 0.5, 0.5])])


class DatasetLoader(Dataset):
    def __init__(self, path, image_names, attr_images):
        super().__init__()
        self.labels = attr_images
        self.images = image_names
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_attr = self.labels[index]
        img_path = join(self.path, img_name)
        image = img.imread(img_path)

        if self.transform is not None:
            image = self.transform(image)

        img_attr = torch.tensor((img_attr + 1) // 2)

        return image, img_attr


def get_data_loaders(cfg):
    images, attr_images = load_data(cfg)
    indexes_train, indexes_test = train_test_split(np.arange(0, len(images) - 1), random_state=42, test_size=0.1)

    loader_train = DataLoader(
        dataset=DatasetLoader(join(cfg["base"], cfg["images"]),
                              images[indexes_train],
                              attr_images[indexes_train]),
        batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["loader_jobs"])

    loader_valid = DataLoader(
        dataset=DatasetLoader(join(cfg["base"], cfg["images"]),
                              images[indexes_test],
                              attr_images[indexes_test]),
        batch_size=16, shuffle=False, num_workers=1)

    return loader_train, loader_valid
