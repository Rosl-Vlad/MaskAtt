import torch
import torch.nn as nn
import matplotlib.image as img
import torchvision.models as models

from os.path import join
from sklearn.metrics import f1_score, accuracy_score

from ResNet.data import transform


class ResNet:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = models.mobilenet_v2(pretrained=cfg["pretrained"])
        self.model.classifier = nn.Linear(1280, cfg["num_attrs"])

        if cfg["model_path"] != "":
            if cfg["GPU"]["enable"]:
                self.model.load_state_dict(torch.load(cfg["model_path"], map_location=cfg["GPU"]["name"]))
            else:
                self.model.load_state_dict(torch.load(cfg["model_path"], map_location=torch.device('cpu')))

        if cfg["GPU"]["enable"]:
            self.model.cuda(cfg["GPU"]["name"])

        self.opt = torch.optim.Adam(self.model.parameters(), lr=0.0001)

    def set_mode(self, mode='eval'):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

    def save(self, it):
        if self.cfg["save_enable"]:
            torch.save(self.model.state_dict(), join(
                self.cfg["log_file"],
                self.cfg["checkpoints"],
                "{}_model.pth".format(it)
            ))

    def metrics(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='samples')

    def step(self, images, attrs):
        pred = self.model(images)
        pred_ = torch.round(torch.sigmoid(pred))

        loss = torch.nn.BCEWithLogitsLoss(pred, attrs)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        ac, f1 = self.metrics(pred_.detach().cpu().numpy(), attrs.detach().cpu().numpy())
        return ac, f1, loss.item()

    def predict(self, images):
        return torch.round(torch.sigmoid(self.model(images)))

    def inference(self, image_path):
        image = img.imread(image_path)
        image = transform(image).unsqueeze(0)
        image = image.cuda(self.cfg["GPU"]["name"]) if self.cfg["GPU"]["enable"] else image

        return self.predict(image)

