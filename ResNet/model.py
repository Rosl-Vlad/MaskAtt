import torch
import torch.nn as nn
import torchvision.models as models

from os.path import join
from sklearn.metrics import f1_score, accuracy_score


class ResNet:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = models.resnet18(pretrained=cfg["pretrained"])
        self.model.fc = nn.Linear(512, cfg["data"]["num_attrs"])

        if cfg["model_path"] != "":
            self.model.load_state_dict(
                torch.load(cfg["model_path"], map_location=cfg["GPU"]["name"]))

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
