import torch
import numpy as np
import torchvision.transforms as transforms

from PIL import Image

from BiseNet.model import BiSeNet


class MaskModel:
    def __init__(self, cfg):
        self.cfg = cfg

        n_classes = 19
        self.model = BiSeNet(n_classes=n_classes)
        if cfg["GPU"]["enable"]:
            self.model.cuda(cfg["GPU"]["name"])
            self.model.load_state_dict(torch.load(cfg["model_path"], map_location=cfg["GPU"]["name"]))
        else:
            self.model.load_state_dict(torch.load(cfg["model_path"], map_location=torch.device('cpu')))

        self.model.eval()

    def predict(self, image_path):
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        with torch.no_grad():
            img = Image.open(image_path)
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda(self.cfg["GPU"]["name"]) if self.cfg["GPU"]["enable"] else img
            out = self.model(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0).astype(np.uint8)
            parsing = torch.nn.functional.interpolate(
                torch.tensor(np.expand_dims(parsing, axis=(0, 1))),
                size=(self.cfg["mask_size"], self.cfg["mask_size"]))[0][0].numpy()
            return parsing.astype(np.uint8)
