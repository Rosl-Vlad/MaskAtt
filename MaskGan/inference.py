import torch
import matplotlib.image as img

from os.path import join
from torchvision import transforms, utils

from MaskGan.mask_generate import get_influence_mask
from MaskGan.gan import GeneratorCustom

from torchvision.utils import save_image


class GAN:
    def __init__(self, cfg):
        self.cfg = cfg

        self.model = GeneratorCustom(
            in_channel=cfg["setting"]["in_channel"],
            n_layers_enc=cfg["setting"]["enc_l"],
            n_layers_dec=cfg["setting"]["dec_l"],
            n_STU=cfg["setting"]["STU"],
            n_inject=cfg["setting"]["inject"],
            n_masks=cfg["setting"]["mask_l"],
            n_attrs=cfg["setting"]["num_attrs"],
        )

        self.model.load_state_dict(torch.load(cfg["model_path"], map_location=cfg["GPU"]["name"])["G"])

        if cfg["GPU"]["enable"]:
            self.model.cuda(cfg["GPU"]["name"])

        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             #transforms.CenterCrop(170),
                                             transforms.Resize((128, 128)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.inv_normalize = transforms.Normalize(
            mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
            std=[1 / 0.5, 1 / 0.5, 1 / 0.5])

        self.model.eval()

    def generate(self, image_path, attr_diff, attr_b, mask=None, mask_path="", path_save="", changes_=[]):
        masks = torch.tensor(
            get_influence_mask(mask_path, attr_diff.cpu().detach().numpy(), mask=mask)
        ).unsqueeze(0)
        img_name = image_path.split('/')[-1]
        image = img.imread(image_path)
        image = self.transform(image).unsqueeze(0)

        image = image.cuda(self.cfg["GPU"]["name"]) if self.cfg["GPU"]["enable"] else image
        masks = masks.cuda(self.cfg["GPU"]["name"]) if self.cfg["GPU"]["enable"] else masks
        attr_b = attr_b.cuda(self.cfg["GPU"]["name"]) if self.cfg["GPU"]["enable"] else attr_b

        pred = self.model(image, attr_b, masks.type(torch.float))

        utils.save_image(pred, join(path_save, "gen_" + img_name), nrow=1, normalize=True)

        #utils.save_image(pred, join(path_save, "gen_" + "_".join(changes_) + "_" + img_name), nrow=1, normalize=True)
