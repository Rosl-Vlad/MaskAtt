import os
import yaml
import torch
import wandb
import matplotlib.image as mpimg

from tqdm import tqdm
from os.path import join
from datetime import datetime
from torchvision import utils
from torch.utils.data import DataLoader

from gan import GAN
from data import DatasetLoader, load_data
from mask_generate import get_influence_mask


def set_wandb(cfg):
    if cfg["enable"]:
        wandb.init(project=cfg["project"], entity=cfg["entity"], name=cfg["run_name"])


def set_log_dirs(cfg):
    os.makedirs(join(config['log_file'], config["run_name"]), exist_ok=True)
    os.makedirs(join(config['log_file'], config["run_name"], config['checkpoints']), exist_ok=True)
    os.makedirs(join(config['log_file'], config["run_name"], 'generated_images'), exist_ok=True)
    with open(join(config['log_file'], config["run_name"], "config.yaml"), 'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)


def get_data_loaders(cfg):
    images, attr_images = load_data(cfg)
    loader_train = DataLoader(
        dataset=DatasetLoader(join(cfg["base"], cfg["images"]),
                              join(cfg["base"], cfg["masks"]),
                              images,
                              attr_images),
        batch_size=32, shuffle=True, num_workers=cfg["loader_jobs"], drop_last=True)

    loader_valid = DataLoader(
        dataset=DatasetLoader(join(cfg["base"], cfg["images"]),
                              join(cfg["base"], cfg["masks"]),
                              images,
                              attr_images,
                              mode='valid',
                              ret_mask_path=True),
        batch_size=16, shuffle=False, num_workers=1)

    return loader_train, loader_valid


def prepare_mask(path, masks, attrs_diff):
    res = []
    for i in range(len(masks)):
        res.append(get_influence_mask(join(path, str(masks[i].item()).zfill(6)), attrs_diff[i].cpu().numpy()))
    return res


def get_valid_samples(cfg, loader):
    fixed_img_a, fixed_att_a, masks = next(iter(loader))
    fixed_img_a = fixed_img_a.cuda(cfg["GPU"]["name"]) if cfg["GPU"]["enable"] else fixed_img_a
    fixed_att_a = fixed_att_a.cuda(cfg["GPU"]["name"]) if cfg["GPU"]["enable"] else fixed_att_a
    fixed_att_a = fixed_att_a.type(torch.float)
    sample_att_b_list = [fixed_att_a]
    empty_masks = torch.zeros((len(fixed_att_a),
                               len(cfg["data"]["attributes"]),
                               cfg["data"]["image_size"],
                               cfg["data"]["image_size"]))
    empty_masks = empty_masks.cuda(cfg["GPU"]["name"]) if cfg["GPU"]["enable"] else empty_masks
    sample_masks = [empty_masks]
    for i in range(len(cfg["data"]["attributes"])):
        tmp = fixed_att_a.clone()
        tmp[:, i] = 1 - tmp[:, i]
        mask_tensor = torch.tensor(prepare_mask(cfg["data"]["masks"], masks, tmp != fixed_att_a)).type(torch.float)
        mask_tensor = mask_tensor.cuda(cfg["GPU"]["name"]) if cfg["GPU"]["enable"] else mask_tensor
        sample_masks.append(mask_tensor)
        sample_att_b_list.append(tmp)

    return fixed_img_a, fixed_att_a, sample_att_b_list, sample_masks


def validate_data(cfg, g, validation_data, epoch, it):
    fixed_img_a, fixed_att_a, sample_att_b_list, sample_masks = validation_data
    with torch.no_grad():
        samples = [fixed_img_a]
        for i, att_b in enumerate(sample_att_b_list):
            att_b_ = (att_b * 2 - 1) * 0.5
            if i > 0:
                att_b_[..., i - 1] = att_b_[..., i - 1] * 1 / 0.5
            samples.append(g.G(fixed_img_a, att_b_, sample_masks[i]))
        samples = torch.cat(samples, dim=3)
        path = join(
            cfg["log_file"], cfg["run_name"], 'generated_images',
            '{}_Epoch_{}_It.jpg'.format(epoch, it))

        utils.save_image(samples, path, nrow=1, normalize=True, range=(-1., 1.))

        if cfg["wandb"]["enable"]:
            wandb.log({"examples": wandb.Image(mpimg.imread(path), caption="Label")})


def get_log_train(logger_g, logger_d):
    return {
        "train clf loss discriminator": logger_d["dc_loss"],
        "train adv loss discriminator": logger_d["df_loss"],
        "train total loss discriminator": logger_d["d_loss"],

        "train clf loss generator": logger_g["gc_loss"],
        "train adv loss generator": logger_g["gf_loss"],
        "train total loss generator": logger_g["g_loss"],
        "train rec loss generator": logger_g["gr_loss"],
    }


def train(cfg):
    train_loader, valid_loader = get_data_loaders(cfg["data"])
    validation_data = get_valid_samples(cfg, valid_loader)
    g = GAN(cfg)

    it = 0
    for i in range(cfg["fit"]["num_epoch"]):
        for images, attr_a, attr_b, mask in tqdm(train_loader):
            it += 1

            img_a = img_a.cuda(cfg["GPU"]["name"]) if cfg["GPU"]["enable"] else img_a
            att_a = att_a.cuda(cfg["GPU"]["name"]) if cfg["GPU"]["enable"] else att_a
            att_b = att_b.cuda(cfg["GPU"]["name"]) if cfg["GPU"]["enable"] else att_b
            masks = masks.cuda(cfg["GPU"]["name"]) if cfg["GPU"]["enable"] else masks

            if it % 5 != 0:
                errD = g.stepD(images, attr_a, attr_b, mask)
            else:
                errG = g.stepG(images, attr_a, attr_b, mask)

        if cfg["wandb"]["enable"] and it % cfg["wandb"]["logs_iter"] == 0:
            wandb.log(get_log_train(errG, errD))

        if (it + 1) % cfg["fit"]["save_interval"] == 0:
            g.save_models(it)

        if (it + 1) % cfg["fit"]["valid_interval"] == 0:
            g.set_mode(mode='eval')
            validate_data(cfg, g, validation_data, i, it)
            g.set_mode(mode='train')


if __name__ == "__main__":
    f = open("config.yaml", 'r')
    config = yaml.safe_load(f)["train"]
    f.close()
    if config["run_name"] == "":
        config["run_name"] = datetime.now().strftime('%dd-%mm-%Hh-%Mmin')

        print(config["wandb"])
        if config["wandb"]["run_name"] == "":
            config["wandb"]["run_name"] = config["run_name"]

    set_wandb(config["wandb"])
    set_log_dirs(config)
    train(config)
