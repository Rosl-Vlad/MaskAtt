import yaml
import torch

from model import ResNet
from data import get_data_loaders


def tune(cfg):
    train_loader, valid_loader = get_data_loaders(cfg["data"])
    model = ResNet(cfg)

    for i in range(cfg["tune"]["num_epoch"]):
        for j, (images, attr) in enumerate(train_loader):
            attr = attr.type(torch.float)

            images = images.cuda(cfg["GPU"]["name"]) if cfg["GPU"]["enable"] else images
            attr = attr.cuda(cfg["GPU"]["name"]) if cfg["GPU"]["enable"] else attr

            ac, f1, loss = model.step(images, attr)
            print("\r Done: {}/{} acc {} f1 {} loss {}".
                format(
                    j * cfg["data"]["batch_size"],
                    len(train_loader) * cfg["data"]["batch_size"],
                    ac, f1, loss),
                end='')
        print()
        model.save(i)
        val_lbl = []
        val_pred = []
        model.set_mode("eval")
        for images, attr in valid_loader:
            attr = attr.type(torch.float)

            images = images.cuda(cfg["GPU"]["name"]) if cfg["GPU"]["enable"] else images
            attr = attr.cuda(cfg["GPU"]["name"]) if cfg["GPU"]["enable"] else attr

            pred = model.predict(images)

            val_lbl.append(attr.detach().cpu())
            val_pred.append(pred.detach().cpu())

        ac, f1 = model.metrics(torch.cat(val_lbl, dim=0).numpy(), torch.cat(val_pred, dim=0).numpy())
        print("{} test acc {}, f1 {}".format(i + 1, ac, f1))
        model.set_mode("train")


if __name__ == "__main__":
    f = open("config.yaml", 'r')
    config = yaml.safe_load(f)["train"]
    f.close()
