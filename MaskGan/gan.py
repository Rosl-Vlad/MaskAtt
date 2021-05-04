import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from os.path import join

from blocs import Conv2d, STU, TransposeConv2d, MaskConv


class GeneratorCustom(nn.Module):
    def __init__(self, in_channel=64, n_layers_enc=5, n_layers_dec=5,
                 n_attrs=12, n_STU=3, n_inject=0, n_masks=3):
        super(GeneratorCustom, self).__init__()
        self.n_attrs = n_attrs
        self.n_skip = n_STU
        self.n_layers_enc = n_layers_enc
        self.n_layers_dec = n_layers_dec
        self.n_inject = n_inject
        self.masks_layers = n_masks

        # <--- encoder ---> #
        self.enc = nn.ModuleList()
        enc_in_channel = 3
        for i in range(n_layers_enc):
            self.enc.append(Conv2d(enc_in_channel, in_channel * 2 ** i, 2, nn.ReLU, nn.BatchNorm2d, 4))
            enc_in_channel = in_channel * 2 ** i

        # <--- STU ---> #
        self.stu = nn.ModuleList()
        stu_in_channel = in_channel * 2 ** (n_layers_enc - n_STU - 1)
        for i in range(n_STU):
            self.stu.append(STU(stu_in_channel, stu_in_channel, 3, self.n_attrs))
            stu_in_channel *= 2

        # <--- decoder ---> #
        self.dec = nn.ModuleList()
        dec_n_channel = 2 ** (int(np.log2(in_channel)) + n_layers_enc - 1)
        dec_in_channel = dec_n_channel + self.n_attrs
        for i in range(n_layers_dec):
            if i + 1 == n_layers_dec:
                self.dec.append(TransposeConv2d(dec_in_channel, 3, is_tanh=True))
                continue

            if self.n_skip >= i > 0:
                inject = 0
                if self.n_inject >= i > 0:
                    inject = self.n_attrs
                self.dec.append(
                    TransposeConv2d(dec_in_channel + dec_in_channel + inject, dec_n_channel // (2 ** (i + 1))))
            else:
                self.dec.append(TransposeConv2d(dec_in_channel, dec_n_channel // (2 ** (i + 1))))

            dec_in_channel = dec_n_channel // (2 ** (i + 1))

        # <--- mask enc ---> #
        mask_in_channel = self.n_attrs
        self.mask_enc = nn.ModuleList()
        for i in range(self.n_layers_enc - 1):
            self.mask_enc.append(Conv2d(mask_in_channel, in_channel * 2 ** i, 2, nn.ReLU, nn.BatchNorm2d, 4))
            mask_in_channel = in_channel * 2 ** i

        # <--- mask gate ---> #
        self.mask_gates = nn.ModuleList()
        dec_n_channel = 2 ** (int(np.log2(in_channel)) + n_layers_enc - 2)
        for i in range(self.masks_layers):
            self.mask_gates.append(MaskConv(dec_n_channel))
            dec_n_channel = dec_n_channel // 2

    def encode(self, x):
        hiddens = []
        for i in range(self.n_layers_enc):
            x = self.enc[i](x)
            hiddens.append(x)

        return hiddens

    def encode_mask(self, mask):
        hiddens = []
        for i in range(self.n_layers_enc - 1):
            mask = self.mask_enc[i](mask)
            hiddens.append(mask)

        return hiddens

    def decode(self, hiddens, attr, mask):
        assert (hiddens is not None)

        hidden_mask = self.encode_mask(mask)

        h_state = hiddens[-1]

        n, _, h, w = h_state.size()
        a = attr.view((n, self.n_attrs, 1, 1)).expand((n, self.n_attrs, h, w))
        z = self.dec[0](torch.cat([h_state, a], dim=1))
        z = self.mask_gates[0](z, hidden_mask[-1])
        for i in range(self.n_layers_dec - 1):
            if i < self.n_skip:
                skip, h_state = self.stu[-(i + 1)](hiddens[-(i + 2)], h_state, attr)
                z = torch.cat([z, skip], dim=1)

            if i < self.n_inject:
                n, _, h, w = z.size()
                a = attr.view((n, self.n_attrs, 1, 1)).expand((n, self.n_attrs, h, w))
                z = torch.cat([z, a], dim=1)

            if i < self.masks_layers - 1:
                z = self.dec[i + 1](z)
                z = self.mask_gates[i + 1](z, hidden_mask[-(i + 2)])
            else:
                z = self.dec[i + 1](z)

        return z

    def forward(self, x=None, a=None, mask=None, mode="enc-dec"):
        if mode == "enc":
            return self.encode(x)

        if mode == "dec":
            return self.decode(x, a, mask)

        if mode == "enc-dec":
            return self.decode(self.encode(x), a, mask)


class Discriminator(nn.Module):
    def __init__(self, image_size=128, n_attrs=12, in_channel=64, fc_dim=1024, n_layers=5):
        super(Discriminator, self).__init__()
        layers = []
        in_channels = 3
        for i in range(n_layers):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channel * 2 ** i, 4, 2, 1),
                nn.InstanceNorm2d(in_channel * 2 ** i, affine=True),
                nn.LeakyReLU()
            ))
            in_channels = in_channel * 2 ** i
        self.conv = nn.Sequential(*layers)
        feature_size = image_size // 2**n_layers
        self.fc_adv = nn.Sequential(
            nn.Linear(in_channel * 2 ** (n_layers - 1) * feature_size ** 2, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, 1)
        )
        self.fc_att = nn.Sequential(
            nn.Linear(in_channel * 2 ** (n_layers - 1) * feature_size ** 2, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, n_attrs),
        )

    def forward(self, x):
        y = self.conv(x)
        y = y.view(y.size()[0], -1)
        logit_adv = self.fc_adv(y)
        logit_att = self.fc_att(y)
        return logit_adv, logit_att


def gradient_penalty(f, real, fake=None, gpu=None):
    def interpolate(a, b=None):
        if b is None:  # interpolation in DRAGAN
            beta = torch.rand_like(a)
            b = a + 0.5 * a.var().sqrt() * beta
        alpha = torch.rand(a.size(0), 1, 1, 1)
        alpha = alpha.cuda(gpu)
        inter = a + alpha * (b - a)
        return inter
    x = interpolate(real, fake).requires_grad_(True)
    pred = f(x)
    if isinstance(pred, tuple):
        pred = pred[0]
    grad = torch.autograd.grad(
        outputs=pred, inputs=x,
        grad_outputs=torch.ones_like(pred),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad = grad.view(grad.size(0), -1)
    norm = grad.norm(2, dim=1)
    gp = ((norm - 1.0) ** 2).mean()
    return gp


class GAN:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lmd1 = cfg["fit"]["lmd1"]
        self.lmd2 = cfg["fit"]["lmd2"]
        self.lmd3 = cfg["fit"]["lmd3"]
        self.lmdGP = cfg["fit"]["lmdGP"]

        self.G = GeneratorCustom(
            in_channel=cfg["generator"]["in_channel"],
            n_layers_enc=cfg["generator"]["enc_l"],
            n_layers_dec=cfg["generator"]["dec_l"],
            n_STU=cfg["generator"]["STU"],
            n_inject=cfg["generator"]["inject"],
            n_masks=cfg["generator"]["mask_l"],
            n_attrs=cfg["data"]["num_attrs"],
        )
        self.D = Discriminator(
            in_channel=cfg["discriminator"]["in_channel"],
            n_layers=cfg["discriminator"]["enc_l"],
            image_size=cfg["image_size"],
            n_attrs=cfg["data"]["num_attrs"],
        )

        if cfg["GPU"]["enable"]:
            self.G.cuda(cfg["GPU"]["name"])
            self.D.cuda(cfg["GPU"]["name"])

        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=cfg["fit"]["lr"], betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=cfg["fit"]["lr"], betas=(0.5, 0.999))
        self.G.train()
        self.D.train()

    def set_mode(self, mode='eval'):
        if mode == 'train':
            self.G.train()
            self.D.train()
        else:
            self.G.eval()
            self.D.eval()

    def save_models(self, it):
        if self.cfg["discriminator"]["save_enable"]:
            torch.save(self.D.state_dict(), join(
                self.cfg["log_file"],
                self.cfg["run_name"],
                self.cfg["checkpoints"],
                "{}_discriminator.pth".format(it)
            ))
        if self.cfg["generator"]["save_enable"]:
            torch.save(self.D.state_dict(), join(
                self.cfg["log_file"],
                self.cfg["run_name"],
                self.cfg["checkpoints"],
                "{}_generator.pth".format(it)
            ))

    def stepG(self, images, attr_a, attr_b, mask):
        for p in self.D.parameters():
            p.requires_grad = False
        for p in self.G.parameters():
            p.requires_grad = True

        attr_a_ = (attr_a * 2 - 1) * 0.5
        attr_b_ = (attr_b * 2 - 1) * 0.5

        h = self.G(images, mode='enc')
        img_fake = self.G(h, attr_b_, mask, mode='dec')
        img_real = self.G(h, attr_a_, mask, mode='dec')
        d_fake, dc_fake = self.D(img_fake)

        gf_loss = -d_fake.mean()
        gc_loss = F.binary_cross_entropy_with_logits(dc_fake, attr_b)
        gr_loss = F.l1_loss(img_real, images)
        g_loss = self.lmd1 * gr_loss + self.lmd2 * gc_loss + self.lmd3 * gf_loss

        self.opt_G.zero_grad()
        g_loss.backward()
        self.opt_G.step()

        errG = {
            'g_loss': g_loss.item(), 'gf_loss': gf_loss.item(),
            'gc_loss': gc_loss.item(), 'gr_loss': gr_loss.item()
        }
        return errG

    def stepD(self, images, attr_a, attr_b, mask):
        for p in self.G.parameters():
            p.requires_grad = False
        for p in self.D.parameters():
            p.requires_grad = True

        attr_b_ = (attr_b * 2 - 1) * 0.5

        img_fake = self.G(images, attr_b_, mask)
        d_real, dc_real = self.D(images)
        d_fake, dc_fake = self.D(img_fake)

        wd = d_real.mean() - d_fake.mean()
        df_loss = -wd
        df_gp = gradient_penalty(self.D, images, img_fake, self.cfg["GPU"]["name"])
        dc_loss = F.binary_cross_entropy_with_logits(dc_real, attr_a)
        d_loss = self.lmd1 * df_loss + self.lmdGP * df_gp + self.lmd3 * dc_loss

        self.opt_D.zero_grad()
        d_loss.backward()
        self.opt_D.step()

        errD = {
            'd_loss': d_loss.item(), 'df_loss': df_loss.item(),
            'df_gp': df_gp.item(), 'dc_loss': dc_loss.item()
        }
        return errD
