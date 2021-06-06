import torch
import torch.nn as nn


class DenseConv(nn.Module):
    def __init__(self, in_channels, out_channel, kernel_size=3):
        super(DenseConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, kernel_size=kernel_size, padding=1,
                      bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([x, self.conv(x)], dim=1)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channel, stride, kernel_size=4, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )

    def forward(self, x):
        return self.conv(x)


class TransposeConv2d(nn.Module):
    def __init__(self, in_channels, out_channel, is_tanh=False):
        super().__init__()

        if is_tanh:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channel, kernel_size=4, stride=2, padding=1, bias=True),
                nn.Tanh(),
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channel, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, n_layer=3, growth_rate=32):
        super(ResidualBlock, self).__init__()
        denses = []
        for i in range(n_layer):
            denses.append(DenseConv(in_channels + growth_rate * i, growth_rate))
        self.denses = nn.Sequential(*denses)

        self.lff = nn.Conv2d(in_channels + growth_rate * n_layer, in_channels, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.denses(x))


class MaskConv(nn.Module):
    def __init__(self, in_channels):
        super(MaskConv, self).__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=4, stride=2),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ResidualBlock(in_channels),
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x, mask):
        h = torch.cat([x, mask], dim=1)
        return self.mask(h) + x


class MaskGate(nn.Module):
    def __init__(self, in_channels):
        super(MaskGate, self).__init__()
        self.mask_enc = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.res = nn.Sequential(
            ResidualBlock(in_channels),
            nn.BatchNorm2d(in_channels),
            ResidualBlock(in_channels),
        )

    def forward(self, x, mask):
        h = torch.cat([x, mask], dim=1)
        h = self.mask_enc(h)
        m = self.res(h)
        return m * x + m


class STU(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, n_attrs):
        super(STU, self).__init__()
        self.n_attrs = n_attrs

        self.upsample = nn.ConvTranspose2d(in_dim * 2 + n_attrs, out_dim, 4, 2, 1, bias=False)
        self.reset_gate = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, out_dim, kernel_size, 1, (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, out_dim, kernel_size, 1, (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid()
        )
        self.hidden = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, out_dim, kernel_size, 1, (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Tanh()
        )

    def forward(self, input, old_state, attr):
        n, _, h, w = old_state.size()
        attr = attr.view((n, self.n_attrs, 1, 1)).expand((n, self.n_attrs, h, w))
        state_hat = self.upsample(torch.cat([old_state, attr], 1))
        r = self.reset_gate(torch.cat([input, state_hat], dim=1))
        z = self.update_gate(torch.cat([input, state_hat], dim=1))
        new_state = r * state_hat
        hidden_info = self.hidden(torch.cat([input, new_state], dim=1))
        output = (1-z) * state_hat + z * hidden_info
        return output, new_state
