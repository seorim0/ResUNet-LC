"""
Same as ResUNet-LC
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional
from einops import rearrange


class INCONV(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(INCONV, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=in_ch,
                                 out_channels=out_ch,
                                 kernel_size=(3, 15),
                                 padding=(1, 7),
                                 stride=(1, 2),
                                 bias=False)
        self.bn1_1 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = functional.leaky_relu(self.bn1_1(self.conv1_1(x)))
        return x


class ResUBlock(nn.Module):
    def __init__(self, out_ch, mid_ch, layers, downsampling=True):
        super(ResUBlock, self).__init__()
        self.downsample = downsampling

        self.conv1 = nn.Conv2d(in_channels=out_ch,
                               out_channels=out_ch,
                               kernel_size=(3, 9),
                               padding=(1, 4),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for idx in range(layers):
            if idx == 0:
                self.encoders.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels=out_ch,
                        out_channels=mid_ch,
                        kernel_size=(3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                        bias=False
                    ),
                    nn.BatchNorm2d(mid_ch),
                    nn.LeakyReLU()
                ))
            else:
                self.encoders.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels=mid_ch,
                        out_channels=mid_ch,
                        kernel_size=(3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                        bias=False
                    ),
                    nn.BatchNorm2d(mid_ch),
                    nn.LeakyReLU()
                ))

            if idx == layers - 1:
                self.decoders.append(nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=mid_ch * 2,
                        out_channels=out_ch,
                        kernel_size=(3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                        output_padding=(0, 1),
                        bias=False
                    ),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU()
                ))
            else:
                self.decoders.append(nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=mid_ch * 2,
                        out_channels=mid_ch,
                        kernel_size=(3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                        output_padding=(0, 1),
                        bias=False
                    ),
                    nn.BatchNorm2d(mid_ch),
                    nn.LeakyReLU()
                ))

            self.bottleneck = nn.Sequential(
                nn.Conv2d(
                    in_channels=mid_ch,
                    out_channels=mid_ch,
                    kernel_size=(3, 9),
                    padding=(1, 4),
                    bias=False
                ),
                nn.BatchNorm2d(mid_ch),
                nn.LeakyReLU()
            )

            if self.downsample:
                self.idfunc_0 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
                self.idfunc_1 = nn.Conv2d(in_channels=out_ch,
                                          out_channels=out_ch,
                                          kernel_size=1,
                                          bias=False)

    def forward(self, x):
        x_in = functional.leaky_relu(self.bn1(self.conv1(x)))

        out = x_in
        encoder_out = []
        for idx, layer in enumerate(self.encoders):
            out = layer(out)
            encoder_out.append(out)
        out = self.bottleneck(out)

        for idx, layer in enumerate(self.decoders):
            out = layer(torch.cat([out, encoder_out[-1 - idx]], dim=1))

        out = out[..., :x_in.size(-1)]
        out += x_in

        if self.downsample:
            out = self.idfunc_0(out)
            out = self.idfunc_1(out)

        return out


class LeadCombiner(nn.Module):
    def __init__(self, lead, out_ch):
        super(LeadCombiner, self).__init__()
        self.conv2_1 = nn.Conv1d(in_channels=lead * out_ch,
                                 out_channels=out_ch,
                                 kernel_size=1,
                                 bias=False)
        self.bn2_1 = nn.BatchNorm1d(out_ch)

        self.conv2_2 = nn.Conv1d(in_channels=lead * out_ch,
                                 out_channels=out_ch,
                                 kernel_size=1,
                                 bias=False)
        self.bn2_2 = nn.BatchNorm1d(out_ch)

        self.pool1 = nn.AdaptiveMaxPool1d(output_size=1)
        self.pool2 = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x):
        x1 = rearrange(x, 'b c l t -> b (c l) t')
        x1 = functional.leaky_relu(self.bn2_1(self.conv2_1(x1)))
        x2 = rearrange(x, 'b c l t -> b (t l) c')
        x2 = functional.leaky_relu(self.bn2_2(self.conv2_2(x2)))

        x1 = functional.dropout(x1, p=0.5, training=self.training)
        x2 = functional.dropout(x2, p=0.5, training=self.training)

        x1 = self.pool1(x1).squeeze(2)
        x2 = self.pool2(x2).squeeze(2)

        x = torch.cat([x1, x2], dim=1)
        return x


class ResU2D_LC(nn.Module):
    def __init__(self, nOUT, in_ch=1, out_ch=128, mid_ch=64, lead=12):
        super(ResU2D_LC, self).__init__()
        self.inconv = INCONV(in_ch=in_ch, out_ch=out_ch)

        self.rub_0 = ResUBlock(out_ch=out_ch, mid_ch=mid_ch, layers=6)
        self.rub_1 = ResUBlock(out_ch=out_ch, mid_ch=mid_ch, layers=5)
        self.rub_2 = ResUBlock(out_ch=out_ch, mid_ch=mid_ch, layers=4)
        self.rub_3 = ResUBlock(out_ch=out_ch, mid_ch=mid_ch, layers=3)

        self.lc = LeadCombiner(lead=lead, out_ch=out_ch)

        self.fc = nn.Linear(out_ch * 2, nOUT)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.inconv(x)

        x = self.rub_0(x)
        x = self.rub_1(x)
        x = self.rub_2(x)
        x = self.rub_3(x)

        x = self.lc(x)

        x = self.fc(x)
        return x
