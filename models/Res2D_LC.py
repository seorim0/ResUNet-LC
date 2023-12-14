"""
2-dimensional Residual Network with a lead combiner
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


class ResBlock(nn.Module):
    def __init__(self, out_ch, downsample):
        super(ResBlock, self).__init__()
        self.downsample = downsample
        self.stride = 2 if self.downsample else 1
        self.conv1 = nn.Conv2d(in_channels=out_ch,
                               out_channels=out_ch,
                               kernel_size=(3, 9),
                               stride=(1, self.stride),
                               padding=(1, 4),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(in_channels=out_ch,
                               out_channels=out_ch,
                               kernel_size=(3, 9),
                               padding=(1, 4),
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if self.downsample:
            self.idfunc_0 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
            self.idfunc_1 = nn.Conv2d(in_channels=out_ch,
                                      out_channels=out_ch,
                                      kernel_size=1,
                                      bias=False)

    def forward(self, x):
        identity = x
        if x.size(2) % 2 != 0:
            identity = functional.pad(identity, (1, 0, 0, 0))
        x = functional.leaky_relu(self.bn1(self.conv1(x)))
        x = functional.leaky_relu(self.bn2(self.conv2(x)))
        if self.downsample:
            identity = self.idfunc_0(identity)
            identity = self.idfunc_1(identity)

        x = x + identity
        return x


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


class Res2D_LC(nn.Module):
    def __init__(self, nOUT, in_ch=1, out_ch=128, lead=12):
        super(Res2D_LC, self).__init__()
        self.inconv = INCONV(in_ch=in_ch, out_ch=out_ch)

        self.rb_0 = ResBlock(out_ch=out_ch, downsample=True)
        self.rb_1 = ResBlock(out_ch=out_ch, downsample=True)
        self.rb_2 = ResBlock(out_ch=out_ch, downsample=True)
        self.rb_3 = ResBlock(out_ch=out_ch, downsample=True)

        self.pool = nn.AdaptiveMaxPool2d(output_size=1)

        self.lc = LeadCombiner(lead=lead, out_ch=out_ch)

        self.fc = nn.Linear(out_ch, nOUT)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.inconv(x)

        x = self.rb_0(x)
        x = self.rb_1(x)
        x = self.rb_2(x)
        x = self.rb_3(x)

        x = self.lc(x)

        x = self.fc(x)
        return x
