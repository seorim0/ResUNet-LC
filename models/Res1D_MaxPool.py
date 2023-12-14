"""
1-dimensional Residual Network with a max-pooling layer
"""
import torch.nn as nn
import torch.nn.functional as functional


class INCONV(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(INCONV, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_ch,
                              out_channels=out_ch,
                              kernel_size=15,
                              padding=7,
                              stride=2,
                              bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        x = functional.leaky_relu(self.bn(self.conv(x)))
        return x


class ResBlock(nn.Module):
    def __init__(self, out_ch, downsample):
        super(ResBlock, self).__init__()
        self.downsample = downsample
        self.stride = 2 if self.downsample else 1
        K = 9
        P = (K - 1) // 2
        self.conv1 = nn.Conv1d(in_channels=out_ch,
                               out_channels=out_ch,
                               kernel_size=K,
                               stride=self.stride,
                               padding=P,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)

        self.conv2 = nn.Conv1d(in_channels=out_ch,
                               out_channels=out_ch,
                               kernel_size=K,
                               padding=P,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        if self.downsample:
            self.idfunc_0 = nn.AvgPool1d(kernel_size=2, stride=2)
            self.idfunc_1 = nn.Conv1d(in_channels=out_ch,
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


class Res1D_MaxPool(nn.Module):
    def __init__(self, nOUT, in_ch=12, out_ch=128):
        super(Res1D_MaxPool, self).__init__()
        self.inconv = INCONV(in_ch=in_ch, out_ch=out_ch)

        self.rb_0 = ResBlock(out_ch=out_ch, downsample=True)
        self.rb_1 = ResBlock(out_ch=out_ch, downsample=True)
        self.rb_2 = ResBlock(out_ch=out_ch, downsample=True)
        self.rb_3 = ResBlock(out_ch=out_ch, downsample=True)

        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

        self.fc = nn.Linear(out_ch, nOUT)

    def forward(self, x):
        x = self.inconv(x)

        x = self.rb_0(x)
        x = self.rb_1(x)
        x = self.rb_2(x)
        x = self.rb_3(x)

        x = functional.dropout(x, p=0.5, training=self.training)

        x = self.pool(x).squeeze(2)  # B, C, D

        x = self.fc(x)
        return x
