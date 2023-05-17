import torch
import torch.nn as nn

class RTGB(nn.Module):
    def __init__(self, out_channel, height, width):
        super(RTGB, self).__init__()
        self.channel_gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )
        self.wide_gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(height, height, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

        self.high_gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(width, width, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        c = self.channel_gap(x)
        h = self.high_gap(x.transpose(-3, -2))
        w = self.wide_gap(x.transpose(-3, -1))
        # h = h.transpose(-3, -2)
        # w = w.transpose(-3, -1)
        # print(c.shape, h.shape, w.shape)
        ch = torch.einsum("...cbd,...hbd->...chbd", c, h)
        # print(ch.shape)
        chw = torch.einsum("...chbd,...wbd->...chw", ch, w)
        # print(chw.shape)
        return chw

class RTGB_SE(nn.Module):
    def __init__(self, out_channel, height, width, ration=16):
        super(RTGB_SE, self).__init__()
        self.channel_gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channel, out_channel//ration, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channel // ration, out_channel, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )
        self.wide_gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(height, height, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(height, height, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

        self.high_gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(width, width, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(width, width, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        c = self.channel_gap(x)
        h = self.high_gap(x.transpose(-3, -2))
        w = self.wide_gap(x.transpose(-3, -1))
        # h = h.transpose(-3, -2)
        # w = w.transpose(-3, -1)
        # print(c.shape, h.shape, w.shape)
        ch = torch.einsum("...cbd,...hbd->...chbd", c, h)
        # print(ch.shape)
        chw = torch.einsum("...chbd,...wbd->...chw", ch, w)
        # print(chw.shape)
        return chw


class RTGBv2(nn.Module):
    def __init__(self, out_channel, height, width):
        super(RTGBv2, self).__init__()
        self.channel_gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=False),
        )
        self.wide_gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(height, height, kernel_size=1, stride=1, bias=False),
        )

        self.high_gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(width, width, kernel_size=1, stride=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c = self.channel_gap(x)
        h = self.high_gap(x.transpose(-3, -2))
        w = self.wide_gap(x.transpose(-3, -1))
        # h = h.transpose(-3, -2)
        # w = w.transpose(-3, -1)
        # print(c.shape, h.shape, w.shape)
        ch = torch.einsum("...cbd,...hbd->...chbd", c, h)
        # print(ch.shape)
        chw = torch.einsum("...chbd,...wbd->...chw", ch, w)
        # print(chw.shape)
        return self.sigmoid(chw)

class RTGBv3(nn.Module):
    def __init__(self, out_channel, height, width):
        super(RTGBv3, self).__init__()
        self.channel_gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=False),
        )
        self.wide_gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(height, height, kernel_size=1, stride=1, bias=False),
        )

        self.high_gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(width, width, kernel_size=1, stride=1, bias=False),
        )


    def forward(self, x):
        c = self.channel_gap(x)
        h = self.high_gap(x.transpose(-3, -2))
        w = self.wide_gap(x.transpose(-3, -1))

        ch = torch.einsum("...cbd,...hbd->...chbd", c, h)

        chw = torch.einsum("...chbd,...wbd->...chw", ch, w)

        return chw

class DRTLM(nn.Module):
    def __init__(self, rank, out_channel, height, width):# rank = 4
        super(DRTLM, self).__init__()
        self.rank = rank
        print(out_channel, height, width)
        self.rtgb = RTGB(out_channel, height, width)
        self.projection = nn.Conv2d(out_channel*rank, out_channel, 1)

    def resblock(self, input):
        xup = self.rtgb(input)
        res = input - xup
        return xup, res

    def forward(self, x):
        (xup, xdn) = self.resblock(x)
        temp_xup = xdn
        output = xup
        for i in range(1, self.rank):
            (temp_xup, temp_xdn) = self.resblock(temp_xup)
            xup = xup + temp_xup
            output = torch.cat((output, xup), 1)
            temp_xup = temp_xdn

        output = self.projection(output)
        return output

class DRTLM_v2(nn.Module):
    def __init__(self, rank, out_channel, height, width):
        super(DRTLM_v2, self).__init__()
        self.rank = rank
        # print(out_channel, height, width)
        self.preconv = nn.Sequential(nn.Conv2d(out_channel, out_channel, kernel_size=1),
                                     nn.BatchNorm2d(out_channel),
                                     nn.ReLU(inplace=True))
        self.rtgb = RTGB(out_channel=out_channel, height=height, width=width)
        self.projection = nn.Conv2d(out_channel*rank, out_channel, 1)
        # self.sigmoid = nn.Sigmoid()

    def resblock(self, input):
        xup = self.rtgb(input)
        res = input - xup
        # xdn = self.rtgb(res)
        # xdn = xdn + xup
        return xup, res

    def forward(self, input, vis=False):

        x = self.preconv(input)
        (xup, xdn) = self.resblock(x)

        temp_xup = xdn
        attention_map = [temp_xup]
        output = xup
        for i in range(1, self.rank):
            (temp_xup, temp_xdn) = self.resblock(temp_xup)
            # xup = xup + temp_xup
            if vis:
                attention_map.append(temp_xup)
            output = torch.cat((output, temp_xup), 1)
            temp_xup = temp_xdn

        output = self.projection(output) * x
        # if vis:
        #     attention_map = output.clone()
        output = input + output
        if vis:
            return output, attention_map
        else:
            return output