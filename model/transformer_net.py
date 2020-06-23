import torch
import torch.nn as nn
import torchvision.models as models

from collections import namedtuple


def norm_fn(norm):
    if norm == "instance":
        return nn.InstanceNorm2d
    elif norm == "group":
        return nn.GroupNorm
    elif norm == "batch":
        return nn.BatchNorm2d
    else:
        print(f'W: norm \"{norm}\" is not supported! default norm \"instance\" is used instead.')
        return nn.InstanceNorm2d


class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels, norm):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = norm_fn(norm)(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = norm_fn(norm)(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class ConvEncoder(nn.Module):
    def __init__(self, norm):
        super(ConvEncoder, self).__init__()
        # Initial convolution layers
        self.model = nn.Sequential()

        self.model.add_module('conv1', ConvLayer(3, 32, kernel_size=9, stride=1))
        self.model.add_module('in1', norm_fn(norm)(32, affine=True))
        self.model.add_module('relu1', nn.ReLU())

        self.model.add_module('conv2', ConvLayer(32, 64, kernel_size=3, stride=2))
        self.model.add_module('in2', norm_fn(norm)(64, affine=True))
        self.model.add_module('relu2', nn.ReLU())

        self.model.add_module('conv3', ConvLayer(64, 128, kernel_size=3, stride=2))
        self.model.add_module('in3', norm_fn(norm)(128, affine=True))
        self.model.add_module('relu3', nn.ReLU())

    def forward(self, x):
        return self.model(x)


class VGG16BonedSkinEncoder(nn.Module):
    def __init__(self, norm):
        super(VGG16BonedSkinEncoder, self).__init__()
        # self.vgg = VGG16()
        self.relu = nn.ReLU()
        transfilters = []
        transfilter1 = nn.Sequential()
        transfilter1.add_module('trans_conv1_1', ConvLayer(64, 16, kernel_size=3, stride=1))
        transfilter1.add_module('trans_norm1_1', norm_fn(norm)(16, affine=True))
        transfilters.append(transfilter1)
        transfilter2 = nn.Sequential()
        transfilter2.add_module('trans_conv2_1', ConvLayer(128, 16, kernel_size=3, stride=1))
        transfilter2.add_module('trans_norm2_1', norm_fn(norm)(16, affine=True))
        transfilters.append(transfilter2)
        transfilter3 = nn.Sequential()
        transfilter3.add_module('trans_conv3_1', ConvLayer(256, 64, kernel_size=3, stride=1))
        transfilter3.add_module('trans_norm3_1',   norm_fn(norm)(64, affine=True))
        transfilters.append(transfilter3)
        # transfilter4 = nn.Sequential()
        # transfilter4.add_module('trans_conv4_1', ConvLayer(512, 32, kernel_size=3, stride=1))
        # transfilters.append(transfilter4)
        self.transfilters = nn.ModuleList(transfilters)

        fusers = []
        fuser1 = nn.Sequential()
        fuser1.add_module('trans_conv1_2', ConvLayer(16, 16, kernel_size=3, stride=2))
        fuser1.add_module('trans_norm1_2',   norm_fn(norm)(16, affine=True))
        fusers.append(fuser1)
        fuser2 = nn.Sequential()
        fuser2.add_module('trans_conv2_2', ConvLayer(32, 64, kernel_size=3, stride=2))
        fuser2.add_module('trans_norm2_2',   norm_fn(norm)(64, affine=True))
        fusers.append(fuser2)
        fuser3 = nn.Sequential()
        fuser3.add_module('trans_conv3_2', ConvLayer(128, 128, kernel_size=3, stride=1))
        fuser3.add_module('trans_norm3_2',   norm_fn(norm)(128, affine=True))
        fusers.append(fuser3)
        # fuser4 = nn.Sequential()
        # fuser4.add_module('trans_conv4_2', ConvLayer(64, 64, kernel_size=3, stride=1))
        # fuser4.add_module('trans_norm4',   norm_fn(norm)(64, affine=True))
        # fusers.append(fuser4)
        self.fusers = nn.ModuleList(fusers)

    def get_num_output_channels(self):
        return 128

    def forward(self, x, features):
        # features = self.vgg(x)
        features = [features[i] for i in range(len(self.transfilters))]

        prev = None
        for f, transfilter, fuser in zip(features, self.transfilters, self.fusers):
            transformed = self.relu(transfilter(f))
            prev = self.relu(fuser(torch.cat((prev, transformed), dim=1))) if prev is not None else fuser(transformed)

        out = prev # torch.cat((features[-1], prev), dim=1)

        return out


class ConvDecoder(nn.Module):
    def __init__(self, norm, in_channels=128):
        super(ConvDecoder, self).__init__()
        # Upsampling Layers
        self.model = nn.Sequential()

        self.model.add_module('deconv1', UpsampleConvLayer(in_channels, 64, kernel_size=3, stride=1, upsample=2))
        self.model.add_module('in4', norm_fn(norm)(64, affine=True))
        self.model.add_module('relu4', nn.ReLU())

        self.model.add_module('deconv2', UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2))
        self.model.add_module('in5', norm_fn(norm)(32, affine=True))
        self.model.add_module('relu5', nn.ReLU())

        self.model.add_module('deconv3', ConvLayer(32, 3, kernel_size=9, stride=1))

    def forward(self, x):
        return self.model(x)


class TransformerNet(nn.Module):
    def __init__(self, norm="instance"):
        super(TransformerNet, self).__init__()
        # Encoder
        self.encoder = ConvEncoder(norm=norm)

        # Residual layers
        self.residual = nn.Sequential()
        for i in range(5):
            self.residual.add_module('resblock_%d' % (i + 1), ResidualBlock(128, norm=norm))

        # Decoder
        self.decoder = ConvDecoder(norm=norm)

    def forward(self, x):
        encoder_output = self.encoder(x)
        residual_output = self.residual(encoder_output)
        decoder_output = self.decoder(residual_output)

        return decoder_output
