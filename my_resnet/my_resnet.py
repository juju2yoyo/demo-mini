import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

__all__ = ['ResNet', 'resnet101', 'resnet152']

model_urls = {
  'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
  'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_channel, out_channel, stride=1, groups=1, dilation=1):
  return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, bias=False, padding=dilation, groups=groups, dilation=dilation)


def conv1x1(in_channel, out_channel, stride=1):
  return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):

  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None,
               groups=1, dilation=1, norm_layer=None):
      super(Bottleneck, self).__init__()
      if norm_layer is None:
          norm_layer = nn.BatchNorm2d
      self.conv1 = conv1x1(inplanes, planes)
      self.bn1 = norm_layer(planes)
      self.conv2 = conv3x3(planes, planes, stride)
      self.bn2 = norm_layer(planes)
      self.conv3 = conv1x1(planes, planes * self.expansion)
      self.bn3 = norm_layer(planes * self.expansion)
      self.relu = nn.ReLU(inplace=True)
      self.stride = stride
      self.downsample = downsample

  def forward(self, x):
    if self.downsample is not None:
        identity = self.downsample(x)
    else:
        identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)
    out = self.relu(out)

    out += identity
    out = self.relu(out)

    return out


class ResNet(nn.Module):

  def __init__(self, block, layers, num_classes=1000, norm_layer=None):
      super(ResNet, self).__init__()
      if norm_layer is None:
          norm_layer = nn.BatchNorm2d
      self._norm_layer = norm_layer

      self.inplanes = 64
      self.dilation = 1
      self.groups = 1
      self.base_width = 64

      self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
      self.bn1 = norm_layer(self.inplanes)
      self.relu = nn.ReLU(inplace=True)
      self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

      self.layer1 = self._make_layer(block, 64, layers[0])
      self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
      self.layer3 = self._make_layer(block, 256, layers[1], stride=2)
      self.layer4 = self._make_layer(block, 512, layers[1], stride=2)

      self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
      self.fc = nn.Linear(512 * block.expansion, num_classes)

      # init weights
      for m in self.modules():
          if isinstance(m, nn.Conv2d):
              nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
          elif isinstance(m, nn.BatchNorm2d):
              nn.init.constant_(m.weight, 1)
              nn.init.constant_(m.bias, 0)

  def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
      if stride != 1:
          downsample = nn.Sequential(
              conv1x1(planes, planes * block.expansion, stride),
              self._norm_layer(planes * block.expansion)
          )

      if stride != 1 or self.inplanes != planes * block.expansion:
          downsample = nn.Sequential(
              conv1x1(self.inplanes, planes * block.expansion, stride),
              self._norm_layer(planes * block.expansion))

      layers = []
      layers.append(block(self.inplanes, planes, stride, downsample))
      self.inplanes = planes * block.expansion

      for _ in range(1, blocks):
          layers.append(block(self.inplanes, planes))
      return nn.Sequential(*layers)

  def forward(self, x):
      x = self.conv1(x)
      x = self.bn1(x)
      x = self.relu(x)
      x = self.maxpool(x)

      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x = self.layer4(x)

      x = self.avgpool(x)
      x = torch.flatten(x, 1)
      x = self.fc(x)

      return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet101(pretrained=False, progress=True, **kwargs):
  return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
  return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)















