"""
Implementation of Attention Augmented Convolutional Networks
https://arxiv.org/pdf/1904.09925.pdf
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import conv1x1, conv3x3
from torchvision.models.densenet import _DenseLayer, _DenseBlock

# --------------------
# Attention augmented convolution
# --------------------

class AAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dk, dv, nh, relative, input_dims, **kwargs):
        super().__init__()
        self.dk = dk
        self.dv = dv
        self.nh = nh
        self.relative = relative

        assert dk % nh == 0, 'nh must divide dk'
        assert dv % nh == 0, 'nh must divide dv'

        # `same` conv since conv and attn are concatenated in the output
        padding = kwargs.pop('padding', None)
        if not padding: padding = kernel_size//2

        self.conv = nn.Conv2d(in_channels, out_channels - dv, kernel_size, stride, padding, bias=False, **kwargs) if out_channels > dv else None
        self.in_proj_qkv = nn.Conv2d(in_channels, 2*dk + dv, kernel_size=1, stride=stride, bias=False)
        self.out_proj = nn.Conv2d(dv, dv, kernel_size=1, bias=False)

        if relative:
            H, W = input_dims
            self.key_rel_h = nn.Parameter(dk**-0.5 + torch.randn(dk//nh, 2*H-1))
            self.key_rel_w = nn.Parameter(dk**-0.5 + torch.randn(dk//nh, 2*W-1))

    def rel_to_abs(self, x):
        B, nh, L, _ = x.shape   # (B, nh, L, 2*L-1)

        # pad to shift from relative to absolute indexing
        x = F.pad(x, (0,1))     # (B, nh, L, 2*L)
        x = x.flatten(2)        # (B, nh, L*2*L)
        x = F.pad(x, (0,L-1))   # (B, nh, L*2*L + L-1)

        # reshape and slice out the padded elements
        x = x.reshape(B, nh, L+1, 2*L-1)
        return x[:,:,:L,L-1:]

    def relative_logits_1d(self, q, rel_k):
        B, nh, H, W, dkh = q.shape

        rel_logits = torch.matmul(q, rel_k)                                     # (B, nh, H, W, 2*W-1)
        # collapse height and heads
        rel_logits = rel_logits.reshape(B, nh*H, W, 2*W-1)
        rel_logits = self.rel_to_abs(rel_logits)                                # (B, nh*H, W, W)
        # shape back and tile height times
        return rel_logits.reshape(B, nh, H, 1, W, W).expand(-1,-1,-1,H,-1,-1)   # (B, nh, H, H, W, W)

    def forward(self, x):
        # compute qkv
        qkv = self.in_proj_qkv(x)
        q, k, v = qkv.split([self.dk, self.dk, self.dv], dim=1)
        # split channels into multiple heads, flatten H,W dims and scale q; out (B, nh, dkh or dvh, HW)
        B, _, H, W = qkv.shape
        flat_q = q.reshape(B, self.nh, self.dk//self.nh, H, W).flatten(3) * (self.dk//self.nh)**-0.5
        flat_k = k.reshape(B, self.nh, self.dk//self.nh, H, W).flatten(3)
        flat_v = v.reshape(B, self.nh, self.dv//self.nh, H, W).flatten(3)

        logits = torch.matmul(flat_q.transpose(2,3), flat_k)    # (B, nh, HW, HW)
        if self.relative:
            q = flat_q.reshape(B, self.nh, self.dk//self.nh, H, W).permute(0,1,3,4,2)  # (B, nh, H, W, dkh)
            # compute relative logits in width dim
            w_rel_logits = self.relative_logits_1d(q, self.key_rel_w)                  # (B, nh, H, H, W, W)
            # repeat for heigh dim by transposing H,W and then permuting output
            h_rel_logits = self.relative_logits_1d(q.transpose(2,3), self.key_rel_h)   # (B, nh, W, W, H, H)
            # permute and reshape for adding to the attention logits
            h_rel_logits = h_rel_logits.permute(0,1,2,4,3,5).reshape(B, self.nh, H*W, H*W)
            w_rel_logits = w_rel_logits.permute(0,1,4,2,5,3).reshape(B, self.nh, H*W, H*W)
            # add to attention logits
            logits += h_rel_logits + w_rel_logits
        self.weights = F.softmax(logits, -1)

        attn_out = torch.matmul(self.weights, flat_v.transpose(2,3)) # (B, nh, HW, dvh)
        attn_out = attn_out.transpose(2,3)                           # (B, nh, dvh, HW)
        attn_out = attn_out.reshape(B, -1 , H, W)                    # (B, dv, H, W)
        attn_out = self.out_proj(attn_out)

        if self.conv is not None:
            return torch.cat([self.conv(x), attn_out], dim=1)
        else:
            return attn_out

    def extra_repr(self):
        return 'dk={dk}, dv={dv}, nh={nh}, relative={relative}'.format(**self.__dict__)


# --------------------
# Attention Augmented ResNet
# --------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, input_dims=None, attn_params=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # attention
        if attn_params is not None:
            nh = attn_params['nh']
            dk = max(20*nh, int((attn_params['k'] * planes // nh)*nh))
            dv = int((attn_params['v'] * planes // nh)*nh)
            relative = attn_params['relative']
            # scale input dims to network HW outputs at this layer
            input_dims = int(attn_params['input_dims'][0] * 16 / planes), int(attn_params['input_dims'][1] * 16 / planes)
            print('BasicBlock attention: dk {}, dv {}, input_dims {}x{}'.format(dk, dv, *input_dims))

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride) if attn_params is None else \
                     AAConv2d(inplanes, planes, 3, stride, dk, dv, nh, relative, input_dims)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, input_dims=None, attn_params=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        # attention
        if attn_params is not None:
            nh = attn_params['nh']
            dk = max(20*nh, int((attn_params['k'] * width // nh)*nh))
            dv = int((attn_params['v'] * width // nh)*nh)
            relative = attn_params['relative']
            # scale input dims to network HW outputs at this layer
            input_dims = int(attn_params['input_dims'][0] * 16 / planes), int(attn_params['input_dims'][1] * 16 / planes)
            print('Bottleneck attention: dk {}, dv {}, input_dims {}x{}'.format(dk, dv, *input_dims))

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation) if attn_params is None else \
                     AAConv2d(width, width, 3, stride, dk, dv, nh, relative, input_dims, groups=groups, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ cf paper -- replaces the first conv3x3 in BasicBlock and Bottleneck of Resnet layers 2,3,4;
    ResNet class from torchvision. """

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, attn_params=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], attn_params=attn_params)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], attn_params=attn_params)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], attn_params=attn_params)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, attn_params=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, attn_params=attn_params))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, attn_params=attn_params))

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
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# --------------------
# Attention augmented WideResNet
# --------------------

class WideResNet(nn.Module):
    """ cf paper -- replaces the first conv3x3 in BasicBlock of WideResnet layers 2,3;
    WideResnet implementation is a modiefied torchvision ResNet class. """

    def __init__(self, block, depth, width, num_classes=100, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, attn_params=None):
        super().__init__()

        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4)//6
        if attn_params:
            # rescale input dims to reuse the same feature maps HW scale calcs at BasicBlock and Bottleneck
            attn_params['input_dims'] = int(attn_params['input_dims'][0] * width), int(attn_params['input_dims'][1] * width)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16 * width, n)
        self.layer2 = self._make_layer(block, 32 * width, n, stride=2, dilate=replace_stride_with_dilation[0], attn_params=attn_params)
        self.layer3 = self._make_layer(block, 64 * width, n, stride=2, dilate=replace_stride_with_dilation[1], attn_params=attn_params)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * width, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, attn_params=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, attn_params=attn_params))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, attn_params=attn_params))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# --------------------
# Attention augmented Densenet
# --------------------

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, attn_params=None):
        super(_Transition, self).__init__()

        # attention
        if attn_params is not None:
            nh = attn_params['nh']
            dk = max(20*nh, int((attn_params['k'] * num_output_features // nh)*nh))
            dv = int((attn_params['v'] * num_output_features // nh)*nh)
            relative = attn_params['relative']
            if True:
                # uses a strided conv to downsample, so attn applied on downsampled features
                input_dims = attn_params['input_dims'][0]//2, attn_params['input_dims'][1]//2
            else:
                # substitutes the 1x1 conv with attn-aug conv before downsampling, so attn applied on pre-downsampled features
                input_dims = attn_params['input_dims'][0], attn_params['input_dims'][1]
            print('Transition layer attention: dk {}, dv {}, input_dims {}x{}'.format(dk, dv, *input_dims))

        if attn_params is None:
            # standard densenet architecture
            self.add_module('norm', nn.BatchNorm2d(num_input_features))
            self.add_module('relu', nn.ReLU(inplace=True))
            self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            if True:
                # use stride 2 conv for downsampling instead of avgpool and attn-augment the strided conv
                self.add_module('norm', nn.InstanceNorm2d(num_input_features))
                self.add_module('relu', nn.ReLU(inplace=True))
                self.add_module('conv', AAConv2d(num_input_features, num_output_features, 3, 2, dk, dv, nh, relative, input_dims))
            else:
                # substitute the standard densenet architecture with an attn-aug conv
                self.add_module('norm', nn.BatchNorm2d(num_input_features))
                self.add_module('relu', nn.ReLU(inplace=True))
                self.add_module('conv', AAConv2d(num_input_features, num_output_features, 1, 1, dk, dv, nh, relative, input_dims))
                self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    """ cf paper -- replaces the conv in every Transition layer after the first (where feature maps are still large in HW);
    Densenet class from torchvision. """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, attn_params=None):

        super(DenseNet, self).__init__()

        # First convolution -- cf paper section 3 implementation details
        #   all densenet have 3 dense blocks except imagenet which uses 4; imagenet config also uses different network stem
        if len(block_config)==4:  # imagenet config
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))
            if attn_params is not None:
                # scale input dims to attn HW outputs after pooling layer
                attn_params['input_dims'] = attn_params['input_dims'][0] // 4, attn_params['input_dims'][1] // 4
        else:  # all densenet configs except imagenet described in the paper
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=5, stride=1, padding=2, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
            ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2,
                                    attn_params=attn_params)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

            # attention -- scale input dims to attn HW outputs after downsampling in transition layer
            if attn_params is not None:
                attn_params['input_dims'] = attn_params['input_dims'][0] // 2, attn_params['input_dims'][1] // 2


        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out




if __name__ == '__main__':
    # NOTE -- assert statements are wrt number of parameters in the respective papers

    if True:
        # Densenet-BC (k=12, L=100) on CIFAR
        m = DenseNet(12, (16, 16, 16), 24, num_classes=10)
        n_params = sum(p.numel() for p in m.parameters())
        print('Densenet-BC (k=12, L=100) params: {:,}'.format(n_params))
        assert round(n_params * 1e-6, 1) == 0.8
        m(torch.rand(1,3,32,32))

        # Densenet-BC (k=24, L=250) on CIFAR
        m = DenseNet(24, (41, 41, 41), 48, num_classes=10)
        n_params = sum(p.numel() for p in m.parameters())
        print('Densenet-BC (k=24, L=250) params: {:,}'.format(n_params))
        assert round(n_params * 1e-6, 1) == 15.3
        m(torch.rand(1,3,32,32))

        # Densenet-BC (k=40, L=190) on CIFAR
        m = DenseNet(40, (31, 31, 31), 80, num_classes=10)
        n_params = sum(p.numel() for p in m.parameters())
        print('Densenet-BC (k=40, L=190) params: {:,}'.format(n_params))
        assert round(n_params * 1e-6, 1) == 25.6
        m(torch.rand(1,3,32,32))

        # Densenet-BC 121 on Imagenet
        m = DenseNet(32, (6, 12, 24, 16), 64)
        n_params = sum(p.numel() for p in m.parameters())
        print('Densenet121 params: {:,}'.format(n_params))
        m(torch.rand(1,3,224,224))

    if True:
        # NOTE -- when augmenting the 3x3 conv in _DenseLayer; its out_channels==growth_rate;
        #         AAConv2d's self.conv outputs out_channels - dv where dv = out_channels * v;
        #           --> v needs to be set s.t. dv = growth_rate * v is large enough, and
        #                                      dvh = dv / dh has sufficient representation power, and
        #                                      self.conv.out_channels = growth_rate - dv is large enough so AAConv2d is not all attention
        #               e.g. at growth_rate=k=12, nh=8, v=0.7; dv = 8 and self.conv.out_channels = 4; but dvh = 1
        #                    at growth_rate=k=12, nh=4, v=0.7; dv = 8 and dvh = 2

        # Densenet-BC (k=12, L=100) on CIFAR
        # v=0.7 -> dv = 8 and dvh = 2
        m = DenseNet(12, (16, 16, 16), 24, num_classes=10, attn_params={'k': 0.2, 'v': 0.7, 'nh': 4, 'relative': True, 'input_dims': (32,32)})
        n_params = sum(p.numel() for p in m.parameters())
        print('AADensenet-BC (k=12, L=100) params: {:,}'.format(n_params))
        m(torch.rand(1,3,32,32))

        # Densenet-BC (k=24, L=250) on CIFAR
        # v=0.7 -> dv = 16 and dvh = 2
        m = DenseNet(24, (41, 41, 41), 48, num_classes=10, attn_params={'k': 0.2, 'v': 0.7, 'nh': 8, 'relative': True, 'input_dims': (32,32)})
        n_params = sum(p.numel() for p in m.parameters())
        print('AADensenet-BC (k=24, L=250) params: {:,}'.format(n_params))
        m(torch.rand(1,3,32,32))

        # AADensenet121 on 224x224
        # v=0.5 -> dv = 16 and dvh = 2
        m = DenseNet(32, (6, 12, 24, 16), 64, attn_params={'k': 0.2, 'v': 0.5, 'nh': 8, 'relative': True, 'input_dims': (224,224)})
        n_params = sum(p.numel() for p in m.parameters())
        print('AADensenet121 params: {:,}'.format(n_params))
        m(torch.rand(1,3,224,224))

        # AADensenet121 on 320x320
        # v=0.5 -> dv = 16 and dvh = 2
        m = DenseNet(32, (6, 12, 24, 16), 64, attn_params={'k': 0.2, 'v': 0.5, 'nh': 8, 'relative': True, 'input_dims': (320,320)})
        n_params = sum(p.numel() for p in m.parameters())
        print('AADensenet121 params: {:,}'.format(n_params))
        m(torch.rand(1,3,320,320))

    if True:
        # WideResnet-28-10
        m = WideResNet(BasicBlock, 28, 10, num_classes=100)
        n_params = sum(p.numel() for p in m.parameters())
        print('WideResNet-28-10 params: {:,}'.format(n_params))
#        assert round(n_params * 1e-6, 1) == 36.3
        m(torch.rand(1,3,32,32))

        # attention augment WideResnet
        m = WideResNet(BasicBlock, 28, 10, num_classes=100, attn_params={'k': 0.2, 'v': 0.1, 'nh': 8, 'relative': True, 'input_dims': (32,32)})
        n_params = sum(p.numel() for p in m.parameters())
        print('AAWideResnet-28-10 -- k=v=0.25 -- params: {:,}'.format(n_params))
        assert round(n_params * 1e-6, 1) == 36.2
        m(torch.rand(1,3,32,32))

    if True:
        # original Resnet
        m = ResNet(BasicBlock, [3,4,6,3])
        n_params = sum(p.numel() for p in m.parameters())
        print('Resnet34 params: {:,}'.format(n_params))
        assert round(n_params * 1e-6, 1) == 21.8
        m(torch.rand(1,3,224,224))

        m = ResNet(Bottleneck, [3,4,6,3])
        n_params = sum(p.numel() for p in m.parameters())
        print('Resnet50 params: {:,}'.format(n_params))
        assert round(n_params * 1e-6, 1) == 25.6
        m(torch.rand(1,3,224,224))

        # attention augment Resnets
        m = ResNet(BasicBlock, [3,4,6,3], attn_params={'k': 0.25, 'v': 0.25, 'nh': 8, 'relative': True, 'input_dims': (224,224)})
        n_params = sum(p.numel() for p in m.parameters())
        print('AAResnet34 -- k=v=0.25 -- params: {:,}'.format(n_params))
        assert round(n_params * 1e-6, 1) == 20.7
        m(torch.rand(1,3,224,224))

        m = ResNet(Bottleneck, [3,4,6,3], attn_params={'k': 0.2, 'v': 0.1, 'nh': 8, 'relative': True, 'input_dims': (224,224)})
        n_params = sum(p.numel() for p in m.parameters())
        print('AAResnet50 -- k=2v=0.2 -- params: {:,}'.format(n_params))
        assert round(n_params * 1e-6, 1) == 25.8
        m(torch.rand(1,3,224,224))

        m = ResNet(Bottleneck, [3,4,6,3], attn_params={'k': 0.25, 'v': 0.25, 'nh': 8, 'relative': True, 'input_dims': (224,224)})
        n_params = sum(p.numel() for p in m.parameters())
        print('AAResnet50 -- k=v=0.25 -- params: {:,}'.format(n_params))
        assert round(n_params * 1e-6, 1) == 24.3
        m(torch.rand(1,3,224,224))

        m = ResNet(Bottleneck, [3,4,6,3], attn_params={'k': 0.5, 'v': 0.5, 'nh': 8, 'relative': True, 'input_dims': (224,224)})
        n_params = sum(p.numel() for p in m.parameters())
        print('AAResnet50 -- k=v=0.5  -- params: {:,}'.format(n_params))
        assert round(n_params * 1e-6, 1) == 22.3
        m(torch.rand(1,3,224,224))

        m = ResNet(Bottleneck, [3,4,6,3], attn_params={'k': 0.75, 'v': 0.75, 'nh': 8, 'relative': True, 'input_dims': (224,224)})
        n_params = sum(p.numel() for p in m.parameters())
        print('AAResnet50 -- k=v=0.75 -- params: {:,}'.format(n_params))
        assert round(n_params * 1e-6, 1) == 20.7
        m(torch.rand(1,3,224,224))

        m = ResNet(Bottleneck, [3,4,6,3], attn_params={'k': 1, 'v': 1, 'nh': 8, 'relative': True, 'input_dims': (224,224)})
        n_params = sum(p.numel() for p in m.parameters())
        print('AAResnet50 -- k=v=1    -- params: {:,}'.format(n_params))
        assert round(n_params * 1e-6, 1) == 19.4
        m(torch.rand(1,3,224,224))

