"""Standalone UPerNet model definition for ONNX export and MindSpore Lite deployment.

This module reimplements the UPerNet (Unified Perceptual Parsing) architecture
without dependencies on custom CUDA operators (PrRoIPool) or SynchronizedBatchNorm.
PrRoIPool2D is replaced with AdaptiveAvgPool2d, and SynchronizedBatchNorm2d is
replaced with standard nn.BatchNorm2d (equivalent in eval mode).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    """Create a 3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    """Create a 3x3 conv + BatchNorm + ReLU sequential module."""
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet (1x1 -> 3x3 -> 1x1 with expansion=4)."""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Forward pass of bottleneck block."""
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNetEncoder(nn.Module):
    """ResNet50 encoder that returns multi-scale feature maps.

    Unlike standard torchvision ResNet, this uses a 3-conv stem
    (conv1/conv2/conv3) matching the original UPerNet implementation.
    """

    def __init__(self, layers=(3, 4, 6, 3)):
        super().__init__()
        self.inplanes = 128
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """Build a residual layer with the given block type."""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass returning list of 4 feature maps at strides 4/8/16/32."""
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        conv_out = []
        x = self.layer1(x); conv_out.append(x)
        x = self.layer2(x); conv_out.append(x)
        x = self.layer3(x); conv_out.append(x)
        x = self.layer4(x); conv_out.append(x)
        return conv_out


class UPerNetDecoder(nn.Module):
    """UPerNet decoder with PPM (Pyramid Pooling Module) and FPN.

    PrRoIPool2D is replaced with AdaptiveAvgPool2d for ONNX compatibility.
    SynchronizedBatchNorm2d is replaced with nn.BatchNorm2d.
    """

    def __init__(self, nr_classes, fc_dim=2048,
                 pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=512):
        super().__init__()

        # PPM Module - use AdaptiveAvgPool2d instead of PrRoIPool2D
        self.ppm_pooling = nn.ModuleList(
            [nn.AdaptiveAvgPool2d((scale, scale)) for scale in pool_scales]
        )
        self.ppm_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ) for _ in pool_scales
        ])
        self.ppm_last_conv = conv3x3_bn_relu(
            fc_dim + len(pool_scales) * 512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True),
            ) for fpn_inplane in fpn_inplanes[:-1]
        ])
        self.fpn_out = nn.ModuleList([
            nn.Sequential(conv3x3_bn_relu(fpn_dim, fpn_dim, 1))
            for _ in range(len(fpn_inplanes) - 1)
        ])
        self.conv_fusion = conv3x3_bn_relu(
            len(fpn_inplanes) * fpn_dim, fpn_dim, 1)

        # Task heads
        self.nr_scene_class = nr_classes['scene']
        self.nr_object_class = nr_classes['object']
        self.nr_part_class = nr_classes['part']
        self.nr_material_class = nr_classes['material']

        self.scene_head = nn.Sequential(
            conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fpn_dim, self.nr_scene_class, kernel_size=1, bias=True),
        )
        self.object_head = nn.Sequential(
            conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, self.nr_object_class, kernel_size=1, bias=True),
        )
        self.part_head = nn.Sequential(
            conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, self.nr_part_class, kernel_size=1, bias=True),
        )
        self.material_head = nn.Sequential(
            conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, self.nr_material_class, kernel_size=1, bias=True),
        )

    def forward(self, conv_out):
        """Forward pass returning raw logits for all four tasks.

        Args:
            conv_out: list of 4 feature maps from encoder [P1..P4].

        Returns:
            Tuple of (scene_logits, object_logits, part_logits, material_logits).
            Scene logits: [B, num_scene, 1, 1].
            Object/Part/Material logits: [B, num_classes, H/4, W/4].
        """
        conv5 = conv_out[-1]
        input_size = conv5.size()
        h_feat, w_feat = input_size[2], input_size[3]

        # PPM: pyramid pooling with adaptive avg pool + bilinear upsample
        ppm_out = [conv5]
        for pool_layer, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            pooled = pool_layer(conv5)
            upsampled = F.interpolate(
                pooled, (h_feat, w_feat), mode='bilinear', align_corners=False)
            ppm_out.append(pool_conv(upsampled))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        # Scene head (operates on PPM output)
        scene_logits = self.scene_head(f)

        # FPN: top-down pathway
        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f = F.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False)
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))
        fpn_feature_list.reverse()  # [P2, P3, P4, P5]

        # Material head (operates on P2, finest resolution)
        material_logits = self.material_head(fpn_feature_list[0])

        # FPN fusion for object and part
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(F.interpolate(
                fpn_feature_list[i], output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_fusion(fusion_out)

        object_logits = self.object_head(x)
        part_logits = self.part_head(x)

        return scene_logits, object_logits, part_logits, material_logits


class UPerNetModel(nn.Module):
    """Full UPerNet model (encoder + decoder) for unified perceptual parsing.

    Outputs raw logits for scene classification, object parsing, part parsing,
    and material parsing. Post-processing (softmax, interpolation, argmax)
    should be applied externally.
    """

    def __init__(self, nr_classes, fc_dim=2048, fpn_dim=512):
        super().__init__()
        self.encoder = ResNetEncoder(layers=(3, 4, 6, 3))
        self.decoder = UPerNetDecoder(
            nr_classes=nr_classes, fc_dim=fc_dim, fpn_dim=fpn_dim)

    def forward(self, img):
        """Forward pass returning raw logits for all four tasks.

        Args:
            img: input image tensor [B, 3, H, W] (BGR, mean-subtracted).

        Returns:
            Tuple of (scene_logits, object_logits, part_logits, material_logits).
        """
        conv_out = self.encoder(img)
        return self.decoder(conv_out)


# Class counts from Broden+ dataset (derived from pretrained weight shapes)
NR_CLASSES = {
    'scene': 365,
    'object': 336,
    'part': 427,
    'material': 26,
}

# BGR mean for preprocessing (matching original UPerNet training)
IMG_MEAN = [102.9801, 115.9465, 122.7717]
IMG_STD = [1.0, 1.0, 1.0]


def build_model(weights_encoder_path, weights_decoder_path):
    """Build UPerNet model and load pretrained weights.

    Args:
        weights_encoder_path: path to encoder .pth file.
        weights_decoder_path: path to decoder .pth file.

    Returns:
        Loaded and eval-mode UPerNetModel.
    """
    model = UPerNetModel(nr_classes=NR_CLASSES, fc_dim=2048, fpn_dim=512)

    enc_state = torch.load(weights_encoder_path, map_location='cpu')
    dec_state = torch.load(weights_decoder_path, map_location='cpu')

    # strict=False to ignore SyncBN extra keys (_tmp_running_mean, etc.)
    model.encoder.load_state_dict(enc_state, strict=False)
    model.decoder.load_state_dict(dec_state, strict=False)

    model.eval()
    return model
