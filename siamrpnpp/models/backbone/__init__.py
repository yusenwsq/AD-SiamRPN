# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from siamrpnpp.models.backbone.alexnet import alexnetlegacy, alexnet, alexnetlegacy2
from siamrpnpp.models.backbone.mobile_v2 import mobilenetv2
from siamrpnpp.models.backbone.resnet_atrous import resnet18, resnet34, resnet50

BACKBONES = {
              'alexnetlegacy': alexnetlegacy,
              'alexnetlegacy2': alexnetlegacy2,
              'mobilenetv2': mobilenetv2,
              'resnet18': resnet18,
              'resnet34': resnet34,
              'resnet50': resnet50,
              'alexnet': alexnet,
            }


def get_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs)
