import torch
import numbers
import numpy as np
import functools
import h5py
import math
from torchvision import models
import pretrainedmodels
import torch.nn.functional as F
import types
import torch
from efficientnet_pytorch import EfficientNet
from collections import OrderedDict
import torch.nn as nn


def Dense121(config):
    return models.densenet121(pretrained=True)

def Dense161(config):
    return models.densenet161(pretrained=True)

def Dense169(config):
    return models.densenet169(pretrained=True)

def Dense201(config):
    return models.densenet201(pretrained=True)

def Resnet50(config):
    return pretrainedmodels.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')

# ✅ FIX: 用 pretrainedmodels 的 resnet101（带 last_linear，兼容你的 train.py/modify_meta）
def Resnet101(config):
    return pretrainedmodels.__dict__['resnet101'](num_classes=1000, pretrained='imagenet')

# ✅ FIX: 用 pretrainedmodels 的 inceptionv3（同样更兼容 last_linear 的写法）
def InceptionV3(config):
    # 兼容不同 pretrainedmodels 版本的 key
    for k in ('inceptionv3', 'inception_v3'):
        if k in pretrainedmodels.__dict__:
            return pretrainedmodels.__dict__[k](num_classes=1000, pretrained='imagenet')
    raise KeyError("pretrainedmodels has no inceptionv3 / inception_v3")

def se_resnext50(config):
    return pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet')

def se_resnext101(config):
    return pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained='imagenet')

def se_resnet50(config):
    return pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained='imagenet')

def se_resnet101(config):
    return pretrainedmodels.__dict__['se_resnet101'](num_classes=1000, pretrained='imagenet')

def se_resnet152(config):
    return pretrainedmodels.__dict__['se_resnet152'](num_classes=1000, pretrained='imagenet')

def resnext101(config):
    return pretrainedmodels.__dict__['resnext101_32x4d'](num_classes=1000, pretrained='imagenet')

def resnext101_64(config):
    return pretrainedmodels.__dict__['resnext101_64x4d'](num_classes=1000, pretrained='imagenet')

def senet154(config):
    return pretrainedmodels.__dict__['senet154'](num_classes=1000, pretrained='imagenet')

def polynet(config):
    return pretrainedmodels.__dict__['polynet'](num_classes=1000, pretrained='imagenet')

def dpn92(config):
    return pretrainedmodels.__dict__['dpn92'](num_classes=1000, pretrained='imagenet+5k')

def dpn68b(config):
    return pretrainedmodels.__dict__['dpn68b'](num_classes=1000, pretrained='imagenet+5k')

def nasnetamobile(config):
    return pretrainedmodels.__dict__['nasnetamobile'](num_classes=1000, pretrained='imagenet')

def resnext101_32_8_wsl(config):
    return torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')

def resnext101_32_16_wsl(config):
    return torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')

def resnext101_32_32_wsl(config):
    return torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x32d_wsl')

def resnext101_32_48_wsl(config):
    return torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl')

def efficientnet_b0(config):
    return EfficientNet.from_pretrained('efficientnet-b0',num_classes=config['numClasses'])

def efficientnet_b1(config):
    return EfficientNet.from_pretrained('efficientnet-b1',num_classes=config['numClasses'])

def efficientnet_b2(config):
    return EfficientNet.from_pretrained('efficientnet-b2',num_classes=config['numClasses'])

def efficientnet_b3(config):
    return EfficientNet.from_pretrained('efficientnet-b3',num_classes=config['numClasses'])

def efficientnet_b4(config):
    return EfficientNet.from_pretrained('efficientnet-b4',num_classes=config['numClasses'])

def efficientnet_b5(config):
    return EfficientNet.from_pretrained('efficientnet-b5',num_classes=config['numClasses'])

def efficientnet_b6(config):
    return EfficientNet.from_pretrained('efficientnet-b6',num_classes=config['numClasses'])

def efficientnet_b7(config):
    return EfficientNet.from_pretrained('efficientnet-b7',num_classes=config['numClasses'])


def modify_meta(mdlParams,model):
    # Define FC layers
    if len(mdlParams['fc_layers_before']) > 1:
        model.meta_before = nn.Sequential(nn.Linear(mdlParams['meta_array'].shape[1],mdlParams['fc_layers_before'][0]),
                                    nn.BatchNorm1d(mdlParams['fc_layers_before'][0]),
                                    nn.ReLU(),
                                    nn.Dropout(p=mdlParams['dropout_meta']),
                                    nn.Linear(mdlParams['fc_layers_before'][0],mdlParams['fc_layers_before'][1]),
                                    nn.BatchNorm1d(mdlParams['fc_layers_before'][1]),
                                    nn.ReLU(),
                                    nn.Dropout(p=mdlParams['dropout_meta']))
    else:
        model.meta_before = nn.Sequential(nn.Linear(mdlParams['meta_array'].shape[1],mdlParams['fc_layers_before'][0]),
                                    nn.BatchNorm1d(mdlParams['fc_layers_before'][0]),
                                    nn.ReLU(),
                                    nn.Dropout(p=mdlParams['dropout_meta']))

    # ---- get cnn feature dim (used by both branches) ----
    if 'efficient' in mdlParams['model_type']:
        num_cnn_features = model._fc.in_features
    elif 'wsl' in mdlParams['model_type']:
        num_cnn_features = model.fc.in_features
    else:
        num_cnn_features = model.last_linear.in_features

    # Define fc layers after
    if len(mdlParams['fc_layers_after']) > 0:
        model.meta_after = nn.Sequential(nn.Linear(mdlParams['fc_layers_before'][-1]+num_cnn_features,mdlParams['fc_layers_after'][0]),
                                    nn.BatchNorm1d(mdlParams['fc_layers_after'][0]),
                                    nn.ReLU())
        classifier_in_features = mdlParams['fc_layers_after'][0]
    else:
        model.meta_after = None
        classifier_in_features = mdlParams['fc_layers_before'][-1] + num_cnn_features

    # Modify classifier
    if 'efficient' in mdlParams['model_type']:
        model._fc = nn.Linear(classifier_in_features, mdlParams['numClasses'])
    elif 'wsl' in mdlParams['model_type']:
        model.fc = nn.Linear(classifier_in_features, mdlParams['numClasses'])
    else:
        model.last_linear = nn.Linear(classifier_in_features, mdlParams['numClasses'])

    # Modify forward pass
    def new_forward(self, inputs):
        x, meta_data = inputs
        # Normal CNN features
        if 'efficient' in mdlParams['model_type']:
            # Convolution layers
            cnn_features = self.extract_features(x)
            # Pooling and final linear layer
            cnn_features = F.adaptive_avg_pool2d(cnn_features, 1).squeeze(-1).squeeze(-1)
            if self._dropout:
                # robust: self._dropout may be float OR nn.Dropout (depends on EfficientNet version)
                p_drop = float(self._dropout.p) if hasattr(self._dropout, 'p') else float(self._dropout)
                cnn_features = F.dropout(cnn_features, p=p_drop, training=self.training)
        elif 'wsl' in mdlParams['model_type']:
            cnn_features = self.conv1(x)
            cnn_features = self.bn1(cnn_features)
            cnn_features = self.relu(cnn_features)
            cnn_features = self.maxpool(cnn_features)

            cnn_features = self.layer1(cnn_features)
            cnn_features = self.layer2(cnn_features)
            cnn_features = self.layer3(cnn_features)
            cnn_features = self.layer4(cnn_features)

            cnn_features = self.avgpool(cnn_features)
            cnn_features = torch.flatten(cnn_features, 1)

        # ✅ FIX: 支持 resnet-style（避免 meta+resnet 直接 AttributeError）
        elif hasattr(self, "conv1") and hasattr(self, "layer1") and hasattr(self, "avgpool"):
            cnn_features = self.conv1(x)
            cnn_features = self.bn1(cnn_features)
            cnn_features = self.relu(cnn_features)
            cnn_features = self.maxpool(cnn_features)

            cnn_features = self.layer1(cnn_features)
            cnn_features = self.layer2(cnn_features)
            cnn_features = self.layer3(cnn_features)
            cnn_features = self.layer4(cnn_features)

            cnn_features = self.avgpool(cnn_features)
            cnn_features = torch.flatten(cnn_features, 1)

        else:
            cnn_features = self.layer0(x)
            cnn_features = self.layer1(cnn_features)
            cnn_features = self.layer2(cnn_features)
            cnn_features = self.layer3(cnn_features)
            cnn_features = self.layer4(cnn_features)
            cnn_features = self.avg_pool(cnn_features)
            if self.dropout is not None:
                cnn_features = self.dropout(cnn_features)
            cnn_features = cnn_features.view(cnn_features.size(0), -1)

        # Meta part
        meta_features = self.meta_before(meta_data)

        # Cat
        features = torch.cat((cnn_features,meta_features),dim=1)
        if self.meta_after is not None:
            features = self.meta_after(features)

        # Classifier
        if 'efficient' in mdlParams['model_type']:
            output = self._fc(features)
        elif 'wsl' in mdlParams['model_type']:
            output = self.fc(features)
        else:
            output = self.last_linear(features)
        return output

    model.forward  = types.MethodType(new_forward, model)
    return model


model_map = OrderedDict([('Dense121',  Dense121),
                        ('Dense161' , Dense161),
                        ('Dense169' , Dense169),
                        ('Dense201' , Dense201),
                        ('Resnet50' , Resnet50),
                        ('Resnet101' , Resnet101),
                        ('InceptionV3', InceptionV3),
                        ('se_resnext50', se_resnext50),
                        ('se_resnext101', se_resnext101),
                        ('se_resnet50', se_resnet50),
                        ('se_resnet101', se_resnet101),
                        ('se_resnet152', se_resnet152),
                        ('resnext101', resnext101),
                        ('resnext101_64', resnext101_64),
                        ('senet154', senet154),
                        ('polynet', polynet),
                        ('dpn92', dpn92),
                        ('dpn68b', dpn68b),
                        ('nasnetamobile', nasnetamobile),
                        ('resnext101_32_8_wsl', resnext101_32_8_wsl),
                        ('resnext101_32_16_wsl', resnext101_32_16_wsl),
                        ('resnext101_32_32_wsl', resnext101_32_32_wsl),
                        ('resnext101_32_48_wsl', resnext101_32_48_wsl),
                        ('efficientnet-b0', efficientnet_b0),
                        ('efficientnet-b1', efficientnet_b1),
                        ('efficientnet-b2', efficientnet_b2),
                        ('efficientnet-b3', efficientnet_b3),
                        ('efficientnet-b4', efficientnet_b4),
                        ('efficientnet-b5', efficientnet_b5),
                        ('efficientnet-b6', efficientnet_b6),
                        ('efficientnet-b7', efficientnet_b7),
                    ])


def getModel(config):
  """Returns a function for a model
  Args:
    config: dictionary, contains configuration
  Returns:
    model: A class that builds the desired model
  Raises:
    ValueError: If model name is not recognized.
  """
  if config['model_type'] in model_map:
    func = model_map[config['model_type'] ]
    @functools.wraps(func)
    def model():
        return func(config)
  else:
      raise ValueError(f"Unknown model_type: {config.get('model_type')}")
  return model
