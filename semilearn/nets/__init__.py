# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .resnet import resnet8, resnet8x4, resnet14, resnet20, resnet32, resnet44, resnet32x4, resnet56, resnet110, resnet18, resnet50
from .wrn import wrn_28_2, wrn_28_8, wrn_var_37_2, wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2, wide_resnet50_2
from .vit import vit_base_patch16_224, vit_small_patch16_224, vit_small_patch2_32, vit_tiny_patch2_32, vit_base_patch16_96
from .bert import bert_base_cased, bert_base_uncased
from .wave2vecv2 import wave2vecv2_base
from .hubert import hubert_base
from .mobilenet import mobilenet
from .shufflenet import shuffleV1, shuffleV2
from .vgg import vgg8, vgg11, vgg13, vgg16, vgg19