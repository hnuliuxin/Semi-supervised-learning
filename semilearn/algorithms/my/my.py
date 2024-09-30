import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from semilearn.core import AlgorithmBase
from semilearn.core.utils import EMA, ALGORITHMS, send_model_cuda
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument
from semilearn.datasets import DistributedSampler


@ALGORITHMS.register('my')
class MyAlgorithm(AlgorithmBase):
    def __init__(self, args, backbone, num_classes, feat_s_shape, feat_t_shape):
        super(MyAlgorithm, self).__init__(args)
