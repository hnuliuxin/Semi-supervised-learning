import os
import math
import torch
import torch.nn as nn
from semilearn.core import AlgorithmBase
from semilearn.algorithms.distiller.distiller_base import DistillerBase
from semilearn.core.utils import EMA, ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument
from semilearn.datasets import DistributedSampler

"""Knowledge distillation via softmax regression representation learning
code:https://github.com/jingyang2017/KD_SRRL
"""

class transfer_conv(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.Connectors = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_feature),
            nn.ReLU())
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, student):
        student = self.Connectors(student)
        return student


class statm_loss(nn.Module):
    def __init__(self):
        super(statm_loss, self).__init__()

    def forward(self,x, y):
        x = x.view(x.size(0),x.size(1),-1)
        y = y.view(y.size(0),y.size(1),-1)
        x_mean = x.mean(dim=2)#BC
        y_mean = y.mean(dim=2)
        mean_gap = (x_mean-y_mean).pow(2).mean(1)
        return mean_gap.mean()




@ALGORITHMS.register('srd')
class SRD(DistillerBase):
    """
        SRD algorithm (https://arxiv.org/pdf/2205.06701.pdf).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for distillation smoothing
            - gamma (`float`, default=1):
                weight for classification
            - alpha (`float`, default=None):
                weight balance for KD
            - beta (`float`, default=None):
                weight balance for other losses
        """
    def __init__(self, args, net_builder, teacher_net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, teacher_net_builder, tb_log, logger) 
        # SRD specified arguments
        self.init(T=args.T, gamma=args.gamma, alpha=args.alpha, beta=args.beta, criterion_kd_weight=args.criterion_kd_weight)

    def init(self, T, gamma=1, alpha=1, beta=1, criterion_kd_weight=1):
        self.T = T
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.criterion_kd_weight = criterion_kd_weight
        #TODO SRD的特征好像是取得倒数第二层，注意再看下
        self.connector = 
    
    