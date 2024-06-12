import os

import torch
from .utils import DashThresholdingHook
from semilearn.core import AlgorithmBase
from semilearn.core.utils import EMA, ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument
from semilearn.datasets import DistributedSampler

class SRD(AlgorithmBase):
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
    