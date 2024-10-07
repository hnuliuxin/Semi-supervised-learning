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


class myNet(nn.Module):
    def __init__(self, backbone, num_classes, feat_s_shape):
        super(myNet, self).__init__()
        self.backbone = backbone
        # print(f"feat_s_shape: {feat_s_shape}")
        # 冻结backbone参数
        for name, param in self.backbone.named_parameters():
            # print(f"Parameter Name: {name}")
            if name != 'fc.weight' and name != 'fc.bias':
                param.requires_grad = False
        self.backbone.fc = nn.Linear(feat_s_shape[1], 300)

        self.fc = nn.Linear(300, 200)

        torch.nn.init.xavier_uniform_(self.backbone.fc.weight)
        nn.init.constant_(self.backbone.fc.bias, 0)

    def forward(self, x, **kwargs):
        result_dict = self.backbone(x)
        result_dict['logits'] = self.fc(result_dict['logits'])
        return result_dict

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher

@ALGORITHMS.register('my')
class MyAlgorithm(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        with torch.no_grad():
            self.model.eval()
            data = torch.randn(2, 3, self.args.img_size, self.args.img_size)
            self.feat_s = self.model(data, only_feat=True)[-1]
            self.model = myNet(self.model, args.num_classes, feat_s_shape=self.feat_s.shape)
            self.ema_model = myNet(self.ema_model, args.num_classes, feat_s_shape=self.feat_s.shape)
            self.ema_model.load_state_dict(self.model.state_dict())
        
    def train_step(self, x_lb, y_lb):
        with self.amp_cm():
            logits_x = self.model(x_lb)['logits']
            sup_loss = self.ce_loss(logits_x, y_lb, reduction='mean')
        
        out_dict = self.process_out_dict(loss=sup_loss)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item())
        return out_dict, log_dict


        


        