import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from semilearn.core import AlgorithmBase
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

def KD_Loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


@ALGORITHMS.register('srd')
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
    def __init__(self, args, net_builder, tb_log=None, logger=None, teacher_net_builder=None):
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
        data = torch.randn(2, 3, 32, 32)
        feat_s = self.model(data, only_feat=True)
        feat_t = self.teacher_model(data, only_feat=True)
        self.connector = transfer_conv(feat_s.shape[1], feat_t.shape[1])
        self.statm_loss = statm_loss()
    
    def train_step(self, x_lb, y_lb, x_ulb_w):
        with self.amp_cm():

            outs_x_lb = self.model(x_lb)
            outs_x_lb_t = self.teacher_model(x_lb)
            logits_x_lb = outs_x_lb['logits']
            feats_x_lb = outs_x_lb['feats']
            logits_x_lb_t = outs_x_lb_t['logits']
            feats_x_lb_t = outs_x_lb_t['feats']

            #特征维度转换
            feats_x_lb = self.connector(feats_x_lb)
            logit_tc = self.teacher_model(x=None, feat_s=feats_x_lb)

            #有标签的损失
            cls_loss = nn.CrossEntropyLoss(logits_x_lb, y_lb)
            div_loss = KD_Loss(logits_x_lb, logits_x_lb_t, self.T)
            kd_loss = self.criterion_kd_weight * self.statm_loss(feats_x_lb, feats_x_lb_t) + F.mse_loss(logit_tc, logits_x_lb_t)

            outs_x_unlb = self.model(x_ulb_w)
            outs_x_unlb_t = self.teacher_model(x_ulb_w)
            feats_x_unlb = outs_x_unlb['feats']
            logits_x_unlb_t = outs_x_unlb_t['logits']
            feats_x_unlb_t = outs_x_unlb_t['feats']


            feats_x_unlb = self.connector(feats_x_unlb)
            logit_tc_unlb = self.teacher_model(x=None, feat_s=feats_x_unlb)
            kd_loss_unlb = self.criterion_kd_weight * self.statm_loss(feats_x_unlb, feats_x_unlb_t) + F.mse_loss(logit_tc_unlb, logits_x_unlb_t)

            #求和
            sup_loss = self.gamma * cls_loss + self.alpha * div_loss + self.beta * kd_loss
            unsup_loss = self.beta * kd_loss_unlb
            total_loss = sup_loss + unsup_loss * self.lambda_u

        out_dict = self.process_out_dict(loss = total_loss)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item())
    
        return out_dict, log_dict


            

