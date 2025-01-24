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
    def __init__(self, backbone, num_classes, feat_s_shape, feat_t_shape):
        super(myNet, self).__init__()
        self.backbone = backbone
        self.feat_s_shape = feat_s_shape
        self.feat_t_shape = feat_t_shape
        # 改变通道数
        self.connector = nn.Sequential(
            nn.Conv2d(feat_s_shape[1], feat_t_shape[1], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feat_t_shape[1]),
            nn.ReLU())
        self.var_estimator = nn.Sequential(
            nn.Linear(feat_t_shape[1], feat_t_shape[1]),
            nn.BatchNorm1d(feat_t_shape[1])
        )
        nn.init.xavier_normal_(self.connector[0].weight)
        nn.init.constant_(self.connector[1].weight, 1)
        nn.init.constant_(self.connector[1].bias, 0)
        nn.init.xavier_normal_(self.var_estimator[0].weight)
        nn.init.constant_(self.var_estimator[1].weight, 1)
        nn.init.constant_(self.var_estimator[1].bias, 0)    

    def forward(self, x, **kwargs):
        feats = self.backbone(x, only_feat=True)
        feats_x2 = feats[-2]
        feats_x = feats[-1]
        logits_x = self.backbone(feats_x, only_fc=True)
        if self.feat_s_shape[1] != self.feat_t_shape[1]:
            feats_x2 = self.connector(feats_x2)

        log_variances = self.var_estimator(feats_x)
        return_dict = {
            'logits': logits_x,
            'feat': feats_x,
            'feat2': feats_x2,
            'log_var': log_variances
        }
        return return_dict

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher

def statm_loss(x, y):
    x = x.view(x.size(0), x.size(1), -1)
    y = y.view(y.size(0), y.size(1), -1)
    x_mean = x.mean(dim=2)  # BC
    y_mean = y.mean(dim=2)
    mean_gap = (x_mean - y_mean).pow(2).mean(1)
    return mean_gap.mean()

def KD_Loss(logits_student, logits_teacher, temperature):# 温度修改为向量(N)
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd
@ALGORITHMS.register('my')
class MyAlgorithm(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None, teacher_net_builder=None):
        super().__init__(args, net_builder, tb_log, logger, teacher_net_builder)
        self.init(T=args.T)
        #查看老师模型所有参数
        # for name, param in self.teacher_model.named_parameters():
        #     print(name, param.size())

    def init(self, T):
        # self.teacher_model.requires_grad_(False)
        self.T = T
        weights = torch.nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0, 10]), requires_grad=True).to(self.args.gpu)
        self.gamma = weights[0]
        self.kl = weights[1]
        # self.pad = weights[2]
        # self.srd = weights[3]
        self.unsup_weight = weights[4]
        self.eps = 1e-6
        with torch.no_grad():
            self.model.eval()
            self.teacher_model.eval()
            data = torch.randn(2, 3, self.args.img_size, self.args.img_size)
            self.feat_s = self.model(data, only_feat=True)[-2]
            self.feat_t = self.teacher_model(data, only_feat=True)[-2]
            self.model = myNet(self.model, num_classes=self.num_classes, feat_s_shape=self.feat_s.shape, feat_t_shape=self.feat_t.shape)
            self.ema_model = myNet(self.ema_model, num_classes=self.num_classes, feat_s_shape=self.feat_s.shape, feat_t_shape=self.feat_t.shape)
            self.ema_model.load_state_dict(self.model.state_dict())
        
    def train_step(self, x_lb, y_lb, x_ulb_w = None):
        with self.amp_cm():
            self.model.train()
            self.teacher_model.eval()

            # concat labeled and unlabeled data
            input = torch.cat([x_lb, x_ulb_w], dim=0) if x_ulb_w is not None else x_lb
            batch_size = x_lb.shape[0]

            # SrdNet特征取倒数第二层
            outs_x = self.model(input)
            logits_x = outs_x['logits']
            feats_x = outs_x['feat']
            feats_x2 = outs_x['feat2']
            log_variances = outs_x['log_var']

            probs = F.softmax(logits_x, dim=1)
            scores = -torch.sum(probs * torch.log(probs), dim=1)
            index = torch.argsort(scores, descending=True)
            
            with torch.no_grad():
                outs_x_t = self.teacher_model(input)
                logits_x_t = outs_x_t['logits']
                feats_x_t2 = outs_x_t['feat'][-2]
                feats_x_t = outs_x_t['feat'][-1]

            #有标签的损失
            sup_loss = self.ce_loss(logits_x[:batch_size], y_lb, reduction='mean')
            unsup_loss = F.mse_loss(logits_x[batch_size:], logits_x_t[batch_size:], reduction='mean') 


            # 因为有include_lb_to_ulb,重复计算了有标签的蒸馏损失
            kl_loss = KD_Loss(logits_x, logits_x_t, self.T)

            # srd_loss = statm_loss(feats_x2[index], feats_x_t2[index]) 
            # srd_loss = F.mse_loss(logits_x, logits_x_t)
            # srd_loss = 0

            # pad_loss = torch.mean(
            #         (feats_x[index] - feats_x_t[index]) ** 2 / (self.eps + torch.exp(log_variances[index]))
            #         + log_variances[index], dim=1).mean()  
            # pad_loss = 0
            
            #求和
            total_loss = (1 / self.gamma) * sup_loss + (1 / self.unsup_weight) * unsup_loss + \
                  (1 / self.kl) * kl_loss + \
                    2 * torch.log(self.gamma * self.unsup_weight * self.kl )

        out_dict = self.process_out_dict(loss = total_loss)
        log_dict = self.process_log_dict(sup_loss=(1 / self.gamma) * sup_loss.item(), 
                                         unsup_loss=(1 / self.unsup_weight) * unsup_loss.item(),
                                         kl_loss=(1 / self.kl) * kl_loss.item(), 
                                         srd_loss=(1 / self.srd) * srd_loss.item(),
                                         pad_loss=(1 / self.pad) * pad_loss.item(),
                                         epoch=self.epoch
                                            )
    
        return out_dict, log_dict


    def get_save_dict(self):
        save_dict = super().get_save_dict() 
        save_dict["T"] = self.T
        save_dict["gamma"] = self.gamma
        save_dict["kl"] = self.kl
        save_dict["pad"] = self.pad
        save_dict["srd"] = self.srd
        save_dict["unsup_weight"] = self.unsup_weight
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.T = checkpoint["T"]
        self.gamma = checkpoint["gamma"]
        self.kl = checkpoint["kl"]
        self.pad = checkpoint["pad"]
        self.srd = checkpoint["srd"]
        self.unsup_weight = checkpoint["unsup_weight"]
        return checkpoint
    
    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--T', default=1, type=float, help='Temperature for distillation smoothing'),
            # SSL_Argument('--gamma', default=1, type=float, help='weight for classification'),
            # SSL_Argument('--kl', default=1, type=float, help='weight for KD'),
            # SSL_Argument('--pad', default=1, type=float, help='weight for PAD'),
            # SSL_Argument('--srd', default=1, type=float, help='weight for SRD'),
        ]


        


        