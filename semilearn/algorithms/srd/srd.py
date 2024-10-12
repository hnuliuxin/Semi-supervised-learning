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

"""Knowledge distillation via softmax regression representation learning
code:https://github.com/jingyang2017/KD_SRRL
"""

# class transfer_conv(nn.Module):
#     def __init__(self, in_feature, out_feature):
#         super().__init__()
#         self.in_feature = in_feature
#         self.out_feature = out_feature
#         self.Connectors = nn.Sequential(
#             nn.Conv2d(in_feature, out_feature, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(out_feature),
#             nn.ReLU())
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#     def forward(self, student):
#         student = self.Connectors(student)
#         return student

class SRDNet(nn.Module):
    def __init__(self, backbone, num_classes, feat_s_shape, feat_t_shape):
        super(SRDNet, self).__init__()
        self.backbone = backbone
        self.feat_s_shape = feat_s_shape
        self.feat_t_shape = feat_t_shape
        # self.connector = transfer_conv(feat_s_shape, feat_t_shape)
        print("feat_s_shape: ", feat_s_shape)
        print("feat_t_shape: ", feat_t_shape)
        # 改变通道数
        self.connector = nn.Sequential(
            nn.Conv2d(feat_s_shape[1], feat_t_shape[1], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feat_t_shape[1]),
            nn.ReLU())
        nn.init.xavier_normal_(self.connector[0].weight)
        nn.init.constant_(self.connector[1].weight, 1)
        nn.init.constant_(self.connector[1].bias, 0)

    def forward(self, x, **kwargs):
        feats_x = self.backbone(x, only_feat=True)[-2]
        logits_x = self.backbone(x, feat_s=feats_x)
        feats_x = self.connector(feats_x)
        # 不同模型输出特征图维度可能不一致，需要插值
        if(self.feat_s_shape[2] != self.feat_t_shape[2]):
            feats_x = F.interpolate(feats_x, size=(self.feat_t_shape[2], self.feat_t_shape[3]), mode='bilinear', align_corners=True)
        return_dict = {
            'logits': logits_x,
            'feat': feats_x
        }
        return return_dict

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher

def statm_loss(x, y):
    """
    计算学生模型和教师模型输出特征的统计差异，作为知识蒸馏的损失函数。

    Args:
        x: 学生模型的输出特征，形状为 (batch_size, feature_num, -1)。
        y: 教师模型的输出特征，形状为 (batch_size, feature_num, -1)。

    Returns:
        损失值，一个标量。
    """
    x = x.view(x.size(0), x.size(1), -1)
    y = y.view(y.size(0), y.size(1), -1)
    x_mean = x.mean(dim=2)  # BC
    y_mean = y.mean(dim=2)
    mean_gap = (x_mean - y_mean).pow(2).mean(1)
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
        super().__init__(args, net_builder, tb_log, logger, teacher_net_builder) 
        # SRD specified arguments
        self.init(T=args.T, gamma=args.gamma, alpha=args.alpha, beta=args.beta, criterion_kd_weight=args.criterion_kd_weight)
        #查看老师模型所有参数
        # for name, param in self.teacher_model.named_parameters():
        #     print(name, param.size())

    def init(self, T, gamma=1, alpha=1, beta=1, criterion_kd_weight=10):
        # self.teacher_model.requires_grad_(False)
        self.T = T
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.criterion_kd_weight = criterion_kd_weight
        with torch.no_grad():
            self.model.eval()
            self.teacher_model.eval()
            data = torch.randn(2, 3, self.args.img_size, self.args.img_size)
            self.feat_s = self.model(data, only_feat=True)[-2]
            self.feat_t = self.teacher_model(data, only_feat=True)[-2]
            self.model = SRDNet(self.model, num_classes=self.num_classes, feat_s_shape=self.feat_s.shape, feat_t_shape=self.feat_t.shape)
            self.ema_model = SRDNet(self.ema_model, num_classes=self.num_classes, feat_s_shape=self.feat_s.shape, feat_t_shape=self.feat_t.shape)
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
            
            with torch.no_grad():
                outs_x_t = self.teacher_model(input)
                logits_x_t = outs_x_t['logits']
                feats_x_t = outs_x_t['feat'][-2]

                logit_tc = self.teacher_model(x=None, feat_s=feats_x)

            #有标签的损失
            cls_loss = self.ce_loss(logits_x[:batch_size], y_lb, reduction='mean')

            # 因为有include_lb_to_ulb,重复计算了有标签的蒸馏损失
            # div_loss = KD_Loss(logits_x, logits_x_t, self.T)
            # kd_loss = 0
            kd_loss = self.criterion_kd_weight * statm_loss(feats_x, feats_x_t) + F.mse_loss(logit_tc, logits_x_t)
            
            #求和
            total_loss =self.gamma * cls_loss +  self.beta * kd_loss

        out_dict = self.process_out_dict(loss = total_loss)
        log_dict = self.process_log_dict(cls_loss=cls_loss.item(), 
                                        #  div_loss=div_loss.item(), 
                                         kd_loss=kd_loss.item()
                                         )
    
        return out_dict, log_dict


    def get_save_dict(self):
        save_dict = super().get_save_dict() 
        save_dict["T"] = self.T
        save_dict["gamma"] = self.gamma
        save_dict["alpha"] = self.alpha
        save_dict["beta"] = self.beta
        save_dict["criterion_kd_weight"] = self.criterion_kd_weight
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.T = checkpoint["T"]
        self.gamma = checkpoint["gamma"]
        self.alpha = checkpoint["alpha"]
        self.beta = checkpoint["beta"]
        self.criterion_kd_weight = checkpoint["criterion_kd_weight"]
        return checkpoint
    
    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--T', default=1, type=float, help='Temperature for distillation smoothing'),
            SSL_Argument('--gamma', default=1, type=float, help='weight for classification'),
            SSL_Argument('--alpha', default=1, type=float, help='weight balance for KD'),
            SSL_Argument('--beta', default=1, type=float, help='weight balance for other losses'),
            SSL_Argument('--criterion_kd_weight', default=10, type=float, help='weight balance for other losses'),
        ]


