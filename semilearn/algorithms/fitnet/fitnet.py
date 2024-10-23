import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS, send_model_cuda
from semilearn.algorithms.utils import SSL_Argument


class ConvReg(nn.Module):
    """Convolutional regression"""

    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1 + s_H - t_H, 1 + s_W - t_W))
        else:
            raise NotImplemented("student size {}, teacher size {}".format(s_H, t_H))
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)


@ALGORITHMS.register('fitnet')
class FitNet(AlgorithmBase):
    """
        FitNet algorithm

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - gamma (`float`, default=1):
                weight for classification
            - alpha (`float`, default=None):
                weight balance for KD
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None, teacher_net_builder=None):
        super().__init__(args, net_builder, tb_log, logger, teacher_net_builder)
        self.gamma = args.gamma
        self.alpha = args.alpha
        with torch.no_grad():
            self.model.eval()
            self.teacher_model.eval()
            data = torch.randn(1, 3, self.args.img_size, self.args.img_size)
            self.feat_s = self.model(data, only_feat=True)[-1]
            self.feat_t = self.teacher_model(data, only_feat=True)[-1]
            # print("student feature size: ", self.feat_s.shape)
            # print("teacher feature size: ", self.feat_t.shape)
            self.conv_reg = ConvReg(self.feat_s.shape, self.feat_t.shape)
            send_model_cuda(self.args, self.conv_reg)
    
    def train_step(self, x_lb, y_lb, x_ulb_w = None):
        with self.amp_cm():
            self.model.train()
            self.teacher_model.eval()

            input = torch.cat([x_lb, x_ulb_w], dim=0) if x_ulb_w is not None else x_lb
            batch_size = x_lb.shape[0]

            outs_x = self.model(input)
            logits_x = outs_x['logits']
            feats_x = outs_x['feat'][-1]
            
            with torch.no_grad():
                outs_x_teacher = self.teacher_model(input)
                feats_x_teacher = outs_x_teacher['feat'][-1]

            feats_x = self.conv_reg(feats_x)
            # print(feats_x.shape, feats_x_teacher.shape)

            sup_loss = self.ce_loss(logits_x[:batch_size], y_lb, reduction='mean')
            kd_loss = self.consistency_loss(feats_x[batch_size:], feats_x_teacher[batch_size:], 'mse')

            total_loss = self.gamma * sup_loss + self.alpha * kd_loss
        
        out_dict = self.process_out_dict(loss=total_loss, feat=feats_x)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         kd_loss=kd_loss.item(), 
                                         total_loss=total_loss.item())
        return out_dict, log_dict
        
            

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        save_dict["gamma"] = self.gamma
        save_dict["alpha"] = self.alpha
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.gamma = checkpoint["gamma"]
        self.alpha = checkpoint["alpha"]
        return checkpoint
    
    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--gamma', default=1, type=float, help='weight for classification'),
            SSL_Argument('--alpha', default=1, type=float, help='weight balance for KD'),
        ]