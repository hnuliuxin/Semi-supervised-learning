import torch
import torch.nn as nn
import torch.nn.functional as F
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument

class PADNet(nn.Module):
    def __init__(self, backbone, feat_s_shape, feat_t_shape):
        super().__init__()
        self.backbone = backbone
        self.feat_s_shape = feat_s_shape
        self.feat_t_shape = feat_t_shape
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
        for param in self.var_estimator.parameters():
            param.requires_grad = False

    def forward(self, x, **kwargs):
        feats_x = self.backbone(x, only_feat=True)[-1]
        logits_x = self.backbone(x, feat_s=feats_x)
        feats_x = self.connector(feats_x)

        log_variances = self.var_estimator(feats_x)

        return_dict = {
            'logits': logits_x,
            'feat': feats_x,
            'log_var': log_variances
        }
        return return_dict
    
    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher

@ALGORITHMS.register('pad')
class PAD(AlgorithmBase):
    """
        PAD algorithm

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
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None, teacher_net_builder=None, **kwargs):
        super().__init__(args, net_builder, tb_log, logger, teacher_net_builder, **kwargs)
        self.init(T=args.T, gamma=args.gamma, alpha=args.alpha)

    def init(self, T, gamma=1, alpha=1):
        self.T = T
        self.gamma = gamma
        self.alpha = alpha
        with torch.no_grad():
            self.model.eval()
            self.teacher_model.eval()
            data = torch.randn(2, 3, self.args.img_size, self.args.img_size)
            self.feat_s = self.model(data, only_feat=True)[-1]
            self.feat_t = self.teacher_model(data, only_feat=True)[-1]
            self.model = PADNet(self.model, feat_s_shape=self.feat_s.shape, feat_t_shape=self.feat_t.shape)
            self.ema_model = PADNet(self.ema_model, feat_s_shape=self.feat_s.shape, feat_t_shape=self.feat_t.shape)
            self.ema_model.load_state_dict(self.model.state_dict())

    def train_step(self, x_lb, y_lb, x_ulb_w = None):
        with self.amp_cm():
            self.model.train()
            self.teacher_model.eval()

            input = torch.cat([x_lb, x_ulb_w], dim=0) if x_ulb_w is not None else x_lb
            batch_size = x_lb.shape[0]

            outs_x_lb = self.model(input)
            logits_x_lb = outs_x_lb['logits']
            feats_x_lb = outs_x_lb['feat']
            log_variances = outs_x_lb['log_var']

            self.teacher_model.eval()
            with torch.no_grad():
                outs_x_teacher = self.teacher_model(input)
                logits_x_teacher = outs_x_teacher['logits']
                feats_x_teacher = outs_x_teacher['feat'][-1]

            sup_loss = self.ce_loss(logits_x_lb[:batch_size], logits_x_teacher[:batch_size])

            if self.it / 2048 < 130:
                kd_loss = self.consistency_loss(feats_x_lb, feats_x_teacher, name = "mse")
            else:
                kd_loss = torch.mean(
                    (feats_x_lb - feats_x_teacher) ** 2 / (self.eps + torch.exp(log_variances))
                    + log_variances, dim=1)

            total_loss = sup_loss * self.gamma + kd_loss.mean() * self.alpha

        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         kd_loss=kd_loss.mean().item())
        return out_dict, log_dict
    
    def get_save_dict(self):
        save_dict = super().get_save_dict()
        save_dict['T'] = self.T
        save_dict['gamma'] = self.gamma
        save_dict['alpha'] = self.alpha
        return save_dict
    
    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.T = checkpoint['T']
        self.gamma = checkpoint['gamma']
        self.alpha = checkpoint['alpha']
        return checkpoint
    
    @staticmethod
    def get_argument():
        return [
            SSL_Argument('T', default=1, type=float, help='Temperature for distillation smoothing'),
            SSL_Argument('gamma', default=1, type=float, help='weight for classification'),
            SSL_Argument('alpha', default=1, type=float, help='weight balance for KD')
        ]

