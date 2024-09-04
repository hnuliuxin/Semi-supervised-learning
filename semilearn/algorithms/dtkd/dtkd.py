import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument
from semilearn.core.criterions import ConsistencyLoss



class DTKDNet(nn.Module):
    def __init__(self, backbone, temprarure, p_cutoff = 0.95):
        super(DTKDNet, self).__init__()
        self.backbone = backbone
        self.p_cutoff = p_cutoff
        self.T = nn.parameter.Parameter(torch.tensor(temprarure, dtype=torch.float32, requires_grad=True))

    def forward(self, x, logits_teacher = None, **kwargs):
        feats_x = self.backbone(x, only_feat=True)[-2]
        logits_x = self.backbone(x, feat_s=feats_x)

        self.T.data.clamp_(0.1, 10)

        probs_x = F.softmax(logits_teacher.detach() if logits_teacher is not None else logits_x.detach(), dim=1)
        max_probs, _ = torch.max(probs_x, dim=-1)
        mask = max_probs.ge(self.p_cutoff).to(max_probs.dtype)
        pseudo_label = torch.softmax(logits_teacher.detach() if logits_teacher is not None else logits_x.detach() / self.T, dim=1)

        unsup_loss = ConsistencyLoss()(logits_x, pseudo_label, name='kl', mask=mask)

        return_dict = {
            'T' : self.T.item(),
            'unsup_loss': unsup_loss,
            'KD_Loss' : self.KD_Loss(logits_x, logits_teacher) if logits_teacher is not None else 0,
            'logits': logits_x,
            'feat': feats_x
        }
        return return_dict
    
    def KD_Loss(self, logits_student, logits_teacher):
        log_pred_student = F.log_softmax(logits_student / self.T, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.T, dim=1)
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
        loss_kd *= self.T**2
        return loss_kd
    
    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher
        
@ALGORITHMS.register('dtkd')
class DTKD(AlgorithmBase):
    """
        Args:
         - args (`argparse`):
            algorithm arguments
        - net_builder (`callable`):
            network loading function
        - tb_log (`TBLog`):
            tensorboard logger
        - logger (`logging.Logger`):
            logger to use
        - p_cutoff(`float`):
            Confidence threshold for generating pseudo-labels
         - T (`float`):
                Initial Temperature for pseudo-label sharpening and KD
        - gamma (`float`, default=1):
                weight for classification
        - alpha (`float`, default=None):
            weight balance for KD
        - beta (`float`, default=None):
            weight balance for SSL
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None, teacher_net_builder=None):
        super().__init__(args, net_builder, tb_log, logger, teacher_net_builder)
        self.T = args.T
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.beta = args.beta
        self.p_cutoff = args.p_cutoff

        self.model = DTKDNet(self.model, self.T)
        self.ema_model = DTKDNet(self.ema_model, self.T)
        self.ema_model.load_state_dict(self.model.state_dict())
        
    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, x_ulb_w = None):
        with self.amp_cm():
            self.model.train()
            self.teacher_model.eval()
            with torch.no_grad():
                logits_x_lb_teacher = self.teacher_model(x_lb)['logits']
                logits_x_ulb_teacher = self.teacher_model(x_ulb_w)['logits']


            outs_x_lb = self.model(x_lb, logits_x_lb_teacher)
            logits_x_lb = outs_x_lb['logits']

            self.bn_controller.freeze_bn(self.model)
            outs_x_ulb = self.model(x_ulb_w, logits_x_ulb_teacher)
            logits_x_ulb = outs_x_ulb['logits']
            feats_x_ulb = outs_x_ulb['feat'][-1]
            self.bn_controller.unfreeze_bn(self.model)

            
                
            cls_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            # mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb)

            # pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
            #                               logits=logits_x_ulb, 
            #                               use_hard_label=False,
            #                               softmax=True)
            
            # # print(logits_x_ulb[0])
            # # print(pseudo_label[0])
            # #TODO 目前无监督的损失还是用的固定的T，需要加到model里面
            # unsup_loss = self.consistency_loss(logits_x_ulb, pseudo_label, 'ce', mask=mask)
            unsup_loss = outs_x_ulb['unsup_loss']

            kd_loss = outs_x_lb['KD_Loss'] + outs_x_ulb['KD_Loss']

            total_loss = cls_loss * self.gamma + unsup_loss * self.beta + kd_loss * self.alpha


        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(cls_loss=cls_loss.item() * self.gamma, 
                                         unsup_loss=unsup_loss.item() * self.beta, 
                                         kd_loss=kd_loss.item() * self.alpha,
                                         Temperature=outs_x_lb['T']
                                         )
        return out_dict, log_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        save_dict.update({'gamma': self.gamma, 'alpha': self.alpha})
        return save_dict
    
    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.gamma = checkpoint['gamma']
        self.alpha = checkpoint['alpha']
        return checkpoint
    
    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--T', float, default=0.1, help='Temperature for pseudo-label sharpening and KD'),
            SSL_Argument('--gamma', float, default=1.0, help='weight for classification'),
            SSL_Argument('--alpha', float, default=1.0, help='weight balance for KD'),
            SSL_Argument('--beta', float, default=1.0, help='weight balance for SSL'),
            SSL_Argument('--p_cutoff', float, default=0.95, help='Confidence threshold for generating pseudo-labels')
        ]