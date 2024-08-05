import numpy as np
import torch
import torch.nn.functional as F
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument

def KD_Loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

@ALGORITHMS.register('kd')
class KD(AlgorithmBase):
    """
        KD algorithm

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
    def __init__(self, args, net_builder, tb_log=None, logger=None, teacher_net_builder=None):
        super().__init__(args, net_builder, tb_log, logger, teacher_net_builder)
        self.init(T=args.T, gamma=args.gamma, alpha=args.alpha)


    def init(self, T, gamma=1, alpha=1, beta=1):
        self.T = T
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    

    def train_step(self, x_lb, y_lb, x_ulb_w = None):
        with self.amp_cm():

            outs_x_lb = self.model(x_lb)
            logits_x_lb = outs_x_lb['logits']

            if x_ulb_w is not None:
                outs_x_unlb = self.model(x_ulb_w)
                logits_x_unlb = outs_x_unlb['logits']

            self.teacher_model.eval()
            with torch.no_grad():
                outs_x_lb_teacher = self.teacher_model(x_lb)
                logits_x_lb_teacher = outs_x_lb_teacher['logits']

                if x_ulb_w is not None:
                    outs_x_unlb_teacher = self.teacher_model(x_ulb_w)
                    logits_x_unlb_teacher = outs_x_unlb_teacher['logits']

            feat_dict = {'x_lb': outs_x_lb['feat'][-1]}
            if x_ulb_w is not None:
                feat_dict['x_unlb'] = outs_x_unlb['feat'][-1]

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            kd_loss = KD_Loss(logits_x_lb, logits_x_lb_teacher, self.T)
            if x_ulb_w is not None:
                unsup_kd_loss = KD_Loss(logits_x_unlb, logits_x_unlb_teacher, self.T)
            else:
                unsup_kd_loss = 0
            total_loss = sup_loss * self.gamma + kd_loss * self.alpha + unsup_kd_loss * self.lambda_u

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         kd_loss=kd_loss.item(), 
                                         unsup_kd_loss=unsup_kd_loss.item() if x_ulb_w is not None else 0,)
        return out_dict, log_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        save_dict["T"] = self.T
        save_dict["gamma"] = self.gamma
        save_dict["alpha"] = self.alpha
        return save_dict
        


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.T = checkpoint["T"]
        self.gamma = checkpoint["gamma"]
        self.alpha = checkpoint["alpha"]
        return checkpoint


    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--T', default=1, type=float, help='Temperature for distillation smoothing'),
            SSL_Argument('--gamma', default=1, type=float, help='weight for classification'),
            SSL_Argument('--alpha', default=1, type=float, help='weight balance for KD'),
        ]