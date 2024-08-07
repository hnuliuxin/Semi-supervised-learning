import torch
import torch.nn as nn
import torch.nn.functional as F
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

@ALGORITHMS.register('dkd')
class dkd(AlgorithmBase):
    """Decoupled Knowledge Distillation(CVPR 2022)"""
    def __init__(self, args, net_builder, tb_log=None, logger=None, teacher_net_builder=None):
        super().__init__(args, net_builder, tb_log, logger, teacher_net_builder)
        self.T = args.T
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.beta = args.beta

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

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            kd_loss = dkd_loss(logits_x_lb, logits_x_lb_teacher, y_lb, self.alpha, self.beta, self.T)
            total_loss = sup_loss * self.gamma + kd_loss
        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         kd_loss=kd_loss.item())
        return out_dict, log_dict
            

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        save_dict['T'] = self.T
        save_dict['gamma'] = self.gamma
        save_dict['alpha'] = self.alpha
        save_dict['beta'] = self.beta
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.T = checkpoint['T']
        self.gamma = checkpoint['gamma']
        self.alpha = checkpoint['alpha']
        self.beta = checkpoint['beta']
        return checkpoint

    def get_argument():
        return [
            SSL_Argument("--T", default=0.1, type=float, help="Temperature for sharpening and KD"),
            SSL_Argument("--gamma", default=1, type=float, help="weight for classification"),
            SSL_Argument("--alpha", default=1, type=float, help="weight balance for tckd_loss"),
            SSL_Argument("--beta", default=1, type=float, help="weight balance for nckd_loss"),
        ]