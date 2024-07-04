import torch
from semilearn.core.algorithmbase import AlgorithmBase

class DistillerBase(AlgorithmBase):
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
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
    """
    def __init__(self, args, net_builder, teacher_net_builder,  tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # teacher model init
        self.teacher_net_builder = teacher_net_builder
        self.teacher_model = self.teacher_net_builder(num_classes=self.num_classes, pretrained=self.args.use_pretrain, pretrained_path=self.args.pretrain_path)
    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        print("DistillerBase train_step")
    