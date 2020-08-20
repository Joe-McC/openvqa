# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from openvqa.core.base_cfgs import BaseCfgs


class Cfgs(BaseCfgs):
    def __init__(self):
        super(Cfgs, self).__init__()

        self.LAYER = 6
        self.HIDDEN_SIZE = 512
        self.BBOXFEAT_EMB_SIZE = 2048
        self.FF_SIZE = 2048
        self.MULTI_HEAD = 8
        self.DROPOUT_R = 0.1
        self.USE_AUX_FEAT = False
        self.USE_BBOX_FEAT = False
        self.BBOX_NORMALIZE = True
