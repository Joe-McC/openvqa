# --------------------------------------------------------
# OpenVQA
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from openvqa.core.path_cfgs import PATH
import os, torch, random
import numpy as np
from types import MethodType


class BaseCfgs(PATH):
    def __init__(self):
        super(BaseCfgs, self).__init__()

        # Set Devices
        # If use multi-gpu training, you can set e.g.'0, 1, 2' instead
        self.GPU = '0'

        # Set Seed For CPU And GPUs
        self.SEED = random.randint(0, 9999999)

        # -------------------------
        # ---- Version Control ----
        # -------------------------

        # You can set a name to start new training
        self.VERSION = str(self.SEED)

        # Use checkpoint to resume training
        self.RESUME = False

        # Resume training version or testing version
        self.CKPT_VERSION = self.VERSION

        # Resume training epoch or testing epoch
        self.CKPT_EPOCH = 0

        # if set 'CKPT_PATH', -> 'CKPT_VERSION' and 'CKPT_EPOCH' will not work any more
        self.CKPT_PATH = None

        # Print loss every iteration
        self.VERBOSE = True


        # ------------------------------
        # ---- Data Provider Params ----
        # ------------------------------

        self.MODEL = 'mcan_small'

        self.DATASET = 'vqa'

        # Run as 'train' 'val' or 'test'
        self.RUN_MODE = 'train'

        # Set True to evaluate offline when an epoch finished
        # (only work when train with 'train' split)
        self.EVAL_EVERY_EPOCH = True

        # Set True to save the prediction vector
        # (use in ensemble)
        self.TEST_SAVE_PRED = False


        # A external method to set train split
        # will override the SPLIT['train']
        self.TRAIN_SPLIT = 'train'

        # Set True to use pretrained GloVe word embedding
        # (GloVe: spaCy https://spacy.io/)
        self.USE_GLOVE = True

        # Word embedding matrix size
        # (token size x WORD_EMBED_SIZE)
        self.WORD_EMBED_SIZE = 300

        # Default training batch size: 64
        self.BATCH_SIZE = 64

        # Multi-thread I/O
        self.NUM_WORKERS = 8

        # Use pin memory
        # (Warning: pin memory can accelerate GPU loading but may
        # increase the CPU memory usage when NUM_WORKS is big)
        self.PIN_MEM = True

        # Large model can not training with batch size 64
        # Gradient accumulate can split batch to reduce gpu memory usage
        # (Warning: BATCH_SIZE should be divided by GRAD_ACCU_STEPS)
        self.GRAD_ACCU_STEPS = 1


        # --------------------------
        # ---- Optimizer Params ----
        # --------------------------

        # The base learning rate
        self.LR_BASE = 0.0001

        # Learning rate decay ratio
        self.LR_DECAY_R = 0.2

        # Learning rate decay at {x, y, z...} epoch
        self.LR_DECAY_LIST = [10, 12]

        # Warmup epoch lr*{1/(n+1), 2/(n+1), ... , n/(n+1)}
        self.WARMUP_EPOCH = 3

        # Max training epoch
        self.MAX_EPOCH = 13

        # Gradient clip
        # (default: -1 means not using)
        self.GRAD_NORM_CLIP = -1

        # Adam optimizer betas and eps
        self.OPT_BETAS_0 = 0.9
        self.OPT_BETAS_1 = 0.98
        self.OPT_EPS = 1e-9


    def parse_to_dict(self, args):
        args_dict = {}
        for arg in dir(args):
            if not arg.startswith('_') and not isinstance(getattr(args, arg), MethodType):
                if getattr(args, arg) is not None:
                    args_dict[arg] = getattr(args, arg)

        return args_dict


    def add_args(self, args_dict):
        for arg in args_dict:
            setattr(self, arg, args_dict[arg])


    def proc(self):
        assert self.RUN_MODE in ['train', 'val', 'test']

        # ------------ Devices setup
        os.environ['CUDA_VISIBLE_DEVICES'] = self.GPU
        self.N_GPU = len(self.GPU.split(','))
        self.DEVICES = [_ for _ in range(self.N_GPU)]
        torch.set_num_threads(2)


        # ------------ Path check
        self.check_path(self.DATASET)


        # ------------ Model setup
        self.MODEL_USE = self.MODEL.split('_')[0]


        # ------------ Seed setup
        # fix pytorch seed
        torch.manual_seed(self.SEED)
        if self.N_GPU < 2:
            torch.cuda.manual_seed(self.SEED)
        else:
            torch.cuda.manual_seed_all(self.SEED)
        torch.backends.cudnn.deterministic = True

        # fix numpy seed
        np.random.seed(self.SEED)

        # fix random seed
        random.seed(self.SEED)

        if self.CKPT_PATH is not None:
            print('Warning: you are now using CKPT_PATH args, '
                  'CKPT_VERSION and CKPT_EPOCH will not work')
            self.CKPT_VERSION = self.CKPT_PATH.split('/')[-1] + '_' + str(random.randint(0, 9999999))


        # ------------ Split setup
        # if not training or train split include val dataset
        # will not trigger the EVAL_EVERY_EPOCH
        self.SPLIT = self.SPLITS[self.DATASET]
        self.SPLIT['train'] = self.TRAIN_SPLIT
        if self.SPLIT['val'] in self.SPLIT['train'].split('+') or self.RUN_MODE not in ['train']:
            self.EVAL_EVERY_EPOCH = False

        if self.RUN_MODE not in ['test']:
            self.TEST_SAVE_PRED = False

        # ------------ Feature setup
        self.FEATURE = self.FEATURES[self.DATASET]


        # ------------ Gradient accumulate setup
        assert self.BATCH_SIZE % self.GRAD_ACCU_STEPS == 0
        self.SUB_BATCH_SIZE = int(self.BATCH_SIZE / self.GRAD_ACCU_STEPS)

        # Set small eval batch size will reduce gpu memory usage
        self.EVAL_BATCH_SIZE = int(self.SUB_BATCH_SIZE / 2)



    def __str__(self):
        for attr in dir(self):
            if not attr.startswith('__') and not isinstance(getattr(self, attr), MethodType):
                print('{ %-17s }->' % attr, getattr(self, attr))

        return ''

#
#
# if __name__ == '__main__':
#     __C = Cfgs()
#     __C.proc()




