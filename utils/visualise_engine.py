import os
from argparse import Namespace

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
import torch
import torch.nn as nn
import torch.utils.data as Data
import yaml
from scipy.misc import imread, imresize
from itertools import chain
from openvqa.datasets.dataset_loader import DatasetLoader
from openvqa.models.model_loader import ModelLoader, CfgLoader


@torch.no_grad()
def visualise_engine(__C):
    # Load parameters
    dataset = DatasetLoader(__C).DataSet()
    if __C.CKPT_PATH is not None:
        print('Warning: you are now using CKPT_PATH args, '
              'CKPT_VERSION and CKPT_EPOCH will not work')

        path = __C.CKPT_PATH
    else:
        path = __C.CKPTS_PATH + \
               '/ckpt_' + __C.CKPT_VERSION + \
               '/epoch' + str(__C.CKPT_EPOCH) + '.pkl'

    print('Loading ckpt from: {}'.format(path))
    state_dict = torch.load(path)['state_dict']
    print('Finish!')

    if __C.N_GPU > 1:
        state_dict = ckpt_proc(state_dict)

    # Store the prediction list
    # qid_list = [ques['question_id'] for ques in dataset.ques_list]
    ans_ix_list = []
    pred_list = []

    data_size = dataset.data_size
    token_size = dataset.token_size
    ans_size = dataset.ans_size
    pretrained_emb = dataset.pretrained_emb

    net = ModelLoader(__C).Net(
        __C,
        pretrained_emb,
        token_size,
        ans_size,
        dataset.token_to_ix
    )
    net.cuda()
    net.eval()

    if __C.N_GPU > 1:
        net = nn.DataParallel(net, device_ids=__C.DEVICES)

    net.load_state_dict(state_dict)

    batch_size = 1
    dataloader = Data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=__C.NUM_WORKERS,
        pin_memory=__C.PIN_MEM
    )

    for step, (
            frcn_feat_iter,
            grid_feat_iter,
            bbox_feat_iter,
            ques_ix_iter,
            ans_iter, image_id, question, words, target_ans
    ) in enumerate(dataloader):
        print("\rEvaluation: [step %4d/%4d]" % (
            step,
            int(data_size / __C.EVAL_BATCH_SIZE),
        ), end='          ')

        frcn_feat_iter = frcn_feat_iter.cuda()
        grid_feat_iter = grid_feat_iter.cuda()
        bbox_feat_iter = bbox_feat_iter.cuda()
        ques_ix_iter = ques_ix_iter.cuda()

        pred, img_attention_map, txt_attention_map = net(
            frcn_feat_iter,
            grid_feat_iter,
            bbox_feat_iter,
            ques_ix_iter
        )
        img_attention_map = img_attention_map[:, :, :, 1:]
        txt_attention_map = txt_attention_map[:, :, :, 1:len(words) + 1]
        pred_np = pred.cpu().data.numpy()
        pred_argmax = np.argmax(pred_np, axis=1)
        ans = dataset.ix_to_ans[pred_argmax[0]]

        visualise_img(question['image_filename'][0], question['question'][0], img_attention_map.cpu().data.numpy()[0],
                      ans, target_ans[0])
        visualise_txt(words, txt_attention_map.cpu().data.numpy()[0])



def visualise_img(image_id, question, attention_map, ans, target_ans):
    img_path = os.path.join('/home/mark/openvqa/data/clevr/raw/images/val', image_id)
    original_img = imread(img_path, mode='RGB')
    img = imresize(original_img, (224, 224), interp='bicubic')
    fig = plt.figure()
    ax1 = fig.add_axes((0.1, 0.2, 0.8, 0.7))
    ax1.imshow(original_img)
    ax1.axis('off')
    fig.text(0.5, 0.1, f'{question} \nAnswer: {target_ans} Prediction: {ans}', wrap=True, horizontalalignment='center',
             fontsize=12, fontname='Droid Sans')
    plt.show()

    plt.figure(figsize=(40, 30))
    count = 1
    for i in range(6):
        for j in range(8):
            plt.subplot(6, 8, count)
            attention = skimage.transform.pyramid_expand(attention_map[i][j].reshape(14, 14), upscale=16, sigma=10)
            plt.imshow(img)
            plt.imshow(attention, alpha=0.6)
            plt.text(0, 1, f'[IMG]-Layer{i}-Head{j}', backgroundcolor='white', fontsize=26)
            plt.text(0, 1, f'[IMG]-Layer{i}-Head{j}', color='black', fontsize=26)
            plt.set_cmap(cm.Greys_r)
            plt.axis('off')
            count += 1
    plt.show()


def visualise_txt(question, attention_map):
    fig, axs = plt.subplots(3, 2, figsize=(30, 20))
    plt.set_cmap(cm.Oranges)
    layer = 0

    for i in range(3):
        for j in range(2):
            axs[i][j].imshow(attention_map[layer])
            axs[i][j].set_title(f'[TXT] Layer {layer}', fontsize=18)
            axs[i][j].set_xticks(np.arange(len(list(chain(*question)))))
            head_labels = [f"Head{x}" for x in range(8)]
            axs[i][j].set_yticks(np.arange(len(head_labels)))
            axs[i][j].set_xticklabels(list(chain(*question)))
            axs[i][j].set_yticklabels(head_labels, fontsize=16)
            axs[i][j].tick_params(axis='x', which='major')
            plt.setp(axs[i][j].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=16)
            layer += 1

    plt.show()


def ckpt_proc(state_dict):
    state_dict_new = {}
    for key in state_dict:
        state_dict_new['module.' + key] = state_dict[key]
        # state_dict.pop(key)

    return state_dict_new


if __name__ == '__main__':
    args = Namespace(BATCH_SIZE=None, CKPT_EPOCH=16, CKPT_PATH=None, CKPT_VERSION='5825257', DATASET='clevr',
                     EVAL_EVERY_EPOCH=None, GPU=None, GRAD_ACCU_STEPS=None, MODEL='vqabert', NUM_WORKERS=None,
                     PIN_MEM=None,
                     RESUME=None, RUN_MODE='visualise', SEED=None, TEST_SAVE_PRED=None, TRAIN_SPLIT=None, VERBOSE=None,
                     VERSION=None, WANDB='False')

    cfg_file = "configs/{}/{}.yml".format(args.DATASET, args.MODEL)
    with open(cfg_file, 'r') as f:
        yaml_dict = yaml.load(f)

    __C = CfgLoader(yaml_dict['MODEL_USE']).load()
    args = __C.str_to_bool(args)
    args_dict = __C.parse_to_dict(args)

    args_dict = {**yaml_dict, **args_dict}
    __C.add_args(args_dict)
    __C.proc()

    print('Hyper Parameters:')
    print(__C)
    visualise_engine(__C)
