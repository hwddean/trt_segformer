from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import mmcv
import cv2
import os
import numpy as np
import torch
from mmseg.models import build_segmentor
from tools.pytorch2onnx import _convert_batchnorm,_demo_mm_inputs
from mmcv.runner import load_checkpoint
import torch.nn as nn
import random


def trans_img(img_path,size):
    print(img_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img,(size,size))
    mean,std = [103.53,116.28,123.67],[57.375,57.12,58.395]
    final = np.transpose((img-mean)/std,(2,0,1))
    final = np.expand_dims(final,axis=0)
    
    return torch.Tensor(final)

def main():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file',type=str)
    parser.add_argument('--config', help='Config file',type=str)
    parser.add_argument('--checkpoint', help='Checkpoint file',type=str)
    parser.add_argument('--batchsize',help='Input batchsize',type=int)
    parser.add_argument('--size',type=int)
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    
    
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    segmentor = build_segmentor(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    # convert SyncBN to BN
    segmentor = _convert_batchnorm(segmentor)

    if args.checkpoint:
        load_checkpoint(segmentor, args.checkpoint, map_location='cuda')

    segmentor.cuda().eval()
   
    
    imgPaths = os.listdir(args.img)
    random.shuffle(imgPaths)
    inputlist = []
    final = trans_img(os.path.join(args.img,imgPaths[0]),args.size)
    for imgP in imgPaths[:min(args.batchsize,len(imgPaths))]:
        inputTensor = trans_img(os.path.join(args.img,imgP),args.size).cuda()
        inputlist.append(inputTensor)
    
    # print(final.shape)
    if isinstance(segmentor.decode_head, nn.ModuleList):
        num_classes = segmentor.decode_head[-1].num_classes
    else:
        num_classes = segmentor.decode_head.num_classes
    mm_inputs = _demo_mm_inputs((1,3,args.size,args.size), num_classes)
    img_metas = mm_inputs.pop('img_metas')
    img_meta_list = [[img_meta] for img_meta in img_metas]
    results=[] 
    for i in range(len(inputlist)):
        result=segmentor([inputlist[i]], img_meta_list, return_loss=False)
        results.append(result[0])
    print(len(img_meta_list))
    print(results[0].shape)
    dict_save = {'imgs':inputlist,'results':results}
    np.save('./demo/batch_{}_size_{}_test.npy'.format(len(inputlist),args.size),dict_save)

if __name__ == '__main__':
    main()
