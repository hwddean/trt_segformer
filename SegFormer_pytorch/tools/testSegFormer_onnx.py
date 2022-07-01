#!/usr/bin/python

import cv2
import numpy as np
import onnx
import onnxruntime as rt
import os
import sys
import random
import time
from glob import glob
from argparse import ArgumentParser

# segFormerPlanFile = 'mit_b2_FP32.plan'
onnx_path = '/app/weights/mit_b2_dy_0618_opt.onnx'
img_path = '/app/demo/demo.png'
save = '/app/demo/demo_result'
segplanFilePath ='/app/demo/'
 

def transfrom_img(img_path,size):
    img=cv2.imread(img_path)
    img = cv2.resize(img,(size,size))
    mean,std = [123.675, 116.28, 103.53], [58.395, 57.12, 57.375]
    final=np.transpose((img-mean)/std,(2,0,1))
    final=np.expand_dims(final,axis=0)
    return final
 
 
 
#onnx推理
def onnx_result(args):
    imgPaths = os.listdir(args.img)
    random.shuffle(imgPaths)
    inputs = transfrom_img(img_path,args.size).astype(np.float32)
    # for i in imgPaths[1:min(args.batchsize,len(imgPaths))]:
         
    #     img = transfrom_img(os.path.join(args.img,i),args.size)
    #     img =img.astype(np.float32)
    #     inputs = np.concatenate((inputs,img),axis=0)
    
    sess = rt.InferenceSession(onnx_path,providers=['CPUExecutionProvider'])

    
    print(inputs.shape)
    t1 = time.time()
    onnx_result = sess.run(None, {'input': inputs})
    
    t2=time.time()
    
    print(onnx_result[0].shape)
    print("Total time:{}s".format(t2-t1))

    np.save(save,onnx_result[0])

 

if __name__=='__main__':

    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file',type=str)

    parser.add_argument('--batchsize',help='Input batchsize',type=int)
    parser.add_argument('--size',type=int)

    args = parser.parse_args()
    
    onnx_result(args)
