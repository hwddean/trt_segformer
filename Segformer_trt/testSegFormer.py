#!/usr/bin/python
"""
测试最终优化的trt模型代码
"""

import cv2
from cv2 import cuda_Event
import numpy as np
import os
import sys
import ctypes
import time
import tensorrt as trt
from cuda import cudart
from glob import glob

# segFormerPlanFile = 'mit_b2_FP32.plan'
imgPath = '/workspace/Segformer_trt/demo.png'
targetPath = '/workspace/Segformer_trt/demo.npz.npy'
segplanFilePath ='/workspace/Segformer_trt/mit_b2_dy_opt_FP32.plan'
segTestOutTable = '/workspace/Segformer_trt/evaluate.txt'
savePreOut = '/workspace/Segformer_trt/preOut.jpg'
saveTarget = '/workspace/Segformer_trt/target.jpg'
soFileList = glob("./*.so")
 

MAX_ITER = 100

def transfrom_img(imgPath):
    img=cv2.imread(imgPath)
    img = cv2.resize(img,(1024,1024))
    mean,std= [123.675, 116.28, 103.53], [58.395, 57.12, 57.375]
    final=np.transpose((img-mean)/std,(2,0,1))
    final=np.expand_dims(final,axis=0)
    return final
 
def check(a,b,weak=False,epsilon = 5000):
    if weak:
        res = np.all(np.abs(a - b) < epsilon)
    
    else:
        res=np.all(a==b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.median(np.abs(a - b) )
    diff2 = np.average(np.abs(a - b))

    return res,diff0,diff1,diff2

def saveOutImg(a,savtePath):
    m = a.astype(np.float)
   
    m = ((m-np.min(m))/(np.max(m)-np.min(m))*255).astype(np.uint8)
    m = np.transpose(m.reshape(m.shape[1:]),(1,2,0))
    cv2.imwrite(savtePath,m)


def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]



def infer_trt():
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger,'')
    
    if len(soFileList) > 0:
        print("Find Plugin %s!"%soFileList)
    else:
        print("No Plugin!")
    for soFile in soFileList:
        ctypes.cdll.LoadLibrary(soFile)

    target = np.load(targetPath)
    with open(segTestOutTable,'w') as f:
        with open(segplanFilePath,'rb') as segPlane:
            engine = trt.Runtime(logger).deserialize_cuda_engine(segPlane.read())
            if engine is None:
                print("Failed loading %s"%segplanFilePath)
                exit()
            print("Succeeded loading %s"%segplanFilePath)

            nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
            nOutput = engine.num_bindings - nInput
            context = engine.create_execution_context()

            inputImg = transfrom_img(imgPath)

            context.set_binding_shape(0,inputImg.shape)

            bufferH=[]
            bufferH.append(inputImg.astype(np.float32).reshape(-1))
            for i in range(nInput,nInput+nOutput):
                bufferH.append(np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))))
            

            bufferD = []

            for i in range(nInput+nOutput):
                bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

            for i in range(nInput):
                cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

            context.execute_v2(bufferD)

            for i in range(nInput,nInput+nOutput):
                cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

            # warm up
            for i in range(20):
                context.execute_v2(bufferD)

            #test inference time
            #t0=time.time()
            start =  checkCudaErrors(cudart.cudaEventCreate())
            end =  checkCudaErrors(cudart.cudaEventCreate())

            checkCudaErrors(cudart.cudaEventRecord(start, 0))
            
            for i in range(MAX_ITER):
                context.execute_v2(bufferD)
            checkCudaErrors(cudart.cudaEventRecord(end,0))
            checkCudaErrors(cudart.cudaEventSynchronize(end))
            msecTotal =  checkCudaErrors(cudart.cudaEventElapsedTime(start, end))
            print("Total elapsed time = {} ms over {} iterations".format(msecTotal, MAX_ITER))
            

            timePerInference = (msecTotal)/MAX_ITER #

            print("Per Time:{:.6}ms".format(timePerInference))

            
            indexOut = engine.get_binding_index('output')#3016
            check0 = check(bufferH[indexOut],target.astype(np.int32),True)

            string = "%4d,%4d,%9.3e,%9.3e,%9.3e,%9.3e"%(1,
                                                  3,
                                                  timePerInference,
                                                  check0[1],
                                                  check0[2],
                                                  check0[3])
            print(string )
            
            f.write(string + "\n")
            saveOutImg(bufferH[indexOut],savePreOut)
            #saveOutImg(target,saveTarget)

            for i in range(nInput + nOutput):                
                cudart.cudaFree(bufferD[i])







if __name__=='__main__':
    infer_trt()
