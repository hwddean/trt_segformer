/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
 #include "LayerNormPlugin.h"

using namespace nvinfer1;

PluginFieldCollection LayerNormPluginCreator::fc_{};
std::vector<PluginField> LayerNormPluginCreator::attr_;

 

template <unsigned TPB>
__global__ void layerNormKernel(float *pInput, float *pOutput)
{
    const int tx = threadIdx.x, index = blockIdx.x * TPB*2 + threadIdx.x;
   
    __shared__ float temp[TPB];

    float value0 = pInput[index];
    float value1 = pInput[index + TPB];

    temp[tx] = value0 + value1;
    __syncthreads();

    for (int stride = (TPB/2); stride >= 1; stride /= 2)
    {
        if (tx < stride)
        {
            temp[tx] += temp[tx + stride];
        }
        __syncthreads();
    }
    float mean = temp[0] / TPB/2;
    __syncthreads();

    temp[tx] = (value0 - mean) * (value0 - mean) + (value1 - mean) * (value1 - mean);
    __syncthreads();

    for (int stride = (TPB/2); stride >= 1; stride /= 2)
    {
        if (tx < stride)
        {
            temp[tx] += temp[tx + stride];
        }
        __syncthreads();
    }
    float var = temp[0] / TPB/2;

    pOutput[index]       = (value0 - mean) * rsqrtf(var + 6e-6);
    pOutput[index + TPB] = (value1 - mean) * rsqrtf(var + 6e-6);
}
 

int32_t LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
    
    if (inputDesc[0].dims.d[2]<=64)
        layerNormKernel<32><<<nBlock, 32, 0, stream>>>((float *)inputs[0], (float *)outputs[0]);
    
    else if (inputDesc[0].dims.d[2]<=128)
        layerNormKernel<64><<<nBlock, 64, 0, stream>>>((float *)inputs[0], (float *)outputs[0]);

    else if (inputDesc[0].dims.d[2]<=256)
        layerNormKernel<128><<<nBlock, 128, 0, stream>>>((float *)inputs[0], (float *)outputs[0]);
    else if (inputDesc[0].dims.d[2]<=512)
        layerNormKernel<256><<<nBlock, 256, 0, stream>>>((float *)inputs[0], (float *)outputs[0]);
    
    
    return 0;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);
