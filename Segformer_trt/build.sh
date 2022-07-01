#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cd ./LayerNormPlugin
make clean
make
cp LayerNorm.so ..
cd ..

python3 ./graph_opt_dy.py


trtexec --onnx=mit_b2_dy_opt_v2.onnx  --minShapes=input:1x3x256x256 --optShapes=input:8x3x512x512  --maxShapes=input:8x3x1024x1024   --workspace=1024000 --saveEngine=mit_b2_dy_opt_FP32.plan --shapes=input:4x3x512x512 --plugins=./LayerNorm.so --verbose

