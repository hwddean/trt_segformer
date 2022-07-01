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
 
python3 graph_opt_st.py

trtexec --onnx=mit_b2_1024_opt.onnx   --workspace=1024 --saveEngine=mit_b2_1024-FP32.plan --plugins=./LayerNorm.so --verbose

