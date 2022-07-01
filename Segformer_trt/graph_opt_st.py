"""
Segformer 动态onnx模型节点融合
"""
from itertools import count
from platform import node
import onnx
import onnxruntime
# import onnx_graphsurgeon as gs
import numpy as np
from onnx.numpy_helper import from_array

 

def updateNode(model):
    """
    对固定输入为1024维度的onnx模型进行转换
    """
    
    nodes = model.graph.node
    
    for i in range(len(nodes)):
        if '2991' in nodes[i].output :
            t_2991 = nodes[i].attribute[0].t
    
    
    for i in range(len(nodes)):
        if '2983' in nodes[i].output :
            #修改节点参数
            t_2983 = nodes[i].attribute[0].t
            t_2983.dims[0] = t_2991.dims[0]
            t_2983.raw_data = np.array([-1],dtype = np.int64).tobytes()
         
        if '2988' in nodes[i].output :
            t_2988 = nodes[i].attribute[0].t
            t_2988.raw_data = np.array([-2],dtype = np.int64).tobytes()

            
            i_2988 = i
    
    for i in range(len(nodes)):
        if nodes[i].name=='Slice_2259':
            #添加slice ，step=-2，starts=-2，ends=-9223372036854775808，
            # print("index_Slice_2259",i)
            
            slice_start = onnx.helper.make_node(
                "Constant",
                inputs=[],
                outputs=['slice_start'],
                value=onnx.helper.make_tensor('values',data_type=onnx.TensorProto.INT64, dims=[1],vals=np.array([-2]).astype(np.int64).tobytes(),raw=True)
            ) 
    
            slice_ends = onnx.helper.make_node(
                "Constant",
                inputs=[],
                outputs=['slice_ends'],
                value=onnx.helper.make_tensor('values',data_type=onnx.TensorProto.INT64, dims=[1],vals=np.array([-922337203685477580]).astype(np.int64).tobytes(),raw=True)
            ) 
           
            slice_new = onnx.helper.make_node(
                'Slice',
                name = 'Slice_new',
                inputs = ['2984','slice_start','slice_ends','2985','2988'],
                outputs = ['slice_out'],
            )
           
            # print(slice_new.input)

            concat_new = onnx.helper.make_node(
                'Concat',
                inputs=[nodes[i].output[0],slice_new.output[0]],
                outputs=['concat_out'],
                axis=0)
            model.graph.node.remove(nodes[i+1])
            model.graph.node.insert(i+1, slice_new)
            slice_new_index =i+1
            model.graph.node.remove(nodes[i+2])
            model.graph.node.insert(i+2, concat_new)
            nodes[i+4].input[0] = nodes[i+2].output[0]
            
    model.graph.node.insert(slice_new_index,slice_start)
    model.graph.node.insert(slice_new_index+1,slice_ends)

    for i in range(len(nodes)-1,-1,-1):
        if nodes[i].name =='Reshape_2262':
            model.graph.node.remove(nodes[i])

    return model
 

def layerNormGen(model):
    """
    onnx合并LayerNorm
    """
     
    nodes = model.graph.node

    del_list = []
    layernorm_count=1

    new_node_list = []
    for i in range(len(nodes)-1):
        if (nodes[i].op_type=='ReduceMean' and \
            nodes[i+1].op_type=='Sub' and \
            nodes[i+2].op_type=='Cast' and\
            nodes[i+3].op_type=='Pow' and\
            nodes[i+4].op_type=='ReduceMean' and\
            nodes[i+6].op_type=='Add' and\
            nodes[i+7].op_type=='Sqrt' and \
            nodes[i+8].op_type=='Div'):
            

                # nodes[i].output[0] = nodes[i+10].output[0]
                # nodes[i].op_type = 'LayerNorm'
                # nodes[i].name ='LayerNorm_'+str(layernorm_count)

            new_node = onnx.helper.make_node(
                'LayerNorm',
                name = 'LayerNorm_'+str(layernorm_count),
                inputs = [nodes[i].input[0]],
                outputs = [nodes[i+8].output[0]]
            )
            layernorm_count+=1
            nodes.remove(nodes[i])
            nodes.insert(i,new_node)
            new_node_list.append(new_node)
                
            del_list.append(i+1)
            del_list.append(i+2) 
            del_list.append(i+3)
            del_list.append(i+4)
            del_list.append(i+5)
            del_list.append(i+6)
            del_list.append(i+7)
            del_list.append(i+8)
          
            
        
    for i in range(len(del_list)-1,-1,-1):
        nodes.remove(nodes[del_list[i]])
      
    
    return model

if __name__=='__main__':
    onnx_path = 'mit_b2_1024.onnx'
    save_onnx_path = 'mit_b2_1024_opt.onnx'
    model = onnx.load(onnx_path)
    model = updateNode(model)
    model = layerNormGen(model)
    onnx.save(model,save_onnx_path)
    