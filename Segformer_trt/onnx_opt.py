from itertools import count
from platform import node
import onnx
import onnxruntime
# import onnx_graphsurgeon as gs
import numpy as np
from onnx.numpy_helper import from_array

 

def updateNode():
    """
    对固定输入为1024维度的onnx模型进行转换
    """
  
    import sys
    onnx_path = 'mit_b2_1024.onnx'
    model = onnx.load(onnx_path)
    nodes = model.graph.node
    del_list = []
    layernorm_count=1

    new_node_list = []
    
    
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
            print("index_Slice_2259",i)
            
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
           
            print(slice_new.input)

            concat_new = onnx.helper.make_node(
                'Concat',
                inputs=[nodes[i].output[0],slice_new.output[0]],
                outputs=['output'],
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

    onnx.save(model,'mit_b2_1024_opt.onnx')

def updateNodeDy():
    """
    对动态输入的onnx模型进行转换
    """
  
    import sys
    onnx_path = 'mit_b2_dy_0618.onnx'
    model = onnx.load(onnx_path)
    nodes = model.graph.node
    del_list = []
    layernorm_count=1

    new_node_list = []
    
    
    for i in range(len(nodes)):
        if '3135' in nodes[i].output :
            t_3135 = nodes[i].attribute[0].t
    
    
    for i in range(len(nodes)):
        if '3127' in nodes[i].output :
            t_3127 = nodes[i].attribute[0].t
            t_3127.dims[0] = t_3135.dims[0]
            t_3127.raw_data = np.array([-1],dtype = np.int64).tobytes()
            
        
        if '3132' in nodes[i].output :
            t_3132 = nodes[i].attribute[0].t
            t_3132.raw_data = np.array([-2],dtype = np.int64).tobytes()

            
            t_3132 = i
    
    for i in range(len(nodes)):
 
        if nodes[i].name=='Slice_2355':
            #添加slice ，step=-2，starts=-2，ends=-9223372036854775808，
            print("index_Slice_2355",nodes[i].output[0])
           
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
                inputs = ['3128','slice_start','slice_ends','3129','3132'],
                outputs = ['slice_out'],
            )
           

            concat_new = onnx.helper.make_node(
                'Concat',
                inputs=[nodes[i].output[0],slice_new.output[0]],
                outputs=['concat_output'],
                axis=0)
            model.graph.node.remove(nodes[i+1])
           
            model.graph.node.insert(i+1, slice_new)
            slice_new_index =i+1
            model.graph.node.remove(nodes[i+2])
            model.graph.node.insert(i+2, concat_new)
            nodes[i+4].input[0] = nodes[i+2].output[0]
            #model.graph.node.remove(nodes[i+3])

            # slice_starts = np.array([0, 0], dtype=np.int64)
            # slice_ends = np.array([3, 10], dtype=np.int64)
            
    model.graph.node.insert(slice_new_index,slice_start)
    model.graph.node.insert(slice_new_index+1,slice_ends)

    for i in range(len(nodes)-1,-1,-1):
        if nodes[i].name =='Reshape_2358':
            model.graph.node.remove(nodes[i])

    onnx.save(model,'mit_b2_dy_0618_opt.onnx')

def visual():

    onnx_path= 'mit_b2_dynamic_opt.onnx'
    model = onnx.load(onnx_path)
    nodes = model.graph.node

    for i in range(len(nodes)):
        if nodes[i].name=='Reshape_31':
            print('index:',i)
            # print(nodes[i-4])
            # print(nodes[i-3])
            print(nodes[i-2])
            print(nodes[i-1])
            print(nodes[i])
            print(nodes[i+1])
            print(nodes[i+2])
            print(nodes[i+3])
            print(nodes[i+4])
            print(nodes[i+5])
            # print(nodes[i+6])
            # print(nodes[i+7])
            # print(nodes[i+8])
            # print(nodes[i+9])
            # print(nodes[i+10])




def dy_onnx_visaul():
    onnx_file = 'mit_b2_dy_0618_opt.onnx'
    model = onnx.load(onnx_file)
    nodes = model.graph.node
    
    for i in range(len(nodes)):
        if nodes[i].name=='Reshape_43':
            id = nodes[i].input[1]
             
        
        # if nodes[i].name=='Reshape_86':
        #     print(nodes[i])
    for i in range(len(nodes)):
        if id in nodes[i].output:
            print(nodes[i])

            t_id = nodes[i].attribute[0].t
            
        
           # tesnor = from_array(np.array([-1,64,-1]).astype(np.int64),'value')
            t_id.raw_data = np.array([-1,64,-1],dtype = np.int64).tobytes()
            print(nodes[i])
            # reshape_out = onnx.helper.make_node(
            #     "Constant",
            #     inputs=[],
            #     outputs=['387'],
            #     value=onnx.helper.make_tensor('values',data_type=onnx.TensorProto.INT64, dims=[3],vals=np.array([-1,64,-1]).astype(np.int64),raw=True)

            # ) 

            # model.graph.node.remove(nodes[i])
            # model.graph.node.insert(reshape_out,nodes[i])
           
    
    # model.graph.node.remove(nodes[indx])

    # for i in range(len(nodes)):
    #     if nodes[i].name=='Reshape_35':
    #         nodes[i].input[1].type = [-1,64,-1]
    #         print(nodes[i].input[1])

    onnx.save(model,'mit_b2_dy_0618.onnx')


def layerNormGen():
    """
    onnx合并LayerNorm
    """
    onnx_file = 'mit_b2_dy_0618_opt.onnx'
    model = onnx.load(onnx_file)
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
            #del nodes[del_list[i]]
        
    # for i in range(len(nodes)-1,-1,-1):
    #     if nodes[i].name.split('_')[0]=='Constant':
    #         nodes.remove(nodes[i])

    onnx.save(model, 'mit_b2_dy_0623_opt.onnx')
def visualLayerNorm():
    
    onnx_file = 'mit_b2_dy_0618_opt.onnx'
    model = onnx.load(onnx_file)
    nodes = model.graph.node

    del_list = []
    layernorm_count=1

    new_node_list = []
    count = 0
    for i in range(len(nodes)-1):
        if (nodes[i].op_type=='ReduceMean' and \
            nodes[i+1].op_type=='Sub' and \
            nodes[i+2].op_type=='Cast' and \
            nodes[i+3].op_type=='Pow' and\
            nodes[i+4].op_type=='ReduceMean' and\
            nodes[i+6].op_type=='Add' and\
            nodes[i+7].op_type=='Sqrt' and \
            nodes[i+8].op_type=='Div' and \
            nodes[i+9].op_type=='Mul' and \
            nodes[i+10].op_type=='Add'):
            count+=1
           
    
    for i in range(len(nodes)):
        if nodes[i].name =="ReduceMean_142":
            print(nodes[i],i)
            print(nodes[i+1],i+1)
            print(nodes[i+2],i+2)
            print(nodes[i+3],i+3)
            print(nodes[i+4],i+4)
            print(nodes[i+5],i+5)
            print(nodes[i+6],i+6)
            print(nodes[i+7],i+7)
            print(nodes[i+8],i+8)
            print(nodes[i+9],i+9)
            print(nodes[i+10],i+10)
            
            
        # if nodes[i].name =="Cast_144":
        #     print(nodes[i],i)
    print(count)


def fusionGelu():
    onnx_file = 'mit_b2_dy_0623_opt.onnx'
    model = onnx.load(onnx_file)
    nodes = model.graph.node
    
    gelu_count,del_list=0,[]
    for i in range(len(nodes)-6):
        if nodes[i].name=='Div_175':
            print(nodes[i-2],i-2)
            print(nodes[i-1],i-1)
            print(nodes[i],i)
            print(nodes[i+1],i+1)
            print(nodes[i+2],i+2)
            print(nodes[i+3],i+3)
            print(nodes[i+4],i+4)
            print(nodes[i+5],i+5)
            print(nodes[i+6],i+6)
    
    for i in range(len(nodes)-6):
        if nodes[i].op_type=='Div' and \
            nodes[i+1].op_type=='Erf' and \
            nodes[i+3].op_type=='Add' and \
            nodes[i+4].op_type=='Mul' and \
            nodes[i+6].op_type=='Mul':
            
            new_node = onnx.helper.make_node(
                'GELU',
                name = 'GELU_'+str(gelu_count),
                inputs = [nodes[i].input[0]],
                outputs = [nodes[i+6].output[0]]
            )
            gelu_count+=1
            nodes.remove(nodes[i])
            nodes.insert(i,new_node)
            
            del_list.append(i-1)
            del_list.append(i+1)
            del_list.append(i+2) 
            del_list.append(i+3)
            del_list.append(i+4)
            del_list.append(i+5)
            del_list.append(i+6)
            
    for i in range(len(del_list)-1,-1,-1):
        nodes.remove(nodes[del_list[i]])
    onnx.save(model,'mit_b2_dy_0623_gelu_opt.onnx')


if __name__=='__main__':
    #updateNode()
    #updateNodeDy()
    layerNormGen()
 