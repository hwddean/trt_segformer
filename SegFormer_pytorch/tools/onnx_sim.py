from onnxsim import simplify
import onnx
input_path = '../weights/mit_b2_dynamic.onnx'
out_path = '../weights/mit_b2_sim.onnx'

onnx_model = onnx.load(input_path)
model_simp, check = simplify(onnx_model,dynamic_input_shape=True,input_shapes={'input':[1,3,256,256]})
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, out_path)
print('finished exporting onnx')
 