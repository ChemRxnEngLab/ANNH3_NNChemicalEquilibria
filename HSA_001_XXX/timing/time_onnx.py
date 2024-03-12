import onnx
import onnxruntime as ort

onnx_model = onnx.load("models/onnx/NH3_net_LU.onnx")

onnx.checker.check_model(onnx_model)

ort_session = ort.InferenceSession("models/onnx/NH3_net_LU.onnx")
