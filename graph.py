import onnx

model_path = "quant/qwen3_int8.onnx"
model = onnx.load(model_path)
graph = model.graph

# 查找输出节点名
output_name = graph.output[0].name
print(f"输出节点名称: {output_name}")

# 逆向遍历最后 10 个算子
print("\n输出节点前的最后 10 个算子:")
for i in range(1, 11):
    node = graph.node[-i]
    print(f"算子类型: {node.op_type: <15} | 节点名称: {node.name}")

# 在 export_stu.py 末尾添加
import onnxruntime as ort
import numpy as np

test_sess = ort.InferenceSession(model_path)
test_out = test_sess.run(None, {
    "input_ids": np.ones((1, 5), dtype=np.int64),
    "attention_mask": np.ones((1, 5), dtype=np.int64)
})
print("--- 验证 FP32 模型输出 ---")
print(f"Logits sum: {test_out[0].sum()}") 
# 如果这里打印的是 0.0 或 NaN，说明导出补丁依然有问题！

import onnx
# 检查模型结构是否合法
onnx.checker.check_model(model_path)